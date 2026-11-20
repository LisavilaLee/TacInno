"""Train the TacInno tokenizer.

    python tools/train_tokenizer.py configs/tokenizer/tacinno-tokenizer_univtac-6task.py [--work-dir DIR]
        [--cfg-options train_cfg.max_steps=200 ...]

Single GPU. The run resumes automatically from ``<work_dir>/ckpt_latest.pt`` (model, optimizer, scheduler, step and all
RNG states, so a resumed run continues exactly like an uninterrupted one). Outputs in ``work_dir``:
``config.py`` (resolved config), ``action_stats.pt`` (action standardisation, needed for export), ``log.jsonl``
(training losses every ``log_interval`` steps and the code-shuffle validation every ``val_interval`` steps),
``ckpt_latest.pt`` and ``model_<step>.pt`` (weights, kept every ``keep_interval`` steps and at the end).
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time

import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from tacinno.config import Config, DictAction  # noqa: E402
from tacinno.datasets.tokenizer_dataset import UniVTACTokenizerData  # noqa: E402,F401  (registers the dataset)
from tacinno.models.tokenizer import TacInnoTokenizer  # noqa: E402,F401  (registers the model)
from tacinno.models.tokenizer.training import (StopFlag, cosine_lambda, load_action_stats,  # noqa: E402
                                               param_groups, perplexity, save_checkpoint, try_resume,
                                               within_group_perm)
from tacinno.registry import DATASETS, MODELS  # noqa: E402

ACTION_STATS = 'action_stats.pt'


def parse_args():
    ap = argparse.ArgumentParser(description='Train the TacInno tokenizer')
    ap.add_argument('config')
    ap.add_argument('--work-dir', help='output directory (default: work_dirs/<config name>)')
    ap.add_argument('--cfg-options', nargs='+', action=DictAction, help='override config entries, key=value')
    ap.add_argument('--stop-after', type=int, default=0, help='debug: save a checkpoint and exit after this step')
    return ap.parse_args()


def build_data(cfg, model, split: str, device: str):
    ds = dict(cfg[f'{split}_dataset'])
    if ds.get('with_vision') is None:
        ds['with_vision'] = model.uses_images
    ds.setdefault('device', device)
    return DATASETS.build(ds)


@torch.no_grad()
def validate(model, data, idx, amean, astd, chunk: int = 256):
    out = []
    for i in range(0, len(idx), chunk):
        b = idx[i:i + chunk]
        X = data.fields_at(b, model.offsets)
        vis = data.vis_at(b, model.offsets) if model.uses_images else None
        act = (data.actions(b, model.horizon) - amean) / astd
        perm = within_group_perm(data.frame_task[b], torch.Generator(device=b.device).manual_seed(7 + i))
        out.append(model.forward_eval(X, vis, act, perm))
    codes = torch.cat([o['codes'] for o in out])
    lt, ls, lc = (torch.cat([o[k] for o in out]).mean(0).tolist() for k in ('true', 'shuffled', 'copy'))
    return dict(val_true_h=lt, val_shuf_h=ls, val_copy_h=lc, val_gap=sum(ls) / len(ls) - sum(lt) / len(lt),
                slot_ppl=perplexity(codes, model.vq.n_codes)[2])


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    work_dir = args.work_dir or cfg.get('work_dir') or osp.join('work_dirs', osp.splitext(osp.basename(args.config))[0])
    os.makedirs(work_dir, exist_ok=True)
    tc, oc = cfg.train_cfg, cfg.optim_cfg
    device = 'cuda'

    torch.manual_seed(tc.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    # The model is built after the data (as for the released checkpoint) so that parameter initialisation sees the same RNG
    # stream; the data only needs the window geometry, read from the config.
    probe = _ModelProbe(cfg.model)
    tr, va = build_data(cfg, probe, 'train', device), build_data(cfg, probe, 'val', device)
    tr_anc, va_anc = tr.anchors(probe.horizon), va.anchors(probe.horizon)
    sp = osp.join(work_dir, ACTION_STATS)
    if not osp.exists(sp):
        am, astd = tr.action_stats(tr_anc, probe.horizon)
        torch.save(dict(mean=am.cpu(), std=astd.cpu(),
                        definition=f'joint[t+i,:8]-joint[t,:8], i=1..{probe.horizon}, per (i, dim)'), sp)
    amean, astd = load_action_stats(sp, device)

    model = MODELS.build(cfg.model).to(device)
    opt = torch.optim.AdamW(param_groups(model, oc.weight_decay), lr=oc.lr, betas=tuple(oc.betas))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_lambda(tc.max_steps, oc.warmup, oc.get('lr_floor', 0.1)))
    gen = torch.Generator(device=device)
    gen.manual_seed(tc.seed)
    step, _ = try_resume(work_dir, model, opt, sched, gen)
    gv = torch.Generator(device=device).manual_seed(1234)
    va_idx = va_anc[torch.randperm(len(va_anc), device=device, generator=gv)[:tc.n_val]]
    cfg.dump(osp.join(work_dir, 'config.py'))
    stop = StopFlag()
    log = open(osp.join(work_dir, 'log.jsonl'), 'a')
    params = list(model.parameters())
    extra = dict(target=model.target, offsets=list(model.offsets), context=model.context)
    t0, acc = time.time(), dict(rec=0., cb=0., commit=0., n=0)
    while step < tc.max_steps:
        model.train()
        idx = tr_anc[torch.randint(0, len(tr_anc), (tc.batch_size,), device=device, generator=gen)]
        with torch.no_grad():
            X = tr.fields_at(idx, model.offsets)
            vis = tr.vis_at(idx, model.offsets) if model.uses_images else None
            act = (tr.actions(idx, model.horizon) - amean) / astd
        out = model.forward_train(X, vis, act)
        opt.zero_grad(set_to_none=True)
        out['loss'].backward()
        gn = torch.nn.utils.clip_grad_norm_(params, oc.grad_clip)
        opt.step()
        sched.step()
        step += 1
        for k in ('rec', 'cb', 'commit'):
            acc[k] += float(out[k])
        acc['n'] += 1
        reset_every = tc.codebook_reset_interval
        n_reset = model.vq.reset_dead(out['z'], gen) if (reset_every and step % reset_every == 0) else None
        if step % tc.log_interval == 0 or n_reset is not None:
            rec = dict(step=step, lr=sched.get_last_lr()[0], gn=float(gn), sec=round(time.time() - t0, 1),
                       **{k: v / max(acc['n'], 1) for k, v in acc.items() if k != 'n'})
            if n_reset is not None:
                rec['codes_reset'] = n_reset
            log.write(json.dumps(rec) + '\n')
            log.flush()
            acc = dict(rec=0., cb=0., commit=0., n=0)
        if step % tc.val_interval == 0 or step == tc.max_steps:
            model.eval()
            rec = dict(step=step, **validate(model, va, va_idx, amean, astd))
            print(json.dumps(rec), flush=True)
            log.write(json.dumps(rec) + '\n')
            log.flush()
        stopping = stop.sig is not None or (args.stop_after and step == args.stop_after)
        if step % tc.ckpt_interval == 0 or step == tc.max_steps or stopping:
            save_checkpoint(work_dir, step, model, opt, sched, gen, extra=extra,
                            keep_step=(step % tc.keep_interval == 0 or step == tc.max_steps))
        if stopping:
            print(f'[train] stopped after step {step}; re-run the same command to resume', flush=True)
            return
    open(osp.join(work_dir, 'DONE'), 'w').write(f'{step}\n')


class _ModelProbe:
    """Window geometry and conditioning of the model config, without building (and initialising) the model."""

    def __init__(self, model_cfg):
        offsets = tuple(model_cfg.get('offsets', (0, 5, 10, 15)))
        context = model_cfg.get('context', 'both')
        self.offsets, self.horizon = offsets, offsets[-1]
        self.uses_images = context in ('both', 'image')


if __name__ == '__main__':
    main()
