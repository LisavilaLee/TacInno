"""Training utilities of the TacInno tokenizer: optimizer groups, learning-rate schedule, code statistics and
checkpoints with exact resume (model, optimizer, scheduler, step, every RNG state and the data-sampling generator)."""
from __future__ import annotations

import math
import os
import os.path as osp
import random
import signal
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

__all__ = [
    "cosine_lambda", "param_groups", "within_group_perm", "perplexity", "save_checkpoint", "try_resume", "StopFlag",
    "LATEST", "load_weights", "find_action_stats", "load_action_stats",
]

LATEST = 'ckpt_latest.pt'
NO_DECAY_SUFFIXES = ('pos', 'gel', 'frame', 'step', 'slot', 'queries', 'out_q', 'prop_t')


def cosine_lambda(steps: int, warmup: int, floor: float = 0.1):
    """Linear warm-up over ``warmup`` steps, then cosine decay from 1 to ``floor`` at ``steps`` (LambdaLR factor)."""
    def f(s):
        warm = min(1.0, (s + 1) / warmup)
        prog = min(s, steps) / steps
        return warm * (floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * prog)))
    return f


def param_groups(model: torch.nn.Module, wd: float, no_decay_names: Sequence[str] = ('codebook',)) -> List[Dict]:
    """Weight decay on matrices only: biases, norms, embeddings (names ending in pos/gel/slot/queries...) and the
    codebook get none."""
    dec, nodec = [], []
    for n, p in model.named_parameters():
        if p.ndim < 2 or any(k in n for k in no_decay_names) or n.endswith(NO_DECAY_SUFFIXES):
            nodec.append(p)
        else:
            dec.append(p)
    return [dict(params=dec, weight_decay=wd), dict(params=nodec, weight_decay=0.0)]


def within_group_perm(groups: torch.Tensor, gen: torch.Generator) -> torch.Tensor:
    """Random permutation that only swaps positions with the same group id (for example the task)."""
    perm = torch.arange(len(groups), device=groups.device)
    for gid in torch.unique(groups):
        pos = torch.nonzero(groups == gid)[:, 0]
        perm[pos] = pos[torch.randperm(len(pos), device=groups.device, generator=gen)]
    return perm


def perplexity(codes: torch.Tensor, n_codes: int = 16) -> Tuple[float, int, List[float]]:
    """codes (N, slots) -> (perplexity over all slots, #codes used, per-slot perplexities)."""
    def pp(c):
        p = torch.bincount(c.flatten(), minlength=n_codes).float()
        p = p / p.sum()
        return float(torch.exp(-(p[p > 0] * p[p > 0].log()).sum()))
    used = int((torch.bincount(codes.flatten(), minlength=n_codes) > 0).sum())
    return pp(codes), used, [pp(codes[:, i]) for i in range(codes.shape[1])]


def _rng_state(gen: torch.Generator) -> Dict:
    state = dict(torch=torch.get_rng_state(), np=np.random.get_state(), py=random.getstate(), gen=gen.get_state())
    if torch.cuda.is_available():
        state['cuda'] = torch.cuda.get_rng_state_all()
    return state


def _set_rng_state(s: Dict, gen: torch.Generator) -> None:
    # RNG states are CPU ByteTensors; torch.load(map_location='cuda') moves them, so bring them back first
    torch.set_rng_state(s['torch'].cpu())
    if 'cuda' in s and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([t.cpu() for t in s['cuda']])
    np.random.set_state(s['np'])
    random.setstate(s['py'])
    gen.set_state(s['gen'].cpu())


def save_checkpoint(work_dir: str, step: int, model, opt, sched, gen, extra: Optional[Dict] = None,
                    keep_step: bool = False) -> None:
    """Write ``ckpt_latest.pt`` atomically (tmp + rename); with ``keep_step`` also ``model_<step>.pt`` (weights only,
    the format of the released checkpoints: ``{'step', 'model', 'extra'}``)."""
    st = dict(step=step, model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), rng=_rng_state(gen),
              extra=extra or {})
    tmp = osp.join(work_dir, LATEST + '.tmp')
    torch.save(st, tmp)
    os.replace(tmp, osp.join(work_dir, LATEST))
    if keep_step:
        torch.save(dict(step=step, model=model.state_dict(), extra=extra or {}), osp.join(work_dir, f'model_{step:06d}.pt'))


def try_resume(work_dir: str, model, opt, sched, gen, map_location='cuda') -> Tuple[int, Optional[Dict]]:
    """-> (step, extra) of ``<work_dir>/ckpt_latest.pt``, or (0, None) when starting fresh."""
    p = osp.join(work_dir, LATEST)
    if not osp.exists(p):
        return 0, None
    st = torch.load(p, map_location=map_location, weights_only=False)
    model.load_state_dict(st['model'])
    opt.load_state_dict(st['opt'])
    sched.load_state_dict(st['sched'])
    _set_rng_state(st['rng'], gen)
    print(f'[ckpt] resumed from {p} at step {st["step"]}', flush=True)
    return st['step'], st['extra']


class StopFlag:
    """Set by SIGTERM / SIGUSR1 (pre-emption, a scheduler's time-limit warning); the training loop checks it after
    every step, saves a checkpoint and exits. Re-running the same command resumes."""

    def __init__(self):
        self.sig = None
        signal.signal(signal.SIGTERM, self._h)
        signal.signal(signal.SIGUSR1, self._h)

    def _h(self, signum, frame):
        self.sig = signum


ACTION_STATS_NAMES = ('action_stats.pt', 'cmd_stats.pt')  # the released checkpoints use the second name


def load_weights(model: torch.nn.Module, path: str, strict: bool = True, map_location='cpu') -> Dict:
    """Load ``model_<step>.pt`` / ``ckpt_latest.pt`` weights into ``model``; returns the checkpoint dict."""
    ck = torch.load(path, map_location=map_location, weights_only=False)
    model.load_state_dict(ck['model'] if 'model' in ck else ck, strict=strict)
    return ck


def find_action_stats(checkpoint: str) -> str:
    """Action standardisation file stored next to a checkpoint."""
    d = osp.dirname(osp.abspath(checkpoint))
    for name in ACTION_STATS_NAMES:
        if osp.exists(osp.join(d, name)):
            return osp.join(d, name)
    raise FileNotFoundError(f'none of {ACTION_STATS_NAMES} found next to {checkpoint}; pass --action-stats')


def load_action_stats(path: str, device) -> Tuple[torch.Tensor, torch.Tensor]:
    """-> (mean, std), each (horizon, 8), on ``device``."""
    s = torch.load(path, map_location='cpu', weights_only=False)
    return s['mean'].to(device), s['std'].to(device)
