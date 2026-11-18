"""Export the tactile innovation code of every demonstration frame with a trained TacInno tokenizer.

    python tools/export_codes.py CONFIG CHECKPOINT --out data/univtac/codes [--episodes 0-99]

Output ``<out>/<task>/<ep:04d>.npz``: ``codes`` (T, 4) int16 (-1 where the window does not fit, the last ``horizon``
frames of an episode) and ``valid`` (T,) bool; ``<out>/codebook.npy`` (slots, 16, 32); ``<out>/EXPORT.json``.
``--continuous`` writes the encoder output before quantisation, ``z`` (T, 4, 32) float16, instead of indices.
The action standardisation is read from ``action_stats.pt`` (or ``cmd_stats.pt``) next to the checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from tacinno.config import Config, DictAction  # noqa: E402
from tacinno.datasets.tokenizer_dataset import UniVTACTokenizerData  # noqa: E402,F401
from tacinno.models.tokenizer import TacInnoTokenizer  # noqa: E402,F401
from tacinno.models.tokenizer.training import find_action_stats, load_action_stats, load_weights  # noqa: E402
from tacinno.registry import DATASETS, MODELS  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser(description='Export tactile innovation codes')
    ap.add_argument('config')
    ap.add_argument('checkpoint')
    ap.add_argument('--out', required=True)
    ap.add_argument('--episodes', default='0-99', help="episodes to export, e.g. '0-99' or '90-99'")
    ap.add_argument('--tasks', nargs='+', help='default: the tasks of the training dataset config')
    ap.add_argument('--action-stats', help='default: action_stats.pt / cmd_stats.pt next to the checkpoint')
    ap.add_argument('--continuous', action='store_true', help='write z before quantisation instead of indices')
    ap.add_argument('--batch-size', type=int, default=256)
    ap.add_argument('--storage-device', default=None, help="'cpu' keeps the frames in host memory")
    ap.add_argument('--cfg-options', nargs='+', action=DictAction)
    return ap.parse_args()


@torch.no_grad()
def main():
    a = parse_args()
    cfg = Config.fromfile(a.config)
    if a.cfg_options:
        cfg.merge_from_dict(a.cfg_options)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = MODELS.build(cfg.model).to(device).eval()
    ck = load_weights(model, a.checkpoint, strict=True, map_location=device)
    amean, astd = load_action_stats(a.action_stats or find_action_stats(a.checkpoint), device)

    ds = dict(cfg.train_dataset)
    ds.update(episodes=a.episodes, with_vision=model.uses_images, device=device, storage_device=a.storage_device)
    if a.tasks:
        ds['tasks'] = a.tasks
    data = DATASETS.build(ds)
    anc = data.anchors(model.horizon)
    out = []
    for i in range(0, len(anc), a.batch_size):
        idx = anc[i:i + a.batch_size]
        # The inputs are assembled inside the bf16 autocast region, as in the export that produced the released codes:
        # the marker in-painting convolution of build_fields then runs in bf16. This only affects frames with missing
        # markers; building them in fp32 instead changes the code of ~0.06 % of the frames (near-ties).
        with model.autocast(device):
            fields = data.fields_at(idx, model.offsets)
            vis = data.vis_at(idx, model.offsets) if model.uses_images else None
            z = model.encode(fields, vis, (data.actions(idx, model.horizon) - amean) / astd)
        out.append(z if a.continuous else model.quantize(z)[1])
    codes = torch.cat(out)
    if a.continuous:
        full = torch.zeros((data.n_frames, codes.shape[1], codes.shape[-1]), dtype=torch.float32, device=device)
    else:
        full = torch.full((data.n_frames, codes.shape[1]), -1, dtype=torch.long, device=device)
    full[anc] = codes
    valid = torch.zeros(data.n_frames, dtype=torch.bool, device=device)
    valid[anc] = True
    full = full.cpu().numpy().astype(np.float16 if a.continuous else np.int16)
    valid = valid.cpu().numpy()
    key = 'z' if a.continuous else 'codes'
    for e in data.meta:
        os.makedirs(osp.join(a.out, e['task']), exist_ok=True)
        sl = slice(e['start'], e['start'] + e['T'])
        assert valid[sl].sum() == max(e['T'] - model.horizon, 0)
        np.savez(osp.join(a.out, e['task'], f'{e["ep"]:04d}.npz'), **{key: full[sl]}, valid=valid[sl])
    np.save(osp.join(a.out, 'codebook.npy'), model.vq.codebook.detach().cpu().numpy())
    json.dump(dict(config=osp.basename(a.config), checkpoint=osp.basename(a.checkpoint), step=ck.get('step'),
                   k=model.horizon, offsets=list(model.offsets), context=model.context,
                   content=('continuous encoder output z (T, 4, 32) float16 before quantisation' if a.continuous
                            else 'code indices (T, 4) int16, -1 invalid'),
                   n_frames=data.n_frames, n_valid=int(valid.sum())),
              open(osp.join(a.out, 'EXPORT.json'), 'w'), indent=1)
    print('exported', data.n_frames, 'frames,', int(valid.sum()), 'valid', flush=True)


if __name__ == '__main__':
    main()
