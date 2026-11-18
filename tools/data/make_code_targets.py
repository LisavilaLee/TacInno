"""Turn exported tactile innovation codes into the per-frame code targets of the policy training data.

    python tools/data/make_code_targets.py CODES_DIR COND_ROOT [--out-root DIR] [--tasks ...] [--splits train val]

``CODES_DIR`` is the output of ``tools/export_codes.py``. ``COND_ROOT/<split>/<task>/meta.json`` (written by the policy
data preparation) lists the episodes of the split and their policy-frame counts; the policy data keeps frames
0 .. T-2 of an episode of T recorded frames, so the codes of frames 0 .. T-2 are kept. Written next to it:

* ``code_tgt.npy`` (N, 4) int16: the code of the window starting at each policy frame, -1 where there is none;
* ``code_tv.npy`` (N,) bool: whether the frame has a code (the window t .. t+horizon fits in the episode).

With ``--out-root`` the targets go to a separate tree (for example one per tokenizer ablation) whose other per-frame
files are symlinked from ``COND_ROOT``. Files are written to a temporary name and renamed.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp

import numpy as np

TASKS = ('insert_hole', 'insert_tube', 'lift_bottle', 'lift_can', 'pull_out_key', 'put_bottle_in_shelf')
TARGET_FILES = ('code_tgt.npy', 'code_tv.npy')
SHARED_TOP_FILES = ('cond_stats.json',)


def parse_args():
    ap = argparse.ArgumentParser(description='Per-frame code targets for policy training')
    ap.add_argument('codes_dir')
    ap.add_argument('cond_root')
    ap.add_argument('--out-root', help='default: write into COND_ROOT')
    ap.add_argument('--tasks', nargs='+', default=list(TASKS))
    ap.add_argument('--splits', nargs='+', default=['train', 'val'])
    ap.add_argument('--codebook-size', type=int, default=16)
    return ap.parse_args()


def _link_shared(src_dir: str, dst_dir: str) -> None:
    for f in sorted(os.listdir(src_dir)):
        if f in TARGET_FILES or f.endswith('.tmp.npy'):
            continue
        dst = osp.join(dst_dir, f)
        if not osp.lexists(dst):
            os.symlink(osp.relpath(osp.join(src_dir, f), dst_dir), dst)


def main():
    a = parse_args()
    out_root = a.out_root or a.cond_root
    for split in a.splits:
        for task in a.tasks:
            src = osp.join(a.cond_root, split, task)
            d = osp.join(out_root, split, task)
            os.makedirs(d, exist_ok=True)
            if osp.abspath(d) != osp.abspath(src):
                _link_shared(src, d)
            meta = json.load(open(osp.join(src, 'meta.json')))
            tg, tv = [], []
            for ep, K in zip(meta['episodes'], meta['lengths']):
                z = np.load(osp.join(a.codes_dir, task, f'{ep:04d}.npz'))
                if len(z['codes']) != K + 1:
                    raise ValueError(f"{task} ep {ep}: codes have {len(z['codes'])} frames, policy episode {K} (+1)")
                tg.append(z['codes'][:K].astype(np.int16))
                tv.append(z['valid'][:K].astype(bool))
            tg, tv = np.concatenate(tg), np.concatenate(tv)
            assert len(tg) == meta['n'], (task, split, len(tg), meta['n'])
            assert tg[tv].min() >= 0 and tg[tv].max() < a.codebook_size and (tg[~tv] == -1).all()
            for name, arr in (('code_tgt', tg), ('code_tv', tv)):
                np.save(osp.join(d, f'{name}.tmp.npy'), arr)
                os.replace(osp.join(d, f'{name}.tmp.npy'), osp.join(d, f'{name}.npy'))
            print(split, task, tg.shape, 'valid %.3f' % tv.mean(), flush=True)
    if osp.abspath(out_root) != osp.abspath(a.cond_root):
        for f in SHARED_TOP_FILES:
            p = osp.join(a.cond_root, f)
            if osp.exists(p) and not osp.lexists(osp.join(out_root, f)):
                os.symlink(osp.relpath(p, out_root), osp.join(out_root, f))
    export = json.load(open(osp.join(a.codes_dir, 'EXPORT.json')))
    src_file = osp.join(out_root, 'code_tgt_SOURCE.json')
    if osp.islink(src_file):
        os.remove(src_file)
    json.dump(dict(source=osp.basename(osp.normpath(a.codes_dir)), export=export), open(src_file, 'w'), indent=1)
    print('DONE')


if __name__ == '__main__':
    main()
