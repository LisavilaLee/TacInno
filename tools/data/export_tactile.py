"""Export the tactile inputs of UniVTAC demonstrations (HDF5) to one compressed npz per episode.

    python tools/data/export_tactile.py --src data/univtac/raw --out data/univtac/tactile [--tasks ...] [--episodes 0-99]

Source: ``<src>/<task>/hdf5/<ep>.hdf5`` with ``tactile/{left,right}_tactile/{depth,marker}``, ``embodiment/joint``
and ``step``. Output ``<out>/<task>/<ep:04d>.npz`` in the canonical gel frame (right gel mirrored along x):

  w96      (T, 2, 96, 128) float16  indentation field clip(28.5 - depth, 0, 6) / 3, area-resampled
  mk       (T, 2, 2, 63, 2) float32 markers [reference, current] x 63 x (x, y), 320 x 240 px
  rest     (2, 63, 2) float32       episode rest grid (median reference)
  valid    (T, 2, 63) bool          marker validity
  m        (T, 2, 14) float32       contact-state summary (tactile_fields.M_NAMES)
  m_valid  (T, 2) bool
  e_c      (T, 2) bool              contact label (min depth < 28.3 mm)
  joint    (T, 9) float32           7 arm joints + 2 finger joints
  step     (T,) int64
plus a JSON sidecar with T and simple statistics.
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys
import time
import traceback
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from tacinno.datasets.tactile_fields import process_gel  # noqa: E402
from tacinno.datasets.tokenizer_dataset import parse_episodes  # noqa: E402

TASKS = ['insert_hole', 'insert_tube', 'lift_bottle', 'lift_can', 'pull_out_key', 'put_bottle_in_shelf']


def export_one(job):
    src, dst = job
    if osp.exists(dst):
        return dst, 'skip'
    import h5py

    t0 = time.time()
    with h5py.File(src, 'r') as f:
        gels = []
        for g, name in enumerate(['left_tactile', 'right_tactile']):
            gels.append(process_gel(f[f'tactile/{name}/depth'][()], f[f'tactile/{name}/marker'][()], g))
        joint = f['embodiment/joint'][()].astype(np.float32)
        step = f['step'][()].astype(np.int64)
    out = dict(w96=np.stack([g['w96'] for g in gels], 1).astype(np.float16),
               mk=np.stack([g['mk'] for g in gels], 1), rest=np.stack([g['rest'] for g in gels]),
               valid=np.stack([g['valid'] for g in gels], 1), m=np.stack([g['m'] for g in gels], 1),
               m_valid=np.stack([g['m_valid'] for g in gels], 1), e_c=np.stack([g['e_c'] for g in gels], 1),
               joint=joint, step=step)
    tmp = dst + '.tmp.npz'
    np.savez_compressed(tmp, **out)
    os.replace(tmp, dst)
    json.dump(dict(T=int(len(joint)), valid_frac=float(out['valid'].mean()), m_valid_frac=float(out['m_valid'].mean()),
                   contact_frac=float(out['e_c'].mean()), sec=round(time.time() - t0, 1)), open(dst[:-4] + '.json', 'w'))
    return dst, 'ok'


def _safe(job):
    try:
        return export_one(job)
    except Exception:
        return job[1], 'FAIL ' + traceback.format_exc().replace('\n', ' | ')


def main():
    ap = argparse.ArgumentParser(description='Export UniVTAC tactile inputs')
    ap.add_argument('--src', required=True, help='root with <task>/hdf5/<ep>.hdf5')
    ap.add_argument('--out', required=True)
    ap.add_argument('--tasks', nargs='+', default=TASKS)
    ap.add_argument('--episodes', default='0-99')
    ap.add_argument('--workers', type=int, default=6)
    a = ap.parse_args()
    jobs = []
    for task in a.tasks:
        os.makedirs(osp.join(a.out, task), exist_ok=True)
        for ep in parse_episodes(a.episodes):
            jobs.append((osp.join(a.src, task, 'hdf5', f'{ep}.hdf5'), osp.join(a.out, task, f'{ep:04d}.npz')))
    print(f'{len(jobs)} episodes -> {a.out}', flush=True)
    n_ok = n_fail = 0
    with Pool(a.workers) as pool:
        for dst, st in pool.imap_unordered(_safe, jobs):
            if st.startswith('FAIL'):
                n_fail += 1
                print(dst, st, flush=True)
            else:
                n_ok += 1
            if (n_ok + n_fail) % 50 == 0:
                print(f'{n_ok + n_fail}/{len(jobs)} ok={n_ok} fail={n_fail}', flush=True)
    print(f'DONE ok={n_ok} fail={n_fail}', flush=True)


if __name__ == '__main__':
    main()
