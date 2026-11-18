"""Normalisation statistics of the tactile inputs, computed on the training episodes of the exported data.

    python tools/data/build_tactile_stats.py --data data/univtac/tactile --out data/univtac/tactile/tactile_stats.json

Uses every ``<data>/<task>/<ep:04d>.npz`` with ep < ``--train-eps`` (the bundled
``tacinno/datasets/stats/univtac_tactile_stats.json`` was computed on episodes 0-89 of eight UniVTAC tasks: the six
evaluated tasks plus grasp_classify and insert_HDMI). Output keys:

  rest_grid      (63, 2) global rest marker grid in 7 x 9 raster order (median of the per-episode, per-gel rest grids
                 after sorting each into the raster)
  marker_order   per gel, the sensor's native marker order -> raster order; rest_native per gel (native order)
  sigma_u        RMS (px) of valid marker displacements on contact frames, pooled over x, y and both gels
  m_mean, m_std  per-dimension statistics of the contact-state summary m over valid frames of both gels
  p_mean, p_std  statistics of p = [q[:8], dq[:8], 8 reserved zeros] (dq = backward difference per frame)
"""
from __future__ import annotations

import argparse
import glob
import json
import os.path as osp
import sys

import numpy as np

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from tacinno.datasets.tactile_fields import M_NAMES, grid_order, sanitize_m  # noqa: E402

P_USED, P_RESERVED = 16, 8
P_DIM = P_USED + P_RESERVED


def proprio(joint):
    """joint (T, >=8) -> p (T, 24): [q[:8], dq[:8], 8 reserved zeros], dq = 0 at t = 0."""
    q = np.asarray(joint)[:, :8].astype(np.float32)
    dq = np.zeros_like(q)
    dq[1:] = q[1:] - q[:-1]
    p = np.zeros((len(q), P_DIM), np.float32)
    p[:, :8], p[:, 8:16] = q, dq
    return p


def main():
    ap = argparse.ArgumentParser(description='Tactile normalisation statistics')
    ap.add_argument('--data', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--train-eps', type=int, default=90)
    a = ap.parse_args()
    files = sorted(glob.glob(osp.join(a.data, '*', '[0-9][0-9][0-9][0-9].npz')))
    rests, ds, ms, ps, n_frames, n_contact = [], [], [], [], 0, 0
    natives, orders = [[], []], [set(), set()]
    for f in files:
        if int(osp.basename(f)[:4]) >= a.train_eps:
            continue
        z = np.load(f)
        rests.append(np.stack([z['rest'][g][grid_order(z['rest'][g])] for g in range(2)]))  # raster order
        for g in range(2):
            natives[g].append(z['rest'][g])
            orders[g].add(tuple(grid_order(z['rest'][g]).tolist()))
        mk, valid, ec = z['mk'], z['valid'], z['e_c']
        for g in range(2):
            d = mk[:, g, 1] - z['rest'][g][None]
            sel = valid[:, g] & ec[:, g][:, None]
            ds.append(d[sel])
            ms.append(sanitize_m(z['m'])[z['m_valid'][:, g], g])
        ps.append(proprio(z['joint']))
        n_frames += len(z['joint'])
        n_contact += int(ec.sum())
    rests = np.stack(rests)  # (E, 2, 63, 2)
    rest_global = np.median(rests.reshape(-1, 63, 2), 0)
    dev = np.abs(rests - rest_global[None, None]).max()
    assert np.array_equal(grid_order(rest_global), np.arange(63)), 'global rest grid is not in raster order'
    d = np.concatenate(ds)
    sigma_u = float(np.sqrt((d ** 2).mean()))
    m = np.concatenate(ms)
    p = np.concatenate(ps)
    p_mean, p_std = p.mean(0), p.std(0)
    p_std[:8] = np.maximum(p_std[:8], 1e-4)
    p_std[8:P_USED] = np.maximum(p_std[8:P_USED], 1e-6)
    p_mean[P_USED:], p_std[P_USED:] = 0.0, 1.0  # reserved slots
    assert all(len(o) == 1 for o in orders), 'native marker order differs between episodes'
    order_gel = [list(next(iter(o))) for o in orders]
    rest_native = [np.median(np.stack(natives[g]), 0) for g in range(2)]
    out = dict(rest_grid=rest_global.tolist(), rest_max_dev_px=float(dev),
               marker_order=order_gel, rest_native=[r.tolist() for r in rest_native],
               sigma_u=sigma_u, sigma_u_def='RMS of valid marker displacement (px) on contact frames, x and y pooled',
               m_names=M_NAMES, m_mean=m.mean(0).tolist(), m_std=np.maximum(m.std(0), 1e-6).tolist(),
               p_def='[q[:8], q[t,:8] - q[t-1,:8], 8 reserved zero slots] (dq = 0 at t = 0); q[7] = finger joint '
                     '(gripper opening)', p_used=P_USED, p_reserved=P_RESERVED, p_dim=P_DIM,
               p_mean=p_mean.tolist(), p_std=p_std.tolist(), n_train_episodes=len(rests), n_frames=n_frames,
               contact_frac=n_contact / (2 * n_frames))
    json.dump(out, open(a.out, 'w'), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k not in ('rest_grid', 'marker_order', 'rest_native')}, indent=1))


if __name__ == '__main__':
    main()
