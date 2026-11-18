#!/usr/bin/env python
"""Build the per-frame condition arrays and contact-state targets of TacInno-VLA, aligned with the zarr frames.

Input: the per-episode tactile exports of ``tools/data/export_tactile.py`` (``<tactile_export>/<task>/<ep:04d>.npz`` with
``joint``, ``w96``, ``mk``, ``rest``, ``valid``, ``m``, ``m_valid``, ``e_c``) and their statistics file.
The zarr converter keeps frames 0 .. T-2 of every episode (the last frame has no next-frame action), so frame j of
zarr episode k is frame j of export episode k. For every kept frame this writes to ``<out>/<split>/<task>/``:

    fields.npy   (N, 2, 3, 96, 128) float16  tactile observation per gel (indentation, u_x / sigma_u, u_y / sigma_u),
                                             tacinno.datasets.tactile_fields.build_fields with the global rest grid
    joints8.npy  (N, 8) float32              7 arm joints + gripper joint, raw units
    ep_start.npy (N,) int64                  index of the first kept frame of the frame's episode
    aux_m.npy    (N, 2, 14) float32          contact properties per gel, standardized (normalize_m), contact-state targets
    aux_mv.npy   (N, 2) bool                 contact properties valid
    aux_ec.npy   (N, 2) bool                 contact label (min depth < 28.3 mm)
    meta.json                                episodes, lengths, n

and ``<out>/cond_stats.json`` (mean / std of joints8 over all training frames, sigma_u). The code targets
(``code_tgt.npy``, ``code_tv.npy``) are added by ``tools/data/make_code_targets.py``.

    python tools/data/make_cond.py --tactile-export data/univtac/tactile --out data/univtac/cond
"""
from __future__ import annotations

import argparse
import json
import os
import os.path as osp
import sys

import numpy as np
import torch

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))
from tacinno.datasets import tactile_fields as TF  # noqa: E402

TASKS = ["insert_hole", "insert_tube", "lift_bottle", "lift_can", "pull_out_key", "put_bottle_in_shelf"]


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tactile-export", required=True, help="directory of the per-episode tactile exports")
    ap.add_argument("--stats", help="tactile statistics (default: the bundled UniVTAC statistics used by the released models)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tasks", nargs="+", default=TASKS)
    ap.add_argument("--train-episodes", default="0-49", help="episode range of the training split")
    ap.add_argument("--val-episodes", default="90-99", help="episode range of the validation split")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return ap.parse_args()


def _range(s: str) -> range:
    a, b = (int(x) for x in s.split("-"))
    return range(a, b + 1)


def main():
    a = parse_args()
    # the released models use the bundled UniVTAC statistics (rest grid, sigma_u, contact-property mean / std)
    stats_path = a.stats or osp.join(osp.dirname(TF.__file__), "stats", "univtac_tactile_stats.json")
    stats = TF.load_tactile_stats(stats_path)
    rest = np.array(stats["rest_grid"], np.float32)
    samp = TF.grid_sampling_coords(torch.tensor(rest).reshape(TF.GRID_R, TF.GRID_C, 2), a.device)
    sigma_u = float(stats["sigma_u"])
    m_mean, m_std = np.asarray(stats["m_mean"], np.float32), np.asarray(stats["m_std"], np.float32)
    pool = []
    for split, eps in (("train", _range(a.train_episodes)), ("val", _range(a.val_episodes))):
        for task in a.tasks:
            od = osp.join(a.out, split, task)
            os.makedirs(od, exist_ok=True)
            F, J, S, M, V, E, lens, n = [], [], [], [], [], [], [], 0
            for ep in eps:
                z = np.load(osp.join(a.tactile_export, task, f"{ep:04d}.npz"))
                T = len(z["joint"])
                K = T - 1  # the zarr keeps frames 0 .. T-2
                mk = z["mk"].astype(np.float32)
                d = np.zeros((T, 2, 63, 2), np.float32)
                v = np.zeros((T, 2, 63), bool)
                for g in range(2):
                    o = TF.grid_order(z["rest"][g])
                    d[:, g] = mk[:, g, 1][:, o] - rest[None]
                    v[:, g] = z["valid"][:, g][:, o]
                w = torch.from_numpy(z["w96"][:K].astype(np.float32)).to(a.device).reshape(K * 2, 96, 128)
                dd = torch.from_numpy(d[:K].reshape(K * 2, TF.GRID_R, TF.GRID_C, 2)).to(a.device)
                vv = torch.from_numpy(v[:K].reshape(K * 2, TF.GRID_R, TF.GRID_C)).to(a.device)
                with torch.no_grad():
                    X = TF.build_fields(w, dd, vv, sigma_u, samp).reshape(K, 2, 3, 96, 128)
                F.append(X.half().cpu().numpy())
                J.append(np.concatenate([z["joint"][:K, :7], z["joint"][:K, 7:8]], 1).astype(np.float32))
                S.append(np.full(K, n, np.int64))
                M.append(TF.normalize_m(z["m"][:K], z["m_valid"][:K], m_mean, m_std))
                V.append(z["m_valid"][:K].astype(bool))
                E.append(z["e_c"][:K].astype(bool))
                lens.append(K)
                n += K
            arrays = dict(fields=np.concatenate(F), joints8=np.concatenate(J), ep_start=np.concatenate(S),
                          aux_m=np.concatenate(M), aux_mv=np.concatenate(V), aux_ec=np.concatenate(E))
            for k, arr in arrays.items():
                np.save(osp.join(od, f"{k}.npy"), arr)
            with open(osp.join(od, "meta.json"), "w") as f:
                json.dump(dict(task=task, split=split, episodes=list(eps), lengths=lens, n=int(n)), f)
            if split == "train":
                pool.append(arrays["joints8"])
            print(split, task, arrays["fields"].shape, "contact %.3f" % arrays["aux_ec"].mean(), flush=True)
    P = np.concatenate(pool)
    with open(osp.join(a.out, "cond_stats.json"), "w") as f:
        json.dump(dict(j_mean=P.mean(0).tolist(), j_std=(P.std(0) + 1e-6).tolist(), sigma_u=sigma_u,
                       n_train_frames=int(len(P))), f, indent=1)
    print("DONE")


if __name__ == "__main__":
    main()
