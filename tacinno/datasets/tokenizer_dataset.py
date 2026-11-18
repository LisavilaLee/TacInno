"""Demonstration frames for the TacInno tokenizer, held in (GPU) memory and indexed by global frame number.

Files (one per episode, produced by ``tools/data/export_tactile.py`` and ``tools/data/extract_image_features.py``):

* ``<tactile_root>/<task>/<ep:04d>.npz``: ``w96`` (T, 2, 96, 128) indentation field, ``mk`` (T, 2, 2, 63, 2) markers
  [reference, current], ``rest`` (2, 63, 2), ``valid`` (T, 2, 63), ``m`` (T, 2, 14), ``m_valid`` (T, 2),
  ``e_c`` (T, 2), ``joint`` (T, 9) (7 arm joints + 2 finger joints; the gripper opening is joint[7]);
* ``<feature_root>/<task>/<ep:04d>.npz``: ``patch`` (T, 2, 16, 768) DINOv2-base features of (head, wrist).

Per frame the class keeps the indentation field W, the marker displacement D from the global rest grid in 7 x 9
raster order, the marker validity V, the joints Q = joint[:8] and the image features VIS. The tactile observation of a
frame is assembled on demand by :func:`tactile_fields.build_fields` (3 x 96 x 128 per gel).

A window anchored at frame t covers frames t + offsets; anchors are the frames with t + horizon inside the episode.
"""
from __future__ import annotations

import os.path as osp
from typing import Iterable, List, Optional, Sequence, Union

import numpy as np
import torch

from tacinno.registry import DATASETS

from .tactile_fields import (GRID_C, GRID_R, build_fields, grid_order, grid_sampling_coords, load_tactile_stats,
                             normalize_m)

__all__ = ["UniVTACTokenizerData", "parse_episodes", "UNIVTAC_TASKS", "DEFAULT_STATS_FILE"]

UNIVTAC_TASKS = ('insert_hole', 'insert_tube', 'lift_bottle', 'lift_can', 'pull_out_key', 'put_bottle_in_shelf')
DEFAULT_STATS_FILE = osp.join(osp.dirname(__file__), 'stats', 'univtac_tactile_stats.json')


def parse_episodes(episodes: Union[str, int, Iterable[int]]) -> List[int]:
    """'0-89' (inclusive) / '0-49,90-99' / 7 / [0, 1, 2] -> list of episode indices."""
    if isinstance(episodes, int):
        return [episodes]
    if isinstance(episodes, str):
        out: List[int] = []
        for part in episodes.split(','):
            part = part.strip()
            if '-' in part:
                a, b = (int(x) for x in part.split('-'))
                out.extend(range(a, b + 1))
            elif part:
                out.append(int(part))
        return out
    return [int(e) for e in episodes]


@DATASETS.register_module()
class UniVTACTokenizerData:
    """Frames of the given tasks and episodes.

    Args:
        tactile_root: directory with ``<task>/<ep:04d>.npz`` tactile exports.
        feature_root: directory with ``<task>/<ep:04d>.npz`` image features (needed when ``with_vision``).
        stats_file: tactile statistics (rest grid, sigma_u, m normalisation); defaults to the bundled file computed on
            the UniVTAC training demonstrations.
        tasks: task ids; episodes: '0-89', a list, ...
        with_vision: load the image features.
        device: device of all returned tensors; storage_device: where the frames are kept (default: ``device``;
            'cpu' saves GPU memory, results are identical).
        skip_missing: skip episodes without a tactile file instead of failing.
    """

    def __init__(self, tactile_root: str, feature_root: Optional[str] = None, stats_file: Optional[str] = None,
                 tasks: Sequence[str] = UNIVTAC_TASKS, episodes: Union[str, Iterable[int]] = '0-89',
                 with_vision: bool = True, device: str = 'cuda', storage_device: Optional[str] = None,
                 skip_missing: bool = False, verbose: bool = True):
        st = load_tactile_stats(stats_file or DEFAULT_STATS_FILE)
        eps = parse_episodes(episodes)
        self.dev = torch.device(device)
        self.store = torch.device(storage_device) if storage_device else self.dev
        self.tasks = list(tasks)
        rest_np = np.array(st['rest_grid'], np.float32)  # (63, 2) global rest grid in 7 x 9 raster order
        self.samp = grid_sampling_coords(torch.tensor(rest_np).reshape(GRID_R, GRID_C, 2), self.dev)
        self.sigma_u = float(st['sigma_u'])
        m_mean, m_std = np.array(st['m_mean'], np.float32), np.array(st['m_std'], np.float32)
        W, D, V, Q, VIS, EC, M, MV, meta = [], [], [], [], [], [], [], [], []
        start = 0
        for ti, task in enumerate(self.tasks):
            for ep in eps:
                path = osp.join(tactile_root, task, f'{ep:04d}.npz')
                if skip_missing and not osp.exists(path):
                    continue
                z = np.load(path)
                T = len(z['joint'])
                mk = z['mk'].astype(np.float32)  # (T, 2 gels, [reference, current], 63, xy)
                d = np.zeros((T, 2, 63, 2), np.float32)
                v = np.zeros((T, 2, 63), bool)
                for g in range(2):
                    o = grid_order(z['rest'][g])  # episode markers -> 7 x 9 raster
                    d[:, g] = mk[:, g, 1][:, o] - rest_np[None]
                    v[:, g] = z['valid'][:, g][:, o]
                W.append(z['w96'].astype(np.float16))
                D.append(d.reshape(T, 2, GRID_R, GRID_C, 2).astype(np.float16))
                V.append(v.reshape(T, 2, GRID_R, GRID_C))
                Q.append(z['joint'][:, :8].astype(np.float32))
                EC.append(z['e_c'].astype(bool))
                M.append(normalize_m(z['m'], z['m_valid'], m_mean, m_std))
                MV.append(z['m_valid'].astype(bool))
                if with_vision:
                    if feature_root is None:
                        raise ValueError('with_vision=True needs feature_root')
                    zv = np.load(osp.join(feature_root, task, f'{ep:04d}.npz'))
                    assert zv['patch'].shape[0] == T, (task, ep, zv['patch'].shape, T)
                    VIS.append(zv['patch'])
                meta.append(dict(task=task, ti=ti, ep=ep, start=start, T=T))
                start += T
        if not meta:
            raise FileNotFoundError(f'no episodes found under {tactile_root} for tasks {self.tasks}')
        cat = lambda a: torch.from_numpy(np.concatenate(a, 0)).to(self.store)  # noqa: E731
        self.W, self.D, self.V, self.Q = cat(W), cat(D), cat(V), cat(Q)
        self.EC, self.M, self.MV = cat(EC), cat(M), cat(MV)
        self.VIS = cat(VIS) if with_vision else None
        self.meta = meta
        self.n_frames = start
        dev = self.dev
        self.frame_ep = torch.cat([torch.full((e['T'],), i, device=dev) for i, e in enumerate(meta)])
        self.frame_t = torch.cat([torch.arange(e['T'], device=dev) for e in meta])
        self.ep_len = torch.tensor([e['T'] for e in meta], device=dev)
        self.ep_task = torch.tensor([e['ti'] for e in meta], device=dev)
        self.frame_task = self.ep_task[self.frame_ep]
        if verbose:
            print(f'[UniVTACTokenizerData] {len(self.tasks)} tasks x {len(eps)} eps: {len(meta)} episodes, '
                  f'{start} frames, vision={with_vision}', flush=True)

    # ---------------------------------------------------------------------------------------------------- helpers
    def _get(self, x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if self.store == self.dev:
            return x[idx]
        return x[idx.to(self.store)].to(self.dev)

    def anchors(self, horizon: int) -> torch.Tensor:
        """Global indices of the frames t with t + horizon inside the same episode."""
        return torch.nonzero(self.frame_t <= self.ep_len[self.frame_ep] - 1 - horizon)[:, 0]

    def fields(self, idx: torch.Tensor, augment: bool = False, gen: Optional[torch.Generator] = None) -> torch.Tensor:
        """idx (N,) global frame indices -> (N, 2 gels, 3, 96, 128) fp32 tactile observations. ``augment`` drops a
        random fraction U(0, 0.4) of the markers of every field (in-painted like tracking anomalies)."""
        N = idx.numel()
        drop = torch.rand(N * 2, device=self.dev, generator=gen) * 0.4 if augment else None
        X = build_fields(self._get(self.W, idx).reshape(N * 2, 96, 128),
                         self._get(self.D, idx).reshape(N * 2, GRID_R, GRID_C, 2),
                         self._get(self.V, idx).reshape(N * 2, GRID_R, GRID_C), self.sigma_u, self.samp,
                         drop_frac=drop)
        return X.reshape(N, 2, 3, 96, 128)

    def fields_at(self, idx: torch.Tensor, offsets: Sequence[int]) -> torch.Tensor:
        """(N,) anchors -> (N, len(offsets), 2, 3, 96, 128) tactile observations at t + offsets."""
        off = torch.as_tensor(offsets, device=idx.device)
        return self.fields((idx[:, None] + off[None]).reshape(-1)).reshape(len(idx), len(offsets), 2, 3, 96, 128)

    def vis_at(self, idx: torch.Tensor, offsets: Sequence[int]) -> Optional[torch.Tensor]:
        """(N,) -> (N, len(offsets), 2, 16, 768) image features of (head, wrist) at t + offsets (None without vision)."""
        if self.VIS is None:
            return None
        off = torch.as_tensor(offsets, device=idx.device)
        return self._get(self.VIS, idx[:, None] + off[None])

    def actions(self, idx: torch.Tensor, horizon: int) -> torch.Tensor:
        """(N,) anchors -> (N, horizon, 8) executed relative joint targets joint[t+i, :8] - joint[t, :8], i = 1..horizon
        (recorded frames are the executed frame targets)."""
        off = torch.arange(1, horizon + 1, device=idx.device)
        return self._get(self.Q, idx[:, None] + off[None]) - self._get(self.Q, idx)[:, None]

    def action_stats(self, anchors: torch.Tensor, horizon: int, n: int = 200000, seed: int = 0):
        """Per (step, joint) mean / std of :meth:`actions` over (a random subset of at most ``n``) anchors."""
        a = anchors
        if len(a) > n:
            g = torch.Generator(device=a.device).manual_seed(seed)
            a = a[torch.randperm(len(a), device=a.device, generator=g)[:n]]
        c = self.actions(a, horizon)
        return c.mean(0), c.std(0).clamp(min=1e-6)
