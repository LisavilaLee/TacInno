"""Per-frame condition inputs and auxiliary targets of TacInno-VLA.

The policy reads, besides the camera images, the language prompt and the current proprioceptive state (handled by the
pi0.5 data pipeline), the following per-frame arrays. They are precomputed by ``tools/data/make_cond.py`` (and the code
targets by ``tools/data/make_code_targets.py``) and stored next to the zarr data, frame-aligned with the zarr frame index
(frame ``j`` of zarr episode ``k`` is frame ``j`` of recorded episode ``k``)::

    <cond_root>/cond_stats.json                 mean / std of the 8 joint values over the training frames
    <cond_root>/<split>/<task>/meta.json        {"n": number of frames}
    <cond_root>/<split>/<task>/joints8.npy      (N, 8)  float  7 arm joints + gripper opening
    <cond_root>/<split>/<task>/fields.npy       (N, 2, 3, 96, 128) float16  tactile observation of both gels
    <cond_root>/<split>/<task>/code_tgt.npy     (N, 4)  int    tactile innovation code of frame t (4 slots x 16 entries)
    <cond_root>/<split>/<task>/code_tv.npy      (N,)    bool   code target valid (t + 15 inside the episode)
    <cond_root>/<split>/<task>/aux_{m,mv,ec}.npy (N, 2, 14) / (N, 2) / (N, 2)  contact-state targets per gel

``split`` and ``task`` are taken from the dataset path ``.../zarr_<split>/<task>``.

Keys (``cond_keys``) and what a sample gets under ``sample["cond"]``:
    proprio     (H, 8)  joints of frames t-H+1 .. t, normalized with cond_stats.json; frames before the episode start
                        repeat the first frame
    field       (2, 3, 96, 128)  tactile observation of frame t (indentation field and the two marker tangential
                        displacement fields of the left and the right gel)
    field_hist  (F, 2, 3, 96, 128)  tactile observations of frames t-o for o in ``cond_tac_hist`` (oldest first),
                        clipped at the episode start
    code_tgt, code_tv   target code of frame t and its validity (training only)
    aux_hist    aux_m (F+1, 2, 14), aux_mv (F+1, 2), aux_ec (F+1, 2): contact-state targets of frames t-o and t
                        (training only)
    aux         contact-state targets of frame t only
    code_prob   (4, 16) externally predicted code probabilities (not used by TacInno-VLA; kept for completeness)
"""
from __future__ import annotations

import json
import os
from typing import Any, Iterable, Sequence

import numpy as np

__all__ = ["ConditionData", "CONDITION_KEYS"]

CONDITION_KEYS = ("proprio", "field", "field_hist", "code_prob", "code_tgt", "aux", "aux_hist")


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, str):
        return [v.strip() for v in value.split(",") if v.strip()]
    return list(value)


class ConditionData:
    """Reads the condition arrays of one zarr dataset (one task and split); see the module docstring."""

    def __init__(self, dataset_path: str, cond_root: str = "", cond_keys: Iterable[str] = (),
                 proprio_hist: int = 8, tac_hist: Sequence[int] = ()):
        keys = [k for k in _as_list(cond_keys)]
        self.enabled = bool(cond_root) and bool(keys)
        if not self.enabled:
            return
        unknown = set(keys) - set(CONDITION_KEYS)
        if unknown:
            raise ValueError(f"unknown condition keys {sorted(unknown)}; valid keys: {CONDITION_KEYS}")
        parts = os.path.normpath(dataset_path).split(os.sep)
        split = parts[-2].replace("zarr_", "")
        task = parts[-1]
        if split not in ("train", "val"):
            raise ValueError(f"condition data: cannot infer the split from {dataset_path} (expected .../zarr_<split>/<task>)")
        self.dir = os.path.join(cond_root, split, task)
        self.keys = keys
        self.hist = int(proprio_hist)
        self.tac_hist = [int(o) for o in _as_list(tac_hist)]
        if ("field_hist" in keys or "aux_hist" in keys) and not self.tac_hist:
            raise ValueError("condition keys field_hist / aux_hist need tactile history offsets (cond_tac_hist)")
        with open(os.path.join(cond_root, "cond_stats.json")) as f:
            stats = json.load(f)
        self.j_mean = np.asarray(stats["j_mean"], np.float32)
        self.j_std = np.asarray(stats["j_std"], np.float32)
        self._arr: dict[str, np.ndarray] = {}
        with open(os.path.join(self.dir, "meta.json")) as f:
            self.n = int(json.load(f)["n"])

    @classmethod
    def from_data_config(cls, dataset_path: str, data_config: Any) -> "ConditionData":
        """Build from the cond_* fields of the openpi data config (absent fields switch the condition data off)."""
        return cls(dataset_path,
                   cond_root=getattr(data_config, "cond_root", "") or "",
                   cond_keys=getattr(data_config, "cond_keys", ()) or (),
                   proprio_hist=getattr(data_config, "cond_proprio_hist", 8),
                   tac_hist=getattr(data_config, "cond_tac_hist", ()) or ())

    def _get(self, name: str) -> np.ndarray:
        if name not in self._arr:  # memory-mapped lazily in each data-loader worker
            self._arr[name] = np.load(os.path.join(self.dir, name), mmap_mode="r")
        return self._arr[name]

    def __call__(self, data_idx: int, start_idx: int) -> dict:
        """Condition inputs of zarr frame ``data_idx`` whose episode starts at frame ``start_idx``."""
        if data_idx >= self.n:
            raise IndexError(f"condition data: frame {data_idx} >= {self.n} in {self.dir}")
        out = {}
        if "proprio" in self.keys:
            idx = np.clip(np.arange(data_idx - self.hist + 1, data_idx + 1), start_idx, None)
            out["proprio"] = (np.asarray(self._get("joints8.npy")[idx], np.float32) - self.j_mean) / self.j_std
        if "field" in self.keys:
            out["field"] = np.asarray(self._get("fields.npy")[data_idx], np.float32)
        if "field_hist" in self.keys:
            idx = [max(data_idx - o, start_idx) for o in self.tac_hist]
            out["field_hist"] = np.asarray(self._get("fields.npy")[idx], np.float32)
        if "code_prob" in self.keys:
            out["code_prob"] = np.asarray(self._get("code_prob.npy")[data_idx], np.float32)
        if "code_tgt" in self.keys:
            out["code_tgt"] = np.asarray(self._get("code_tgt.npy")[data_idx], np.int64)
            out["code_tv"] = np.float32(self._get("code_tv.npy")[data_idx])
        if "aux" in self.keys:
            out["aux_m"] = np.asarray(self._get("aux_m.npy")[data_idx], np.float32)
            out["aux_mv"] = np.asarray(self._get("aux_mv.npy")[data_idx], np.float32)
            out["aux_ec"] = np.asarray(self._get("aux_ec.npy")[data_idx], np.float32)
        if "aux_hist" in self.keys:
            idx = [max(data_idx - o, start_idx) for o in self.tac_hist] + [data_idx]
            for k in ("aux_m", "aux_mv", "aux_ec"):
                out[k] = np.asarray(self._get(f"{k}.npy")[idx], np.float32)
        return out
