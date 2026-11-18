"""Request decoding, tactile-off substitution and metadata of the policy server (no model, CPU)."""
from __future__ import annotations

import types
import zlib

import numpy as np
import pytest

pytest.importorskip("tacinno.datasets.tactile_fields")
from tacinno.apis.server import PolicyServer  # noqa: E402


class FakePolicy:
    real_robot = False
    checkpoint_dir = "/ckpt/run/ema_19999"
    prompts = {"lift_can": "p0", "insert_hole": "p1"}
    horizon = 32
    cameras = ("head", "wrist")
    H = 8
    use_field = True
    tactile_offsets = [8, 4, 0]
    model_config = types.SimpleNamespace(cond_code=True, cond_code_zero=True)

    def __init__(self):
        self.task, self.calls = "lift_can", []

    def select(self, task):
        self.task = task

    def build_cond(self, qhist=None, tac_frames=None):
        self.calls.append((np.asarray(qhist), tac_frames))
        return {"field": np.zeros((2, 3, 96, 128), np.float32)}

    def infer_model8(self, head, wrist, qpos, cond=None, noise_seed=None):
        self.last_seed = noise_seed
        return np.zeros((32, 8), np.float32)


def _frame(rng, packed=True):
    f = {"l_marker": rng.normal(size=(2, 64, 2)).astype(np.float32), "r_marker": rng.normal(size=(2, 64, 2)).astype(np.float32)}
    for side in ("l", "r"):
        d = rng.normal(size=(240, 320)).astype(np.float32)
        if packed:
            f[f"{side}_depth_z"] = zlib.compress(d.tobytes(), 1)
            f[f"{side}_depth_shape"] = np.array(d.shape)
        else:
            f[f"{side}_depth"] = d
        f[f"_{side}_ref"] = d
    return f


def test_request_decoding_and_metadata():
    rng = np.random.default_rng(0)
    p = FakePolicy()
    srv = PolicyServer(p, [("lift_can", 1), ("insert_hole", 2)])
    frames = [_frame(rng), _frame(rng, packed=False), _frame(rng)]
    q = rng.normal(size=(8,)).astype(np.float32)
    obs = {"head": np.zeros((240, 320, 3), np.uint8), "wrist": np.zeros((240, 320, 3), np.uint8), "qpos": q,
           "qhist": np.stack([q] * 3), "tac_hist": frames, "seed": 3, "k": 7}
    out = srv.infer("insert_hole", obs)
    assert out["model8"].shape == (32, 8) and p.task == "insert_hole" and p.last_seed == 3007
    qhist, tac = p.calls[-1]
    assert qhist.shape == (3, 8) and len(tac) == 3
    for got, f in zip(tac, frames):
        assert np.array_equal(got[0], f["_l_ref"]) and np.array_equal(got[2], f["_r_ref"])
        assert np.array_equal(got[1], f["l_marker"]) and np.array_equal(got[3], f["r_marker"])
    meta = srv.metadata("lift_can")
    for k in ("task", "checkpoint", "action_rep", "action_horizon", "proprio_hist", "use_field", "tac_hist_offsets",
              "tactile_off"):
        assert k in meta
    assert meta["tac_hist_offsets"] == [8, 4, 0] and meta["tactile_off"] is False and meta["proprio_hist"] == 8


def test_tactile_off_replaces_every_frame():
    rng = np.random.default_rng(1)
    rest = tuple(rng.normal(size=s).astype(np.float32) for s in ((240, 320), (2, 64, 2), (240, 320), (2, 64, 2)))
    p = FakePolicy()
    srv = PolicyServer(p, [("lift_can", 1)], rest_frame=rest)
    obs = {"head": np.zeros((8, 8, 3), np.uint8), "wrist": np.zeros((8, 8, 3), np.uint8),
           "qpos": np.zeros(8, np.float32), "tac_hist": [_frame(rng) for _ in range(3)]}
    srv.infer("lift_can", obs)
    _, tac = p.calls[-1]
    assert all(t is rest for t in tac)
    assert srv.metadata("lift_can")["tactile_off"] is True
