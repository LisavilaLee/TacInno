"""The policy configs reproduce the settings of the released TacInno-VLA checkpoints; data and engine utilities."""
from __future__ import annotations

import glob
import json
import os
import os.path as osp

import numpy as np
import pytest

from tacinno.config import Config
from tacinno.datasets.condition_data import ConditionData

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
CFG = osp.join(ROOT, "configs")

# model_config.json values shared by every released TacInno-VLA checkpoint
COMMON_MODEL = dict(action_dim=120, action_horizon=32, max_token_len=150, paligemma_variant="gemma_2b",
                    action_expert_variant="gemma_300m", state_input_mode="adarms", use_tactile_input=False,
                    attn_implementation="eager", cond_tactile=True, cond_code=True, cond_gate_init=0.1,
                    cond_code_zero=True, cond_code_per_slot=False, cond_tac_aux=True, cond_tac_hist="8,4",
                    cond_active="tac,code", cond_code_to_action=False, cond_tac_tokens=10, cond_code_head=True)
JOINT_LAMBDA = dict(cond_code_ce_weight=0.11862334646115223, cond_tac_aux_weight=0.46727895124740054)
TASK_LAMBDA = dict(cond_code_ce_weight=0.119, cond_tac_aux_weight=0.467)
FT = dict(num_train_steps=17000, warmup_steps=200, peak_lr=5e-6, decay_lr=5e-6)
SINGLE = dict(num_train_steps=20000, warmup_steps=500, peak_lr=5e-5, decay_lr=5e-6)
JOINT = dict(num_train_steps=17000, warmup_steps=500, peak_lr=5e-5, decay_lr=5e-6)
# config -> (proprio tokens, lambdas, schedule, tasks, init type, EMA)
RECIPES = {
    "tacinno_vla/init/tacinno-vla_univtac-lift-can_init.py": (8, JOINT_LAMBDA, JOINT, 6, "pi05_base", 0.0),
    "tacinno_vla/init/tacinno-vla_univtac-insert-hole_init.py": (0, JOINT_LAMBDA, JOINT, 6, "pi05_base", 0.0),
    "tacinno_vla/tacinno-vla_univtac-lift-can.py": (8, TASK_LAMBDA, FT, ["lift_can"], "checkpoint", 0.99),
    "tacinno_vla/tacinno-vla_univtac-insert-hole.py": (0, TASK_LAMBDA, FT, ["insert_hole"], "checkpoint", 0.99),
    "tacinno_vla/tacinno-vla_univtac-insert-tube.py": (0, TASK_LAMBDA, SINGLE, ["insert_tube"], "pi05_base", 0.99),
    "tacinno_vla/tacinno-vla_univtac-lift-bottle.py": (0, TASK_LAMBDA, SINGLE, ["lift_bottle"], "pi05_base", 0.99),
    "tacinno_vla/tacinno-vla_univtac-pull-out-key.py": (0, TASK_LAMBDA, SINGLE, ["pull_out_key"], "pi05_base", 0.99),
    "tacinno_vla/tacinno-vla_univtac-put-bottle.py": (0, TASK_LAMBDA, SINGLE, ["put_bottle_in_shelf"], "pi05_base", 0.99),
    "real_robot/tacinno-vla_real-4task.py": (
        0, dict(cond_code_ce_weight=0.11798233046789909, cond_tac_aux_weight=0.40614703135347074), SINGLE, 4, "pi05_base", 0.99),
}


def test_every_policy_config_is_covered():
    found = {osp.relpath(p, CFG) for d in ("tacinno_vla", "real_robot")
             for p in glob.glob(osp.join(CFG, d, "**", "*.py"), recursive=True)}
    assert found == set(RECIPES)


@pytest.mark.parametrize("name", sorted(RECIPES))
def test_recipe(name, tmp_path):
    pytest.importorskip("openpi.training.config")
    from tacinno.engine.train_config import build_train_config, write_dataset_config

    proprio, lam, sched, tasks, init, ema = RECIPES[name]
    cfg = Config.fromfile(osp.join(CFG, name))
    ds = write_dataset_config(cfg.data, str(tmp_path / "dataset_config.json"))
    tc = build_train_config(cfg, str(tmp_path / "run"), resume=False, dataset_config_path=ds)
    model = tc.model
    for k, v in {**COMMON_MODEL, **lam, "cond_proprio_hist": proprio}.items():
        assert getattr(model, k) == v, k
    assert tc.batch_size == 64 and tc.seed == 42 and tc.action_joint_rep == "mix" and tc.norm_type == "zscore"
    assert tc.num_train_steps == sched["num_train_steps"]
    s = tc.lr_schedule
    assert (s.warmup_steps, s.peak_lr, s.decay_lr, s.decay_steps) == (
        sched["warmup_steps"], sched["peak_lr"], sched["decay_lr"], sched["num_train_steps"])
    o = tc.optimizer
    assert (o.b1, o.b2, o.eps, o.weight_decay, o.clip_gradient_norm) == (0.9, 0.95, 1e-8, 1e-10, 1.0)
    n_tasks = tasks if isinstance(tasks, int) else len(tasks)
    entries = json.load(open(ds))["datasets"]
    assert len(entries) == n_tasks
    if not isinstance(tasks, int):
        assert [e["name"].split("_", 1)[1] for e in entries] == tasks
    keys = set(tc.cond_keys)
    assert keys == ({"proprio"} if proprio else set()) | {"field", "field_hist", "code_tgt", "aux_hist"}
    assert tuple(tc.cond_tac_hist) == (8, 4)
    assert cfg.init_weights["type"] == init and float(cfg.schedule["ema_decay"]) == ema
    assert tc.checkpoint_dir == (tmp_path / "run").resolve()


def test_put_bottle_uses_head_camera_only():
    cfg = Config.fromfile(osp.join(CFG, "tacinno_vla/tacinno-vla_univtac-put-bottle.py"))
    assert tuple(cfg.deploy["cameras"]) == ("head",) and cfg.data["zarr_dir"].startswith("head_cam")


def _write_cond(root, n_ep=2, length=12):
    d = osp.join(root, "train", "lift_can")
    os.makedirs(d)
    n = n_ep * length
    rng = np.random.default_rng(0)
    np.save(osp.join(d, "joints8.npy"), rng.normal(size=(n, 8)).astype(np.float32))
    np.save(osp.join(d, "fields.npy"), np.arange(n, dtype=np.float16)[:, None, None, None, None]
            * np.ones((1, 2, 3, 96, 128), np.float16))
    np.save(osp.join(d, "code_tgt.npy"), rng.integers(0, 16, (n, 4)))
    np.save(osp.join(d, "code_tv.npy"), np.ones(n, bool))
    for k, shape in (("aux_m", (n, 2, 14)), ("aux_mv", (n, 2)), ("aux_ec", (n, 2))):
        np.save(osp.join(d, f"{k}.npy"), rng.normal(size=shape).astype(np.float32))
    json.dump(dict(n=n), open(osp.join(d, "meta.json"), "w"))
    json.dump(dict(j_mean=[0.0] * 8, j_std=[1.0] * 8), open(osp.join(root, "cond_stats.json"), "w"))
    return n


def test_condition_data_history_is_clipped_at_episode_start(tmp_path):
    _write_cond(str(tmp_path))
    cd = ConditionData(str(tmp_path / "zarr_train" / "lift_can"), cond_root=str(tmp_path),
                       cond_keys=("proprio", "field", "field_hist", "code_tgt", "aux_hist"), proprio_hist=8,
                       tac_hist=(8, 4))
    # second episode starts at frame 12; frame 14 -> history frames clipped to 12
    out = cd(14, 12)
    assert out["proprio"].shape == (8, 8)
    j = np.load(tmp_path / "train" / "lift_can" / "joints8.npy")
    assert np.allclose(out["proprio"][:6], j[12]) and np.allclose(out["proprio"][-1], j[14])
    assert out["field"][0, 0, 0, 0] == 14
    assert [float(f[0, 0, 0, 0]) for f in out["field_hist"]] == [12.0, 12.0]  # t-8, t-4 clipped
    assert out["aux_m"].shape == (3, 2, 14) and out["code_tgt"].shape == (4,)
    off = ConditionData(str(tmp_path / "zarr_train" / "lift_can"), cond_root="", cond_keys=("field",))
    assert not off.enabled


def test_lr_schedule():
    pytest.importorskip("openpi.training.data_loader")
    from tacinno.engine.trainer import lr_at_step

    kw = dict(warmup_steps=500, peak_lr=5e-5, decay_steps=20000, end_lr=5e-6)
    assert lr_at_step(0, **kw) == pytest.approx(5e-5 / 501)
    assert lr_at_step(500, **kw) == pytest.approx(5e-5)
    assert lr_at_step(20000, **kw) == pytest.approx(5e-6)
    const = dict(warmup_steps=200, peak_lr=5e-6, decay_steps=17000, end_lr=5e-6)
    assert lr_at_step(9000, **const) == pytest.approx(5e-6)


def test_ema_update_and_export(tmp_path):
    import torch

    from tacinno.engine.ema import PolicyEMA

    torch.manual_seed(0)
    m = torch.nn.Linear(4, 3)
    ema = PolicyEMA(m, 0.9)
    w0 = m.weight.detach().clone()
    with torch.no_grad():
        m.weight.add_(1.0)
    ema.update(m)
    assert torch.allclose(ema.shadow["weight"], 0.9 * w0 + 0.1 * m.weight.detach(), atol=1e-6)
    ema.save(tmp_path / "ema.safetensors")
    other = PolicyEMA(m, 0.9)
    other.load(tmp_path / "ema.safetensors", "cpu")
    assert other.updates == 1 and torch.equal(other.shadow["weight"], ema.shadow["weight"])
    src = tmp_path / "19999"
    src.mkdir()
    (src / "model_config.json").write_text("{}")
    (src / "optimizer.pt").write_text("x")
    out = ema.export_deploy(m, src, tmp_path / "ema_19999")
    assert (out / "model_config.json").exists() and not (out / "optimizer.pt").exists()
    import safetensors.torch
    assert torch.allclose(safetensors.torch.load_file(str(out / "model.safetensors"))["weight"], ema.shadow["weight"])


def test_resumable_sampler_skips_once():
    pytest.importorskip("openpi.training.data_loader")
    from openpi.training.data_loader import ResumableDistributedSampler

    data = list(range(40))
    s = ResumableDistributedSampler(data, num_replicas=2, rank=0, shuffle=True, drop_last=True)
    full = list(iter(s))
    s.set_skip(6)
    assert list(iter(s)) == full[6:]
    assert list(iter(s)) == full  # one-shot
