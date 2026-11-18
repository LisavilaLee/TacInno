"""Translate a TacInno policy config (``configs/tacinno_vla/*.py``) into the vendored pi0.5 training config.

Config sections read here (see ``configs/_base_``)::

    model    = dict(type="TacInnoVLA", proprio_history=..., ...)      # tacinno.models.policy.build_model_config
    data     = dict(data_root, zarr_dir, cond_dir, norm_root, repo_id, domain_prefix, tasks, cond_keys, ...)
    schedule = dict(num_steps, batch_size, warmup_steps, peak_lr, end_lr, optimizer=dict(...), ema_decay, ema_dtype)
    runtime  = dict(seed, precision, use_torch_compile, log_interval, save_interval, keep_period, keep_last, pre_delete)
    init_weights = dict(type="pi05_base" | "checkpoint", path=...)   # weights of a fresh run

The learning rate warms up linearly for ``warmup_steps`` from peak / (warmup + 1) to ``peak_lr`` and then follows a
cosine from ``peak_lr`` to ``end_lr`` at ``num_steps`` (``peak_lr == end_lr`` gives a constant rate after warm-up).
"""
from __future__ import annotations

import json
import os
import os.path as osp
from typing import Any, Mapping

import openpi.training.config as _openpi_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as _weight_loaders
from openpi.models_pytorch.ftp1_model_config import FTP1ModelConfig

from tacinno.models.policy.tacinno_vla import build_model_config

__all__ = ["build_model_config_from_cfg", "build_train_config", "write_dataset_config", "condition_keys"]


def _abs(root: str, path: str) -> str:
    return path if osp.isabs(path) else osp.join(root, path)


def build_model_config_from_cfg(model_cfg: Mapping[str, Any], assets_dir: str | None = None) -> FTP1ModelConfig:
    if model_cfg.get("type", "TacInnoVLA") != "TacInnoVLA":
        raise ValueError(f"model.type must be TacInnoVLA, got {model_cfg.get('type')!r}")
    kw = {k: v for k, v in dict(model_cfg).items() if k != "type"}
    return build_model_config(assets_dir=assets_dir, **kw)


def condition_keys(model_cfg: Mapping[str, Any], data_cfg: Mapping[str, Any]) -> tuple[str, ...]:
    """Condition arrays the data loader reads: 'proprio' when the model has proprio tokens, then data.cond_keys."""
    keys = list(data_cfg.get("cond_keys", ("field", "field_hist", "code_tgt", "aux_hist")))
    if int(model_cfg.get("proprio_history", 0)) > 0 and "proprio" not in keys:
        keys.insert(0, "proprio")
    if int(model_cfg.get("proprio_history", 0)) == 0 and "proprio" in keys:
        keys.remove("proprio")
    return tuple(keys)


def write_dataset_config(data_cfg: Mapping[str, Any], out_path: str) -> str:
    """Write the zarr dataset list (one entry per task) read by the pi0.5 data pipeline; returns its path."""
    zarr_dir = _abs(data_cfg["data_root"], data_cfg.get("zarr_dir", "zarr_train"))
    prefix = data_cfg.get("domain_prefix", "UniVTAC_")
    entries = [dict(name=f"{prefix}{t}", path=osp.join(zarr_dir, t), use_trajectory_ratio=1.0, enabled=True)
               for t in data_cfg["tasks"]]
    os.makedirs(osp.dirname(osp.abspath(out_path)), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(dict(datasets=entries, default_use_trajectory_ratio=1.0,
                       description=data_cfg.get("description", "TacInno-VLA training data")), f, indent=1)
    return out_path


def build_train_config(cfg, work_dir: str, *, resume: bool, dataset_config_path: str):
    """Return the vendored ``FTP1TrainConfig`` (finalized) of a TacInno policy config; checkpoints go to ``work_dir``."""
    data, sched, rt = cfg["data"], cfg["schedule"], cfg.get("runtime", {})
    opt = sched.get("optimizer", {})
    norm_root = _abs(data["data_root"], data.get("norm_root", "norm"))
    model_config = build_model_config_from_cfg(cfg["model"], assets_dir=norm_root)
    init = cfg.get("init_weights") or {}
    weight_path = None if resume else (init.get("path") or None)
    work_dir = osp.abspath(work_dir)
    steps = int(sched["num_steps"])
    tac_hist = tuple(int(o) for o in cfg["model"].get("tactile_history", (8, 4))) if cfg["model"].get("use_tactile", True) else ()
    train_cfg = _openpi_config.FTP1TrainConfig(
        name="",  # checkpoints go to <checkpoint_base_dir>/<exp_name> = work_dir, statistics to <assets_base_dir>/<repo_id>
        exp_name=osp.basename(work_dir),
        checkpoint_base_dir=osp.dirname(work_dir),
        assets_base_dir=norm_root,
        model=model_config,
        weight_loader=_weight_loaders.NoOpWeightLoader(),
        pytorch_weight_path=weight_path,
        resume=bool(resume),
        repo_id=data["repo_id"],
        dataset_config_path=dataset_config_path,
        check_norm_params_snapshot=bool(data.get("check_norm_params_snapshot", False)),
        batch_size=int(sched.get("batch_size", 64)),
        action_down_sample_steps=1,
        num_train_steps=steps,
        log_interval=int(rt.get("log_interval", 50)),
        val_interval=1_000_000_000,
        val_ratio=0.0,
        save_interval=int(rt.get("save_interval", 5000)),
        keep_period=int(rt.get("keep_period", 100_000)),
        num_workers=int(data.get("num_workers", 12)),
        val_num_workers=2,
        use_val_dataset=True,
        create_train_val_split=True,
        norm_type="zscore",
        norm_image_tactile_mode="channel_wise",
        pytorch_training_precision=rt.get("precision", "bfloat16"),
        proprioception_pose_rep="relative",
        action_pose_rep="relative",
        proprioception_joint_rep="abs",
        action_joint_rep="mix",
        lr_schedule=_optimizer.CosineDecaySchedule(warmup_steps=int(sched.get("warmup_steps", 500)),
                                                   peak_lr=float(sched.get("peak_lr", 5e-5)), decay_steps=steps,
                                                   decay_lr=float(sched.get("end_lr", 5e-6))),
        optimizer=_optimizer.AdamW(b1=float(opt.get("b1", 0.9)), b2=float(opt.get("b2", 0.95)),
                                   eps=float(opt.get("eps", 1e-8)), weight_decay=float(opt.get("weight_decay", 1e-10)),
                                   clip_gradient_norm=float(opt.get("clip_grad_norm", 1.0))),
        lr_decay_till_end=True,
        ema_decay=float(sched.get("ema_decay", 0.99) or 0.99),
        wandb_enabled=False,
        use_torch_compile=bool(rt.get("use_torch_compile", True)),
        gradient_checkpointing_enable=bool(rt.get("gradient_checkpointing", False)),
        seed=int(rt.get("seed", 42)),
        cond_root=_abs(data["data_root"], data.get("cond_dir", "cond")),
        cond_keys=condition_keys(cfg["model"], data),
        cond_proprio_hist=int(cfg["model"].get("proprio_history", 0)) or 8,
        cond_tac_hist=tac_hist,
    )
    train_cfg.finalize_config()
    return train_cfg
