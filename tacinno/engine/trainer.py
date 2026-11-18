"""Policy trainer of TacInno-VLA (single process or ``torchrun``; DDP, bf16, torch.compile).

One training step: take one batch (all samples of a batch come from one task, as in pi0.5 multi-task training),
compute ``L = L_flow + lambda_contact * L_contact + lambda_code * L_code`` (model forward), backward, clip the gradient
norm, AdamW step, EMA update. Checkpoints (``<work_dir>/<step>/``) hold the model, the optimizer, the EMA weights,
every rank's RNG states and the configs and normalization statistics; resuming restores all of them and the exact data
position (epoch = step // batches per epoch, skipping step % batches per epoch batches), and continues at step s + 1.
The last step also writes the deployable EMA checkpoint ``<work_dir>/ema_<step>/``.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import random
import shutil
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.parallel

import openpi.training.config as _openpi_config
import openpi.training.data_loader as _data
from openpi.training.ftp1_train_utils import (
    _unwrap_compile_model,
    check_train_config_consistency,
    collect_trainable_parameters,
    get_ftp1_latest_checkpoint_step,
    load_ftp1_hpt_checkpoint,
    load_ftp1_model,
    load_partial_weights,
    move_to_device,
    print_model_statistics,
    save_ftp1_hpt_checkpoint,
)

from tacinno.engine.dist import cleanup_ddp, log_memory_usage, set_seed, setup_ddp
from tacinno.engine.ema import PolicyEMA
from tacinno.models.policy.tacinno_vla import TacInnoVLA

__all__ = ["PolicyTrainer", "lr_at_step"]


def lr_at_step(step: int, *, warmup_steps: int, peak_lr: float, decay_steps: int, end_lr: float) -> float:
    """Linear warm-up from peak / (warmup + 1) to peak, then cosine decay from peak to end_lr at decay_steps."""
    if step < warmup_steps:
        init_lr = peak_lr / (warmup_steps + 1)
        return init_lr + (peak_lr - init_lr) * step / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, decay_steps - warmup_steps))
    return end_lr + (peak_lr - end_lr) * 0.5 * (1 + np.cos(np.pi * progress))


def _barrier(device: torch.device) -> None:
    if device.type == "cuda":
        dist.barrier(device_ids=[device.index])
    else:  # gloo (CPU) process group
        dist.barrier()


def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    m = _unwrap_compile_model(model)
    return m.module if isinstance(m, torch.nn.parallel.DistributedDataParallel) else m


def _configure_compile_cache(base_dir: str | None, rank: int, local_rank: int) -> None:
    """Rank-local torch.compile / triton caches (avoids directory collisions between ranks)."""
    if not base_dir:
        return
    suffix = f"rank{rank}_local{local_rank}_pid{os.getpid()}"
    root = os.path.join(base_dir, "torch_compile_cache")
    for env_name, sub in (("TORCH_COMPILE_DIR", "compile"), ("TRITON_CACHE_DIR", "triton"),
                          ("TORCHINDUCTOR_CACHE_DIR", "inductor")):
        path = os.path.join(root, sub, suffix)
        os.makedirs(path, exist_ok=True)
        os.environ[env_name] = path


class PolicyTrainer:
    """Runs one training from a finalized vendored ``FTP1TrainConfig`` (see :func:`tacinno.engine.build_train_config`).

    Args:
        config: the vendored training config (data, model, schedule, checkpoint directory).
        ema_decay: policy-weight EMA decay (0 disables the EMA).
        ema_dtype: "fp32", "bf16" or "auto" (fp32 unless less than 20 GB would stay free on the device).
        keep_last / pre_delete: checkpoint retention (keep the newest ``keep_last`` step directories plus multiples of
            ``keep_period`` and the final step; ``pre_delete`` removes the older one before writing a new one).
        compile_cache_dir: base directory of the torch.compile caches (default: $TMPDIR).
        init_type: how ``config.pytorch_weight_path`` is loaded for a fresh run: "pi05_base" (pi0.5 base weights, keys
            matching by name and shape; the TacInno modules keep their initialization) or "checkpoint" (a TacInno-VLA
            checkpoint directory, e.g. the joint stage).
    """

    def __init__(self, config: _openpi_config.FTP1TrainConfig, *, ema_decay: float = 0.99, ema_dtype: str = "auto",
                 keep_last: int = 1, pre_delete: bool = False, compile_cache_dir: str | None = None,
                 init_type: str = "pi05_base"):
        if init_type not in ("pi05_base", "checkpoint"):
            raise ValueError(f"init_weights.type must be pi05_base or checkpoint, got {init_type!r}")
        self.config = config
        self.init_type = init_type
        self.ema_decay = float(ema_decay)
        self.ema_dtype = ema_dtype
        self.keep_last = int(keep_last)
        self.pre_delete = bool(pre_delete)
        self.compile_cache_dir = compile_cache_dir if compile_cache_dir is not None else os.environ.get("TMPDIR")

    # ------------------------------------------------------------------------------------------------------------ setup
    def _find_resume_step(self) -> int | None:
        cfg = self.config
        if not cfg.resume:
            return None
        if not cfg.checkpoint_dir.exists():
            raise FileNotFoundError(f"checkpoint directory {cfg.checkpoint_dir} does not exist for resume")
        latest_step, _ = get_ftp1_latest_checkpoint_step(cfg.checkpoint_dir)
        if latest_step is None:
            raise FileNotFoundError(f"no valid checkpoint in {cfg.checkpoint_dir} for resume")
        return latest_step

    def _check_norm_snapshot(self, data_config) -> None:
        cfg = self.config
        path = cfg.assets_dirs / data_config.repo_id / "norm_params_snapshot.json"
        if not path.exists():
            raise ValueError(f"normalization snapshot {path} not found: run tools/data/compute_norm_stats.py first")
        with open(path) as f:
            saved = json.load(f)
        diffs = _openpi_config.check_norm_params_snapshot_matches(saved, _openpi_config.get_norm_params_snapshot(cfg))
        if diffs:
            raise ValueError("normalization statistics were computed with other settings:\n" + "\n".join(diffs))

    def _build_model(self, device, model_cfg, tactile_input_config, resuming: bool) -> torch.nn.Module:
        cfg = self.config
        model = TacInnoVLA(model_config=model_cfg).to(device)
        if cfg.gradient_checkpointing_enable:
            model.gradient_checkpointing_enable()
        if resuming and cfg.pytorch_weight_path is not None:
            raise ValueError("init weights must be unset when resuming")
        if cfg.pytorch_weight_path is not None:
            path = cfg.pytorch_weight_path
            if self.init_type == "pi05_base":
                load_partial_weights(model, os.path.join(path, "model.safetensors"))  # pi0.5 base: matching keys only
                logging.info("loaded pi0.5 base weights from %s", path)
            else:
                load_ftp1_model(checkpoint_dir=path, tactile_input_config_file=tactile_input_config, device=device,
                                model=model, load_hpt_tokenizers_strict=False, allow_partial_load=cfg.allow_partial_load)
                logging.info("loaded TacInno-VLA weights from checkpoint %s", path)
        return model

    # ------------------------------------------------------------------------------------------------------------- run
    def train(self) -> None:
        cfg = self.config
        use_ddp, local_rank, device = setup_ddp()
        is_main = (not use_ddp) or dist.get_rank() == 0
        world_size = dist.get_world_size() if use_ddp else 1
        rank = dist.get_rank() if use_ddp else 0
        if cfg.use_torch_compile:
            _configure_compile_cache(self.compile_cache_dir, rank, local_rank)
        set_seed(cfg.seed, local_rank)
        logging.info("Running on %s | rank %d/%d | batch %d per GPU (global %d)", platform.node(), rank, world_size,
                     cfg.batch_size // world_size, cfg.batch_size)

        latest_step = self._find_resume_step()
        resuming = latest_step is not None
        if not resuming:
            if cfg.overwrite and cfg.checkpoint_dir.exists():
                shutil.rmtree(cfg.checkpoint_dir)
            cfg.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        elif is_main:
            check_train_config_consistency(cfg, cfg.checkpoint_dir, latest_step, strict=False)

        data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
        if cfg.check_norm_params_snapshot:
            self._check_norm_snapshot(data_config)
        train_loader, _ = _data.create_torch_data_loader(
            data_config, model_config=cfg.model, action_horizon=cfg.model.action_horizon, batch_size=cfg.batch_size,
            shuffle=True, num_batches=None, num_workers=cfg.num_workers, val_num_workers=cfg.val_num_workers,
            seed=cfg.seed, framework="pytorch", domain_batch_split=True, load_zarr_norm_stats=True)
        model_cfg = cfg.model
        object.__setattr__(model_cfg, "dtype", cfg.pytorch_training_precision)
        base_dataset = _data._unwrap_dataset(train_loader._data_loader.torch_loader.dataset)
        tactile_input_config = base_dataset.default_tactile_input_config_path
        object.__setattr__(model_cfg, "tactile_input_config_file", tactile_input_config)

        model = self._build_model(device, model_cfg, tactile_input_config, resuming)
        if is_main:
            print_model_statistics(model)
            log_memory_usage(device, 0, "after_model_creation")
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        trainable_parameters = collect_trainable_parameters(model)

        if use_ddp:  # some parameters are unused in a given configuration, hence find_unused_parameters
            model = torch.nn.parallel.DistributedDataParallel(
                model, device_ids=[device.index] if device.type == "cuda" else None, find_unused_parameters=True,
                gradient_as_bucket_view=True, static_graph=False)
        if cfg.use_torch_compile:
            model = self._compile(model, train_loader, device, use_ddp, is_main)

        sched = cfg.lr_schedule
        optim = torch.optim.AdamW(trainable_parameters, lr=sched.peak_lr, betas=(cfg.optimizer.b1, cfg.optimizer.b2),
                                  eps=cfg.optimizer.eps, weight_decay=cfg.optimizer.weight_decay)
        for pg in optim.param_groups:
            pg["initial_lr"] = sched.peak_lr
            pg.setdefault("group_name", "default")

        global_step, resume_state = 0, None
        if resuming:
            saved_step = load_ftp1_hpt_checkpoint(model, optim, cfg.checkpoint_dir, device, global_step=latest_step)
            global_step = saved_step + 1  # checkpoint s is written after the update of step s
            rs_path = cfg.checkpoint_dir / f"{latest_step}" / "resume_state.pt"
            if rs_path.exists():
                resume_state = torch.load(rs_path, map_location="cpu", weights_only=False)
            else:
                logging.warning("%s missing: RNG states are not restored", rs_path)
            logging.info("resumed from step %d; next step %d", saved_step, global_step)

        ema, ema_model = None, _unwrap(model)
        if self.ema_decay > 0 and is_main:
            ema = PolicyEMA(ema_model, self.ema_decay, "bf16" if self.ema_dtype == "bf16" else "fp32")
            if self.ema_dtype == "auto" and torch.cuda.is_available() and torch.cuda.mem_get_info(device)[0] < 20 * 2**30:
                del ema
                torch.cuda.empty_cache()
                ema = PolicyEMA(ema_model, self.ema_decay, "bf16")
            if resuming:
                ema_file = cfg.checkpoint_dir / f"{latest_step}" / "ema.safetensors"
                if ema_file.exists():
                    ema.load(ema_file, device)
                else:
                    logging.warning("resuming without %s: the EMA restarts from the resumed weights", ema_file)
            logging.info("EMA decay %s dtype %s (%d tensors, %d updates so far)", self.ema_decay, ema.dtype_name,
                         len(ema.names), ema.updates)

        # data position: every step consumes one batch of this rank's stream; multi-task runs use the per-task batch
        # sampler (epochs reshuffle), single-task DDP runs a DistributedSampler (same order every epoch)
        batch_sampler = train_loader._data_loader._batch_sampler
        sampler = getattr(train_loader._data_loader.torch_loader, "sampler", None)
        local_bs = cfg.batch_size // world_size
        if batch_sampler is not None:
            n_b = len(batch_sampler)
        elif hasattr(sampler, "set_skip"):
            n_b = len(sampler) // local_bs
        else:  # single process, single task: no exact data position on resume
            n_b = None
        if resume_state is not None:
            rng = resume_state["rng"]
            if len(rng) == world_size:
                random.setstate(rng[rank]["python"])
                np.random.set_state(rng[rank]["numpy"])
                torch.set_rng_state(rng[rank]["torch"])
                if torch.cuda.is_available() and rng[rank]["cuda"] is not None:
                    torch.cuda.set_rng_state(rng[rank]["cuda"], device)
            else:
                logging.warning("RNG states saved for %d ranks, running %d: not restored", len(rng), world_size)

        model.train()
        t0 = time.time()
        while global_step < cfg.num_train_steps:
            epoch = global_step // n_b if n_b else 0
            train_loader._data_loader.set_epoch(epoch)
            if batch_sampler is not None:
                batch_sampler.set_skip(global_step % n_b)
            elif n_b:
                sampler.set_skip((global_step % n_b) * local_bs)
            for observation, actions in train_loader:
                if global_step >= cfg.num_train_steps:
                    break
                observation = move_to_device(observation, device)
                actions = actions.to(torch.float32).to(device)
                lr = lr_at_step(global_step, warmup_steps=sched.warmup_steps, peak_lr=sched.peak_lr,
                                decay_steps=sched.decay_steps, end_lr=sched.decay_lr)
                for pg in optim.param_groups:
                    pg["lr"] = lr * pg.get("initial_lr", sched.peak_lr) / sched.peak_lr if sched.peak_lr > 0 else lr
                output = model(observation, actions)
                loss = output["loss"] if isinstance(output, dict) else output
                extras = {k: v for k, v in output.items() if k != "loss"} if isinstance(output, dict) else {}
                loss_value = float(loss.item())
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=cfg.optimizer.clip_gradient_norm)
                optim.step()
                optim.zero_grad(set_to_none=True)
                if ema is not None:
                    ema.update(ema_model)

                if is_main and (global_step % cfg.log_interval == 0 or global_step < 3):
                    enc = getattr(ema_model, "cond_encoder", None)
                    gates = enc.gates() if enc is not None else {}
                    logging.info("STEP %d loss %.6f lr %.3e grad_norm %.4f (%.1fs since last log) gates %s extras %s",
                                 global_step, loss_value, optim.param_groups[0]["lr"], float(grad_norm), time.time() - t0,
                                 {k: round(v, 5) for k, v in gates.items()},
                                 {k: round(float(v), 5) for k, v in extras.items()})
                    t0 = time.time()

                self._maybe_save(model, optim, global_step, data_config, is_main, use_ddp, rank, world_size, device, ema,
                                 ema_model)
                global_step += 1
        cleanup_ddp()

    # ------------------------------------------------------------------------------------------------------------ parts
    def _compile(self, model, train_loader, device, use_ddp: bool, is_main: bool):
        dyn = torch._dynamo.config
        dyn.recompile_limit = max(int(getattr(dyn, "recompile_limit", 64)), 64)
        dyn.accumulated_recompile_limit = max(int(getattr(dyn, "accumulated_recompile_limit", 1024)), 1024)
        dyn.cache_size_limit = max(int(getattr(dyn, "cache_size_limit", 64)), 64)
        # single GPU: reduce-overhead (CUDA graphs); DDP: default mode (CUDA graphs are incompatible with DDP)
        model = torch.compile(model, mode="default" if use_ddp else "reduce-overhead")
        if use_ddp:  # make every rank finish compiling before training starts
            model.train()
            try:
                observation, actions = next(iter(train_loader))
                observation = move_to_device(observation, device)
                actions = actions.to(torch.float32).to(device)
                with torch.no_grad():
                    for _ in range(2):
                        model(observation, actions)
            except Exception as exc:  # noqa: BLE001
                logging.warning("compile warm-up failed (continuing): %s", exc)
            dist.barrier()
            if is_main:
                logging.info("torch.compile warm-up done on all ranks")
        return model

    def _maybe_save(self, model, optim, step, data_config, is_main, use_ddp, rank, world_size, device, ema, ema_model):
        cfg = self.config
        will_save = (step % cfg.save_interval == 0 and step > 0) or step == cfg.num_train_steps - 1
        extra = None
        if will_save:
            rng_dir = cfg.checkpoint_dir / "_rng"
            rng_dir.mkdir(parents=True, exist_ok=True)
            torch.save({"python": random.getstate(), "numpy": np.random.get_state(), "torch": torch.get_rng_state(),
                        "cuda": torch.cuda.get_rng_state(device) if torch.cuda.is_available() else None},
                       rng_dir / f"{step}_rank{rank}.pt")
            if use_ddp:
                _barrier(device)
            extra = {"rng_dir": str(rng_dir), "global_step": step, "world_size": world_size}
        save_ftp1_hpt_checkpoint(model, optim, step, cfg, is_main, data_config, extra_state=extra, ema=ema,
                                 keep_last=self.keep_last, pre_delete=self.pre_delete)
        if ema is not None and step == cfg.num_train_steps - 1:
            ema.export_deploy(ema_model, cfg.checkpoint_dir / f"{step}", cfg.checkpoint_dir / f"ema_{step}")
        if will_save and use_ddp:
            _barrier(device)
