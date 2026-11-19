#!/usr/bin/env python
"""Train TacInno-VLA.

    # one GPU
    python tools/train_policy.py configs/tacinno_vla/tacinno-vla_univtac-insert-tube.py
    # N GPUs (global batch stays schedule.batch_size)
    bash tools/dist_train.sh configs/tacinno_vla/tacinno-vla_univtac-insert-tube.py 2

The run resumes automatically from the newest complete checkpoint in the work directory (``--no-resume`` to refuse).
"""
from __future__ import annotations

import argparse
import os
import os.path as osp
import sys

# JAX is only used for tree utilities in the data pipeline; keep it off the GPUs
os.environ.setdefault("JAX_PLATFORMS", "cpu")

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))

from tacinno.config import Config, DictAction  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Train TacInno-VLA")
    p.add_argument("config", help="config file")
    p.add_argument("--work-dir", help="checkpoint / log directory (default: work_dirs/<config name>)")
    p.add_argument("--no-resume", action="store_true", help="fail instead of resuming when the work dir has checkpoints")
    p.add_argument("--cfg-options", nargs="+", action=DictAction, help="override config keys, e.g. schedule.num_steps=10")
    return p.parse_args()


def has_checkpoint(work_dir: str) -> bool:
    if not osp.isdir(work_dir):
        return False
    for name in os.listdir(work_dir):
        d = osp.join(work_dir, name)
        if name.isdigit() and all(osp.exists(osp.join(d, f)) for f in ("model.safetensors", "optimizer.pt", "metadata.pt")):
            return True
    return False


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    work_dir = osp.abspath(args.work_dir or cfg.get("work_dir") or
                           osp.join("work_dirs", osp.splitext(osp.basename(args.config))[0]))

    from tacinno.engine import PolicyTrainer, build_train_config, write_dataset_config
    from tacinno.engine.dist import init_logging

    init_logging()
    resume = has_checkpoint(work_dir)
    if resume and args.no_resume:
        raise SystemExit(f"{work_dir} already has checkpoints (remove --no-resume to continue the run)")
    rank = int(os.environ.get("RANK", "0"))
    ds_path = osp.join(work_dir, "dataset_config.json")
    if rank == 0:
        os.makedirs(work_dir, exist_ok=True)
        write_dataset_config(cfg.data, ds_path)
        cfg.dump(osp.join(work_dir, "tacinno_config.py"))
    else:  # other ranks wait for rank 0 to write the dataset list
        import time
        for _ in range(600):
            if osp.exists(ds_path):
                break
            time.sleep(0.5)
    train_cfg = build_train_config(cfg, work_dir, resume=resume, dataset_config_path=ds_path)
    rt, sched = cfg.get("runtime", {}), cfg.schedule
    PolicyTrainer(train_cfg, ema_decay=float(sched.get("ema_decay", 0.0)), ema_dtype=sched.get("ema_dtype", "auto"),
                  keep_last=int(rt.get("keep_last", 1)), pre_delete=bool(rt.get("pre_delete", False)),
                  compile_cache_dir=rt.get("compile_cache_dir"),
                  init_type=(cfg.get("init_weights") or {}).get("type", "pi05_base")).train()


if __name__ == "__main__":
    main()
