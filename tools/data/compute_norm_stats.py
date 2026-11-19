#!/usr/bin/env python
"""Compute the per-task normalization statistics of a TacInno-VLA config (z-score of state and action per task).

    python tools/data/compute_norm_stats.py configs/tacinno_vla/init/tacinno-vla_univtac-lift-can_init.py

Writes ``<data_root>/<norm_root>/<repo_id>/`` (one folder per task with ``independent_norm_stats_all_t0_zscore.json``
and ``train_val_split.json``, plus ``norm_params_snapshot.json``). The released models use the statistics of the
six-task data set for every recipe (``repo_id = univtac_6task``); a single-task config reuses that folder.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import os.path as osp
import sys

# JAX is only used for tree utilities in the data pipeline; keep it off the GPUs
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import tempfile

ROOT = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.insert(0, ROOT)

from tacinno.config import Config, DictAction  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description="TacInno-VLA normalization statistics")
    ap.add_argument("config")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = ap.parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)

    from openpi.training.compute_norm_stats import main as compute
    from tacinno.engine.train_config import build_train_config, write_dataset_config

    with tempfile.TemporaryDirectory() as tmp:
        ds = write_dataset_config(cfg.data, osp.join(tmp, "dataset_config.json"))
        tc = build_train_config(cfg, osp.join(tmp, "norm"), resume=False, dataset_config_path=ds)
        tc = dataclasses.replace(tc, batch_size=32, norm_sample_ratio=1.0, norm_batch_size=64,
                                 norm_num_workers=args.num_workers, skip_contact_detection=False,
                                 cond_root="", cond_keys=())
        tc.finalize_config()
        compute(tc)


if __name__ == "__main__":
    main()
