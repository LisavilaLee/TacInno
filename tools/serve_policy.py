#!/usr/bin/env python
"""Serve a TacInno-VLA checkpoint over websockets (one port per task).

    # simulation (UniVTAC client): six tasks share one checkpoint only if it was trained on all of them
    python tools/serve_policy.py CKPT_DIR --task lift_can:36850
    # deployment settings (cameras, domain prefix) from the training config
    python tools/serve_policy.py CKPT_DIR --config configs/tacinno_vla/tacinno-vla_pi05_20k-headcam_univtac-put-bottle.py \
        --task put_bottle_in_shelf:36851
    # tactile-off test: every tactile frame is replaced by a no-contact frame
    python tools/serve_policy.py CKPT_DIR --task lift_can:36850 --tactile-off no_contact_frame.npz
    # real robot (four tasks, one Xense gel)
    python tools/serve_policy.py CKPT_DIR --real-robot --task real_wipe_board:39101 --task real_lift_can:39102

The joint statistics (``--cond-stats``) and the tactile statistics (``--tactile-stats``) of the released models are
bundled in ``tacinno/datasets/stats/`` and used by default; pass the files of your own condition data when serving a
policy trained on other data.
"""
from __future__ import annotations

import argparse
import logging
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.abspath(__file__))))


def parse_args():
    p = argparse.ArgumentParser(description="TacInno-VLA policy server")
    p.add_argument("checkpoint", help="deployable checkpoint directory (ema_<step>/)")
    p.add_argument("--task", action="append", required=True, help="TASK:PORT, repeat for several tasks")
    p.add_argument("--config", help="training config; its deploy section sets cameras and the domain prefix")
    p.add_argument("--cond-stats", help="cond_stats.json of the training condition data "
                   "(default: the bundled statistics of the released UniVTAC or real-robot models)")
    p.add_argument("--tactile-stats", help="tactile statistics (default: the bundled UniVTAC or real-robot statistics)")
    p.add_argument("--cameras", nargs="+", choices=["head", "wrist"], help="cameras the model reads (default head wrist)")
    p.add_argument("--real-robot", action="store_true", help="real-robot request format and prompts")
    p.add_argument("--tactile-off", help="npz with a no-contact frame (l_depth, l_marker, r_depth, r_marker)")
    p.add_argument("--bind", default="127.0.0.1")
    p.add_argument("--device", default="cuda")
    p.add_argument("--group", default="tacinno", help="name reported in the metadata")
    return p.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, force=True)
    from tacinno.apis.inference import TacInnoPolicy
    from tacinno.apis.server import PolicyServer, load_rest_frame
    from tacinno.config import Config
    from tacinno.datasets import stats_path

    deploy = {}
    if args.config:
        deploy = dict(Config.fromfile(args.config).get("deploy", {}) or {})
    cameras = tuple(args.cameras or deploy.get("cameras", ("head", "wrist")))
    prefix = deploy.get("domain_prefix", "Real_" if args.real_robot else "UniVTAC_")
    entries = [(e.split(":")[0], int(e.split(":")[1])) for e in args.task]
    cond_stats = args.cond_stats or stats_path("real_robot_cond_stats" if args.real_robot else "univtac_cond_stats")
    tactile_stats = args.tactile_stats or (stats_path("real_robot_tactile_stats") if args.real_robot else None)
    policy = TacInnoPolicy(args.checkpoint, entries[0][0], cond_stats=cond_stats, tactile_stats=tactile_stats,
                           device=args.device, cameras=cameras, domain_prefix=prefix, real_robot=args.real_robot)
    rest = load_rest_frame(args.tactile_off) if args.tactile_off else None
    PolicyServer(policy, entries, bind=args.bind, rest_frame=rest, group=args.group).serve_forever()


if __name__ == "__main__":
    main()
