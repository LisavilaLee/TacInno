#!/usr/bin/env python
"""Serve a TacInno-VLA checkpoint over websockets (one port per task).

    # simulation (UniVTAC client): six tasks share one checkpoint only if it was trained on all of them
    python tools/serve_policy.py CKPT_DIR --task lift_can:36850
    # a policy trained with another config: its deploy section sets the cameras and the domain prefix
    python tools/serve_policy.py CKPT_DIR --config configs/tacinno_vla/<config>.py --task lift_can:36850
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

ROOT = osp.dirname(osp.dirname(osp.abspath(__file__)))
sys.path.insert(0, ROOT)

# training config of the released checkpoint of each UniVTAC task (used when --config is not given)
RELEASED_CONFIGS = {
    "insert_hole": "tacinno-vla_univtac-insert-hole.py",
    "insert_tube": "tacinno-vla_univtac-insert-tube.py",
    "lift_bottle": "tacinno-vla_univtac-lift-bottle.py",
    "lift_can": "tacinno-vla_univtac-lift-can.py",
    "pull_out_key": "tacinno-vla_univtac-pull-out-key.py",
    "put_bottle_in_shelf": "tacinno-vla_univtac-put-bottle.py",
}
REAL_ROBOT_CONFIG = osp.join("real_robot", "tacinno-vla_real-4task.py")


def default_config(task: str, real_robot: bool):
    name = REAL_ROBOT_CONFIG if real_robot else RELEASED_CONFIGS.get(task)
    return osp.join(ROOT, "configs", "real_robot" if real_robot else "tacinno_vla", osp.basename(name)) if name else None


def parse_args():
    p = argparse.ArgumentParser(description="TacInno-VLA policy server")
    p.add_argument("checkpoint", help="deployable checkpoint directory (ema_<step>/)")
    p.add_argument("--task", action="append", required=True, help="TASK:PORT, repeat for several tasks")
    p.add_argument("--config", help="training config; its deploy section sets cameras and the domain prefix "
                   "(default: the config of the released checkpoint of the task)")
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

    entries = [(e.split(":")[0], int(e.split(":")[1])) for e in args.task]
    config = args.config or default_config(entries[0][0], args.real_robot)
    deploy = dict(Config.fromfile(config).get("deploy", {}) or {}) if config else {}
    cameras = tuple(args.cameras or deploy.get("cameras", ("head", "wrist")))
    prefix = deploy.get("domain_prefix", "Real_" if args.real_robot else "UniVTAC_")
    cond_stats = args.cond_stats or stats_path("real_robot_cond_stats" if args.real_robot else "univtac_cond_stats")
    tactile_stats = args.tactile_stats or (stats_path("real_robot_tactile_stats") if args.real_robot else None)
    policy = TacInnoPolicy(args.checkpoint, entries[0][0], cond_stats=cond_stats, tactile_stats=tactile_stats,
                           device=args.device, cameras=cameras, domain_prefix=prefix, real_robot=args.real_robot)
    rest = load_rest_frame(args.tactile_off) if args.tactile_off else None
    PolicyServer(policy, entries, bind=args.bind, rest_frame=rest, group=args.group).serve_forever()


if __name__ == "__main__":
    main()
