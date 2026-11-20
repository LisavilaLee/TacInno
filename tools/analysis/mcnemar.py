#!/usr/bin/env python
"""Paired comparison of two policies on the same evaluation seeds (exact two-sided McNemar test).

Each side is one or more ``episodes.json`` files or run directories (several runs of one policy are merged)::

    python tools/analysis/mcnemar.py \
        --a eval_result/TacInnoVLA/lift_can/deploy_lift_can \
        --b eval_result/TacInnoVLA/lift_can/deploy_lift_can_tactile_off

Each policy is evaluated on its first 100 valid episodes from the start seed, and the seeds both policies completed
are paired.
"""
from __future__ import annotations

import argparse
import os.path as osp
import sys

sys.path.insert(0, osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from tacinno.evaluation.results import PAPER_SEEDS, compare, first_results, first_valid_seeds, success_rate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--a", nargs="+", required=True, help="results of policy A")
    parser.add_argument("--b", nargs="+", required=True, help="results of policy B")
    parser.add_argument("--start-seed", type=int, default=PAPER_SEEDS.start)
    parser.add_argument("--num", type=int, default=len(PAPER_SEEDS))
    args = parser.parse_args()

    ra, rb = first_results(args.a), first_results(args.b)
    sa = first_valid_seeds(ra, args.start_seed, args.num)
    sb = first_valid_seeds(rb, args.start_seed, args.num)
    for name, r, seeds in (("A", ra, sa), ("B", rb, sb)):
        s = success_rate(r, seeds)
        print(f"{name}: success {s['success']}/{s['valid']} = {100 * s['rate']:.1f}%")
    print(compare(ra, rb, sorted(set(sa) & set(sb))))


if __name__ == "__main__":
    main()
