"""Closed-loop result bookkeeping: success rates and exact McNemar tests on paired seeds.

Protocol used in the paper
--------------------------
* Seeds ``1000000 .. 1000099`` (100 episodes per task), identical for every policy.
* The action limit of every task is twice the benchmark default (``UNIVTAC_STEP_LIM_SCALE=2``).
* Episodes that raise (simulator or planning errors) are skipped by the evaluator and filled with the next seed; they
  never count as failures. Only ``success`` / ``failed`` episodes are results.
* If a seed was evaluated more than once (for example, evaluation split over several processes that overlap), the
  result that finished first counts.
* Two policies are compared on the seeds both completed with the exact two-sided McNemar test on the discordant pairs.

The evaluator writes one ``episodes.json`` per run directory (``eval_result/<policy>/<task>/<deploy>/<time>/``).
"""
from __future__ import annotations

import datetime
import glob
import json
import os
from dataclasses import dataclass
from math import comb
from typing import Dict, Iterable, List, Optional, Sequence, Union

__all__ = [
    "PAPER_SEEDS",
    "load_episode_files",
    "first_results",
    "success_rate",
    "mcnemar_exact",
    "PairedComparison",
    "compare",
]

#: the 100 evaluation seeds of the paper
PAPER_SEEDS = range(1000000, 1000100)
FINISHED = ("success", "failed")
PathLike = Union[str, os.PathLike]


def load_episode_files(paths: Iterable[PathLike]) -> List[str]:
    """``episodes.json`` files given directly or found (recursively) under the given directories, sorted."""
    files: List[str] = []
    for p in paths:
        p = os.fspath(p)
        if os.path.isfile(p):
            files.append(p)
        elif os.path.isdir(p):
            files.extend(glob.glob(os.path.join(p, "**", "episodes.json"), recursive=True))
        else:
            raise FileNotFoundError(p)
    return sorted(set(files))


def _run_start(run_dir: str, fallback: float) -> float:
    try:
        return datetime.datetime.strptime(os.path.basename(run_dir), "%Y-%m-%d_%H:%M:%S").timestamp()
    except ValueError:
        return fallback


def first_results(paths: Iterable[PathLike], seeds: Optional[Sequence[int]] = None) -> Dict[int, dict]:
    """Map seed -> the first finished episode (``success`` or ``failed``) among all given runs.

    The finish time is the episode's ``finished_at`` (unix seconds) when present; otherwise the start time encoded in
    the run directory name plus the cumulative episode durations of that run (the order in which the run finished
    its episodes). Each returned episode has an extra ``source`` key with the file it came from.
    """
    wanted = None if seeds is None else set(int(s) for s in seeds)
    best: Dict[int, tuple] = {}
    for f in load_episode_files(paths):
        try:
            with open(f) as fh:
                episodes = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        t = _run_start(os.path.dirname(f), os.stat(f).st_mtime)
        for ep in episodes:
            t += float(ep.get("seconds", 0) or 0)
            if ep.get("result") not in FINISHED:
                continue
            seed = int(ep["seed"])
            if wanted is not None and seed not in wanted:
                continue
            finished = float(ep["finished_at"]) if ep.get("finished_at") is not None else t
            if seed not in best or finished < best[seed][0]:
                best[seed] = (finished, dict(ep, source=f))
    return {s: e for s, (_, e) in sorted(best.items())}


def _ok(ep: dict) -> bool:
    return ep.get("result") == "success"


def success_rate(results: Dict[int, dict], seeds: Optional[Sequence[int]] = None) -> dict:
    """``{"success": k, "valid": n, "rate": k / n, "missing": [...]}`` over ``seeds`` (default: all seeds present)."""
    seeds = sorted(results) if seeds is None else list(seeds)
    valid = [s for s in seeds if s in results]
    k = sum(_ok(results[s]) for s in valid)
    return {"success": k, "valid": len(valid), "rate": k / len(valid) if valid else float("nan"),
            "missing": [s for s in seeds if s not in results]}


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """Exact two-sided McNemar p-value from the discordant counts (binomial test with p = 1/2)."""
    n = only_a + only_b
    if n == 0:
        return 1.0
    tail = sum(comb(n, i) for i in range(min(only_a, only_b) + 1))
    return min(1.0, 2.0 * tail / 2 ** n)


@dataclass
class PairedComparison:
    paired: int
    success_a: int
    success_b: int
    only_a: int
    only_b: int
    p_value: float

    def __str__(self) -> str:
        return (f"paired seeds {self.paired}: A {self.success_a} vs B {self.success_b} | only A {self.only_a}, "
                f"only B {self.only_b} | exact McNemar p = {self.p_value:.3g}")


def compare(results_a: Dict[int, dict], results_b: Dict[int, dict],
            seeds: Optional[Sequence[int]] = None) -> PairedComparison:
    """Paired comparison of two policies on the seeds that both completed."""
    pool = sorted(set(results_a) & set(results_b)) if seeds is None else [s for s in seeds
                                                                          if s in results_a and s in results_b]
    oa = sum(_ok(results_a[s]) and not _ok(results_b[s]) for s in pool)
    ob = sum(_ok(results_b[s]) and not _ok(results_a[s]) for s in pool)
    return PairedComparison(paired=len(pool), success_a=sum(_ok(results_a[s]) for s in pool),
                            success_b=sum(_ok(results_b[s]) for s in pool), only_a=oa, only_b=ob,
                            p_value=mcnemar_exact(oa, ob))
