"""Closed-loop evaluation: the UniVTAC client of TacInno-VLA and result statistics.

The client (:mod:`tacinno.evaluation.univtac_client`) is imported lazily because it needs ``websockets`` and the
simulator-side packages; the statistics only need the standard library.
"""
from .results import PAPER_SEEDS, compare, first_results, load_episode_files, mcnemar_exact, success_rate

__all__ = ["PAPER_SEEDS", "compare", "first_results", "load_episode_files", "mcnemar_exact", "success_rate"]
