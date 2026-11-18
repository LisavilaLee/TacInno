"""Data of TacInno.

* :mod:`tacinno.datasets.tactile_fields`: tactile observation of a vision-based tactile sensor (indentation field and
  marker tangential displacement field) from the simulated depth image and marker positions.
* :mod:`tacinno.datasets.tokenizer_dataset`: demonstration windows for the TacInno tokenizer.
* :mod:`tacinno.datasets.condition_data`: per-frame condition inputs and targets of TacInno-VLA.
* :mod:`tacinno.datasets.policy_dataset`: the TacInno-VLA training set (pi0.5 zarr pipeline, imported on use).

The statistics of the released models are bundled in ``tacinno/datasets/stats/``; :func:`stats_path` returns them.
"""
import os.path as _osp

from . import condition_data, tactile_fields
from .condition_data import ConditionData

STATS_DIR = _osp.join(_osp.dirname(_osp.abspath(__file__)), "stats")
BUNDLED_STATS = ("univtac_tactile_stats", "univtac_cond_stats", "real_robot_tactile_stats", "real_robot_cond_stats")


def stats_path(name: str) -> str:
    """Path of a bundled statistics file.

    * ``univtac_tactile_stats`` / ``real_robot_tactile_stats``: rest marker grid, ``sigma_u`` and the mean / std of the
      contact properties of the training tactile data;
    * ``univtac_cond_stats`` / ``real_robot_cond_stats``: mean / std of the 8 joint values over the training frames.
    """
    if name not in BUNDLED_STATS:
        raise KeyError(f"unknown statistics {name!r}; bundled: {', '.join(BUNDLED_STATS)}")
    return _osp.join(STATS_DIR, f"{name}.json")


__all__ = ["condition_data", "tactile_fields", "ConditionData", "STATS_DIR", "BUNDLED_STATS", "stats_path"]
