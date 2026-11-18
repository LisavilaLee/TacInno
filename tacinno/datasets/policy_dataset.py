"""Policy training data of TacInno-VLA (UniVTAC simulation and real robot).

The samples come from the pi0.5 zarr pipeline (``third_party/openpi``): per task one zarr of demonstrations
(``<data_root>/<zarr_dir>/<task>``), normalized per task with the statistics in ``<norm_root>/<repo_id>/``. Each
sample holds the images (head and wrist, 224 x 224), the tokenized prompt, the 120-D state, the 32-step action chunk
and, under ``cond``, the condition inputs and targets of :class:`tacinno.datasets.ConditionData`.

    from tacinno.config import Config
    from tacinno.registry import DATASETS
    import tacinno.datasets.policy_dataset  # registers the builder
    cfg = Config.fromfile("configs/tacinno_vla/tacinno-vla_pi05_20k_univtac-insert-tube.py")
    dataset = DATASETS.build(dict(type="TacInnoPolicyDataset", cfg=cfg))
    sample = dataset[0]          # dict: image, image_mask, state, actions, tokenized_prompt, ..., cond
"""
from __future__ import annotations

import os.path as osp
import tempfile

from tacinno.registry import DATASETS

__all__ = ["build_policy_dataset"]


@DATASETS.register_module(name="TacInnoPolicyDataset")
def build_policy_dataset(cfg, split: str = "train", work_dir: str | None = None, model_transforms: bool = True):
    """Build the (transformed) training dataset of a TacInno-VLA config.

    Args:
        cfg: a loaded config (``tacinno.config.Config``) with ``model``, ``data``, ``schedule`` and ``runtime``.
        split: "train" (the datasets of ``data.zarr_dir``).
        work_dir: where the dataset list is written (default: a temporary directory).
        model_transforms: apply the model transforms (prompt tokenization) as in training.
    """
    import openpi.training.data_loader as _data

    from tacinno.engine.train_config import build_train_config, write_dataset_config

    if split != "train":
        raise ValueError("only the training split is built here; validation episodes live in a separate zarr")
    work_dir = work_dir or tempfile.mkdtemp(prefix="tacinno_data_")
    ds_path = write_dataset_config(cfg["data"], osp.join(work_dir, "dataset_config.json"))
    tc = build_train_config(cfg, osp.join(work_dir, "run"), resume=False, dataset_config_path=ds_path)
    data_config = tc.data.create(tc.assets_dirs, tc.model)
    dataset, _ = _data.create_torch_dataset(data_config, tc.model.action_horizon, tc.model)
    dataset.load_norm_stats(independent_norm_mode=data_config.independent_norm_mode,
                            norm_time_dim=data_config.norm_time_dim, norm_type=data_config.norm_type)
    if model_transforms:
        dataset = _data.transform_dataset(dataset, data_config, only_model_transforms=True)
    return dataset
