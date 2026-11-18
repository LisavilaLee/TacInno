"""Training engine of TacInno-VLA: config translation, trainer (DDP, exact resume, EMA) and checkpoint I/O."""
from .ema import PolicyEMA
from .train_config import build_model_config_from_cfg, build_train_config, condition_keys, write_dataset_config
from .trainer import PolicyTrainer, lr_at_step

__all__ = ["PolicyEMA", "PolicyTrainer", "build_train_config", "build_model_config_from_cfg", "condition_keys",
           "write_dataset_config", "lr_at_step"]
