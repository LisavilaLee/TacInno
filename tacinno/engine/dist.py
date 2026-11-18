"""Process-group setup, seeding and logging helpers of the policy trainer (``torchrun`` compatible)."""
from __future__ import annotations

import datetime
import logging
import os

import numpy as np
import torch
import torch.distributed as dist

__all__ = ["setup_ddp", "cleanup_ddp", "set_seed", "init_logging", "log_memory_usage"]


def init_logging() -> None:
    np.set_printoptions(suppress=True, precision=12, floatmode="fixed")
    level_mapping = {"DEBUG": "D", "INFO": "I", "WARNING": "W", "ERROR": "E", "CRITICAL": "C"}

    class _Formatter(logging.Formatter):
        def format(self, record):
            record.levelname = level_mapping.get(record.levelname, record.levelname)
            return super().format(record)

    formatter = _Formatter(fmt="%(asctime)s.%(msecs)03d [%(levelname)s] %(message)-80s (%(process)d:%(filename)s:%(lineno)s)",
                           datefmt="%H:%M:%S")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    else:
        logger.handlers[0].setFormatter(formatter)


def setup_ddp(timeout_min: int = 60) -> tuple[bool, int, torch.device]:
    """Initialize the NCCL process group when launched by torchrun (WORLD_SIZE > 1); returns (use_ddp, local_rank,
    device). ``set_device`` is called before ``init_process_group`` and the timeout is long enough for checkpoint saves."""
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    use_ddp = world_size > 1
    local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
    if use_ddp and not dist.is_initialized():
        backend = "nccl" if torch.cuda.is_available() else "gloo"
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
        dist.init_process_group(backend=backend, init_method="env://", timeout=datetime.timedelta(minutes=timeout_min))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    return use_ddp, local_rank, device


def cleanup_ddp() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


def set_seed(seed: int, local_rank: int) -> None:
    torch.manual_seed(seed + local_rank)
    np.random.seed(seed + local_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed + local_rank)


def log_memory_usage(device: torch.device, step: int, phase: str = "unknown") -> None:
    if not torch.cuda.is_available():
        return
    stats = torch.cuda.memory_stats(device)
    logging.info("Step %d (%s): GPU memory allocated %.2f GB, reserved %.2f GB, peak allocated %.2f GB", step, phase,
                 torch.cuda.memory_allocated(device) / 1e9, torch.cuda.memory_reserved(device) / 1e9,
                 stats.get("allocated_bytes.all.peak", 0) / 1e9)
