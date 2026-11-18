"""Inference and serving APIs of TacInno-VLA."""
from .inference import REAL_ROBOT_PROMPTS, UNIVTAC_PROMPTS, TacInnoPolicy, init_policy, resolve_abs

__all__ = ["TacInnoPolicy", "init_policy", "resolve_abs", "UNIVTAC_PROMPTS", "REAL_ROBOT_PROMPTS", "PolicyServer"]


def __getattr__(name):
    if name == "PolicyServer":
        from .server import PolicyServer
        return PolicyServer
    raise AttributeError(name)
