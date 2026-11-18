"""TacInno-VLA policy: pi0.5 + tactile condition tokens + code queries (code loss) + contact-state head.

Submodules are imported lazily: the vendored pi0.5 model imports :mod:`.condition_tokens`, and :mod:`.tacinno_vla`
imports the vendored model, so an eager import here would be circular.
"""
import importlib

_EXPORTS = {
    "TacInnoVLA": "tacinno_vla", "build_model_config": "tacinno_vla", "model_config_to_dict": "tacinno_vla",
    "CondTokenEncoder": "condition_tokens", "TactileFieldEncoder": "condition_tokens", "code_ce_loss": "condition_tokens",
    "contact_state_loss": "condition_tokens", "open_residual": "condition_tokens",
}
__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _EXPORTS:
        return getattr(importlib.import_module(f"{__name__}.{_EXPORTS[name]}"), name)
    raise AttributeError(name)
