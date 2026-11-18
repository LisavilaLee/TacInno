"""Registration of all TacInno components (MMDetection style).

Components register themselves in :mod:`tacinno.registry` when their module is imported. Tools call
:func:`register_all_modules` once so that every ``type=...`` in a config can be built::

    from tacinno.utils import register_all_modules
    register_all_modules()                      # tokenizer, datasets and TacInno-VLA
    register_all_modules(include_policy=False)  # tokenizer only, no openpi needed
"""
from __future__ import annotations

import importlib

_TOKENIZER_MODULES = ("tacinno.models.tokenizer", "tacinno.datasets.tokenizer_dataset")
_POLICY_MODULES = ("tacinno.datasets.policy_dataset", "tacinno.models.policy.tacinno_vla")


def register_all_modules(include_policy: bool = True) -> None:
    """Import the modules that register datasets and models.

    Args:
        include_policy: also register TacInno-VLA and its dataset. These import the vendored openpi package
            (``third_party/openpi``); set False to use the tokenizer alone.
    """
    for name in _TOKENIZER_MODULES + (_POLICY_MODULES if include_policy else ()):
        importlib.import_module(name)
