"""Component registries.

Every buildable component (datasets, transforms, models, losses, hooks) is registered under a string name and built
from a config dict whose ``type`` key names it, in the style of MMEngine / MMDetection::

    @MODELS.register_module()
    class TacInnoTokenizer(nn.Module):
        ...

    model = MODELS.build(dict(type="TacInnoTokenizer", num_slots=4, codebook_size=16))

The implementation is intentionally small and dependency free.
"""
from __future__ import annotations

import copy
import inspect
from typing import Any, Callable, Dict, Iterator, Mapping, Optional

__all__ = [
    "Registry",
    "build_from_cfg",
    "DATASETS",
    "TRANSFORMS",
    "MODELS",
    "LOSSES",
    "HOOKS",
    "RUNNERS",
]


class Registry:
    """A name -> class (or function) mapping with a ``build`` helper."""

    def __init__(self, name: str):
        self._name = name
        self._modules: Dict[str, Callable] = {}

    @property
    def name(self) -> str:
        return self._name

    def __len__(self) -> int:
        return len(self._modules)

    def __contains__(self, key: str) -> bool:
        return key in self._modules

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._modules))

    def __repr__(self) -> str:
        items = ", ".join(sorted(self._modules))
        return f"Registry(name={self._name!r}, items=[{items}])"

    def keys(self):
        return sorted(self._modules)

    def get(self, key: str) -> Callable:
        if key not in self._modules:
            raise KeyError(
                f"{key!r} is not registered in the {self._name!r} registry. "
                f"Registered: {', '.join(sorted(self._modules)) or '(none)'}. "
                "Did you forget to import the module that defines it?"
            )
        return self._modules[key]

    def _register(self, module: Callable, name: Optional[str], force: bool) -> None:
        if not (inspect.isclass(module) or inspect.isfunction(module)):
            raise TypeError(f"only classes or functions can be registered, got {type(module)}")
        key = name or module.__name__
        if not force and key in self._modules and self._modules[key] is not module:
            raise KeyError(f"{key!r} is already registered in {self._name!r}")
        self._modules[key] = module

    def register_module(self, name: Optional[str] = None, module: Optional[Callable] = None, force: bool = False):
        """Register ``module`` directly, or return a decorator when ``module`` is None."""
        if module is not None:
            self._register(module, name, force)
            return module

        def _decorator(cls: Callable) -> Callable:
            self._register(cls, name, force)
            return cls

        return _decorator

    def build(self, cfg: Mapping[str, Any], **default_args: Any) -> Any:
        return build_from_cfg(cfg, self, **default_args)


def build_from_cfg(cfg: Mapping[str, Any], registry: Registry, **default_args: Any) -> Any:
    """Instantiate ``cfg['type']`` from ``registry`` with the remaining keys as keyword arguments.

    ``default_args`` are used only for keys that ``cfg`` does not set.
    """
    if not isinstance(cfg, Mapping):
        raise TypeError(f"cfg must be a mapping, got {type(cfg)}")
    if "type" not in cfg:
        raise KeyError(f"cfg must contain the key 'type', got keys {list(cfg)}")
    args = copy.deepcopy(dict(cfg))
    obj_type = args.pop("type")
    for key, value in default_args.items():
        args.setdefault(key, value)
    obj_cls = registry.get(obj_type) if isinstance(obj_type, str) else obj_type
    try:
        return obj_cls(**args)
    except Exception as exc:  # keep the original exception, add which component failed
        if hasattr(exc, "add_note"):
            exc.add_note(f"while building {getattr(obj_cls, '__name__', obj_cls)!s} from the {registry.name!r} registry")
        raise


DATASETS = Registry("dataset")
TRANSFORMS = Registry("transform")
MODELS = Registry("model")
LOSSES = Registry("loss")
HOOKS = Registry("hook")
RUNNERS = Registry("runner")
