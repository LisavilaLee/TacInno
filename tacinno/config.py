"""Python config files with ``_base_`` inheritance (MMEngine style, dependency free).

A config is a plain Python file whose top-level variables form the config::

    # configs/tacinno_vla/tacinno-vla_pi05_univtac-lift-can.py
    _base_ = ["../_base_/models/tacinno_pi05.py", "../_base_/schedules/policy_ft17k.py"]
    task = "lift_can"
    model = dict(use_proprio_tokens=True)          # merged into the base ``model`` dict

Rules
-----
* ``_base_`` is a path or a list of paths relative to the file; bases are merged in order, then the file itself.
* Dicts merge recursively; any other value (lists included) replaces the base value.
* ``dict(_delete_=True, ...)`` replaces the base dict instead of merging into it.
* Names starting with ``_`` (other than ``_base_``), modules, functions and classes are not part of the config.
* ``{{fileDirname}}`` in strings expands to the directory of the config file.
"""
from __future__ import annotations

import argparse
import ast
import copy
import os
import os.path as osp
import pprint
import runpy
import types
from typing import Any, Dict, Iterable, List, Mapping, Optional

__all__ = ["Config", "ConfigDict", "DictAction"]

BASE_KEY = "_base_"
DELETE_KEY = "_delete_"


class ConfigDict(dict):
    """``dict`` with attribute access; nested dicts are converted on assignment."""

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__()
        for key, value in dict(*args, **kwargs).items():
            self[key] = value

    @staticmethod
    def _wrap(value: Any) -> Any:
        if isinstance(value, ConfigDict):
            return value
        if isinstance(value, Mapping):
            return ConfigDict(value)
        if isinstance(value, list):
            return [ConfigDict._wrap(v) for v in value]
        if isinstance(value, tuple):
            return tuple(ConfigDict._wrap(v) for v in value)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        super().__setitem__(key, self._wrap(value))

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(f"config has no key {name!r}") from exc

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __delattr__(self, name: str) -> None:
        try:
            del self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __deepcopy__(self, memo: Dict[int, Any]) -> "ConfigDict":
        return ConfigDict(copy.deepcopy(dict(self), memo))

    def to_dict(self) -> Dict[str, Any]:
        def _unwrap(value: Any) -> Any:
            if isinstance(value, Mapping):
                return {k: _unwrap(v) for k, v in value.items()}
            if isinstance(value, list):
                return [_unwrap(v) for v in value]
            if isinstance(value, tuple):
                return tuple(_unwrap(v) for v in value)
            return value

        return _unwrap(self)


def _merge(base: Dict[str, Any], new: Mapping[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``new`` into a copy of ``base``."""
    out = copy.deepcopy(dict(base))
    for key, value in new.items():
        if isinstance(value, Mapping):
            value = dict(value)
            delete = value.pop(DELETE_KEY, False)
            if not delete and isinstance(out.get(key), Mapping):
                out[key] = _merge(out[key], value)
            else:
                out[key] = _strip_delete(value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _strip_delete(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {k: _strip_delete(v) for k, v in value.items() if k != DELETE_KEY}
    if isinstance(value, list):
        return [_strip_delete(v) for v in value]
    return value


def _expand(value: Any, file_dir: str) -> Any:
    if isinstance(value, str):
        return value.replace("{{fileDirname}}", file_dir)
    if isinstance(value, Mapping):
        return {k: _expand(v, file_dir) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v, file_dir) for v in value]
    if isinstance(value, tuple):
        return tuple(_expand(v, file_dir) for v in value)
    return value


def _is_config_value(name: str, value: Any) -> bool:
    if name.startswith("_"):
        return False
    return not isinstance(value, (types.ModuleType, types.FunctionType, types.BuiltinFunctionType, type))


class Config:
    """A loaded config: ``cfg.model.type``, ``cfg['model']``, ``cfg.get('work_dir')``."""

    def __init__(self, cfg_dict: Optional[Mapping[str, Any]] = None, filename: Optional[str] = None):
        object.__setattr__(self, "_cfg", ConfigDict(cfg_dict or {}))
        object.__setattr__(self, "_filename", filename)

    # ------------------------------------------------------------------ loading
    @staticmethod
    def _load_dict(filename: str, _stack: Optional[List[str]] = None) -> Dict[str, Any]:
        filename = osp.abspath(osp.expanduser(filename))
        if not osp.isfile(filename):
            raise FileNotFoundError(f"config file not found: {filename}")
        if not filename.endswith(".py"):
            raise ValueError(f"only Python config files are supported, got {filename}")
        stack = list(_stack or [])
        if filename in stack:
            raise RecursionError("circular _base_ inheritance: " + " -> ".join(stack + [filename]))
        stack.append(filename)

        namespace = runpy.run_path(filename)
        file_dir = osp.dirname(filename)
        own = {k: v for k, v in namespace.items() if _is_config_value(k, v)}
        own = _expand(own, file_dir)

        bases = namespace.get(BASE_KEY, [])
        if isinstance(bases, str):
            bases = [bases]
        merged: Dict[str, Any] = {}
        for base in bases:
            merged = _merge(merged, Config._load_dict(osp.join(file_dir, base), stack))
        return _merge(merged, own)

    @staticmethod
    def fromfile(filename: str) -> "Config":
        return Config(Config._load_dict(filename), filename=osp.abspath(filename))

    # ------------------------------------------------------------------ access
    @property
    def filename(self) -> Optional[str]:
        return self._filename

    def __getattr__(self, name: str) -> Any:
        return getattr(self._cfg, name)

    def __setattr__(self, name: str, value: Any) -> None:
        self._cfg[name] = value

    def __getitem__(self, name: str) -> Any:
        return self._cfg[name]

    def __setitem__(self, name: str, value: Any) -> None:
        self._cfg[name] = value

    def __contains__(self, name: str) -> bool:
        return name in self._cfg

    def __iter__(self):
        return iter(self._cfg)

    def __len__(self) -> int:
        return len(self._cfg)

    def get(self, name: str, default: Any = None) -> Any:
        return self._cfg.get(name, default)

    def keys(self):
        return self._cfg.keys()

    def items(self):
        return self._cfg.items()

    def to_dict(self) -> Dict[str, Any]:
        return self._cfg.to_dict()

    def copy(self) -> "Config":
        return Config(copy.deepcopy(self._cfg.to_dict()), self._filename)

    # ------------------------------------------------------------------ overrides
    def merge_from_dict(self, options: Mapping[str, Any]) -> None:
        """Apply ``{'a.b.c': value}`` overrides (as given by ``--cfg-options``)."""
        cfg = self._cfg
        for dotted, value in options.items():
            keys = dotted.split(".")
            node = cfg
            for key in keys[:-1]:
                if key not in node or not isinstance(node[key], Mapping):
                    node[key] = ConfigDict()
                node = node[key]
            node[keys[-1]] = value

    # ------------------------------------------------------------------ export
    @property
    def pretty_text(self) -> str:
        lines = []
        for key, value in self._cfg.to_dict().items():
            lines.append(f"{key} = {pprint.pformat(value, sort_dicts=False, width=110)}")
        return "\n".join(lines) + "\n"

    def dump(self, path: str) -> None:
        os.makedirs(osp.dirname(osp.abspath(path)), exist_ok=True)
        with open(path, "w") as f:
            f.write(f"# Resolved config (generated from {self._filename})\n")
            f.write(self.pretty_text)

    def __repr__(self) -> str:
        return f"Config (path: {self._filename}):\n{self.pretty_text}"


class DictAction(argparse.Action):
    """argparse action for ``--cfg-options a.b=1 c='x' d=[1,2]`` (values parsed as Python literals when possible)."""

    @staticmethod
    def _parse(value: str) -> Any:
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            lowered = value.lower()
            if lowered in ("true", "false"):
                return lowered == "true"
            if lowered in ("none", "null"):
                return None
            return value

    def __call__(self, parser, namespace, values: Iterable[str], option_string=None):
        options = dict(getattr(namespace, self.dest, None) or {})
        for item in values:
            if "=" not in item:
                raise argparse.ArgumentError(self, f"expected KEY=VALUE, got {item!r}")
            key, value = item.split("=", 1)
            options[key] = self._parse(value)
        setattr(namespace, self.dest, options)
