import argparse
import os.path as osp

import pytest

from tacinno.config import Config, DictAction
from tacinno.registry import Registry

CFG_DIR = osp.join(osp.dirname(__file__), "data", "configs")


def test_inheritance_and_merge():
    cfg = Config.fromfile(osp.join(CFG_DIR, "child.py"))
    assert cfg.model.type == "Toy"                      # inherited
    assert cfg.model.width == 16                        # overridden
    assert cfg.model.head == {"type": "MLP", "hidden": 32}   # _delete_ replaces the base dict
    assert cfg.model.tags == ["c"]                      # lists replace
    assert cfg.train == {"lr": 1e-3, "steps": 200, "warmup": 10}  # later base wins, dicts merge
    assert cfg.data_root.endswith(osp.join("configs", "_base_", "data"))
    assert "math" not in cfg and "_private" not in cfg and "_base_" not in cfg


def test_cfg_options_and_dump(tmp_path):
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg-options", nargs="+", action=DictAction)
    args = parser.parse_args(["--cfg-options", "model.width=64", "train.name=run1", "new.flag=true", "x=[1,2]"])
    cfg = Config.fromfile(osp.join(CFG_DIR, "child.py"))
    cfg.merge_from_dict(args.cfg_options)
    assert cfg.model.width == 64 and cfg.train.name == "run1" and cfg.new.flag is True and cfg.x == [1, 2]
    out = tmp_path / "resolved.py"
    cfg.dump(str(out))
    again = Config.fromfile(str(out))
    assert again.to_dict() == cfg.to_dict()


def test_registry_build():
    reg = Registry("toy")

    @reg.register_module()
    class Toy:
        def __init__(self, width, depth=1):
            self.width, self.depth = width, depth

    obj = reg.build(dict(type="Toy", width=3), depth=5)
    assert (obj.width, obj.depth) == (3, 5)
    obj = reg.build(dict(type="Toy", width=3, depth=2), depth=5)
    assert obj.depth == 2
    with pytest.raises(KeyError):
        reg.build(dict(type="Missing"))
    with pytest.raises(KeyError):
        reg.register_module(name="Toy", module=lambda: None)
