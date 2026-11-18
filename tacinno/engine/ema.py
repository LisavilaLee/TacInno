"""Policy-weight EMA used for every released TacInno-VLA checkpoint.

One EMA copy of every trainable parameter is kept on rank 0 (parameters are identical on all DDP ranks)::

    e <- decay * e + (1 - decay) * w        after every optimizer step (torch lerp_ with weight 1 - decay)

starting from the weights the run starts from. Every checkpoint gets ``ema.safetensors`` next to
``model.safetensors`` (read back on resume). At the end :meth:`PolicyEMA.export_deploy` writes the deployable
``ema_<step>/`` directory (the step directory's files, ``model.safetensors`` holding the EMA weights, no optimizer or
resume state), which :func:`tacinno.apis.init_policy` loads.
"""
from __future__ import annotations

import json
import logging
import pathlib
import shutil

import safetensors
import safetensors.torch
import torch

__all__ = ["PolicyEMA"]


class PolicyEMA:
    def __init__(self, model: torch.nn.Module, decay: float, dtype: str = "fp32"):
        self.decay = float(decay)
        self.dtype = torch.float32 if dtype == "fp32" else torch.bfloat16
        self.names = [n for n, p in model.named_parameters() if p.requires_grad]
        params = dict(model.named_parameters())
        self.shadow = {n: params[n].detach().to(self.dtype).clone() for n in self.names}
        self.updates = 0

    @property
    def dtype_name(self) -> str:
        return "fp32" if self.dtype == torch.float32 else "bf16"

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        w = 1.0 - self.decay
        params = dict(model.named_parameters())
        for n in self.names:
            self.shadow[n].lerp_(params[n].detach().to(self.dtype), w)
        self.updates += 1

    def save(self, path) -> None:
        safetensors.torch.save_file({n: t.contiguous() for n, t in self.shadow.items()}, str(path),
                                    metadata={"decay": str(self.decay), "dtype": self.dtype_name, "updates": str(self.updates)})

    def load(self, path, device) -> None:
        sd = safetensors.torch.load_file(str(path), device=str(device))
        with safetensors.safe_open(str(path), framework="pt") as f:
            meta = f.metadata() or {}
        missing = [n for n in self.names if n not in sd]
        if missing:
            raise KeyError(f"EMA file {path} lacks {len(missing)} parameters, e.g. {missing[:3]}")
        for n in self.names:
            self.shadow[n].copy_(sd[n].to(self.dtype))
        self.updates = int(meta.get("updates", 0))

    @torch.no_grad()
    def export_deploy(self, model: torch.nn.Module, src_dir, out_dir) -> pathlib.Path:
        """Copy the checkpoint directory ``src_dir`` to ``out_dir`` without optimizer / resume / EMA files and with
        ``model.safetensors`` holding the EMA weights (written through the model, like the trainer's checkpoints)."""
        src, out = pathlib.Path(src_dir), pathlib.Path(out_dir)
        tmp = out.parent / f"tmp_{out.name}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(src, tmp, ignore=shutil.ignore_patterns("optimizer.pt", "resume_state.pt", "ema.safetensors",
                                                                 "model.safetensors"))
        params = dict(model.named_parameters())
        backup = {n: params[n].detach().to("cpu", copy=True) for n in self.names}
        try:
            for n in self.names:
                params[n].copy_(self.shadow[n].to(params[n].dtype))
            safetensors.torch.save_model(model, str(tmp / "model.safetensors"))
        finally:
            for n in self.names:
                params[n].copy_(backup[n].to(params[n].device))
        with open(tmp / "EMA_INFO.json", "w") as f:
            json.dump(dict(decay=self.decay, dtype=self.dtype_name, updates=self.updates, source=src.name), f, indent=1)
        shutil.rmtree(out, ignore_errors=True)
        tmp.rename(out)
        logging.info("EMA deploy checkpoint (%d updates, decay %s, %s) -> %s", self.updates, self.decay, self.dtype_name, out)
        return out
