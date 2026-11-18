"""Websocket policy server of TacInno-VLA (one process, one checkpoint, one port per task).

Protocol (msgpack-numpy frames, the format of ``openpi_client.msgpack_numpy``), stateless: the client keeps the
histories and sends them with every request.

* On connect the server sends a metadata dict: ``task``, ``checkpoint`` (the client compares it with the checkpoint it
  expects), ``action_rep`` ("mix"), ``action_horizon``, ``proprio_hist``, ``use_field``, ``tac_hist_offsets`` (control-step
  offsets of the tactile frames to send, oldest first, e.g. [8, 4, 0]), ``tactile_off`` (tactile-off test active).
* Simulation request: ``head``, ``wrist`` (uint8 RGB), ``qpos`` (8,), ``qhist`` (n, 8), ``tac_hist`` (list of frames
  for ``tac_hist_offsets``; each frame a dict with ``l_marker``, ``r_marker`` and ``l_depth`` / ``r_depth`` or the
  zlib-packed ``*_depth_z`` + ``*_depth_shape``), optional ``seed`` and ``k`` (noise seed = seed * 1000 + k).
* Real-robot request: as above, but each ``tac_hist`` frame holds the workstation's pre-processed left-gel arrays
  ``w96`` (96, 128), ``mk`` (2, 63, 2), ``valid`` (63,).
* Reply: ``model8`` (32, 8) model-space chunk (arm deltas w.r.t. qpos, gripper absolute), ``server_timing``; the real-robot
  reply also carries ``abs8`` (absolute targets).

Tactile-off test: with ``rest_frame`` every tactile frame the model reads is replaced by a fixed no-contact frame
(the rest of the request is unchanged), and the metadata reports ``tactile_off = True``.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import traceback
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from tacinno.apis.inference import TacInnoPolicy, resolve_abs

__all__ = ["PolicyServer", "load_rest_frame"]


def load_rest_frame(npz_path: str):
    """No-contact tactile frame (l_depth, l_marker, r_depth, r_marker) for the tactile-off test."""
    r = np.load(npz_path)
    return (r["l_depth"], r["l_marker"], r["r_depth"], r["r_marker"])


def _depth(d: dict, side: str) -> np.ndarray:
    if f"{side}_depth_z" in d:
        shape = tuple(int(x) for x in np.asarray(d[f"{side}_depth_shape"]).reshape(-1))
        return np.frombuffer(zlib.decompress(d[f"{side}_depth_z"]), np.float32).reshape(shape)
    return d[f"{side}_depth"]


class PolicyServer:
    """Serves one :class:`TacInnoPolicy` on ``entries`` = [(task, port), ...]; requests of all ports are handled one at
    a time by one asyncio loop (inference is synchronous on the GPU)."""

    def __init__(self, policy: TacInnoPolicy, entries: Sequence[tuple[str, int]], *, bind: str = "127.0.0.1",
                 rest_frame=None, group: str = "tacinno"):
        self.policy = policy
        self.entries = [(t, int(p)) for t, p in entries]
        self.bind = bind
        self.rest = rest_frame
        self.group = group
        self.n = 0
        for t, _ in self.entries:  # warm the per-task normalization cache
            policy.select(t)
        policy.select(self.entries[0][0])
        if policy.real_robot and rest_frame is not None:
            raise ValueError("the tactile-off test is implemented for the simulation protocol only")

    # ---------------------------------------------------------------------------------------------------- protocol
    def metadata(self, task: str) -> dict:
        p = self.policy
        ckpt = os.path.normpath(p.checkpoint_dir)
        if p.real_robot:
            return dict(task=task, prompt=p.prompts[task], checkpoint=ckpt, exp=Path(ckpt).parent.name,
                        action_rep="mix", action_horizon=p.horizon, control_hz=10, tac_hist_offsets=p.tactile_offsets,
                        tactile="left Xense pre-processed w96 / mk / valid per frame; right gel = copy of left",
                        vision_only=not p.use_field,
                        images="head = cam_front, wrist = cam_high, uint8 RGB, any size (resized to 224 here)")
        return dict(task=task, checkpoint=ckpt, exp=Path(ckpt).parent.name, group=self.group, action_rep="mix",
                    use_wrist="wrist" in p.cameras, action_horizon=p.horizon, proprio_hist=p.H, use_field=p.use_field,
                    use_code=bool(getattr(p.model_config, "cond_code", False)),
                    code_zero=bool(getattr(p.model_config, "cond_code_zero", False)),
                    tac_hist_offsets=p.tactile_offsets, tactile_off=self.rest is not None)

    def _sim_frame(self, d: dict):
        if self.rest is not None:
            return self.rest
        return (_depth(d, "l"), d["l_marker"], _depth(d, "r"), d["r_marker"])

    def infer(self, task: str, obs: dict) -> dict:
        p = self.policy
        p.select(task)
        q = np.asarray(obs["qpos"], np.float32).reshape(8)
        frames = None
        if p.use_field:
            if "tac_hist" in obs:
                raw = list(obs["tac_hist"])
            elif "tac" in obs:
                raw = [obs["tac"]]
            else:
                raise ValueError("request lacks tac_hist")
            frames = raw if p.real_robot else [self._sim_frame(d) for d in raw]
        cond = p.build_cond(qhist=np.asarray(obs.get("qhist", q[None]), np.float32), tac_frames=frames)
        noise = int(obs["seed"]) * 1000 + int(obs["k"]) if "seed" in obs and "k" in obs else None
        model8 = p.infer_model8(np.asarray(obs["head"]), np.asarray(obs["wrist"]) if "wrist" in obs else None, q,
                                cond=cond, noise_seed=noise)
        self.n += 1
        if self.n <= 3 or self.n % 500 == 0:
            logging.info("infer #%d task %s seed %s k %s", self.n, task, obs.get("seed"), obs.get("k"))
        out = {"model8": model8}
        if p.real_robot:
            out["abs8"] = resolve_abs(model8, q)
        return out

    # ---------------------------------------------------------------------------------------------------- serving
    def _handler(self, task: str):
        import websockets
        import websockets.frames
        from openpi_client import msgpack_numpy

        meta = self.metadata(task)

        async def handler(ws):
            packer = msgpack_numpy.Packer()
            await ws.send(packer.pack(meta))
            while True:
                try:
                    obs = msgpack_numpy.unpackb(await ws.recv())
                    t0 = time.monotonic()
                    out = self.infer(task, obs)
                    out["server_timing"] = {"infer_ms": (time.monotonic() - t0) * 1000}
                    await ws.send(packer.pack(out))
                except websockets.ConnectionClosed:
                    break
                except Exception:
                    await ws.send(traceback.format_exc())
                    await ws.close(code=websockets.frames.CloseCode.INTERNAL_ERROR, reason="Internal server error")
                    raise
        return handler, meta

    async def _main(self) -> None:
        import websockets.asyncio.server as _server

        servers = []
        for task, port in self.entries:
            handler, meta = self._handler(task)
            servers.append(await _server.serve(handler, self.bind, port, compression=None, max_size=None))
            logging.info("serving %s on %s:%d meta %s", task, self.bind, port, meta)
        logging.info("SERVING_ALL %d ports", len(servers))
        await asyncio.gather(*(s.serve_forever() for s in servers))

    def serve_forever(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        asyncio.run(self._main())
