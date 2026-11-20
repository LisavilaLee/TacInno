"""UniVTAC closed-loop client for TacInno-VLA.

This module is the policy that the official UniVTAC evaluator (``scripts/eval_policy.py``) instantiates. It is
self-contained (NumPy, OpenCV, PyTorch, ``websockets`` and ``msgpack``; no ``tacinno`` install needed inside the
simulator environment) and talks to a TacInno-VLA policy server (``tools/serve_policy.py``) over a websocket.
``tools/eval_univtac.py install`` copies it into a UniVTAC checkout as ``policy/TacInnoVLA``.

Protocol (msgpack with NumPy extension, see :mod:`msgpack_numpy`)
-----------------------------------------------------------------
On connect the server sends its metadata (``task``, ``checkpoint``, ``proprio_hist`` H, ``use_field``,
``tac_hist_offsets``, ``action_horizon``, ...). Each request carries the observation of one control step:

=============  ==========================  ==============================================================
key            value                       meaning
=============  ==========================  ==============================================================
``head``       uint8 (224, 224, 3)         head camera, resized with ``cv2.INTER_LINEAR`` (the server would
                                           apply the identical resize, so the payload is just smaller)
``wrist``      uint8 (224, 224, 3)         wrist camera (ignored by head-camera-only checkpoints)
``qpos``       float32 (8,)                7 arm joints + gripper of the current step
``qhist``      float32 (h, 8)              joint states of the last h = min(H, steps so far) control steps,
                                           oldest first (the server repeats the first row up to H)
``tac_hist``   list of tactile packets     raw readings of steps t-8, t-4, t (``tac_hist_offsets``), clipped
                                           at the episode start; one packet per step (see ``pack_tactile``)
``seed``       int                         episode seed; with ``k`` it seeds the flow-matching noise, so a
``k``          int                         rollout is reproducible
``step``       int                         control step (logging only)
=============  ==========================  ==============================================================

The response holds ``model8``: float32 (32, 8), an action chunk in model space whose row 0 is the current step.
Row i (i >= 1) is the joint-target offset of step t+i from the joint state at inference time for the 7 arm joints and
the absolute gripper command for the last entry.

Execution
---------
Chunked execution: the policy is queried at step s and executes rows 1 .. ``replan`` of the chunk at steps
s .. s+replan-1, all resolved against the joint state of step s; then it queries again. The paper uses
``replan = 31`` (the whole chunk) for Insert Hole, Insert Tube, Pull-out Key and Put Bottle, and ``replan = 10`` for
Lift Can and Lift Bottle. One control step is one ``task.take_action`` (one recorded data frame).

Tactile-off test
----------------
``tactile_off: true`` replaces both gels' readings by a fixed no-contact frame at every step while every other input
stays live, so any change of the success rate is attributable to tactile input. The frame is read from
``no_contact_frame`` (an ``.npz`` with ``l_depth``, ``l_marker``, ``r_depth``, ``r_marker``) or, if unset, the first
frame of each episode is used (valid only for tasks that start without contact).
"""
from __future__ import annotations

import os
import time
import zlib
from typing import Any, Dict, List, Optional

import cv2
import numpy as np
import torch
import websockets.sync.client

try:  # works both inside the tacinno package and when copied into UniVTAC's policy/ directory
    from . import msgpack_numpy
except ImportError:  # pragma: no cover
    from openpi_client import msgpack_numpy  # type: ignore

__all__ = [
    "InfrastructureError",
    "ServerUnreachable",
    "PolicyError",
    "PolicyServerClient",
    "TacInnoVLAPolicy",
    "TactileOff",
    "Policy",
    "pack_tactile",
    "resize224",
    "DEFAULT_REPLAN",
]

GELS = (("l", "left_tactile"), ("r", "right_tactile"))

#: action execution horizon of the released checkpoints (rows of the chunk executed per inference)
DEFAULT_REPLAN: Dict[str, int] = {
    "insert_hole": 31,
    "insert_tube": 31,
    "lift_bottle": 10,
    "lift_can": 10,
    "pull_out_key": 31,
    "put_bottle_in_shelf": 31,
}


class InfrastructureError(RuntimeError):
    """A setup problem of the evaluation (the server serves another task or checkpoint, or cannot be reached). The
    evaluator records the episode as an error; fix the setup and rerun."""


class PolicyError(RuntimeError):
    """The policy server returned an error or an invalid action chunk. The client then holds the robot still for the
    rest of the episode, so the episode counts as a failure of the policy."""


class ServerUnreachable(InfrastructureError):
    """The policy server could not be reached within the reconnect budget."""


# ----------------------------------------------------------------------------------------------------------- helpers
def _np(x: Any, dtype=None) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu().numpy()
    x = np.asarray(x)
    return x.astype(dtype) if dtype is not None else x


def _to_uint8_hwc(x: Any) -> np.ndarray:
    a = _np(x)
    if a.ndim != 3 or a.shape[-1] != 3:
        raise ValueError(f"expected an HWC RGB image, got shape {a.shape}")
    if a.dtype != np.uint8:
        if np.issubdtype(a.dtype, np.floating) and a.size and float(np.nanmax(a)) <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)


def resize224(x: Any) -> np.ndarray:
    """Camera image as uint8 (224, 224, 3), resized exactly as the server does (``cv2.INTER_LINEAR``)."""
    a = _to_uint8_hwc(x)
    return a if a.shape[:2] == (224, 224) else cv2.resize(a, (224, 224), interpolation=cv2.INTER_LINEAR)


def pack_tactile(tactile: Dict[str, Any]) -> Dict[str, Any]:
    """Raw readings of both gels for one control step.

    ``tactile[<gel>]["depth"]`` is the indentation depth map (float32, losslessly zlib-compressed on the wire) and
    ``tactile[<gel>]["marker"]`` the marker positions (float32). The server turns them into the tactile observation
    (indentation field + marker tangential displacement field) exactly as in training.
    """
    out: Dict[str, Any] = {}
    for side, name in GELS:
        depth = np.ascontiguousarray(_np(tactile[name]["depth"], np.float32))
        out[f"{side}_depth_z"] = zlib.compress(depth.tobytes(), 1)
        out[f"{side}_depth_shape"] = np.asarray(depth.shape, np.int64)
        out[f"{side}_marker"] = _np(tactile[name]["marker"], np.float32)
    return out


# ------------------------------------------------------------------------------------------------------------ client
class PolicyServerClient:
    """Blocking websocket client of one policy server port.

    Args:
        host, port: server address.
        expect_checkpoint: if set, the ``checkpoint`` announced by the server must end with this path (guards against
            evaluating the wrong model when several servers run side by side).
        expect_tactile_off: if not None, the server's tactile-off flag must equal it.
        reconnect_s: how long to keep retrying an unreachable server.
    """

    def __init__(self, host: str, port: int, expect_checkpoint: str = "", expect_tactile_off: Optional[bool] = None,
                 reconnect_s: float = 1200.0):
        self.uri = f"ws://{host}:{port}"
        self.expect_checkpoint = os.path.normpath(expect_checkpoint) if expect_checkpoint else ""
        self.expect_tactile_off = expect_tactile_off
        self.reconnect_s = float(reconnect_s)
        self.packer = msgpack_numpy.Packer()
        self.ws = None
        self.meta: Dict[str, Any] = {}
        self._connect()

    @staticmethod
    def _tactile_off_flag(meta: Dict[str, Any]) -> bool:
        """Server-side tactile-off flag."""
        return bool(meta.get("tactile_off", False))

    def _check_meta(self, meta: Dict[str, Any]) -> None:
        served = os.path.normpath(str(meta.get("checkpoint", "")))
        if self.expect_checkpoint and not served.endswith(self.expect_checkpoint):
            raise InfrastructureError(f"server {self.uri} serves {served}, expected {self.expect_checkpoint}")
        if self.expect_tactile_off is not None and self._tactile_off_flag(meta) != self.expect_tactile_off:
            raise InfrastructureError(f"server {self.uri} tactile-off={self._tactile_off_flag(meta)}, "
                                      f"expected {self.expect_tactile_off}")

    def _connect(self) -> None:
        t0 = time.time()
        while True:
            try:
                self.ws = websockets.sync.client.connect(self.uri, compression=None, max_size=None, open_timeout=30)
                meta = msgpack_numpy.unpackb(self.ws.recv(timeout=60))
                self._check_meta(meta)
                self.meta = meta
                return
            except InfrastructureError:
                raise
            except Exception as exc:  # server still starting, network hiccup, ...
                if time.time() - t0 > self.reconnect_s:
                    raise ServerUnreachable(f"policy server {self.uri} unreachable: {exc!r}") from exc
                print(f"[TacInnoVLA] server {self.uri} not reachable ({exc!r}); retrying in 10 s", flush=True)
                time.sleep(10)

    def infer(self, request: Dict[str, Any]) -> Dict[str, Any]:
        for attempt in range(2):
            try:
                self.ws.send(self.packer.pack(request))
                reply = self.ws.recv(timeout=300)
            except Exception as exc:
                if attempt == 1:
                    raise ServerUnreachable(f"policy server {self.uri}: {exc!r}") from exc
                print(f"[TacInnoVLA] inference failed ({exc!r}); reconnecting", flush=True)
                self.close()
                self._connect()
                continue
            if isinstance(reply, str):  # the server sends its traceback as text
                raise PolicyError(f"policy server error:\n{reply}")
            return msgpack_numpy.unpackb(reply)
        raise AssertionError("unreachable")

    def close(self) -> None:
        try:
            if self.ws is not None:
                self.ws.close()
        except Exception:
            pass


# ------------------------------------------------------------------------------------------------------------ policy
class TacInnoVLAPolicy:
    """UniVTAC policy (``reset`` / ``eval(task, observation)`` / ``close``) backed by a TacInno-VLA server.

    The client keeps the per-episode history the stateless server needs: the joint states of the last H steps and
    the raw tactile readings of the steps listed in ``tac_hist_offsets``.

    Deploy-config keys: ``host``, ``port`` (both overridable by the environment variables ``TACINNO_HOST`` /
    ``TACINNO_PORT``), ``task_name`` (set by the evaluator), ``replan`` (default from :data:`DEFAULT_REPLAN`),
    ``expect_checkpoint``, ``expect_tactile_off``.
    """

    def __init__(self, args: Dict[str, Any], client: Optional[PolicyServerClient] = None):
        self.task_name = str(args.get("task_name"))
        host = os.environ.get("TACINNO_HOST", str(args.get("host", "127.0.0.1")))
        port = int(os.environ.get("TACINNO_PORT", args.get("port", 8000)))
        self.replan = int(args.get("replan") or DEFAULT_REPLAN.get(self.task_name, 10))
        expect_off = args.get("expect_tactile_off", None)
        if expect_off is not None:
            expect_off = str(expect_off).lower() in ("1", "true", "yes")
        self.client = client or PolicyServerClient(host, port, str(args.get("expect_checkpoint", "") or ""),
                                                   expect_off)
        meta = self.client.meta
        if meta.get("task") is not None and str(meta.get("task")) != self.task_name:
            raise InfrastructureError(f"server task {meta.get('task')!r} != evaluated task {self.task_name!r}")
        self.horizon = int(meta.get("action_horizon", 32) or 32)
        if not 1 <= self.replan < self.horizon:
            raise ValueError(f"replan must be in [1, {self.horizon - 1}], got {self.replan}")
        self.proprio_hist = max(1, int(meta.get("proprio_hist", 0) or 0))
        self.use_field = bool(meta.get("use_field", False))
        self.tac_hist_offsets: List[int] = [int(o) for o in meta.get("tac_hist_offsets", []) or []]
        self.keep = max([self.proprio_hist] + [o + 1 for o in self.tac_hist_offsets])
        print(f"[TacInnoVLA] {self.client.uri} task={self.task_name} replan={self.replan} H={self.proprio_hist} "
              f"tactile offsets={self.tac_hist_offsets}", flush=True)
        self.reset()

    # -------------------------------------------------------------------- episode state
    def reset(self) -> None:
        self.q_hist: List[np.ndarray] = []
        self.tac_hist: List[Dict[str, Any]] = []
        self.chunk: Optional[np.ndarray] = None
        self.chunk_step = -1
        self.chunk_q: Optional[np.ndarray] = None
        self.seed = -1
        self.step = 0
        self.k = 0
        self.latency_s: List[float] = []
        self.halted = False

    def build_request(self, observation: Dict[str, Any], q: np.ndarray) -> Dict[str, Any]:
        cams = observation["observation"]
        request: Dict[str, Any] = {
            "head": resize224(cams["head"]["rgb"]),
            "wrist": resize224(cams["wrist"]["rgb"]),
            "qpos": q,
            # up to H rows; at the start of an episode fewer rows are sent and the server repeats the first one
            "qhist": np.stack(self.q_hist[-self.proprio_hist:], 0),
            "seed": int(self.seed),
            "k": int(self.k),
            "step": int(self.step),
        }
        if self.tac_hist_offsets:
            n = len(self.tac_hist)
            request["tac_hist"] = [self.tac_hist[max(n - 1 - o, 0)] for o in self.tac_hist_offsets]
        elif self.use_field:
            request["tac"] = self.tac_hist[-1]
        return request

    def _infer_chunk(self, observation: Dict[str, Any], q: np.ndarray) -> np.ndarray:
        request = self.build_request(observation, q)
        self.k += 1
        t0 = time.perf_counter()
        chunk = np.asarray(self.client.infer(request)["model8"], dtype=np.float32)
        self.latency_s.append(time.perf_counter() - t0)
        if chunk.ndim != 2 or chunk.shape[1] != 8 or chunk.shape[0] <= self.replan or not np.all(np.isfinite(chunk)):
            raise PolicyError(f"invalid action chunk of shape {chunk.shape}")
        return chunk

    @staticmethod
    def resolve(action8: np.ndarray, q_ref: np.ndarray) -> np.ndarray:
        """Joint target of one chunk row: arm offsets are added to the reference joint state, gripper is absolute."""
        out = action8.astype(np.float32).copy()
        out[:7] += q_ref[:7]
        return out

    def act(self, observation: Dict[str, Any], seed: int) -> np.ndarray:
        """Joint target (8,) for the current control step; updates the history and queries the server if needed."""
        if self.step == 0:
            self.seed = int(seed)
        q = _np(observation["embodiment"]["joint"][:8], np.float32).reshape(8)
        self.q_hist = (self.q_hist + [q.copy()])[-self.keep:]
        if self.use_field or self.tac_hist_offsets:
            self.tac_hist = (self.tac_hist + [pack_tactile(observation["tactile"])])[-self.keep:]
        e = self.step
        if not self.halted and (self.chunk is None or e - self.chunk_step >= self.replan):
            try:
                self.chunk, self.chunk_step, self.chunk_q = self._infer_chunk(observation, q), e, q.copy()
            except PolicyError as exc:
                print(f"[TacInnoVLA] seed {self.seed} step {e}: {exc}; holding still, the episode counts as a failure",
                      flush=True)
                self.halted, self.hold_q = True, q.copy()
        if self.halted:
            self.step += 1
            return self.hold_q.copy()
        action = self.resolve(self.chunk[1 + e - self.chunk_step], self.chunk_q)
        if e % 100 == 0:
            ms = 1000 * float(np.mean(self.latency_s[-20:])) if self.latency_s else float("nan")
            print(f"[TacInnoVLA] seed {self.seed} step {e} inference {ms:.0f} ms", flush=True)
        self.step += 1
        return action

    def eval(self, task: Any, observation: Dict[str, Any]) -> None:
        action = self.act(observation, seed=int(task.cfg.seed))
        task.take_action(torch.as_tensor(action, device=task.device, dtype=torch.float32), action_type="qpos")

    def close(self) -> None:
        self.client.close()


class TactileOff:
    """Tactile-off test wrapper: the wrapped policy receives a fixed no-contact tactile frame at every step."""

    def __init__(self, inner: TacInnoVLAPolicy, no_contact_frame: Optional[str] = None):
        self.inner = inner
        self.fixed: Optional[Dict[str, Dict[str, np.ndarray]]] = None
        if no_contact_frame:
            z = np.load(no_contact_frame)
            self.fixed = {name: {"depth": z[f"{side}_depth"], "marker": z[f"{side}_marker"]} for side, name in GELS}
        self._episode_frame: Optional[Dict[str, Dict[str, np.ndarray]]] = None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.inner, name)

    def reset(self) -> None:
        self._episode_frame = None
        self.inner.reset()

    def _frame(self, observation: Dict[str, Any]) -> Dict[str, Dict[str, np.ndarray]]:
        if self.fixed is not None:
            return self.fixed
        if self._episode_frame is None:  # first frame of the episode
            self._episode_frame = {name: {"depth": _np(observation["tactile"][name]["depth"], np.float32).copy(),
                                          "marker": _np(observation["tactile"][name]["marker"], np.float32).copy()}
                                   for _, name in GELS}
        return self._episode_frame

    def eval(self, task: Any, observation: Dict[str, Any]) -> None:
        obs = dict(observation)
        obs["tactile"] = self._frame(observation)
        self.inner.eval(task, obs)

    def close(self) -> None:
        self.inner.close()


def Policy(args: Dict[str, Any]):  # noqa: N802  (name required by the UniVTAC evaluator)
    """Factory called by UniVTAC's ``eval_policy.py`` with the parsed deploy config."""
    policy = TacInnoVLAPolicy(args)
    if str(args.get("tactile_off", False)).lower() in ("1", "true", "yes"):
        print("[TacInnoVLA] tactile-off test: tactile input replaced by a fixed no-contact frame", flush=True)
        return TactileOff(policy, args.get("no_contact_frame") or None)
    return policy
