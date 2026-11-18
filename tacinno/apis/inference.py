"""Inference API of TacInno-VLA.

    from tacinno.apis import init_policy
    from tacinno.datasets import stats_path
    policy = init_policy("checkpoints/sim/lift_can", task="lift_can", cond_stats=stats_path("univtac_cond_stats"))
    chunk = policy.infer(head=rgb_head, wrist=rgb_wrist, qpos=q8, qhist=q8_history, tac_frames=[tac_t8, tac_t4, tac_t])

Inputs of one decision (simulation, UniVTAC; one control step = one recorded data frame):

* ``head``, ``wrist``: uint8 RGB images of any size, resized to 224 x 224 (cv2 bilinear). Policies trained with the
  head camera only (Put Bottle) ignore the wrist image.
* ``qpos`` (8,): 7 arm joints [rad] + gripper opening; ``qhist`` (n, 8): joints of the control steps up to t (oldest
  first), used only by models with proprio tokens (frames before the episode start repeat the first one).
* ``tac_frames``: raw tactile frames of control steps t-8, t-4, t (clipped at the episode start), each a tuple
  ``(left_depth (240, 320), left_marker (2, 64, 2), right_depth, right_marker)``. They become tactile observations
  (2, 3, 96, 128) through :func:`tacinno.datasets.tactile_fields.deploy_frame` and ``build_fields`` and are rounded
  through float16 like the stored training arrays.
* ``noise_seed``: optional; the flow-matching noise is drawn after ``torch.manual_seed(noise_seed)``.

Output: (32, 8) chunk in model space: arm joint targets relative to ``qpos`` at inference time and absolute gripper
opening; :func:`resolve_abs` turns it into absolute joint targets.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import cv2
import numpy as np
import torch

from tacinno.datasets import tactile_fields as TF

__all__ = ["TacInnoPolicy", "init_policy", "UNIVTAC_PROMPTS", "REAL_ROBOT_PROMPTS", "resolve_abs", "default_tactile_stats"]

# 120-D unified state / action layout: arm joints at 9:16, gripper at 16 + 28
GRIPPER_SLOT = 28
ARM = slice(9, 16)
GRIP = 16 + GRIPPER_SLOT

# Training prompts (UniVTAC instruction map of the data parser)
UNIVTAC_PROMPTS = {
    "insert_hole": "insert the stick to the hole.",
    "insert_tube": "insert the tube to the fixed slot.",
    "lift_can": "grasp the can and lifts it vertically without slippage.",
    "lift_bottle": "grasp the bottle and lift it vertically, keeping its final base within 5 cm of the wall.",
    "pull_out_key": "pull out the key.",
    "put_bottle_in_shelf": "grasp the bottle, then position it into the shelf cavity.",
}
REAL_ROBOT_PROMPTS = {
    "real_wipe_board": "wipe the whiteboard clean with the eraser.",
    "real_lift_can": "pick up the lying can and stand it upright on the table.",
    "real_insert_plug": "insert the plug into the power strip.",
    "real_unscrew_cap": "unscrew the cap of the medicine bottle.",
}


def default_tactile_stats() -> str:
    """Statistics of the UniVTAC training tactile data (rest marker grid, sigma_u, contact-property mean / std)."""
    return str(Path(TF.__file__).resolve().parent / "stats" / "univtac_tactile_stats.json")


def resize224(img: np.ndarray) -> np.ndarray:
    img = np.asarray(img)
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    if img.shape[0] == 224 and img.shape[1] == 224:
        return img
    return cv2.resize(img, (224, 224), interpolation=cv2.INTER_LINEAR)


def build_state(qpos8: np.ndarray, state_dim: int) -> np.ndarray:
    q = np.asarray(qpos8, dtype=np.float32).reshape(8)
    s = np.zeros((1, state_dim), dtype=np.float32)
    s[0, ARM] = q[:7]
    s[0, GRIP] = q[7]
    return s


def chunk_to_model8(chunk: np.ndarray) -> np.ndarray:
    out = np.zeros((chunk.shape[0], 8), dtype=np.float32)
    out[:, :7] = chunk[:, ARM]
    out[:, 7] = chunk[:, GRIP]
    return out


def resolve_abs(model8: np.ndarray, qpos8_base: np.ndarray) -> np.ndarray:
    """Model-space chunk -> absolute targets: arm = qpos at inference + delta, gripper as is."""
    a = np.asarray(model8, dtype=np.float32).copy()
    a[..., :7] += np.asarray(qpos8_base, np.float32).reshape(8)[:7]
    return a


class ConditionBuilder:
    """Builds the condition inputs of one decision exactly like the training arrays (see the module docstring)."""

    def __init__(self, cond_stats: str, tactile_stats: str, device: str = "cuda"):
        cs = json.loads(Path(cond_stats).read_text())
        self.j_mean = np.asarray(cs["j_mean"], np.float32)
        self.j_std = np.asarray(cs["j_std"], np.float32)
        self.stats = json.loads(Path(tactile_stats).read_text())
        self.device = device
        rest = torch.tensor(self.stats["rest_grid"], dtype=torch.float32).reshape(TF.GRID_R, TF.GRID_C, 2)
        self.samp = TF.grid_sampling_coords(rest, device)
        self.sigma_u = float(self.stats["sigma_u"])
        if "sigma_u" in cs and abs(self.sigma_u - float(cs["sigma_u"])) > 1e-9:
            raise ValueError("sigma_u of the tactile statistics differs from the one used for the training fields")
        self.last_contact = None

    def proprio(self, qhist8: np.ndarray, H: int) -> np.ndarray:
        """qhist8 (n, 8) raw joints of control steps up to t (oldest first, n >= 1) -> (H, 8) normalized."""
        q = np.asarray(qhist8, np.float32).reshape(-1, 8)
        if len(q) < H:
            q = np.concatenate([np.repeat(q[:1], H - len(q), 0), q], 0)
        return (q[-H:] - self.j_mean) / self.j_std

    @torch.no_grad()
    def field(self, depth_l, marker_l, depth_r, marker_r) -> np.ndarray:
        """Raw simulator frame of both gels -> tactile observation (2, 3, 96, 128), rounded through float16."""
        gels = [TF.deploy_frame(np.asarray(d, np.float32), np.asarray(m, np.float32), g, self.stats)
                for g, (d, m) in enumerate(((depth_l, marker_l), (depth_r, marker_r)))]
        mv = np.array([g["m_valid"] for g in gels])
        self.last_contact = dict(  # contact properties of this frame, standardized like the training targets
            m=TF.normalize_m(np.stack([g["m"] for g in gels]), mv, np.asarray(self.stats["m_mean"], np.float32),
                             np.asarray(self.stats["m_std"], np.float32)),
            mv=mv, ec=np.array([g["e_c"] for g in gels]))
        w = torch.from_numpy(np.stack([g["w96"] for g in gels]).astype(np.float32)).to(self.device)
        d = torch.from_numpy(np.stack([g["d"] for g in gels]).astype(np.float32)).to(self.device)
        v = torch.from_numpy(np.stack([g["valid"] for g in gels])).to(self.device)
        X = TF.build_fields(w, d, v, self.sigma_u, self.samp)  # (2, 3, 96, 128)
        return X.half().float().cpu().numpy()


class RealRobotConditionBuilder(ConditionBuilder):
    """Real robot (one Xense gel): frames come pre-processed by the workstation (``w96`` (96, 128) indentation field,
    ``mk`` (2, 63, 2) [reference, current] marker positions, ``valid`` (63,)); the right gel is a copy of the left.
    Fields are assembled on the CPU, as for the real-robot training data."""

    def __init__(self, cond_stats: str, tactile_stats: str):
        super().__init__(cond_stats, tactile_stats, device="cpu")
        self.order = np.asarray(self.stats["marker_order"][0])
        self.rest = np.asarray(self.stats["rest_grid"], np.float32)

    @torch.no_grad()
    def field_from_frame(self, fr: Mapping) -> np.ndarray:
        w1 = np.asarray(fr["w96"], np.float32).reshape(96, 128)
        mk = np.asarray(fr["mk"], np.float32).reshape(2, 63, 2)
        v1 = np.asarray(fr["valid"]).astype(bool).reshape(63)[self.order].reshape(TF.GRID_R, TF.GRID_C)
        d1 = (mk[1][self.order] - self.rest).reshape(TF.GRID_R, TF.GRID_C, 2)
        w = torch.from_numpy(np.stack([w1, w1]))
        d = torch.from_numpy(np.stack([d1, d1]).astype(np.float32))
        v = torch.from_numpy(np.stack([v1, v1]))
        return TF.build_fields(w, d, v, self.sigma_u, self.samp).half().float().numpy()


class TacInnoPolicy:
    """A loaded TacInno-VLA checkpoint (one set of weights serves every task it was trained on; selecting a task only
    switches the per-task normalization statistics and the prompt).

    Args:
        checkpoint_dir: deployable checkpoint (``ema_<step>/``: model.safetensors, model_config.json,
            train_config.json, normalization/).
        task: initial task id; ``prompts`` maps task ids to the training prompts.
        cond_stats: ``cond_stats.json`` of the training condition data (joint mean / std).
        tactile_stats: tactile statistics of the training data (default: the UniVTAC statistics).
        cameras: ("head", "wrist") or ("head",) for models trained with the head camera only.
        domain_prefix: prefix of the normalization domain names ("UniVTAC_" or "Real_").
        num_steps: Euler steps of the flow-matching sampler (10).
    """

    def __init__(self, checkpoint_dir: str, task: str, *, cond_stats: str, tactile_stats: str | None = None,
                 device: str = "cuda", num_steps: int = 10, cameras: Sequence[str] = ("head", "wrist"),
                 domain_prefix: str = "UniVTAC_", prompts: Mapping[str, str] | None = None, real_robot: bool = False):
        from openpi.policies.ftp1_inference_wrapper import FTP1InferenceWrapper

        self.prompts = dict(prompts or (REAL_ROBOT_PROMPTS if real_robot else UNIVTAC_PROMPTS))
        if task not in self.prompts:
            raise KeyError(f"unknown task {task!r}; known: {sorted(self.prompts)}")
        self.checkpoint_dir = checkpoint_dir
        self.domain_prefix = domain_prefix
        self.cameras = tuple(cameras)
        self.task, self.prompt = task, self.prompts[task]
        self.wrapper = FTP1InferenceWrapper(checkpoint_dir=checkpoint_dir, domain_name=f"{domain_prefix}{task}",
                                            device=device, num_inference_steps=num_steps)
        mc = self.wrapper.model_config
        if bool(getattr(mc, "use_tactile_input", False)):
            raise ValueError("expected a TacInno-VLA checkpoint (the FTP-1 tactile branch must be off)")
        self.model_config = mc
        self.state_dim = int(self.wrapper.get_state_dim())
        self.H = int(getattr(mc, "cond_proprio_hist", 0) or 0)
        self.use_field = bool(getattr(mc, "cond_tactile", False))
        self.tac_hist = [int(o) for o in str(getattr(mc, "cond_tac_hist", "") or "").split(",") if o.strip()]
        self.horizon = int(mc.action_horizon)
        stats = tactile_stats or default_tactile_stats()
        self.cb = (RealRobotConditionBuilder(cond_stats, stats) if real_robot
                   else ConditionBuilder(cond_stats, stats, device)) if (self.H or self.use_field) else None
        self.real_robot = real_robot
        self._norm_cache = {task: (self.wrapper.domain_norm_stats, self.wrapper.shared_norm_stats)}

    @property
    def model(self) -> torch.nn.Module:
        return self.wrapper.model

    @property
    def tactile_offsets(self) -> list[int]:
        """Control-step offsets of the tactile frames the model reads, oldest first (e.g. [8, 4, 0])."""
        return (self.tac_hist + [0]) if self.use_field else []

    def select(self, task: str) -> None:
        if task == self.task:
            return
        if task not in self.prompts:
            raise KeyError(f"unknown task {task!r}")
        w = self.wrapper
        if task not in self._norm_cache:
            w.domain_name = f"{self.domain_prefix}{task}"
            w._load_norm_stats()
            self._norm_cache[task] = (w.domain_norm_stats, w.shared_norm_stats)
        w.domain_name = f"{self.domain_prefix}{task}"
        w.domain_norm_stats, w.shared_norm_stats = self._norm_cache[task]
        self.task, self.prompt = task, self.prompts[task]

    def build_cond(self, qhist=None, tac_frames: Iterable | None = None) -> dict | None:
        """Condition inputs of one decision; ``tac_frames`` holds the frames of ``tactile_offsets`` (oldest first)."""
        if self.cb is None:
            return None
        out = {}
        if self.H:
            out["proprio"] = self.cb.proprio(qhist, self.H)
        if self.use_field:
            frames = list(tac_frames or [])
            if len(frames) != len(self.tactile_offsets):
                raise ValueError(f"the model reads {len(self.tactile_offsets)} tactile frames "
                                 f"(offsets {self.tactile_offsets}), got {len(frames)}")
            if self.real_robot:
                fields = [self.cb.field_from_frame(f) for f in frames]
            else:
                fields = [self.cb.field(*f) for f in frames]
            out["field"] = fields[-1]
            if self.tac_hist:
                out["field_hist"] = np.stack(fields[:-1])
        return out

    def infer_model8(self, head, wrist, qpos, cond=None, noise_seed: int | None = None) -> np.ndarray:
        """One decision -> (horizon, 8) model-space chunk (arm joint deltas w.r.t. qpos, gripper absolute)."""
        images = {"camera_ego_rgb_0": resize224(head)}
        if "wrist" in self.cameras:
            images["right_wrist_camera_rgb_0"] = resize224(wrist)
        if noise_seed is not None:
            torch.manual_seed(int(noise_seed))
        chunk = self.wrapper.infer(images=images, state=build_state(qpos, self.state_dim), prompt=self.prompt, cond=cond)
        return chunk_to_model8(np.asarray(chunk, dtype=np.float32))

    def infer(self, head, wrist, qpos, qhist=None, tac_frames=None, noise_seed: int | None = None) -> np.ndarray:
        q = np.asarray(qpos, np.float32).reshape(8)
        cond = self.build_cond(qhist=q[None] if qhist is None else qhist, tac_frames=tac_frames)
        return self.infer_model8(head, wrist, q, cond=cond, noise_seed=noise_seed)


def init_policy(checkpoint_dir: str, task: str, **kwargs) -> TacInnoPolicy:
    """Load a deployable TacInno-VLA checkpoint (see :class:`TacInnoPolicy` for the arguments)."""
    return TacInnoPolicy(checkpoint_dir, task, **kwargs)
