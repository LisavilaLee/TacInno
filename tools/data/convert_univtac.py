"""
UniVTAC HDF5 -> pi0.5 (FTP-1) zarr converter used for TacInno-VLA training data.

Usage (one task, one split; TacInno uses demonstrations 0-49 for training and 90-99 for validation):

    python tools/data/convert_univtac.py --base_dir data/univtac/raw/train --task_list lift_can \\
        --save_dir data/univtac/zarr_train/lift_can --image_size 224

TacInno-VLA reads the head and the wrist camera for every task (camera_type "all" in TASK_SETTINGS), except the
Put Bottle recipe, which is trained on a head-camera-only copy (``--camera_type head``).


- Reads UniVTAC HDF5 episodes containing joint state, RGB observations, and
  tactile image streams.
- Writes RGB tensors to zarr; raw JPEG decoding yields BGR, which is converted
  to RGB before export for consistency with eval/inference pipelines.
- Exports only the joint-only control representation:
  7 arm dimensions + 1 gripper dimension.
- Supports both tactile key conventions:
  `left/right_tactile` and `left/right_gsmini`.
- Supports `--episodes_per_task` to cap the number of trajectories per task.
- If an earlier version produced BGR zarr outputs, regenerate them with this
  script before training.
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional

import cv2
import h5py
import numpy as np
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[2]
for _p in (_ROOT, _ROOT / "third_party" / "openpi" / "src"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from openpi.replay_buffer import ReplayBuffer  # noqa: E402
import json

# Camera setting per task: "all" = head + wrist, "head" = head only. TacInno-VLA uses head + wrist for all six tasks
# (the Put Bottle recipe uses a head-only copy, --camera_type head).
TASK_SETTINGS = {t: {"camera_type": "all", "downsample": 1} for t in (
    "insert_hole", "insert_tube", "lift_bottle", "lift_can", "pull_out_key", "put_bottle_in_shelf", "place_hidden_support")}
TASK_SETTINGS.update({t: {"camera_type": "head", "downsample": 1} for t in (
    "insert_HDMI", "insert_card", "insert_lean", "grasp_classify")})

# Default FAAS gripper slot, aligned with other FTP1 parsers.
DEFAULT_GRIPPER_IDX = 28
# Tactile sensor name used by T3 pretraining mapping.
TACTILE_SENSOR_NAME = "GelSightMini"
INSTRUCTION_PAD_LEN = 100
IMAGE_SIZE = 224
VIDEO_FPS = 10


def _save_video(
    frames: np.ndarray,
    base_path: Path,
    episode_idx: int,
    fps: int = VIDEO_FPS,
) -> Optional[Path]:
    """Write a `camera_ego_rgb` frame sequence to mp4."""
    if frames is None or len(frames) == 0:
        return None
    out_file = f"{base_path.stem}_ep{episode_idx:04d}.mp4"
    T, H, W = frames.shape[0], frames.shape[1], frames.shape[2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_file), fourcc, fps, (W, H))
    for t in range(T):
        frame = frames[t]
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(frame)
    writer.release()
    return out_file


def _stream_to_img(data: np.ndarray, out_size: tuple[int, int] | None = (IMAGE_SIZE, IMAGE_SIZE)) -> np.ndarray:
    """Decode JPEG byte streams from HDF5 into `(N, H, W, 3)` uint8 RGB arrays."""
    flat = data.ravel()
    imgs = []
    for buf in flat:
        if isinstance(buf, (bytes, bytearray)):
            arr = np.frombuffer(buf, dtype=np.uint8)
        elif isinstance(buf, np.ndarray) and buf.dtype == np.uint8:
            arr = buf
        else:
            try:
                arr = np.frombuffer(bytes(buf), dtype=np.uint8)
            except Exception as e:
                raise TypeError(f"Unsupported buffer type: {type(buf)}") from e
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Failed to decode image from buffer")
        if out_size:
            img = cv2.resize(img, (out_size[1], out_size[0]))  # (width, height)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        imgs.append(img)
    return np.stack(imgs, axis=0).astype(np.uint8)


def _load_univtac_episode(
    hdf5_path: str | Path,
    downsample: int = 1,
    image_size: int = IMAGE_SIZE,
    use_wrist: bool = True,
) -> dict[str, np.ndarray] | None:
    """Load one UniVTAC HDF5 episode and return an FTP1-compatible episode dict."""
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.exists():
        return None

    with h5py.File(hdf5_path, "r") as f:
        if "embodiment" not in f or "joint" not in f["embodiment"]:
            return None
        joint = np.array(f["embodiment"]["joint"][()], dtype=np.float32)  # (T, 9)
        T_full = joint.shape[0]
        if T_full < 2:
            return None

        # downsampled indices [0, downsample, 2 * downsample, ...]; the last frame is kept for the action
        arange = np.arange(0, T_full - 1, downsample)
        if len(arange) == 0:
            return None
        # state = joint[t], action = joint[t + 1]: len(arange) frames are kept, the state aligned with the observation
        # as UniVTAC-ACT: 8-D = 7 arm joints + 1 gripper joint, joint[:8] (the 8th entry is one finger, no averaging)
        T = len(arange)
        joint_state = joint[arange]   # (T, 9) or (T, 8)
        joint_8 = joint_state[:, :8].astype(np.float32)   # (T, 8)：7 arm + 1 gripper

        def _read_rgb(*path_parts: str):
            """Read and decode an image stream of the HDF5 file; raises (never returns None) on a missing key or a decoding error."""
            path_str = "/".join(path_parts)
            try:
                node = f
                for p in path_parts:
                    node = node[p]
                raw = node[()]
                return _stream_to_img(raw, (image_size, image_size))
            except (KeyError, TypeError) as e:
                raise KeyError(f"HDF5 missing or invalid key: {path_str} in {hdf5_path}") from e

        def _read_rgb_with_fallbacks(path_options: list[tuple[str, ...]]) -> np.ndarray:
            last_error = None
            for option in path_options:
                try:
                    return _read_rgb(*option)
                except KeyError as e:
                    last_error = e
            tried = ", ".join("/".join(p) for p in path_options)
            raise KeyError(f"HDF5 missing or invalid key, tried: {tried} in {hdf5_path}") from last_error

        head_rgb = _read_rgb("observation", "head", "rgb")[...,::-1]
        # the wrist camera is read only when use_wrist is True
        wrist_rgb = None
        if use_wrist:
            wrist_rgb = _read_rgb("observation", "wrist", "rgb")[...,::-1]
        left_tac = _read_rgb_with_fallbacks(
            [
                ("tactile", "left_gsmini", "rgb_marker"),
                ("tactile", "left_tactile", "rgb_marker"),
            ]
        )[..., ::-1]
        right_tac = _read_rgb_with_fallbacks(
            [
                ("tactile", "right_gsmini", "rgb_marker"),
                ("tactile", "right_tactile", "rgb_marker"),
            ]
        )[..., ::-1]
        # from PIL import Image
        # Image.fromarray(head_rgb[0]).save("head_rgb.jpg")
        # Image.fromarray(wrist_rgb[0]).save("wrist_rgb.jpg")
        # Image.fromarray(left_tac[0]).save("left_tac.jpg")
        # Image.fromarray(right_tac[0]).save("right_tac.jpg")

        if head_rgb is None or head_rgb.shape[0] != T_full:
            return None
        head_rgb = head_rgb[arange]   # (T, H, W, 3), already image_size from _read_rgb
        if wrist_rgb is not None:
            wrist_rgb = wrist_rgb[arange]
        right_tac = right_tac[arange][:, np.newaxis, ...]   # (T, 1, H, W, 3)
        left_tac = left_tac[arange][:, np.newaxis, ...]   # (T, 1, H, W, 3)

        # the two tactile pads of the gripper are stacked as N = 2: (T, 2, H, W, 3), areas [0, 1]
        # left first, then right (left = thumb, right = index finger)
        right_tac = np.concatenate([left_tac, right_tac], axis=1)   # (T, 2, H, W, 3)

    # timestamps: the step index when the file has no timestamps
    timestamps = np.arange(T, dtype=np.int64)

    right_arm_joints = joint_8[:, :7].astype(np.float32)
    right_hand_joints = joint_8[:, 7:8].astype(np.float32)
    right_hand_joints_idx = np.full((T, 1), DEFAULT_GRIPPER_IDX, dtype=np.int32)

    # tactile: the two gripper pads (left_tactile / right_tactile in the HDF5 file), stacked as N = 2, areas [0, 1]
    right_tactile_area_gripper = np.tile(np.array([[0, 1]], dtype=np.int64), (T, 1))   # (T, 2)
    right_tactile_sensor_gripper = np.array([TACTILE_SENSOR_NAME] * T)
    right_tactile_type_gripper = np.array(["image"] * T)

    episode = {
        "timestamps": timestamps,
        "camera_ego_rgb": head_rgb,
        "right_arm_joints": right_arm_joints,
        "right_hand_joints": right_hand_joints,
        "right_hand_joints_idx": right_hand_joints_idx,
        "right_tactile_data_gripper": right_tac.astype(np.uint8),
        "right_tactile_area_gripper": right_tactile_area_gripper,
        "right_tactile_sensor_gripper": right_tactile_sensor_gripper,
        "right_tactile_type_gripper": right_tactile_type_gripper,
    }
    if wrist_rgb is not None:
        episode["right_wrist_camera_rgb"] = wrist_rgb

    return episode


# data layout: base_dir/<task>/demo/hdf5/<id>.hdf5
HDF5_SUBDIR = "demo/hdf5"


def _discover_task_dirs(base_dir: Path, task_list: list[str] | None) -> list[tuple[str, Path]]:
    """
    Find the task directories containing demo/hdf5/*.hdf5 (layout base_dir/<task>/demo/hdf5/<id>.hdf5, one directory
    per task). Returns [(task_id, task_dir), ...] with task_dir = base_dir/<task>.
    """
    base_dir = Path(base_dir)
    if not base_dir.is_dir():
        return []

    results: list[tuple[str, Path]] = []
    try:
        for first in sorted(base_dir.iterdir()):
            if not first.is_dir():
                continue
            clean_dir = first / HDF5_SUBDIR
            if not clean_dir.is_dir():
                continue
            if not list(clean_dir.glob("*.hdf5")):
                continue
            task_id = first.name
            if task_list is not None and task_id not in task_list:
                continue
            results.append((task_id, first))
    except OSError:
        pass
    return results


def _run(
    base_dir: str,
    save_dir: str,
    episodes_per_task: int | None = None,
    task_list: list[str] | None = None,
    downsample: int = 1,
    image_size: int = IMAGE_SIZE,
    camera_type: str = "auto",
) -> None:
    base_dir = Path(base_dir)
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # camera setting per task (camera_type "all" keeps the wrist camera)
    task_settings = TASK_SETTINGS
    task_dirs = _discover_task_dirs(base_dir, task_list)
    if not task_dirs:
        print(f"No task directories with {HDF5_SUBDIR}/*.hdf5 found under {base_dir}")
        print(f"  Expected layout: base_dir/<task>/{HDF5_SUBDIR}/<id>.hdf5")
        return
    
    instruction_map = {
        "grasp_classify": "use grasped tool for tactile sensing to move to target surface.",
        "insert_HDMI": "insert the HDMI to the fixed slot.",
        "insert_hole": "insert the stick to the hole.",
        "insert_tube": "insert the tube to the fixed slot.",
        "lift_can": "grasp the can and lifts it vertically without slippage.",
        "lift_bottle": "grasp the bottle and lift it vertically, keeping its final base within 5 cm of the wall.",
        "pull_out_key": "pull out the key.",
        "put_bottle_in_shelf": "grasp the bottle, then position it into the shelf cavity.",
        "place_hidden_support": "lower the can into the receiver and release it once it rests on the support.",
    }

    for task_id, task_dir in task_dirs:
        hdf5_dir = task_dir / HDF5_SUBDIR
        def _episode_sort_key(p: Path) -> int:
            try:
                return int(p.stem)
            except ValueError:
                return 0
        hdf5_files = sorted(hdf5_dir.glob("*.hdf5"), key=_episode_sort_key)
        if not hdf5_files:
            continue

        count = 0
        cam_cfg = task_settings.get(task_id, {})
        camera = camera_type if camera_type != "auto" else cam_cfg.get("camera_type", "head")
        use_wrist = camera == "all"

        # one zarr file per task ('/' in the task id replaced by '_')
        safe_id = task_id.replace("/", "_")
        zarr_path = save_dir / f"{safe_id}"
        if use_wrist:
            zarr_path = str(zarr_path) + "_all.zarr"
        else:
            zarr_path = str(zarr_path) + "_head.zarr"
        if os.path.exists(zarr_path):
            shutil.rmtree(zarr_path)
        replay_buffer = ReplayBuffer.create_from_path(str(zarr_path), mode="a")

        for h5_path in tqdm(hdf5_files, desc=f"UniVTAC {task_id}"):
            if episodes_per_task is not None and count >= episodes_per_task:
                break
            episode = _load_univtac_episode(
                h5_path,
                downsample=downsample,
                image_size=image_size,
                use_wrist=use_wrist,
            )
            if episode is None:
                continue
            instruction_org = instruction_map.get(task_id, f"solve the task: {task_id}.")
            instruction = instruction_org.ljust(INSTRUCTION_PAD_LEN)
            episode["sub_task_instruction"] = np.array([instruction] * len(episode["timestamps"]))
            
            # print(f"====================================== Task: {task_id} ======================================")
            # _save_video(episode["camera_ego_rgb"], zarr_path, count, fps=VIDEO_FPS)
            # for key in episode.keys():
            #     print(f"{key}: {episode[key].shape}")
            # print(f"instruction: {instruction}")
            # timestamps: (56,)
            # camera_ego_rgb: (56, 224, 224, 3)
            # right_arm_joints: (56, 7)
            # right_hand_joints: (56, 1)
            # right_hand_joints_idx: (56, 1)
            # right_tactile_data_gripper: (56, 2, 224, 224, 3)
            # right_tactile_area_gripper: (56, 2)
            # right_tactile_sensor_gripper: (56,)
            # right_tactile_type_gripper: (56,)
            # right_wrist_camera_rgb: (56, 224, 224, 3)
            # sub_task_instruction: (56,)
            replay_buffer.add_episode(episode, compressors="disk")
            count += 1
            # break

        print(f"  {task_id}: wrote {count} episodes -> {zarr_path}")


def main():
    parser = argparse.ArgumentParser(description="UniVTAC HDF5 -> FTP1 Zarr (joint control only).")
    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="UniVTAC data root containing per-task HDF5 directories.",
    )
    parser.add_argument("--save_dir", type=str, required=True, help="Output directory for exported zarr files.")
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=None,
        help="Optional cap on the number of trajectories retained per task.",
    )
    parser.add_argument(
        "--task_list",
        type=str,
        default=None,
        help="Optional comma-separated subset of task names to process.",
    )
    parser.add_argument("--downsample", type=int, default=1, help="Temporal downsample factor.")
    parser.add_argument("--image_size", type=int, default=IMAGE_SIZE, help="Output size for RGB and tactile images.")
    parser.add_argument("--camera_type", choices=["auto", "all", "head"], default="auto",
                        help="auto: TASK_SETTINGS (head + wrist for the six TacInno tasks); head: head camera only")
    args = parser.parse_args()

    task_list = None
    if args.task_list:
        task_list = [s.strip() for s in args.task_list.split(",") if s.strip()]

    _run(
        base_dir=args.base_dir,
        save_dir=args.save_dir,
        episodes_per_task=args.episodes_per_task,
        task_list=task_list,
        downsample=args.downsample,
        image_size=args.image_size,
        camera_type=args.camera_type,
    )


if __name__ == "__main__":
    main()
