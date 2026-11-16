"""Raw synthetic signs whose classes differ in motion, and web-demo recordings made from them, for
the live-diagnosis tests.

Training geometry mimics the phone videos: shoulders 0.3 apart at y 0.55, the nose 0.12 above
their midpoint, so r = 0.4. The dominant (right) hand follows a class-specific path; class 1 is
class 0's vertical oscillation 2.5 times larger, so stretching y by 2.5 (a landscape webcam seen
with k = 0.4) turns one class into another.
"""

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from conftest import long_frame
from edgevision.data.landmarks import N_HAND, N_POSE
from edgevision.live.aspect import RawFrames, correct_frames
from edgevision.live.recordings import (
    RECORDINGS_FORMAT,
    RECORDINGS_SCHEMA_VERSION,
    RecordingsFile,
    replay,
)

FloatArray = npt.NDArray[np.float64]

SIGNS = ("small_nod", "large_nod", "shake", "high_hold", "low_hold")
PARTICIPANTS = (101, 202, 303, 404, 505, 606, 707, 808)
R_GEOMETRY = 0.4
SHOULDER_Y = 0.55
SHOULDER_HALF_WIDTH = 0.15
NOSE_HEIGHT = R_GEOMETRY * 2 * SHOULDER_HALF_WIDTH
HAND_SHAPE = np.random.default_rng(7).uniform(-0.03, 0.03, (N_HAND, 2))


def hand_path(label: int, t: FloatArray) -> FloatArray:
    """(T, 2) wrist path of the right hand (image coordinates)."""
    base = np.array([0.42, 0.45])
    wave = np.sin(2 * np.pi * 2 * t)
    offsets = {
        0: np.stack([np.zeros_like(t), 0.03 * wave], axis=1),
        1: np.stack([np.zeros_like(t), 0.075 * wave], axis=1),
        2: np.stack([0.06 * wave, np.zeros_like(t)], axis=1),
        3: np.stack([np.zeros_like(t), np.full_like(t, -0.15)], axis=1),
        4: np.stack([np.zeros_like(t), np.full_like(t, 0.03)], axis=1),
    }
    return base + offsets[label]


def motion_blocks(label: int, n_frames: int, rng: np.random.Generator) -> dict[str, FloatArray]:
    """Per-type (T, n, 3) blocks: pose, right hand on the class path, no left hand."""
    t = np.linspace(0.0, 1.0, n_frames)
    mid_y = SHOULDER_Y + rng.uniform(-0.02, 0.02)
    mid_x = 0.5 + rng.uniform(-0.02, 0.02)
    nose_height = NOSE_HEIGHT * (1.0 + rng.uniform(-0.03, 0.03))
    pose = np.empty((n_frames, N_POSE, 3))
    pose[:] = rng.uniform(-0.04, 0.04, (N_POSE, 3)) + np.array([mid_x, mid_y + 0.1, 0.0])
    pose[:, 0:11, :2] = rng.uniform(-0.03, 0.03, (11, 2)) + np.array([mid_x, mid_y - nose_height])
    pose[:, 0, :2] = (mid_x, mid_y - nose_height)
    pose[:, 11, :2] = (mid_x + SHOULDER_HALF_WIDTH, mid_y)
    pose[:, 12, :2] = (mid_x - SHOULDER_HALF_WIDTH, mid_y)
    pose[..., :2] += rng.normal(0.0, 0.002, (n_frames, N_POSE, 2))
    wrist = hand_path(label, t) + np.array([mid_x - 0.5, mid_y - SHOULDER_Y])
    right = np.zeros((n_frames, N_HAND, 3))
    right[..., :2] = (
        wrist[:, None, :] + HAND_SHAPE[None] + rng.normal(0.0, 0.002, (n_frames, N_HAND, 2))
    )
    pose[:, 16, :2] = wrist  # The right wrist follows the hand.
    return {
        "left_hand": np.full((n_frames, N_HAND, 3), np.nan),
        "pose": pose,
        "right_hand": right,
    }


def build_motion_dataset(root: Path, *, per_cell: int = 4, seed: int = 0) -> Path:
    """train.csv, the sign map and one parquet per sequence, in the asl-signs layout."""
    rng = np.random.default_rng(seed)
    rows = []
    ordinal = 0
    for participant in PARTICIPANTS:
        for label, sign in enumerate(SIGNS):
            for _ in range(per_cell):
                sequence_id = 2_000_000_000 + ordinal
                relative = f"train_landmark_files/{participant}/{sequence_id}.parquet"
                target = root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                blocks = motion_blocks(label, int(rng.integers(20, 41)), rng)
                long_frame(blocks).to_parquet(target, engine="pyarrow", index=False)
                rows.append(f"{relative},{participant},{sequence_id},{sign}")
                ordinal += 1
    (root / "train.csv").write_text(
        "path,participant_id,sequence_id,sign\n" + "\n".join(rows) + "\n", encoding="utf-8"
    )
    sign_map = {sign: index for index, sign in enumerate(SIGNS)}
    (root / "sign_to_prediction_index_map.json").write_text(json.dumps(sign_map), encoding="utf-8")
    return root


def distort(frames: RawFrames, k_true: float) -> RawFrames:
    """What a camera that needs correction k_true shows: y scaled by 1 / k_true about each
    frame's shoulder midpoint, so correcting with k_true restores the original."""
    return correct_frames(frames, 1.0 / k_true)


def raw_from_blocks(blocks: dict[str, FloatArray]) -> RawFrames:
    return RawFrames(blocks["left_hand"], blocks["right_hand"], blocks["pose"])


def _flat(block: FloatArray) -> list[list[float | None] | None]:
    out: list[list[float | None] | None] = []
    for frame in block:
        if np.isnan(frame).all():
            out.append(None)
        else:
            out.append([None if np.isnan(v) else float(v) for v in frame.ravel()])
    return out


def synthetic_recordings(
    meta: dict[str, Any],
    classify: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
    *,
    k_true: float,
    per_class: int = 4,
    seed: int = 99,
) -> RecordingsFile:
    """Recordings of fresh synthetic signs seen through a camera needing k_true, captured by an
    app with aspect correction off (k = 1); the app results are what the app would compute."""
    rng = np.random.default_rng(seed)
    labels = meta["labels"]
    pre = meta["preprocessing"]
    threshold = float(meta["threshold"]["threshold"])
    recordings = []
    for index in range(per_class * len(SIGNS)):
        label = index % len(SIGNS)
        n_frames = int(rng.integers(20, 41))
        frames = distort(raw_from_blocks(motion_blocks(label, n_frames, rng)), k_true)
        result = replay(
            frames,
            1.0,
            pose_indices=pre["pose_indices"],
            num_frames=pre["num_frames"],
            min_hand_frames=pre["min_hand_frames"],
        )
        assert result.features is not None and result.hand_presence is not None
        flat = result.features.reshape(result.features.shape[0], -1)
        features = np.concatenate([flat, result.hand_presence], axis=1).astype(np.float32)
        model_input = features.copy()
        model_input[:, : flat.shape[1]] = flat.astype(np.float16).astype(np.float32)
        logits = classify(model_input[None])[0].astype(np.float64)
        probabilities = np.exp(logits - logits.max())
        probabilities /= probabilities.sum()
        top = np.argsort(-probabilities, kind="stable")[:5]
        t_ms = [1000.0 + 33.3 * t for t in range(n_frames)]
        recordings.append(
            {
                "id": index,
                "session": 0,
                "intended_label": labels[label],
                "intended_index": label,
                "recorded_at": "2026-11-07T10:00:00.000Z",
                "camera": {"width": 640, "height": 480, "frame_rate": 30.0},
                "framing": "full",
                "crop": {"sx": 0, "sy": 0, "sw": 640, "sh": 480},
                "swap_hands": False,
                "pose_every": 1,
                "min_hand_frames": pre["min_hand_frames"],
                "calibration": {
                    "mode": "off",
                    "k": 1.0,
                    "r_live": None,
                    "r_train": meta["aspect"]["r_train"],
                    "frames": 0,
                    "status": "off",
                },
                "segment": {
                    "capture_mode": "auto",
                    "reason": "hands_lowered",
                    "start_ms": t_ms[0],
                    "end_ms": t_ms[-1],
                    "frames": n_frames,
                    "hand_frames": n_frames,
                },
                "frames": {
                    "t_ms": t_ms,
                    "left_hand": _flat(frames.left_hand),
                    "right_hand": _flat(frames.right_hand),
                    "pose": _flat(frames.pose),
                    "pose_visibility": [[0.99] * N_POSE for _ in range(n_frames)],
                    "pose_ran": [True] * n_frames,
                },
                "app_result": {
                    "status": "classified",
                    "reason": None,
                    "info": dict(result.info),
                    "features": features.tolist(),
                    "model_input": model_input.tolist(),
                    "top5": [
                        {
                            "index": int(i),
                            "label": labels[int(i)],
                            "probability": float(probabilities[i]),
                        }
                        for i in top
                    ],
                    "prediction": labels[int(top[0])],
                    "max_probability": float(probabilities[top[0]]),
                    "accepted": bool(probabilities[top[0]] >= threshold),
                    "threshold": threshold,
                },
            }
        )
    return RecordingsFile.model_validate(
        {
            "format": RECORDINGS_FORMAT,
            "schema_version": RECORDINGS_SCHEMA_VERSION,
            "created_at": "2026-11-07T10:05:00.000Z",
            "user_agent": "synthetic recordings (tests/live_support.py)",
            "app": {
                "model_name": meta["model_name"],
                "variant": meta["variant"],
                "run": meta["run"],
                "model_sha256": meta["model_sha256"],
                "feature_set": meta["feature_set"],
                "pose_indices": pre["pose_indices"],
                "pose_swap_pairs": pre["pose_swap_pairs"],
                "num_frames": pre["num_frames"],
                "min_hand_frames": pre["min_hand_frames"],
                "threshold": threshold,
                "num_classes": meta["num_classes"],
                "r_train": meta["aspect"]["r_train"],
            },
            "recordings": recordings,
        }
    )
