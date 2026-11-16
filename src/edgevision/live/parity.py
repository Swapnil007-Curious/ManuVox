"""Golden fixtures for the Python / browser parity test (web/tests/unit/liveParity.test.ts).

Twenty val sequences of the processed dataset are turned back into raw landmarks and pushed
through the Python reference pipeline; the browser must reproduce every model input within
TOLERANCE and the same top-1. The raw asl-signs parquet files are not needed: each processed
sequence (shoulder-normalised coordinates of 32 frames) is placed back into image coordinates.

- dataset cases: long rows in the Holistic convention (NaN rows for missing hands, every other
  case mirrored so the dominance flip runs) -> preprocess_sequence -> storage step -> logits;
- live cases: the same sequences as MediaPipe Tasks would report them for an unmirrored
  640 x 480 camera (the y spread undone by 1 / k, detected hands in shuffled order with wrong or
  duplicate handedness labels on a quarter of the frames, 33 pose points with visibility,
  irregular frame times including a gap above max_gap_ms) -> hand assignment -> aspect
  correction -> preprocessing on the kept-frame clock -> storage step -> logits;
- a stream: several live cases joined by still and hands-down stretches -> motion energy per
  frame -> the motion segmenter -> each segment's model input and logits.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from edgevision.data.landmarks import FIRST_POSE_SLOT, N_HAND, N_POSE, mirror_permutation
from edgevision.data.preprocess import preprocess_sequence
from edgevision.live.aspect import frame_k
from edgevision.live.pipeline import (
    DetectedHand,
    LivePipeline,
    PipelineConfig,
    assign_hands,
    live_frame,
    preprocess_segment,
    to_model_input,
)

FloatArray = npt.NDArray[np.float64]
TOLERANCE = 1e-4
CASES = 20
STREAM_CASES = 4
TWO_HANDED_CASES = 2
WIDTH, HEIGHT = 640, 480
CENTER = (0.5, 0.55)
SHOULDER_WIDTH = 0.24  # Image units of the training convention.
FRAME_STEPS_MS = (33.0, 33.0, 34.0, 50.0, 66.0, 41.0)
LONG_GAP_MS = 300.0  # Above max_gap_ms, so the clock cap runs.
DECIMALS = 6
INPUT_DECIMALS = 5  # Model inputs; the parity tolerance is 1e-4.


def select_cases(
    coords: FloatArray,
    presence: npt.NDArray[np.uint8],
    y: npt.NDArray[np.int64],
    seed: int,
    n: int = CASES,
) -> list[int]:
    """n val rows with distinct labels, every selected pose point present in every frame and a
    hand in at least half the frames. The first TWO_HANDED_CASES have both hands in at least
    half the frames, so two-hand assignment and both hand blocks are exercised: random rows are
    almost always one-handed (Holistic rarely tracks the second hand; 4 of the 12573 val
    sequences qualify). Ties of the dominant-hand rule are covered by the synthetic preprocessing
    goldens (web/tests/golden/preprocess_cases*.json)."""
    rng = np.random.default_rng(seed)
    pose_present = (np.abs(coords[:, :, FIRST_POSE_SLOT:, :]).sum(axis=-1) > 0).all(axis=(1, 2))
    hands = presence.any(axis=2).sum(axis=1) >= coords.shape[1] // 2
    both = presence.all(axis=2).sum(axis=1) >= coords.shape[1] // 2
    chosen: list[int] = []
    seen: set[int] = set()
    for pool, limit in (
        (np.flatnonzero(pose_present & both), TWO_HANDED_CASES),
        (np.flatnonzero(pose_present & hands), n),
    ):
        for row in rng.permutation(pool):
            if len(chosen) == limit:
                break
            if int(y[row]) in seen:
                continue
            chosen.append(int(row))
            seen.add(int(y[row]))
    return sorted(chosen)


def image_blocks(
    coords: FloatArray, presence: npt.NDArray[np.uint8], pose_indices: Sequence[int]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """(T, 21, 2), (T, 21, 2), (T, 33, 2) image coordinates of the training convention from
    shoulder-normalised (T, L, 2) coordinates; NaN for absent hands and unselected pose points."""
    frames = coords.shape[0]
    image = np.asarray(CENTER) + SHOULDER_WIDTH * coords
    left = np.where(presence[:, 0, None, None].astype(bool), image[:, :N_HAND], np.nan)
    right = np.where(presence[:, 1, None, None].astype(bool), image[:, N_HAND : 2 * N_HAND], np.nan)
    pose = np.full((frames, N_POSE, 2), np.nan)
    for position, index in enumerate(pose_indices):
        pose[:, index] = image[:, FIRST_POSE_SLOT + position]
    return left, right, pose


def mirrored(
    left: FloatArray, right: FloatArray, pose: FloatArray, pose_indices: Sequence[int]
) -> tuple[FloatArray, FloatArray, FloatArray]:
    """The same frames seen in a mirror: x -> 1 - x, hands and pose pairs swapped."""
    flipped_pose = pose.copy()
    selection = list(pose_indices)
    permutation = mirror_permutation(selection)[FIRST_POSE_SLOT:] - FIRST_POSE_SLOT
    flipped_pose[:, selection] = pose[:, [selection[p] for p in permutation]]
    out = []
    for block in (right, left, flipped_pose):
        block = block.copy()
        block[..., 0] = 1.0 - block[..., 0]
        out.append(block)
    return out[0], out[1], out[2]


def long_rows(
    left: FloatArray, right: FloatArray, pose: FloatArray, pose_indices: Sequence[int]
) -> pd.DataFrame:
    """Holistic-style long rows: 21 rows per hand per frame (NaN when absent), the selected pose
    points, z = 0."""
    parts = []
    frames = left.shape[0]
    for kind, block, points in (
        ("left_hand", left, list(range(N_HAND))),
        ("right_hand", right, list(range(N_HAND))),
        ("pose", pose, list(pose_indices)),
    ):
        values = block[:, points]
        parts.append(
            pd.DataFrame(
                {
                    "frame": np.repeat(np.arange(frames, dtype=np.int64), len(points)),
                    "type": kind,
                    "landmark_index": np.tile(np.asarray(points, dtype=np.int64), frames),
                    "x": values[..., 0].ravel(),
                    "y": values[..., 1].ravel(),
                    "z": np.where(np.isnan(values[..., 0]), np.nan, 0.0).ravel(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def full_pose(pose: FloatArray) -> FloatArray:
    """Fill the unselected pose points (17-32) the way a landmarker always reports them: hips,
    knees and feet below the shoulders, fingers at the wrists."""
    out = pose.copy()
    shoulders = 0.5 * (pose[:, 11] + pose[:, 12])
    width = np.abs(pose[:, 11, 0] - pose[:, 12, 0])[:, None]
    for index, (dx, dy) in {
        23: (0.3, 1.2),
        24: (-0.3, 1.2),
        25: (0.3, 2.2),
        26: (-0.3, 2.2),
        27: (0.3, 3.1),
        28: (-0.3, 3.1),
        29: (0.35, 3.2),
        30: (-0.35, 3.2),
        31: (0.3, 3.3),
        32: (-0.3, 3.3),
    }.items():
        out[:, index] = shoulders + width * np.array([dx, dy])
    for base, fingers in ((15, (17, 19, 21)), (16, (18, 20, 22))):
        for finger in fingers:
            out[:, finger] = pose[:, base]
    return out


def to_live(block: FloatArray, y_mid: FloatArray, k: float) -> FloatArray:
    """Undo the aspect correction: the y spread a 640 x 480 camera shows."""
    out = block.copy()
    out[..., 1] = y_mid[:, None] + (block[..., 1] - y_mid[:, None]) / k
    return out


def frame_times(rng: np.random.Generator, frames: int, start: float = 0.0) -> FloatArray:
    steps = rng.choice(FRAME_STEPS_MS, size=frames)
    steps[0] = 0.0
    if frames > 10:
        steps[frames // 2] = LONG_GAP_MS
    return start + np.cumsum(steps)


def rounded(values: npt.ArrayLike) -> Any:
    array = np.round(np.asarray(values, dtype=np.float64), DECIMALS)
    return [
        [None if np.isnan(v) else float(v) for v in row]
        for row in array.reshape(array.shape[0], -1)
    ]


def block_json(block: FloatArray) -> list[list[float] | None]:
    """Per frame, the flat [x0, y0, x1, y1, ...] of a block, or null when it is missing."""
    out: list[list[float] | None] = []
    for frame in block:
        if np.isnan(frame[:, 0]).all():
            out.append(None)
        else:
            out.append([float(v) for v in np.round(frame[:, :2], DECIMALS).ravel()])
    return out


def detected_hands(
    rng: np.random.Generator, left: FloatArray, right: FloatArray
) -> list[dict[str, Any]]:
    """Tasks-style results for one frame: present hands in shuffled order, the true label, or on
    a quarter of the frames a wrong or duplicated label."""
    hands = []
    for block, label in ((left, "Left"), (right, "Right")):
        if np.isfinite(block[:, :2]).all():
            hands.append({"label": label, "landmarks": block})
    if rng.random() < 0.25 and hands:
        if len(hands) == 2 and rng.random() < 0.5:
            hands[1]["label"] = hands[0]["label"]
        else:
            hands[0]["label"] = "Right" if hands[0]["label"] == "Left" else "Left"
    if len(hands) == 2 and rng.random() < 0.5:
        hands.reverse()
    return hands


def live_frames(
    rng: np.random.Generator,
    left: FloatArray,
    right: FloatArray,
    pose: FloatArray,
    times: FloatArray,
    k: float,
) -> list[dict[str, Any]]:
    pose33 = full_pose(pose)
    y_mid = 0.5 * (pose33[:, 11, 1] + pose33[:, 12, 1])
    left_l, right_l, pose_l = (to_live(block, y_mid, k) for block in (left, right, pose33))
    frames = []
    for t in range(times.size):
        frames.append(
            {
                "t": float(times[t]),
                "hands": [
                    {"label": hand["label"], "landmarks": rounded(hand["landmarks"])}
                    for hand in detected_hands(rng, left_l[t], right_l[t])
                ],
                "pose": rounded(pose_l[t]),
                "visibility": [0.95] * N_POSE,
            }
        )
    return frames


def assigned_frame(frame: dict[str, Any]) -> Any:
    """The Python reference's LiveFrame for one golden frame (hands assigned)."""
    pose = np.c_[np.array(frame["pose"], dtype=np.float64).reshape(N_POSE, 2), np.zeros(N_POSE)]
    hands = [
        DetectedHand(
            np.c_[
                np.array(hand["landmarks"], dtype=np.float64).reshape(N_HAND, 2), np.zeros(N_HAND)
            ],
            hand["label"],
        )
        for hand in frame["hands"]
    ]
    left, right = assign_hands(hands, pose, "nearest_pose_wrist")
    return live_frame(frame["t"], left, right, pose)


def build_live_parity(
    processed_dir: Path,
    splits_path: Path,
    meta: dict[str, Any],
    classify: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
    seed: int,
) -> dict[str, Any]:
    from edgevision.train.data import load_split_arrays

    val = load_split_arrays(processed_dir, splits_path, ["val"])["val"]
    pre = meta["preprocessing"]
    pose_indices = [int(i) for i in pre["pose_indices"]]
    num_frames, min_hand_frames = int(pre["num_frames"]), int(pre["min_hand_frames"])
    width = 2 * (FIRST_POSE_SLOT + len(pose_indices))
    coords_all = (
        val.X[..., :width].reshape(val.X.shape[0], val.X.shape[1], -1, 2).astype(np.float64)
    )
    presence_all = val.X[..., width:].astype(np.uint8)
    rows = select_cases(coords_all, presence_all, val.y, seed)
    rng = np.random.default_rng(seed)
    k = frame_k(WIDTH, HEIGHT)

    dataset_cases, live_cases = [], []
    for number, row in enumerate(rows):
        left, right, pose = image_blocks(coords_all[row], presence_all[row], pose_indices)
        if number % 2 == 1:
            left, right, pose = mirrored(left, right, pose, pose_indices)
        # The fixture stores these values, so Python must consume exactly the rounded ones.
        left, right, pose = (np.round(block, DECIMALS) for block in (left, right, pose))
        frame_rows = long_rows(left, right, pose, pose_indices)
        result = preprocess_sequence(
            frame_rows,
            num_frames=num_frames,
            use_z=False,
            pose_indices=pose_indices,
            min_hand_frames=min_hand_frames,
        )
        model_input = to_model_input(result)
        assert model_input is not None, f"val row {row} was excluded"
        dataset_cases.append(
            {
                "sequence_id": int(val.sequence_id[row]),
                "label": int(val.y[row]),
                "mirrored_input": number % 2 == 1,
                # Long rows are these blocks, kind by kind, frame by frame, point by point
                # (long_rows); z is 0 where x is finite.
                "blocks": {
                    "left_hand": block_json(left),
                    "right_hand": block_json(right),
                    "pose": block_json(pose[:, pose_indices]),
                },
                "info": dict(result.info),
                "model_input": np.round(model_input, INPUT_DECIMALS).tolist(),
                "logits": np.round(classify(model_input), DECIMALS).tolist(),
            }
        )
        times = frame_times(rng, coords_all.shape[1])
        frames = live_frames(rng, left, right, pose, times, k)
        assigned = [assigned_frame(frame) for frame in frames]
        live_result, live_input = preprocess_segment(
            assigned,
            k,
            pose_indices=pose_indices,
            num_frames=num_frames,
            min_hand_frames=min_hand_frames,
            resampling="time",
        )
        assert live_input is not None, f"val row {row} was excluded live"
        live_cases.append(
            {
                "sequence_id": int(val.sequence_id[row]),
                "label": int(val.y[row]),
                "frames": frames,
                "info": dict(live_result.info),
                "model_input": np.round(live_input, INPUT_DECIMALS).tolist(),
                "logits": np.round(classify(live_input), DECIMALS).tolist(),
            }
        )

    stream_frames: list[dict[str, Any]] = []
    clock = 0.0
    for case in live_cases[:STREAM_CASES]:
        first = case["frames"][0]
        # 600 ms holding the first pose still, then the sign, then 600 ms with the hands down.
        for _ in range(18):
            clock += 33.0
            stream_frames.append({**first, "t": clock})
        offset = clock - case["frames"][0]["t"] + 33.0
        for frame in case["frames"]:
            stream_frames.append({**frame, "t": frame["t"] + offset})
        clock = stream_frames[-1]["t"]
        for _ in range(18):
            clock += 33.0
            stream_frames.append(
                {"t": clock, "hands": [], "pose": first["pose"], "visibility": first["visibility"]}
            )
    pipeline = LivePipeline(
        PipelineConfig(
            pose_indices=tuple(pose_indices),
            num_frames=num_frames,
            min_hand_frames=min_hand_frames,
            width=WIDTH,
            height=HEIGHT,
        ),
        classify,
    )
    assigned_stream = [assigned_frame(frame) for frame in stream_frames]
    by_identity = {id(item): i for i, item in enumerate(assigned_stream)}
    energies: list[float | None] = []
    segments = []
    for frame in assigned_stream:
        outcome = pipeline.push(frame)
        energies.append(None if np.isnan(pipeline.last_energy) else float(pipeline.last_energy))
        if outcome is not None:
            segments.append(outcome)
    final = pipeline.flush()
    if final is not None:
        segments.append(final)
    stream_segments = []
    for outcome in segments:
        event = outcome.event
        indices = [by_identity[id(item.frame)] for item in event.frames]
        stream_segments.append(
            {
                "kind": event.kind,
                "reason": event.reason,
                "first_frame": indices[0] if indices else None,
                "last_frame": indices[-1] if indices else None,
                "frames": len(indices),
                "hand_frames": event.hand_frames,
                "model_input": None
                if outcome.model_input is None
                else np.round(outcome.model_input, INPUT_DECIMALS).tolist(),
                "logits": None
                if outcome.logits is None
                else np.round(outcome.logits, DECIMALS).tolist(),
            }
        )
    return {
        "description": __doc__,
        "tolerance": TOLERANCE,
        "seed": seed,
        "model_sha256": meta["model_sha256"],
        "camera": {"width": WIDTH, "height": HEIGHT, "k": k},
        "preprocessing": {
            "pose_indices": pose_indices,
            "pose_swap_pairs": pre["pose_swap_pairs"],
            "num_frames": num_frames,
            "min_hand_frames": min_hand_frames,
        },
        "dataset_cases": dataset_cases,
        "live_cases": live_cases,
        "stream": {"frames": stream_frames, "energies": energies, "segments": stream_segments},
    }


def write_live_parity(path: Path, golden: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(golden, separators=(",", ":")) + "\n", encoding="utf-8")
