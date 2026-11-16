"""Derive the live pipeline's data-dependent constants from the processed asl-signs splits.

Only the train and val splits are read; never the test split, and never any live clip.

- aspect.train_width_over_height (train split): the training videos' effective width / height.
  f = |y_mouth_mid - y_eye_mid| / |x_left_eye - x_right_eye| (pose 2, 5, 9, 10) equals the face's
  physical ratio times width / height in normalised image coordinates, and shoulder
  normalisation keeps the ratio, so width / height = median f / face_ratio_reference (the ratio
  of MediaPipe's canonical face mesh, in features_spec.json).
- live.segmenter (val split): motion energy (edgevision.live.pipeline.MotionTracker) over each
  val sequence, its 32 frames placed (t_kept - 1) / 31 source frames apart at the assumed
  dataset frame rate. start_energy: the 10th percentile of the per-sequence peak energy, so 90%
  of val signs trigger a segment. still_energy: the largest value (0.01 grid) at which at most
  5% of val sequences hold a still run of still_ms between their first and last frame above
  start_energy. preroll_ms: the median time from a sequence's first frame to its first frame
  above start_energy. tail_ms: the median time from its last frame above still_energy to its
  last frame.

    python scripts/calibrate_live.py --processed-dir data/processed_v2 [--write]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from edgevision import spec
from edgevision.data.landmarks import FIRST_POSE_SLOT, N_HAND, N_POSE
from edgevision.live.pipeline import MotionTracker

DATASET_FPS = 30.0  # asl-signs records no frame rate; its phone videos are taken as 30 fps.
START_QUANTILE = 10.0
MAX_FALSE_END_FRACTION = 0.05
STILL_GRID = np.round(np.arange(0.01, 3.0, 0.01), 2)


def load_split(processed_dir: Path, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    data = np.load(processed_dir / "features.npz")
    splits = json.loads(
        (processed_dir / "splits_signer_independent.json").read_text(encoding="utf-8")
    )
    if split not in ("train", "val"):
        raise ValueError("only the train and val splits may be read")
    wanted = np.isin(data["sequence_id"], np.asarray(splits[split]))
    return data["X"][wanted], data["presence"][wanted], data["sequence_id"][wanted]


def train_aspect(processed_dir: Path, pose_indices: list[int]) -> dict[str, Any]:
    coords, _, _ = load_split(processed_dir, "train")
    slot = {index: FIRST_POSE_SLOT + position for position, index in enumerate(pose_indices)}
    eye_l, eye_r, mouth_l, mouth_r = (
        coords[:, :, slot[i], :].astype(np.float64) for i in (2, 5, 9, 10)
    )
    present = np.ones(coords.shape[:2], dtype=bool)
    for block in (eye_l, eye_r, mouth_l, mouth_r):
        present &= np.abs(block).sum(axis=-1) > 0  # (0, 0) marks a missing point.
    height = np.abs(
        0.5 * (mouth_l[..., 1] + mouth_r[..., 1]) - 0.5 * (eye_l[..., 1] + eye_r[..., 1])
    )
    width = np.abs(eye_l[..., 0] - eye_r[..., 0])
    ratio = (height / np.where(width > 0, width, np.nan))[present]
    ratio = ratio[np.isfinite(ratio)]
    reference = float(spec.ASPECT["face_ratio_reference"])
    q25, median, q75 = np.percentile(ratio, [25, 50, 75])
    return {
        "frames": int(ratio.size),
        "f_median": float(median),
        "train_width_over_height": round(float(median / reference), 3),
        "train_width_over_height_iqr": [
            round(float(q25 / reference), 3),
            round(float(q75 / reference), 3),
        ],
    }


def sequence_energy(
    coords: np.ndarray, presence: np.ndarray, t_kept: int, pose_indices: list[int]
) -> tuple[np.ndarray, float]:
    """(32,) energy (NaN without a hand) and the frame spacing in ms of one val sequence."""
    frames = coords.shape[0]
    step_ms = max(t_kept - 1, 1) / (frames - 1) * 1000.0 / DATASET_FPS
    tracker = MotionTracker()
    energy = np.full(frames, np.nan)
    pose = np.full((N_POSE, 2), np.nan)
    for t in range(frames):
        frame = coords[t].astype(np.float64)
        frame_pose = pose.copy()
        for position, index in enumerate(pose_indices):
            point = frame[FIRST_POSE_SLOT + position]
            if point.any():
                frame_pose[index] = point
        left = frame[:N_HAND] if presence[t, 0] else None
        right = frame[N_HAND : 2 * N_HAND] if presence[t, 1] else None
        energy[t] = tracker.update(t * step_ms, left, right, frame_pose)
    return energy, step_ms


def longest_still_run(energy: np.ndarray, threshold: float, start: float, step_ms: float) -> float:
    """Longest run (ms) of frames below ``threshold`` between the first and last frame whose
    energy reaches ``start``."""
    above = np.flatnonzero(energy >= start)
    if above.size < 2:
        return 0.0
    inside = energy[above[0] : above[-1] + 1]
    longest = run = 0
    for value in inside:
        run = run + 1 if (np.isfinite(value) and value < threshold) else 0
        longest = max(longest, run)
    return longest * step_ms


def segmenter_constants(processed_dir: Path, pose_indices: list[int]) -> dict[str, Any]:
    coords, presence, sequence_ids = load_split(processed_dir, "val")
    manifest = pd.read_csv(processed_dir / "manifest.csv").set_index("sequence_id")
    t_kept = manifest.loc[sequence_ids, "t_kept"].to_numpy()
    energies, steps = [], []
    for index in range(coords.shape[0]):
        energy, step = sequence_energy(
            coords[index], presence[index], int(t_kept[index]), pose_indices
        )
        energies.append(energy)
        steps.append(step)
    peaks = np.array([np.nanmax(e) if np.isfinite(e).any() else 0.0 for e in energies])
    start = float(np.round(np.percentile(peaks, START_QUANTILE), 2))
    still_ms = float(spec.LIVE["segmenter"]["still_ms"])
    still = float(STILL_GRID[0])
    for candidate in STILL_GRID:
        if candidate >= start:
            break
        false_ends = np.mean(
            [
                longest_still_run(e, candidate, start, s) >= still_ms
                for e, s in zip(energies, steps, strict=True)
            ]
        )
        if false_ends > MAX_FALSE_END_FRACTION:
            break
        still = float(candidate)
    leads, tails = [], []
    for energy, step in zip(energies, steps, strict=True):
        above_start = np.flatnonzero(energy >= start)
        moving = np.flatnonzero(energy >= still)
        if above_start.size:
            leads.append(above_start[0] * step)
        if moving.size:
            tails.append((energy.size - 1 - moving[-1]) * step)
    return {
        "val_sequences": int(coords.shape[0]),
        "peak_energy_percentiles": {
            str(q): round(float(np.percentile(peaks, q)), 3) for q in (5, 10, 25, 50, 75)
        },
        "start_energy": start,
        "still_energy": still,
        "preroll_ms": float(np.round(np.median(leads))),
        "tail_ms": float(np.round(np.median(tails))),
    }


def format_json(value: Any, indent: int = 0) -> str:
    """JSON with two-space indents and lists of scalars on one line."""
    pad = "  " * indent
    if isinstance(value, dict):
        if not value:
            return "{}"
        items = [
            f"{pad}  {json.dumps(key)}: {format_json(item, indent + 1)}"
            for key, item in value.items()
        ]
        return "{\n" + ",\n".join(items) + f"\n{pad}}}"
    if isinstance(value, list):
        if all(not isinstance(item, (dict, list)) for item in value) or all(
            isinstance(item, list) and all(not isinstance(x, (dict, list)) for x in item)
            for item in value
        ):
            return json.dumps(value, separators=(", ", ": "))
        items = [f"{pad}  {format_json(item, indent + 1)}" for item in value]
        return "[\n" + ",\n".join(items) + f"\n{pad}]"
    return json.dumps(value)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("--processed-dir", type=Path, required=True)
    parser.add_argument(
        "--write", action="store_true", help="store the values in features_spec.json"
    )
    args = parser.parse_args(argv)
    report = json.loads((args.processed_dir / "preprocess_report.json").read_text(encoding="utf-8"))
    pose_indices = [int(i) for i in report["config"]["preprocess"]["pose_indices"]]
    if not {2, 5, 9, 10, 11, 12} <= set(pose_indices):
        print(
            "error: needs a processed dir with pose 0-16 (experiments/preprocess_v2.yaml)",
            file=sys.stderr,
        )
        return 1
    aspect = train_aspect(args.processed_dir, pose_indices)
    segmenter = segmenter_constants(args.processed_dir, pose_indices)
    print(json.dumps({"aspect": aspect, "segmenter": segmenter}, indent=2))
    if args.write:
        path = spec.spec_path()
        document = json.loads(path.read_text(encoding="utf-8"))
        document["aspect"]["train_width_over_height"] = aspect["train_width_over_height"]
        document["aspect"]["train_width_over_height_iqr"] = aspect["train_width_over_height_iqr"]
        block = document["live"]["segmenter"]
        for key in ("start_energy", "still_energy", "preroll_ms", "tail_ms"):
            block[key] = segmenter[key]
        block["derived"] = (
            f"scripts/calibrate_live.py on the val split ({segmenter['val_sequences']} sequences, "
            f"frames {DATASET_FPS:g} fps apart before resampling): "
            f"start_energy = P{START_QUANTILE:g} of the per-sequence peak energy; "
            f"still_energy = the largest value with at most {MAX_FALSE_END_FRACTION:.0%} of "
            "sequences still for still_ms inside the sign; preroll_ms "
            "and tail_ms = median lead-in before start_energy and median tail after the last frame "
            "above still_energy"
        )
        path.write_text(format_json(document) + "\n", encoding="utf-8")
        print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
