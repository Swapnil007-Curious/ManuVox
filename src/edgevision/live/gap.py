"""Live-gap evaluation: the configs of experiments/live_gap.json, the Python (Holistic) pipeline
over recorded clips, and the scoring both pipelines share.

A run of one config over the clips is a JSON file of segment records:
    {"pipeline", "config", "clips": [{"clip", "label", "label_index", "duration_ms",
     "segments": [{"kind", "reason", "start_ms", "end_ms", "frames", "hand_frames",
                   "excluded", "probabilities": [...] | null}], ...}], ...}
web/scripts/live-gap.mjs writes the same structure for the browser pipeline.
"""

import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from edgevision.live.aspect import calibrate, calibration_frames, frame_ratios
from edgevision.live.pipeline import (
    LivePipeline,
    PipelineConfig,
    SegmentOutcome,
    live_frame,
    softmax,
    summary,
)
from edgevision.runtime import OrtModel

FloatArray = npt.NDArray[np.float64]
CALIBRATION_MS = 2000.0
TOP_K = 3


def load_configs(path: Path) -> dict[str, Any]:
    document: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    names = [config["name"] for config in document["configs"]]
    if len(set(names)) != len(names):
        raise ValueError(f"{path}: config names are not unique")
    return document


def load_meta(model_dir: Path) -> dict[str, Any]:
    meta: dict[str, Any] = json.loads((model_dir / "meta.json").read_text(encoding="utf-8"))
    return meta


def decision_params(meta: dict[str, Any], decision: str) -> tuple[float, float]:
    """(temperature, threshold) of a config's decision rule."""
    if decision == "calibrated":
        return float(meta["calibration"]["temperature"]), float(meta["calibration"]["threshold"])
    return 1.0, float(meta["threshold"]["threshold"])


def body_ratio_k(pose: FloatArray, times: FloatArray, r_train: float) -> float:
    """The first demo's calibration: k = r_train / median r over the visible frames of the
    first CALIBRATION_MS."""
    window = times < CALIBRATION_MS
    usable = calibration_frames(pose[window, :, :3], pose[window, :, 3])
    return calibrate(frame_ratios(pose[window][usable, :, :3]), r_train).k


def run_clip(
    holistic: dict[str, Any],
    config: dict[str, Any],
    meta: dict[str, Any],
    classify: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
) -> list[dict[str, Any]]:
    """Segment records of one clip's Holistic landmarks under one config."""
    times = np.asarray(holistic["t_ms"], dtype=np.float64)
    pose = np.asarray(holistic["pose"], dtype=np.float64)
    left = np.asarray(holistic["left_hand"], dtype=np.float64)
    right = np.asarray(holistic["right_hand"], dtype=np.float64)
    if config["hands"] == "label_inverted":
        # Holistic's blocks are right; the first demo's mapping swapped every hand.
        left, right = right, left
    elif config["hands"] != "nearest_pose_wrist":
        raise ValueError(f"unknown hands setting {config['hands']!r}")
    if config["mirrored"]:
        raise ValueError("mirrored detection needs the browser pipeline")
    calibrated_k = 1.0
    if config["aspect"] == "calibrated":
        calibrated_k = body_ratio_k(pose, times, float(meta["aspect"]["r_train"]))
    pre = meta["preprocessing"]
    pipeline = LivePipeline(
        PipelineConfig(
            pose_indices=tuple(pre["pose_indices"]),
            num_frames=int(pre["num_frames"]),
            min_hand_frames=int(pre["min_hand_frames"]),
            width=float(holistic["width"]),
            height=float(holistic["height"]),
            hand_rule="nearest_pose_wrist",
            aspect=config["aspect"],
            calibrated_k=calibrated_k,
            resampling=config["resampling"],
            segmenter=config["segmenter"],
        ),
        classify,
    )
    temperature, _ = decision_params(meta, config["decision"])
    records: list[dict[str, Any]] = []

    def record(outcome: SegmentOutcome | None) -> None:
        if outcome is None:
            return
        item = summary(outcome)
        item["k"] = pipeline.k
        item["probabilities"] = (
            None if outcome.logits is None else softmax(outcome.logits, temperature).tolist()
        )
        records.append(item)

    for index in range(times.size):
        hand_l = left[index] if np.isfinite(left[index, :, :2]).all() else None
        hand_r = right[index] if np.isfinite(right[index, :, :2]).all() else None
        body = pose[index, :, :3] if np.isfinite(pose[index, :, :2]).any() else None
        record(pipeline.push(live_frame(float(times[index]), hand_l, hand_r, body)))
    record(pipeline.flush())
    return records


def run_python(
    configs: dict[str, Any],
    config: dict[str, Any],
    clips_dir: Path,
    repo: Path,
) -> dict[str, Any]:
    """One config over every clip's .holistic.npz with the Python pipeline."""
    model_dir = repo / configs["models"][config["model"]]
    meta = load_meta(model_dir)
    model = OrtModel(model_dir / meta["model_file"], intra_op_threads=1)

    def classify(model_input: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
        logits: npt.NDArray[np.float32] = model.run(model_input[None])[0]
        return logits

    clips = []
    manifest = json.loads((clips_dir / "clips.json").read_text(encoding="utf-8"))
    for entry in manifest["clips"]:
        name = Path(entry["y4m"]).stem
        data = dict(np.load(clips_dir / f"{name}.holistic.npz"))
        clips.append(
            {
                "clip": name,
                "label": entry["label"],
                "label_index": meta["labels"].index(entry["label"]),
                "duration_ms": float(data["t_ms"][-1]),
                "duplicate_of": entry["duplicate_of"],
                "segments": run_clip(data, config, meta, classify),
            }
        )
    temperature, threshold = decision_params(meta, config["decision"])
    return {
        "pipeline": "python_holistic",
        "config": config,
        "model_sha256": meta["model_sha256"],
        "temperature": temperature,
        "threshold": threshold,
        "clips": clips,
    }


def top_k_hit(probabilities: Sequence[float], label: int, k: int) -> bool:
    order = np.argsort(-np.asarray(probabilities), kind="stable")[:k]
    return bool(label in order)


def score_run(run: dict[str, Any], exclude: Sequence[str] = ()) -> dict[str, Any]:
    """Clip- and segment-level top-1 / top-3, coverage and selective accuracy of one run."""
    threshold = float(run["threshold"])
    clip_top1 = clip_top3 = clips = 0
    seg_top1 = seg_top3 = segments = accepted = accepted_right = discarded = excluded = 0
    per_clip = {}
    for clip in run["clips"]:
        if clip["clip"] in exclude:
            continue
        clips += 1
        label = int(clip["label_index"])
        classified = [
            s["probabilities"] for s in clip["segments"] if s["probabilities"] is not None
        ]
        discarded += sum(1 for s in clip["segments"] if s["kind"] == "discarded")
        excluded += sum(
            1 for s in clip["segments"] if s["kind"] == "segment" and s["probabilities"] is None
        )
        prediction = None
        if classified:
            mean = np.mean(np.asarray(classified, dtype=np.float64), axis=0)
            clip_top1 += top_k_hit(mean, label, 1)
            clip_top3 += top_k_hit(mean, label, TOP_K)
            prediction = int(np.argmax(mean))
        for probabilities in classified:
            segments += 1
            seg_top1 += top_k_hit(probabilities, label, 1)
            seg_top3 += top_k_hit(probabilities, label, TOP_K)
            if max(probabilities) >= threshold:
                accepted += 1
                accepted_right += top_k_hit(probabilities, label, 1)
        per_clip[clip["clip"]] = {"segments": len(classified), "prediction": prediction}
    return {
        "clips": clips,
        "clip_top1": clip_top1 / clips if clips else float("nan"),
        "clip_top3": clip_top3 / clips if clips else float("nan"),
        "segments": segments,
        "segment_top1": seg_top1 / segments if segments else float("nan"),
        "segment_top3": seg_top3 / segments if segments else float("nan"),
        "coverage": accepted / segments if segments else float("nan"),
        "selective_accuracy": accepted_right / accepted if accepted else float("nan"),
        "discarded": discarded,
        "excluded": excluded,
        "per_clip": per_clip,
    }
