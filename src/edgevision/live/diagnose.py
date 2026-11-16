"""Offline diagnosis of web-demo recordings: how far live input is from the training data.

For each recording the raw landmarks are replayed through Python preprocessing (the source of
truth) and classified with the exported model under three aspect corrections:

- none: k = 1, the raw landmarks as captured;
- app: the k the app recorded (its 2-second calibration, a manual value, or 1 when off);
- offline: k = r_train / r_live with r_live the median over every visible frame of the
  recording's session, clamped like the app's.

It reports the app-vs-Python feature parity, accuracy per condition, and the input statistics
that differ from training: predicted-label concentration, normalisation method, shoulder and
hand visibility, segment lengths, the aspect ratio, and the standardised mean shift of each
feature group against the model's own FeatureBlock statistics.
"""

import json
import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import onnx
import pandas as pd
import torch
from onnx import numpy_helper

from edgevision.data.landmarks import N_HAND
from edgevision.layout import InputLayout
from edgevision.live.aspect import (
    K_MAX,
    K_MIN,
    calibrate,
    calibration_frames,
    frame_ratios,
    shoulder_frames,
)
from edgevision.live.recordings import (
    Recording,
    RecordingsFile,
    pose_visibility,
    raw_frames,
    replay,
)
from edgevision.models.features import STD_FLOOR, FeatureBlock, StandardisedFeatures
from edgevision.models.features_v2 import BODY_POINTS, BODY_TARGETS, HAND_PAIRS, FeatureBlockV2
from edgevision.runtime import OrtModel

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

PARITY_TOLERANCE = 1e-4
CONDITIONS = ("none", "app", "offline")
PARITY_TEXT = {True: "PASSED", False: "FAILED", None: "not checked (no classified segments)"}
TOP_PREDICTIONS = 10
TOP_SHIFTED = 3
QUANTILES = {"p10": 10, "median": 50, "p90": 90}
REPORT_FILE = "preprocess_report.json"
MANIFEST_FILE = "manifest.csv"
FEATURES_FILE = "features.npz"
SPLITS_FILE = "splits_signer_independent.json"
# A std at the floor marks a feature that is constant in training; its z-scores are not
# meaningful, so it is counted but left out of the group summaries.
FLOORED_STD = STD_FLOOR * 1.0001


def quantiles(values: npt.ArrayLike) -> dict[str, float | None]:
    data = np.asarray(values, dtype=np.float64)
    data = data[np.isfinite(data)]
    if data.size == 0:
        return dict.fromkeys(QUANTILES)
    return {
        name: float(value)
        for name, value in zip(
            QUANTILES, np.percentile(data, list(QUANTILES.values())), strict=True
        )
    }


# ---------------------------------------------------------------------------------------------
# Model and training reference


@dataclass(frozen=True)
class ModelBundle:
    meta: dict[str, Any]
    labels: list[str]
    layout: InputLayout
    feature_set: str
    model: OrtModel
    mean: FloatArray
    std: FloatArray
    sha256: str

    @property
    def pose_indices(self) -> list[int]:
        return list(self.layout.pose_indices)


def feature_buffers(model_path: Path) -> tuple[FloatArray, FloatArray]:
    """The first FeatureBlock's mean and std initializers (an ensemble's first member)."""
    graph = onnx.load(str(model_path)).graph
    arrays = {item.name: numpy_helper.to_array(item) for item in graph.initializer}
    means = sorted(name for name in arrays if name.endswith("features.mean"))
    for name in means:
        std_name = name[: -len("mean")] + "std"
        if std_name in arrays:
            return (
                np.asarray(arrays[name], dtype=np.float64).ravel(),
                np.asarray(arrays[std_name], dtype=np.float64).ravel(),
            )
    raise ValueError(f"{model_path}: no features.mean / features.std initializers")


def load_model(meta_path: Path, model_path: Path, sha256: str) -> ModelBundle:
    with meta_path.open(encoding="utf-8") as handle:
        meta = json.load(handle)
    if meta.get("model_sha256") != sha256:
        raise ValueError(f"{model_path} does not match the model_sha256 in {meta_path}")
    layout = InputLayout.for_pose_indices(meta["preprocessing"]["pose_indices"])
    mean, std = feature_buffers(model_path)
    return ModelBundle(
        meta=meta,
        labels=list(meta["labels"]),
        layout=layout,
        feature_set=str(meta.get("feature_set", "v1")),
        model=OrtModel(model_path, intra_op_threads=1),
        mean=mean,
        std=std,
        sha256=sha256,
    )


@dataclass(frozen=True)
class TrainingReference:
    r_train: float | None
    r_train_source: str
    aspect_stats: dict[str, Any] | None
    norm_method_fractions: dict[str, float] | None
    t_raw: dict[str, float | None]
    t_kept: dict[str, float | None]
    presence_mean: FloatArray
    presence_std: FloatArray
    presence_fraction: dict[str, float]
    split_source: str


def training_reference(processed_dir: Path, meta: dict[str, Any]) -> TrainingReference:
    """Training statistics from the processed dir: the train split when its split file is
    there, otherwise every kept sequence. r_train comes from the processed dir's aspect_stats,
    else from meta.json's aspect block."""
    report: dict[str, Any] = {}
    if (processed_dir / REPORT_FILE).is_file():
        report = json.loads((processed_dir / REPORT_FILE).read_text(encoding="utf-8"))
    aspect_stats = report.get("aspect_stats")
    meta_aspect = meta.get("aspect") or {}
    if aspect_stats and aspect_stats.get("median") is not None:
        r_train, source = float(aspect_stats["median"]), f"{processed_dir / REPORT_FILE}"
    elif meta_aspect.get("r_train") is not None:
        r_train, source = float(meta_aspect["r_train"]), f"meta.json ({meta_aspect.get('source')})"
        aspect_stats = meta_aspect.get("stats")
    else:
        r_train, source = None, "unavailable"

    manifest = pd.read_csv(processed_dir / MANIFEST_FILE)
    kept = manifest.loc[~manifest["excluded"].astype(bool)]
    with np.load(processed_dir / FEATURES_FILE) as arrays:
        presence = np.asarray(arrays["presence"], dtype=np.float64)
        sequence_ids = np.asarray(arrays["sequence_id"])
    split_source = "every kept sequence (no split file)"
    if (processed_dir / SPLITS_FILE).is_file():
        splits = json.loads((processed_dir / SPLITS_FILE).read_text(encoding="utf-8"))
        train_ids = np.asarray(splits["train"])
        presence = presence[np.isin(sequence_ids, train_ids)]
        kept = kept.loc[kept["sequence_id"].isin(train_ids)]
        split_source = f"train split of {processed_dir / SPLITS_FILE}"
    frames = presence.reshape(-1, 2)
    return TrainingReference(
        r_train=r_train,
        r_train_source=source,
        aspect_stats=aspect_stats,
        norm_method_fractions=report.get("norm_method_fractions"),
        t_raw=quantiles(kept["t_raw"]),
        t_kept=quantiles(kept["t_kept"]),
        presence_mean=frames.mean(axis=0),
        presence_std=frames.std(axis=0),
        presence_fraction={
            "left": float(frames[:, 0].mean()),
            "right": float(frames[:, 1].mean()),
        },
        split_source=split_source,
    )


# ---------------------------------------------------------------------------------------------
# Feature groups


def feature_block(feature_set: str, layout: InputLayout) -> StandardisedFeatures:
    return FeatureBlockV2(layout) if feature_set == "v2" else FeatureBlock(layout)


def landmark_labels(layout: InputLayout) -> list[str]:
    hands = [f"{side}_{i}" for side in ("left_hand", "right_hand") for i in range(N_HAND)]
    return hands + [f"pose_{index}" for index in layout.pose_indices]


def feature_groups(feature_set: str, layout: InputLayout) -> list[tuple[str, slice, list[str]]]:
    """(group, slice of the raw feature axis, feature names) in FeatureBlock order."""
    coords = layout.num_coords
    points = landmark_labels(layout)

    def per_axis(suffix: str) -> list[str]:
        return [f"{point}.{suffix}{axis}" for point in points for axis in ("x", "y")]

    if feature_set == "v2":
        pairs = [f"right_hand_{a}-right_hand_{b}" for a, b in HAND_PAIRS]
        body = [f"right_hand_{p}-{target}" for p in BODY_POINTS for target in BODY_TARGETS]
        start = 3 * coords
        return [
            ("coordinates", slice(0, coords), per_axis("")),
            ("velocity", slice(coords, 2 * coords), per_axis("d")),
            ("acceleration", slice(2 * coords, start), per_axis("dd")),
            ("hand_shape", slice(start, start + len(pairs)), pairs),
            ("hand_to_body", slice(start + len(pairs), start + len(pairs) + len(body)), body),
        ]
    return [
        ("coordinates", slice(0, coords), per_axis("")),
        ("presence", slice(coords, coords + 2), ["presence_left", "presence_right"]),
        ("velocity", slice(coords + 2, 2 * coords + 2), per_axis("d")),
    ]


def shift_summary(
    live_mean: FloatArray, train_mean: FloatArray, train_std: FloatArray, names: list[str]
) -> dict[str, Any]:
    """Standardised mean shift z = (live mean - train mean) / train std per feature."""
    floored = train_std <= FLOORED_STD
    z = (live_mean - train_mean) / np.maximum(train_std, STD_FLOOR)
    usable = ~floored
    summary: dict[str, Any] = {
        "features": len(names),
        "floored_std": int(floored.sum()),
        "mean_z": None,
        "mean_abs_z": None,
        "max_abs_z": None,
        "top": [],
    }
    if usable.any():
        values = z[usable]
        order = np.argsort(-np.abs(values), kind="stable")[:TOP_SHIFTED]
        usable_names = [name for name, keep in zip(names, usable, strict=True) if keep]
        summary.update(
            mean_z=float(values.mean()),
            mean_abs_z=float(np.abs(values).mean()),
            max_abs_z=float(np.abs(values).max()),
            top=[{"feature": usable_names[i], "z": float(values[i])} for i in order],
        )
    return summary


def feature_shift(
    model_inputs: Float32Array, bundle: ModelBundle, reference: TrainingReference
) -> dict[str, Any]:
    """Per-group shift of the live model inputs' raw features against the FeatureBlock's
    standardisation buffers; v2 features carry no presence, so its presence group is measured
    against the processed dir's train presence instead."""
    if model_inputs.shape[0] == 0:
        return {}
    block = feature_block(bundle.feature_set, bundle.layout)
    with torch.no_grad():
        raw = block.raw_features(torch.from_numpy(model_inputs)).double().numpy()
    live_mean = raw.reshape(-1, raw.shape[-1]).mean(axis=0)
    if live_mean.shape != bundle.mean.shape:
        raise ValueError(
            f"{live_mean.size} raw features but {bundle.mean.size} FeatureBlock statistics"
        )
    shifts = {
        name: shift_summary(live_mean[part], bundle.mean[part], bundle.std[part], names)
        for name, part, names in feature_groups(bundle.feature_set, bundle.layout)
    }
    if bundle.feature_set == "v2":
        presence = model_inputs[..., bundle.layout.num_coords :].reshape(-1, 2).mean(axis=0)
        shifts["presence"] = shift_summary(
            presence.astype(np.float64),
            reference.presence_mean,
            reference.presence_std,
            ["presence_left", "presence_right"],
        )
    return shifts


# ---------------------------------------------------------------------------------------------
# Recordings


@dataclass
class LiveRecording:
    """One recording with its source file, session key and per-frame measurements."""

    file: str
    file_index: int
    recording: Recording
    frames: Any
    ratios: FloatArray
    # Frames whose nose and shoulders are visible (they calibrate), and whose shoulders are.
    visible: npt.NDArray[np.bool_]
    shoulders: npt.NDArray[np.bool_]

    @property
    def session_key(self) -> tuple[int, int]:
        return self.file_index, self.recording.session


@dataclass
class ConditionResult:
    name: str
    inputs: list[Float32Array] = field(default_factory=list)
    # Packed float32 features before the float16 storage step, as the app records them.
    features: list[Float32Array] = field(default_factory=list)
    rows: list[int] = field(default_factory=list)
    norm_methods: list[str | None] = field(default_factory=list)
    infos: list[dict[str, Any]] = field(default_factory=list)
    k: list[float] = field(default_factory=list)


def collect(files: Sequence[tuple[str, RecordingsFile]]) -> list[LiveRecording]:
    items = []
    for index, (name, payload) in enumerate(files):
        for recording in payload.recordings:
            frames = raw_frames(recording.frames)
            visibility = pose_visibility(recording.frames)
            visible = calibration_frames(frames.pose, visibility)
            shoulders = shoulder_frames(frames.pose, visibility)
            ratios = np.where(visible, frame_ratios(frames.pose), np.nan)
            items.append(LiveRecording(name, index, recording, frames, ratios, visible, shoulders))
    return items


def session_calibrations(
    items: Sequence[LiveRecording], r_train: float | None
) -> dict[tuple[int, int], dict[str, Any]]:
    sessions: dict[tuple[int, int], list[LiveRecording]] = {}
    for item in items:
        sessions.setdefault(item.session_key, []).append(item)
    result = {}
    for key, members in sessions.items():
        ratios = np.concatenate([item.ratios for item in members])
        if r_train is None:
            k, r_live, status = 1.0, None, "no_r_train"
            frames = int(np.isfinite(ratios).sum())
        else:
            calibration = calibrate(ratios, r_train)
            k, r_live, status = calibration.k, calibration.r_live, calibration.status
            frames = calibration.frames
        first = members[0].recording
        result[key] = {
            "file": members[0].file,
            "session": key[1],
            "recordings": len(members),
            "frames_measured": frames,
            "r_live": r_live,
            "k_offline": k,
            "status": status,
            "app_k": sorted({item.recording.calibration.k for item in members}),
            "app_calibration_modes": sorted({item.recording.calibration.mode for item in members}),
            "camera": first.camera.model_dump(),
            "framing": first.framing,
        }
    return result


def condition_k(item: LiveRecording, condition: str, sessions: dict[Any, dict[str, Any]]) -> float:
    if condition == "none":
        return 1.0
    if condition == "app":
        return float(item.recording.calibration.k)
    return float(sessions[item.session_key]["k_offline"])


def pack(features: npt.NDArray[Any], presence: npt.NDArray[Any]) -> Float32Array:
    """(T, L, 2) coordinates and (T, 2) presence -> (T, 2L + 2) float32."""
    flat = np.asarray(features, dtype=np.float32).reshape(features.shape[0], -1)
    return np.concatenate([flat, np.asarray(presence, dtype=np.float32)], axis=1)


def model_input(features: npt.NDArray[Any], presence: npt.NDArray[Any]) -> Float32Array:
    """The float16 storage step training data took, then packing (layout.to_model_input)."""
    with np.errstate(over="ignore"):
        stored = np.asarray(features, dtype=np.float32).astype(np.float16)
    return pack(stored.astype(np.float32), presence)


def evaluate(
    items: Sequence[LiveRecording], bundle: ModelBundle, sessions: dict[Any, dict[str, Any]]
) -> dict[str, ConditionResult]:
    cfg = bundle.meta["preprocessing"]
    results = {name: ConditionResult(name) for name in CONDITIONS}
    for row, item in enumerate(items):
        for name in CONDITIONS:
            k = condition_k(item, name, sessions)
            replayed = replay(
                item.frames,
                k,
                pose_indices=bundle.pose_indices,
                num_frames=int(cfg["num_frames"]),
                min_hand_frames=item.recording.min_hand_frames,
            )
            outcome = results[name]
            outcome.k.append(k)
            outcome.infos.append(dict(replayed.info))
            outcome.norm_methods.append(replayed.info["norm_method"])
            if replayed.features is not None and replayed.hand_presence is not None:
                packed = model_input(replayed.features, replayed.hand_presence)
                if np.isfinite(packed).all():
                    outcome.inputs.append(packed)
                    outcome.features.append(pack(replayed.features, replayed.hand_presence))
                    outcome.rows.append(row)
    return results


def parity(items: Sequence[LiveRecording], results: dict[str, ConditionResult]) -> dict[str, Any]:
    """App features against Python's replay with the app's own k, for classified segments."""
    app = results["app"]
    replays = dict(zip(app.rows, app.features, strict=True))
    worst = 0.0
    info_mismatches: list[dict[str, Any]] = []
    per_recording: list[dict[str, Any]] = []
    for row, item in enumerate(items):
        recorded = item.recording.app_result
        if recorded.status != "classified" or recorded.features is None:
            continue
        info = app.infos[row]
        expected = recorded.info.model_dump() if recorded.info is not None else None
        if expected != info:
            info_mismatches.append({"id": item.recording.id, "app": expected, "python": info})
        if row not in replays:
            per_recording.append({"id": item.recording.id, "max_abs_diff": None})
            worst = math.inf
            continue
        features = np.asarray(recorded.features, dtype=np.float64)
        python = replays[row].astype(np.float64)
        diff = (
            float(np.abs(features - python).max()) if features.shape == python.shape else math.inf
        )
        worst = max(worst, diff)
        per_recording.append({"id": item.recording.id, "file": item.file, "max_abs_diff": diff})
    checked = len(per_recording)
    return {
        "checked": checked,
        "tolerance": PARITY_TOLERANCE,
        "max_abs_diff": None if checked == 0 else (None if math.isinf(worst) else worst),
        # None when no segment was classified by the app, so there was nothing to compare.
        "passed": None if checked == 0 else worst <= PARITY_TOLERANCE and not info_mismatches,
        "info_mismatches": info_mismatches,
        "per_recording": per_recording,
    }


def accuracy(
    outcome: ConditionResult, items: Sequence[LiveRecording], bundle: ModelBundle
) -> dict[str, Any]:
    n = len(items)
    predictions: list[str | None] = [None] * n
    top1 = top5 = 0
    if outcome.inputs:
        logits = bundle.model.run(np.stack(outcome.inputs).astype(np.float32))
        order = np.argsort(-logits, axis=1, kind="stable")[:, :5]
        for row, top in zip(outcome.rows, order, strict=True):
            intended = items[row].recording.intended_index
            predictions[row] = bundle.labels[int(top[0])]
            top1 += int(top[0] == intended)
            top5 += int(intended in top)
    methods = Counter(method or "excluded" for method in outcome.norm_methods)
    distribution = Counter(label for label in predictions if label is not None)
    return {
        "k": quantiles(outcome.k),
        "evaluated": len(outcome.rows),
        "excluded": n - len(outcome.rows),
        "top1": top1 / n if n else None,
        "top5": top5 / n if n else None,
        "prediction_distribution": distribution.most_common(TOP_PREDICTIONS),
        "distinct_predictions": len(distribution),
        "norm_method_fractions": {key: value / n for key, value in sorted(methods.items())},
        "predictions": predictions,
    }


def live_statistics(items: Sequence[LiveRecording], outcome: ConditionResult) -> dict[str, Any]:
    frames = [item.frames for item in items]
    total = sum(block.pose.shape[0] for block in frames)

    def fraction(count: float) -> float | None:
        return count / total if total else None

    left = sum(int(np.isfinite(b.left_hand[..., :2]).all(axis=(1, 2)).sum()) for b in frames)
    right = sum(int(np.isfinite(b.right_hand[..., :2]).all(axis=(1, 2)).sum()) for b in frames)
    either = sum(
        int(
            (
                np.isfinite(b.left_hand[..., :2]).all(axis=(1, 2))
                | np.isfinite(b.right_hand[..., :2]).all(axis=(1, 2))
            ).sum()
        )
        for b in frames
    )
    pose = sum(int(np.isfinite(b.pose[:, 11:13, :2]).all(axis=(1, 2)).sum()) for b in frames)
    visible = sum(int(item.shoulders.sum()) for item in items)
    ratios = np.concatenate([item.ratios for item in items]) if items else np.empty(0)
    resampled: dict[str, float | None] = {"left": None, "right": None}
    if outcome.inputs:
        presence = np.stack(outcome.inputs)[..., -2:].reshape(-1, 2).mean(axis=0)
        resampled = {"left": float(presence[0]), "right": float(presence[1])}
    return {
        "frames": total,
        "pose_shoulders_finite_fraction": fraction(pose),
        "shoulders_visible_fraction": fraction(visible),
        "hand_presence_fraction": {
            "left": fraction(left),
            "right": fraction(right),
            "either": fraction(either),
        },
        "resampled_presence_fraction": resampled,
        "segment_frames": quantiles([b.pose.shape[0] for b in frames]),
        "t_kept": quantiles([info["t_kept"] for info in outcome.infos]),
        "r_live": quantiles(ratios),
        "camera_fps": quantiles([item.recording.camera.frame_rate or np.nan for item in items]),
    }


def diagnose(
    files: Sequence[tuple[str, RecordingsFile]],
    bundle: ModelBundle,
    reference: TrainingReference,
) -> dict[str, Any]:
    items = collect(files)
    if not items:
        raise ValueError("the recordings files hold no recordings")
    for name, payload in files:
        if payload.app.model_sha256 != bundle.sha256:
            raise ValueError(f"{name} was recorded with another model ({payload.app.model_sha256})")
    sessions = session_calibrations(items, reference.r_train)
    results = evaluate(items, bundle, sessions)
    conditions = {}
    for name, outcome in results.items():
        summary = accuracy(outcome, items, bundle)
        summary["feature_shift"] = (
            feature_shift(np.stack(outcome.inputs), bundle, reference) if outcome.inputs else {}
        )
        conditions[name] = summary
    app_status = Counter(item.recording.app_result.status for item in items)
    per_recording: list[dict[str, Any]] = []
    for row, item in enumerate(items):
        recording = item.recording
        info = results["app"].infos[row]
        per_recording.append(
            {
                "file": item.file,
                "id": recording.id,
                "session": recording.session,
                "intended": recording.intended_label,
                "app_prediction": recording.app_result.prediction,
                "app_status": recording.app_result.status,
                "frames": int(item.frames.pose.shape[0]),
                "t_kept": info["t_kept"],
                "norm_method": info["norm_method"],
                "shoulders_visible": (
                    float(item.shoulders.mean()) if item.shoulders.size else None
                ),
                "r_live": quantiles(item.ratios)["median"],
                "k": {name: results[name].k[row] for name in CONDITIONS},
                "predictions": {name: conditions[name]["predictions"][row] for name in CONDITIONS},
            }
        )
    for summary in conditions.values():
        del summary["predictions"]
    return {
        "format": "edge-vision-live-diagnosis",
        "schema_version": 1,
        "model": {
            "name": bundle.meta.get("model_name"),
            "variant": bundle.meta.get("variant"),
            "run": bundle.meta.get("run"),
            "sha256": bundle.sha256,
            "feature_set": bundle.feature_set,
            "pose_indices": bundle.pose_indices,
        },
        "k_limits": [K_MIN, K_MAX],
        "training": {
            "r_train": reference.r_train,
            "r_train_source": reference.r_train_source,
            "aspect_stats": reference.aspect_stats,
            "norm_method_fractions": reference.norm_method_fractions,
            "t_raw": reference.t_raw,
            "t_kept": reference.t_kept,
            "resampled_presence_fraction": reference.presence_fraction,
            "split_source": reference.split_source,
        },
        "recordings": {
            "count": len(items),
            "files": [name for name, _ in files],
            "app_status": dict(sorted(app_status.items())),
            "intended_labels": dict(
                sorted(Counter(item.recording.intended_label for item in items).items())
            ),
        },
        "parity": parity(items, results),
        "sessions": list(sessions.values()),
        "live": live_statistics(items, results["app"]),
        "conditions": conditions,
        "per_recording": per_recording,
    }


# ---------------------------------------------------------------------------------------------
# Markdown


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{100 * value:.1f}%"


def _quantile_cell(values: dict[str, Any]) -> str:
    return " / ".join(_fmt(values.get(name), 2) for name in QUANTILES)


def render_markdown(report: dict[str, Any]) -> str:
    """A readable summary of a diagnosis report."""
    model, training, live = report["model"], report["training"], report["live"]
    parity_block = report["parity"]
    lines = [
        "# Live input diagnosis",
        "",
        f"Created {report['created_at']}. Model {model['name']} ({model['variant']}, run "
        f"{model['run']}, feature set {model['feature_set']}), {report['recordings']['count']} "
        f"recordings from {len(report['recordings']['files'])} file(s).",
        "",
        "## Parity: app features vs Python preprocessing",
        "",
        f"Checked {parity_block['checked']} classified segments: max |diff| "
        f"{_fmt(parity_block['max_abs_diff'], 7)} (tolerance {parity_block['tolerance']}), "
        f"info mismatches {len(parity_block['info_mismatches'])}: "
        f"**{PARITY_TEXT[parity_block['passed']]}**.",
        "",
        "## Accuracy by aspect correction",
        "",
        "| Condition | k (p10 / median / p90) | Evaluated | Excluded | Top-1 | Top-5 | Distinct "
        "predictions | Most frequent predictions |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for name, block in report["conditions"].items():
        common = ", ".join(
            f"{label} ({count})" for label, count in block["prediction_distribution"][:5]
        )
        lines.append(
            f"| {name} | {_quantile_cell(block['k'])} | {block['evaluated']} | "
            f"{block['excluded']} | {_pct(block['top1'])} | {_pct(block['top5'])} | "
            f"{block['distinct_predictions']} | {common or 'none'} |"
        )
    lines += [
        "",
        "## Live input vs training",
        "",
        "| Statistic | Live | Training |",
        "|---|---|---|",
        f"| Aspect ratio r (p10 / median / p90) | {_quantile_cell(live['r_live'])} | "
        f"{_quantile_cell(training['aspect_stats'] or {})} |",
        f"| Shoulders visible (in frame, visibility >= 0.5) | "
        f"{_pct(live['shoulders_visible_fraction'])} | n/a |",
        f"| Pose shoulders finite | {_pct(live['pose_shoulders_finite_fraction'])} | "
        f"{_pct((training['aspect_stats'] or {}).get('shoulders_finite_fraction'))} |",
        f"| Raw frames with left / right / either hand | "
        f"{_pct(live['hand_presence_fraction']['left'])} / "
        f"{_pct(live['hand_presence_fraction']['right'])} / "
        f"{_pct(live['hand_presence_fraction']['either'])} | n/a |",
        f"| Resampled presence left / right | "
        f"{_pct(live['resampled_presence_fraction']['left'])} / "
        f"{_pct(live['resampled_presence_fraction']['right'])} | "
        f"{_pct(training['resampled_presence_fraction']['left'])} / "
        f"{_pct(training['resampled_presence_fraction']['right'])} |",
        f"| Segment frames t_raw (p10 / median / p90) | {_quantile_cell(live['segment_frames'])} "
        f"| {_quantile_cell(training['t_raw'])} |",
        f"| Frames with a hand t_kept (p10 / median / p90) | {_quantile_cell(live['t_kept'])} | "
        f"{_quantile_cell(training['t_kept'])} |",
        f"| Camera fps (p10 / median / p90) | {_quantile_cell(live['camera_fps'])} | n/a |",
        "",
        f"r_train {_fmt(training['r_train'], 4)} from {training['r_train_source']}. Training "
        f"lengths and presence: {training['split_source']}.",
        "",
        "Normalisation method (app condition): "
        + ", ".join(
            f"{key} {_pct(value)}"
            for key, value in report["conditions"]["app"]["norm_method_fractions"].items()
        )
        + "; training: "
        + ", ".join(
            f"{key} {_pct(value)}"
            for key, value in (training["norm_method_fractions"] or {}).items()
        )
        + ".",
        "",
        "## Sessions",
        "",
        "| File | Session | Recordings | Camera | Framing | r_live | k offline | App k "
        "| App mode |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for session in report["sessions"]:
        camera = session["camera"]
        lines.append(
            f"| {Path(session['file']).name} | {session['session']} | {session['recordings']} | "
            f"{camera['width']}x{camera['height']} | {session['framing']} | "
            f"{_fmt(session['r_live'], 4)} | {_fmt(session['k_offline'], 3)} | "
            f"{', '.join(_fmt(k, 3) for k in session['app_k'])} | "
            f"{', '.join(session['app_calibration_modes'])} |"
        )
    lines += [
        "",
        "## Feature shift (live mean - train mean) / train std",
        "",
        "Mean |z| per feature group; the largest shifts in brackets. Features that are constant "
        "in training (std at the floor) are left out.",
        "",
        "| Group | " + " | ".join(report["conditions"]) + " |",
        "|---|" + "---|" * len(report["conditions"]),
    ]
    groups: list[str] = []
    for block in report["conditions"].values():
        groups += [name for name in block["feature_shift"] if name not in groups]
    for group in groups:
        cells = []
        for block in report["conditions"].values():
            shift = block["feature_shift"].get(group)
            if not shift or shift["mean_abs_z"] is None:
                cells.append("n/a")
                continue
            top = ", ".join(f"{item['feature']} {item['z']:+.1f}" for item in shift["top"])
            cells.append(f"{shift['mean_abs_z']:.2f} ({top})")
        lines.append(f"| {group} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Recordings",
        "",
        "| File | Id | Intended | App prediction | Frames | t_kept | Norm | Shoulders visible | "
        "r_live | none | app | offline |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for item in report["per_recording"]:
        predictions = item["predictions"]
        lines.append(
            f"| {Path(item['file']).name} | {item['id']} | {item['intended']} | "
            f"{item['app_prediction'] or item['app_status']} | {item['frames']} | "
            f"{item['t_kept']} | {item['norm_method'] or 'excluded'} | "
            f"{_pct(item['shoulders_visible'])} | {_fmt(item['r_live'], 3)} | "
            f"{predictions['none'] or '-'} | {predictions['app'] or '-'} | "
            f"{predictions['offline'] or '-'} |"
        )
    return "\n".join(lines) + "\n"
