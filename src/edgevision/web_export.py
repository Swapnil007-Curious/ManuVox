"""Export one trained model, or the seed ensemble of its experiment, for the browser demo in web/.

Writes ``<out>/model.onnx`` (one ONNX variant, chosen on val by VARIANT_RULE), ``<out>/meta.json``
(labels, feature set, input layout, preprocessing settings including the pose swap pairs, the
training aspect statistics, hashes, the variant candidates and the confidence threshold) and the
golden vectors that the web test suite compares the browser against. Only the val split is read;
asking for any other split raises.
"""

import hashlib
import json
import platform
import shutil
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, NamedTuple, Self

import numpy as np
import numpy.typing as npt
import onnx
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgevision import spec
from edgevision.config import PreprocessConfig
from edgevision.data.io import to_storage
from edgevision.data.landmarks import N_HAND, NOSE, SHOULDERS, pose_swap_pairs
from edgevision.data.preprocess import MIN_SCALE, X_GRID, preprocess_sequence, snap_x
from edgevision.evaluation.metrics import top_k_accuracy
from edgevision.export.ensemble import export_ensemble_onnx
from edgevision.export.onnx_export import export_onnx, supports_embedding
from edgevision.layout import EMBEDDING_NAME, NUM_FRAMES, POSE_INDICES, InputLayout, to_model_input
from edgevision.live.aspect import K_MAX, K_MIN, MIN_VISIBILITY
from edgevision.live.golden import build_aspect_golden, example_recordings
from edgevision.models.registry import checkpoint_layout, load_checkpoint
from edgevision.quant.variants import (
    VARIANT_SCHEMA_VERSIONS,
    convert_fp16,
    quantize_int8_dynamic,
    read_metadata,
    save_with_metadata,
)
from edgevision.runs import (
    CHECKPOINT_FILE,
    QUANT_REPORT_FILE,
    SUMMARY_FILE,
    variant_file,
)
from edgevision.runtime import (
    META_INPUT_LAYOUT,
    META_LABEL_MAP,
    META_SCHEMA_VERSION,
    META_SOURCE_SHA256,
    META_VARIANT,
    OrtModel,
    label_map_from_metadata,
    onnxruntime_version,
)
from edgevision.train.data import SplitArrays, load_split_arrays

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

WebVariant = Literal["fp32", "fp16", "int8_dynamic", "int8_static"]
WEB_VARIANTS: tuple[WebVariant, ...] = ("fp32", "fp16", "int8_dynamic", "int8_static")
# asl-signs+asl-citizen: the v3 model, trained on both (kaggle/train_v3.ipynb).
DataSource = Literal["asl-signs", "synthetic", "asl-signs+asl-citizen"]
DATA_SOURCES: tuple[DataSource, ...] = ("asl-signs", "synthetic", "asl-signs+asl-citizen")
ASL_SIGNS_CLASSES = 250

FeatureSetName = Literal["v1", "v2"]

EVAL_SPLIT = "val"
DEFAULT_TARGET_PRECISION = 0.80
FALLBACK_MIN_COVERAGE = 0.2
VARIANT_RULE_MAX_DROP_PP = 0.5
# Absorbs float noise in (fp32 - int8) * 100, e.g. 0.85 - 0.84 = 0.0100000000000000089.
_PP_EPSILON = 1e-9
MODEL_CASES = 16
WEB_META_SCHEMA_VERSION: Literal[2] = 2
GOLDEN_SCHEMA_VERSION: Literal[2] = 2
INPUT_DECIMALS = 6
GOLDEN_SEED = 2026
V2_GOLDEN_POSE_INDICES: tuple[int, ...] = tuple(range(17))
CALIBRATION_SECONDS = 2.0
META_ENSEMBLE_MEMBERS = "edgevision.ensemble_members"

MODEL_FILE = "model.onnx"
META_FILE = "meta.json"
PREPROCESS_CASES_FILE = "preprocess_cases.json"
PREPROCESS_CASES_V2_FILE = "preprocess_cases_v2.json"
ASPECT_CASES_FILE = "aspect_cases.json"
RECORDING_EXAMPLE_FILE = "recording_example.json"
MODEL_CASES_FILE = "model_cases.json"
PREPROCESS_REPORT_FILE = "preprocess_report.json"
SHA256_PATTERN = r"^[0-9a-f]{64}$"

VARIANT_RULE = (
    "among fp32, fp16, int8_dynamic and int8_static, the smallest file whose val top-1 is "
    f"within {VARIANT_RULE_MAX_DROP_PP:.1f} percentage points of fp32 val top-1; a variant that "
    "onnxruntime-web (wasm) cannot run falls back to the next smallest such file"
)
FLOAT_ENCODING = (
    "float32 arrays are written as the shortest decimal that rounds to the same float32; "
    "raw input coordinates are exact float64 values (rounded to 6 decimals when generated); "
    "null is NaN"
)


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------------------------
# Split access


def load_eval_split(processed_dir: Path, splits_path: Path, split: str = EVAL_SPLIT) -> SplitArrays:
    """The val split; any other split name raises, so the export cannot read test."""
    if split != EVAL_SPLIT:
        raise ValueError(f"web export reads only the {EVAL_SPLIT} split, not {split!r}")
    return load_split_arrays(processed_dir, splits_path, [EVAL_SPLIT])[EVAL_SPLIT]


# ---------------------------------------------------------------------------------------------
# Variant rule


class WasmCheck(_StrictModel):
    """Whether onnxruntime-web ran the exported file (the browser model-parity test)."""

    status: Literal["passed", "failed", "not_run"]
    detail: str | None = None


class VariantCandidate(_StrictModel):
    variant: WebVariant
    available: bool
    file_bytes: int | None
    val_top1: float | None
    drop_pp: float | None
    eligible: bool
    note: str | None
    wasm: WasmCheck


class Fallback(_StrictModel):
    variant: WebVariant
    reason: str


class VariantSelection(_StrictModel):
    rule: str
    max_drop_pp: float
    source: Literal["quant_report", "export_build"]
    fp32_val_top1: float
    candidates: list[VariantCandidate]
    # Eligible variants, smallest file first: the order in which they are tried.
    order: list[WebVariant]
    recommended: WebVariant
    chosen: WebVariant
    forced: bool
    fallbacks: list[Fallback]
    follows_rule: bool


class CandidateInput(NamedTuple):
    """What is known about one variant before the rule runs."""

    path: Path | None
    file_bytes: int | None
    val_top1: float | None
    note: str | None


def rank_candidates(
    inputs: Mapping[WebVariant, CandidateInput],
) -> tuple[list[VariantCandidate], list[WebVariant], float]:
    """Apply VARIANT_RULE: the candidates in WEB_VARIANTS order, the eligible variants smallest
    file first (ties in WEB_VARIANTS order), and fp32 val top-1."""
    fp32 = inputs.get("fp32")
    if fp32 is None or fp32.path is None or fp32.val_top1 is None:
        raise ValueError("no fp32 variant with a val top-1; rerun scripts/quantize.py")
    candidates = []
    for variant in WEB_VARIANTS:
        item = inputs.get(variant) or CandidateInput(None, None, None, "not built")
        available = item.path is not None and item.val_top1 is not None
        drop = None if item.val_top1 is None else (fp32.val_top1 - item.val_top1) * 100.0
        within = drop is not None and drop <= VARIANT_RULE_MAX_DROP_PP + _PP_EPSILON
        eligible = available and within
        note = item.note
        if available and not eligible and drop is not None:
            note = f"val top-1 {drop:.2f} pp below fp32 (limit {VARIANT_RULE_MAX_DROP_PP:.1f} pp)"
        candidates.append(
            VariantCandidate(
                variant=variant,
                available=available,
                file_bytes=item.file_bytes,
                val_top1=item.val_top1,
                drop_pp=drop,
                eligible=eligible,
                note=note,
                wasm=WasmCheck(status="not_run"),
            )
        )
    ranked = sorted(
        (c for c in candidates if c.eligible),
        key=lambda c: (c.file_bytes or 0, WEB_VARIANTS.index(c.variant)),
    )
    return candidates, [c.variant for c in ranked], fp32.val_top1


def candidates_from_report(
    report: Mapping[str, Any], quant_dir: Path, checkpoint_sha256: str
) -> dict[WebVariant, CandidateInput]:
    """The quant_report.json variants whose files exist and were built from this checkpoint."""
    variants = report.get("variants") or {}
    inputs: dict[WebVariant, CandidateInput] = {}
    for variant in WEB_VARIANTS:
        entry = variants.get(variant)
        path = quant_dir / variant_file(variant)
        if entry is None:
            inputs[variant] = CandidateInput(None, None, None, "not in quant_report.json")
        elif entry.get("error"):
            inputs[variant] = CandidateInput(None, None, None, f"build failed: {entry['error']}")
        elif not path.is_file():
            inputs[variant] = CandidateInput(None, None, None, f"{path} not found")
        elif read_metadata(path).get(META_SOURCE_SHA256) != checkpoint_sha256:
            inputs[variant] = CandidateInput(None, None, None, f"{path} was not built from best.pt")
        else:
            top1 = (entry.get("val") or {}).get("top1")
            inputs[variant] = CandidateInput(
                path,
                path.stat().st_size,
                None if top1 is None else float(top1),
                None if top1 is not None else "no val top-1 in quant_report.json",
            )
    return inputs


# ---------------------------------------------------------------------------------------------
# Confidence threshold


class ThresholdBlock(_StrictModel):
    """Accept a prediction when its softmax max-probability is >= ``threshold``."""

    computed_on: Literal["val"]
    target_precision: float = Field(gt=0.0, le=1.0)
    threshold: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    accuracy_above_threshold: float = Field(ge=0.0, le=1.0)
    overall_val_top1: float = Field(ge=0.0, le=1.0)
    val_sequences: int = Field(ge=1)
    method: Literal["target_precision", "fallback_max_accuracy"]
    fallback_min_coverage: float
    warning: str | None


def softmax(logits: npt.NDArray[np.floating[Any]]) -> FloatArray:
    """Row-wise softmax in float64."""
    values = np.asarray(logits, dtype=np.float64)
    exp = np.exp(values - values.max(axis=1, keepdims=True))
    result: FloatArray = exp / exp.sum(axis=1, keepdims=True)
    return result


def select_threshold(
    max_prob: npt.NDArray[np.floating[Any]],
    correct: npt.NDArray[np.bool_],
    target_precision: float = DEFAULT_TARGET_PRECISION,
    min_coverage: float = FALLBACK_MIN_COVERAGE,
) -> ThresholdBlock:
    """The smallest t with top-1 accuracy >= ``target_precision`` among samples with p >= t.

    Candidates are the distinct max-probabilities, so accuracy above t is exactly what was
    measured. If no candidate reaches the target, the candidate with the highest accuracy among
    those accepting at least ``min_coverage`` of the samples is used (the smallest such t on
    ties) and a warning is recorded.
    """
    p = np.asarray(max_prob, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    if p.ndim != 1 or p.shape != ok.shape or p.size == 0:
        raise ValueError(f"expected two equal non-empty 1-D arrays, got {p.shape} and {ok.shape}")
    if not 0.0 < target_precision <= 1.0:
        raise ValueError(f"target_precision must be in (0, 1], got {target_precision}")
    order = np.argsort(p, kind="stable")
    p_sorted, ok_sorted = p[order], ok[order]
    candidates, first = np.unique(p_sorted, return_index=True)
    n = p.size
    accepted = n - first
    correct_from = np.cumsum(ok_sorted[::-1])[::-1]
    accuracy = correct_from[first] / accepted
    coverage = accepted / n

    reaching = np.flatnonzero(accuracy >= target_precision)
    warning = None
    method: Literal["target_precision", "fallback_max_accuracy"] = "target_precision"
    if reaching.size:
        index = int(reaching[0])
    else:
        eligible = np.flatnonzero(coverage >= min_coverage)
        index = int(eligible[np.argmax(accuracy[eligible])])
        method = "fallback_max_accuracy"
        warning = (
            f"no threshold reaches top-1 accuracy {target_precision:.2f} on val; using the "
            f"threshold with the highest accuracy among those accepting at least "
            f"{min_coverage:.0%} of val sequences"
        )
    return ThresholdBlock(
        computed_on="val",
        target_precision=target_precision,
        threshold=float(candidates[index]),
        coverage=float(coverage[index]),
        accuracy_above_threshold=float(accuracy[index]),
        overall_val_top1=float(ok.mean()),
        val_sequences=n,
        method=method,
        fallback_min_coverage=min_coverage,
        warning=warning,
    )


def threshold_from_logits(
    logits: Float32Array,
    labels: npt.NDArray[np.int64],
    target_precision: float = DEFAULT_TARGET_PRECISION,
) -> ThresholdBlock:
    probabilities = softmax(logits)
    correct = probabilities.argmax(axis=1) == labels
    return select_threshold(probabilities.max(axis=1), correct, target_precision)


# ---------------------------------------------------------------------------------------------
# Calibration: temperature scaling and the "not sure" threshold

TEMPERATURE_BOUNDS = (0.05, 20.0)
ECE_BINS = 15
CALIBRATION_CRITERION = (
    "maximise coverage x (2 x selective accuracy - 1) on val: an accepted right sign gains 1, an "
    "accepted wrong sign costs 1, 'not sure' scores 0 (the smallest such threshold on ties)"
)


class CalibrationBlock(_StrictModel):
    """softmax(logits / temperature); a prediction whose top probability is >= ``threshold`` is
    shown, otherwise the app says "not sure". Both are fitted on val only."""

    computed_on: Literal["val"]
    method: Literal["temperature_scaling"]
    temperature: float = Field(gt=0.0)
    nll_before: float
    nll_after: float
    ece_before: float = Field(ge=0.0, le=1.0)
    ece_after: float = Field(ge=0.0, le=1.0)
    criterion: str
    threshold: float = Field(ge=0.0, le=1.0)
    coverage: float = Field(ge=0.0, le=1.0)
    selective_accuracy: float = Field(ge=0.0, le=1.0)
    net_score: float = Field(ge=-1.0, le=1.0)
    val_top1: float = Field(ge=0.0, le=1.0)
    val_top3: float = Field(ge=0.0, le=1.0)
    val_sequences: int = Field(ge=1)


def _log_softmax(logits: npt.NDArray[np.floating[Any]], temperature: float) -> FloatArray:
    values = np.asarray(logits, dtype=np.float64) / temperature
    values = values - values.max(axis=1, keepdims=True)
    result: FloatArray = values - np.log(np.exp(values).sum(axis=1, keepdims=True))
    return result


def nll(
    logits: npt.NDArray[np.floating[Any]], labels: npt.NDArray[np.int64], temperature: float
) -> float:
    """Mean negative log-likelihood of ``labels`` under softmax(logits / temperature)."""
    log_p = _log_softmax(logits, temperature)
    return float(-log_p[np.arange(labels.size), labels].mean())


def expected_calibration_error(
    max_prob: npt.NDArray[np.floating[Any]], correct: npt.NDArray[np.bool_], bins: int = ECE_BINS
) -> float:
    """Equal-width ECE of the top-1 confidence."""
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(max_prob, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        mask = which == b
        if mask.any():
            total += mask.mean() * abs(float(correct[mask].mean()) - float(max_prob[mask].mean()))
    return float(total)


def fit_temperature(logits: npt.NDArray[np.floating[Any]], labels: npt.NDArray[np.int64]) -> float:
    """The temperature minimising val NLL: golden-section search on log T (NLL is convex in
    1 / T, so unimodal in log T)."""
    low, high = float(np.log(TEMPERATURE_BOUNDS[0])), float(np.log(TEMPERATURE_BOUNDS[1]))
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a, b = high - ratio * (high - low), low + ratio * (high - low)
    fa, fb = nll(logits, labels, float(np.exp(a))), nll(logits, labels, float(np.exp(b)))
    for _ in range(80):
        if fa < fb:
            high, b, fb = b, a, fa
            a = high - ratio * (high - low)
            fa = nll(logits, labels, float(np.exp(a)))
        else:
            low, a, fa = a, b, fb
            b = low + ratio * (high - low)
            fb = nll(logits, labels, float(np.exp(b)))
    return float(np.exp(0.5 * (low + high)))


def select_net_threshold(
    max_prob: npt.NDArray[np.floating[Any]], correct: npt.NDArray[np.bool_]
) -> tuple[float, float, float, float]:
    """(threshold, coverage, selective accuracy, net score) maximising coverage x (2 x
    selective accuracy - 1) over the distinct probabilities; the smallest threshold on ties."""
    p = np.asarray(max_prob, dtype=np.float64)
    ok = np.asarray(correct, dtype=bool)
    if p.ndim != 1 or p.shape != ok.shape or p.size == 0:
        raise ValueError(f"expected two equal non-empty 1-D arrays, got {p.shape} and {ok.shape}")
    order = np.argsort(p, kind="stable")
    p_sorted, ok_sorted = p[order], ok[order]
    candidates, first = np.unique(p_sorted, return_index=True)
    n = p.size
    accepted = n - first
    right = np.cumsum(ok_sorted[::-1])[::-1][first]
    net = (2 * right - accepted) / n
    index = int(np.argmax(net))
    return (
        float(candidates[index]),
        float(accepted[index] / n),
        float(right[index] / accepted[index]),
        float(net[index]),
    )


def calibration_from_logits(
    logits: npt.NDArray[np.floating[Any]], labels: npt.NDArray[np.int64]
) -> CalibrationBlock:
    temperature = fit_temperature(logits, labels)
    raw = softmax(np.asarray(logits, dtype=np.float64))
    scaled = np.exp(_log_softmax(logits, temperature))
    correct = scaled.argmax(axis=1) == labels
    threshold, coverage, selective, net = select_net_threshold(scaled.max(axis=1), correct)
    return CalibrationBlock(
        computed_on="val",
        method="temperature_scaling",
        temperature=temperature,
        nll_before=nll(logits, labels, 1.0),
        nll_after=nll(logits, labels, temperature),
        ece_before=expected_calibration_error(raw.max(axis=1), raw.argmax(axis=1) == labels),
        ece_after=expected_calibration_error(scaled.max(axis=1), correct),
        criterion=CALIBRATION_CRITERION,
        threshold=threshold,
        coverage=coverage,
        selective_accuracy=selective,
        net_score=net,
        val_top1=top_k_accuracy(np.asarray(logits), labels, 1),
        val_top3=top_k_accuracy(np.asarray(logits), labels, 3),
        val_sequences=int(labels.size),
    )


# ---------------------------------------------------------------------------------------------
# meta.json


class InputLayoutBlock(_StrictModel):
    """Mirrors edgevision.layout.InputLayout.to_dict(); extra or missing keys fail validation."""

    num_frames: int
    num_landmarks: int
    coords_per_landmark: int
    frame_dim: int
    coordinate_order: Literal["landmark_major_xy"]
    landmark_order: list[str]
    pose_indices: list[int]
    presence_left_index: int
    presence_right_index: int
    missing_point: str
    input_name: str
    output_name: str


class PreprocessingSettings(_StrictModel):
    num_frames: int = Field(ge=1)
    use_z: Literal[False]
    pose_indices: list[int]
    # The left/right pose pairs mirroring swaps, restricted to pose_indices.
    pose_swap_pairs: list[tuple[int, int]]
    min_hand_frames: int = Field(ge=1)
    min_scale: float
    x_grid: float
    storage_dtype: Literal["float16"]
    source: str


class AspectBlock(_StrictModel):
    """Training aspect ratio r (preprocess_report.json aspect_stats) and the calibration
    settings the app applies: k = r_train / r_live, clamped to [k_min, k_max]."""

    r_train: float = Field(gt=0.0)
    stats: dict[str, Any]
    source: str
    k_min: float
    k_max: float
    calibration_seconds: float
    min_visibility: float


class SourceCheckpoint(_StrictModel):
    run: str
    sha256: str = Field(pattern=SHA256_PATTERN)


class EmbeddingBlock(_StrictModel):
    """The model's second output, the penultimate embedding Teach Dex compares."""

    output_name: str
    dim: int = Field(ge=1)
    source: str


class PersonalizationBlock(_StrictModel):
    """Teach Dex: logits / T + weight * (s - offset) for taught signs, s the highest cosine
    similarity to the user's examples of that sign; weight and offset fitted on val
    (scripts/calibrate_teach.py)."""

    computed_on: Literal["val"]
    method: str
    weight: float = Field(ge=0.0)
    offset: float
    examples_per_sign: int = Field(ge=1)
    influence_threshold: float = Field(ge=0.0, le=1.0)
    simulation: dict[str, Any]


class WebMeta(_StrictModel):
    schema_version: Literal[2]
    model_name: str
    variant: WebVariant
    data_source: DataSource
    run: str
    seed: int | None
    seeds: list[int] = Field(min_length=1)
    ensemble: bool
    feature_set: FeatureSetName
    num_classes: int = Field(ge=2)
    labels: list[str]
    num_frames: int
    landmark_order: list[str]
    pose_indices: list[int]
    input_layout: InputLayoutBlock
    preprocessing: PreprocessingSettings
    aspect: AspectBlock | None
    opset: int
    model_file: str
    model_bytes: int = Field(ge=1)
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    source_checkpoints: list[SourceCheckpoint] = Field(min_length=1)
    threshold: ThresholdBlock
    calibration: CalibrationBlock
    features_spec_version: int
    features_spec_sha256: str = Field(pattern=SHA256_PATTERN)
    variant_selection: VariantSelection
    tool_versions: dict[str, str]
    # Optional, so meta.json files written before Teach Dex still validate.
    embedding: EmbeddingBlock | None = None
    personalization: PersonalizationBlock | None = None

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        problems = []
        if len(self.labels) != self.num_classes:
            problems.append(f"{len(self.labels)} labels for {self.num_classes} classes")
        if len(set(self.labels)) != len(self.labels):
            problems.append("labels are not unique")
        frames = {self.num_frames, self.input_layout.num_frames, self.preprocessing.num_frames}
        if len(frames) != 1:
            problems.append("num_frames differs between fields")
        if len(self.landmark_order) != self.input_layout.num_landmarks:
            problems.append("landmark_order length differs from num_landmarks")
        pose = (self.pose_indices, self.input_layout.pose_indices, self.preprocessing.pose_indices)
        if pose[0] != pose[1] or pose[0] != pose[2]:
            problems.append("pose_indices differ between fields")
        if [list(pair) for pair in self.preprocessing.pose_swap_pairs] != [
            list(pair) for pair in pose_swap_pairs(self.pose_indices)
        ]:
            problems.append("pose_swap_pairs do not match pose_indices")
        if self.variant != self.variant_selection.chosen:
            problems.append("variant differs from variant_selection.chosen")
        if self.ensemble != (len(self.source_checkpoints) > 1):
            problems.append("an ensemble has several source checkpoints, a single run one")
        if problems:
            raise ValueError("; ".join(problems))
        return self


def landmark_names(pose_indices: Sequence[int] = POSE_INDICES) -> list[str]:
    """Selected landmark order: left_hand_0..20, right_hand_0..20, pose_<index>..."""
    hands = [f"{side}_{index}" for side in ("left_hand", "right_hand") for index in range(N_HAND)]
    return hands + [f"pose_{index}" for index in pose_indices]


def read_preprocess_report(processed_dir: Path) -> dict[str, Any] | None:
    path = processed_dir / PREPROCESS_REPORT_FILE
    return read_json(path) if path.is_file() else None


def preprocessing_settings(processed_dir: Path, layout: InputLayout) -> PreprocessingSettings:
    """Settings from the processed dir's preprocess_report.json, else PreprocessConfig defaults
    with the layout's pose selection.

    Refuses settings the model's input layout cannot represent.
    """
    report = read_preprocess_report(processed_dir)
    if report is not None:
        values = dict(report["config"]["preprocess"])
        if "landmarks" in report:  # The effective selection, recorded since v2.
            values["pose_indices"] = report["landmarks"]["pose_indices"]
        source = PREPROCESS_REPORT_FILE
    else:
        fields = PreprocessConfig.model_fields
        values = {
            "num_frames": fields["num_frames"].default,
            "use_z": fields["use_z"].default,
            "pose_indices": list(layout.pose_indices),
            "min_hand_frames": fields["min_hand_frames"].default,
        }
        source = f"PreprocessConfig defaults (no {PREPROCESS_REPORT_FILE} in the processed dir)"
    if (
        values["num_frames"] != layout.num_frames
        or list(values["pose_indices"]) != list(layout.pose_indices)
        or values["use_z"]
    ):
        raise ValueError(
            f"preprocessing settings {values} do not match the model input layout "
            f"(num_frames {layout.num_frames}, pose_indices {list(layout.pose_indices)}, "
            "use_z False)"
        )
    pose_indices = [int(index) for index in values["pose_indices"]]
    return PreprocessingSettings(
        num_frames=int(values["num_frames"]),
        use_z=False,
        pose_indices=pose_indices,
        pose_swap_pairs=list(pose_swap_pairs(pose_indices)),
        min_hand_frames=int(values["min_hand_frames"]),
        min_scale=MIN_SCALE,
        x_grid=X_GRID,
        storage_dtype="float16",
        source=source,
    )


# Report fields that must agree for another preprocessing run's aspect_stats to stand in: the
# measurement depends only on the raw data, the kept-frame rule (a frame with a full hand) and
# which sequences are kept, and it reads only the nose and the shoulders.
ASPECT_PREPROCESS_KEYS = ("data_root", "num_frames", "use_z", "min_hand_frames", "limit", "seed")
ASPECT_POINTS = (NOSE, *SHOULDERS)


def _aspect_block(stats: Mapping[str, Any], source: str) -> AspectBlock:
    return AspectBlock(
        r_train=float(stats["median"]),
        stats=dict(stats),
        source=source,
        k_min=K_MIN,
        k_max=K_MAX,
        calibration_seconds=CALIBRATION_SECONDS,
        min_visibility=MIN_VISIBILITY,
    )


def aspect_compatibility(own: Mapping[str, Any], other: Mapping[str, Any]) -> list[str]:
    """Why ``other``'s aspect_stats cannot stand in for a run preprocessed as ``own``."""
    problems = []
    for key in ("counts", "per_sign_kept", "mirrored_fraction", "norm_method_fractions"):
        if own.get(key) != other.get(key):
            problems.append(f"{key} differs")
    own_pre, other_pre = own["config"]["preprocess"], other["config"]["preprocess"]
    for key in ASPECT_PREPROCESS_KEYS:
        if own_pre.get(key) != other_pre.get(key):
            problems.append(f"config.preprocess.{key} differs")
    if own["config"].get("split") != other["config"].get("split"):
        problems.append("config.split differs")
    for name, pre in (("this run", own_pre), ("the other run", other_pre)):
        missing = [index for index in ASPECT_POINTS if index not in pre["pose_indices"]]
        if missing:
            problems.append(f"{name} does not select pose {missing}")
    if not other.get("aspect_stats") or other["aspect_stats"].get("median") is None:
        problems.append("the other report has no aspect_stats")
    return problems


def aspect_settings(
    processed_dir: Path, aspect_report: Path | None, data_source: DataSource
) -> AspectBlock | None:
    """The processed dir's aspect_stats; else those of ``aspect_report``, a preprocessing run of
    the same raw data, after aspect_compatibility checks; else None for synthetic data."""
    report = read_preprocess_report(processed_dir)
    if report is not None and (report.get("aspect_stats") or {}).get("median") is not None:
        return _aspect_block(
            report["aspect_stats"], (processed_dir / PREPROCESS_REPORT_FILE).as_posix()
        )
    if aspect_report is not None:
        if report is None:
            raise ValueError(f"no {PREPROCESS_REPORT_FILE} in {processed_dir} to compare with")
        other = read_json(aspect_report)
        problems = aspect_compatibility(report, other)
        if problems:
            raise ValueError(
                f"{aspect_report} cannot stand in for {processed_dir}: " + "; ".join(problems)
            )
        return _aspect_block(
            other["aspect_stats"],
            f"{aspect_report.as_posix()} (another preprocessing run of the same raw data: "
            "identical counts, "
            "kept sequences, split settings and kept-frame rule; the measurement reads only pose "
            f"{list(ASPECT_POINTS)}, which both selections hold)",
        )
    if data_source == "synthetic":
        return None
    raise ValueError(
        f"{processed_dir / PREPROCESS_REPORT_FILE} has no aspect_stats (written by preprocessing "
        "since the v2 change); rerun scripts/preprocess.py, or pass --aspect-report with the "
        "report of another preprocessing run of the same raw data"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def onnx_opset(path: Path) -> int:
    model = onnx.load(str(path), load_external_data=False)
    versions = [entry.version for entry in model.opset_import if entry.domain in ("", "ai.onnx")]
    if not versions:
        raise ValueError(f"{path}: no default-domain opset")
    return int(versions[0])


def tool_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "onnx": onnx.__version__,
        "onnxruntime": onnxruntime_version(),
    }


# ---------------------------------------------------------------------------------------------
# Golden vectors


class GoldenRows(_StrictModel):
    """Long-format input rows, column-wise."""

    frame: list[int]
    type: list[str]
    landmark_index: list[int]
    x: list[float | None]
    y: list[float | None]
    z: list[float | None]

    @model_validator(mode="after")
    def _equal_lengths(self) -> Self:
        lengths = {len(getattr(self, name)) for name in type(self).model_fields}
        if len(lengths) != 1:
            raise ValueError(f"row columns differ in length: {sorted(lengths)}")
        return self


class GoldenInfo(_StrictModel):
    t_raw: int
    t_kept: int
    mirrored: bool
    norm_method: Literal["shoulders", "hand_fallback"] | None
    excluded: bool
    reason: Literal["insufficient_hand_frames", "degenerate_scale"] | None


def _check_matrix(name: str, value: list[list[float]] | None, columns: int) -> None:
    if value is None:
        return
    if len(value) != NUM_FRAMES or any(len(row) != columns for row in value):
        raise ValueError(f"{name} must be {NUM_FRAMES} x {columns}")


def frame_dim(pose_indices: Sequence[int]) -> int:
    return InputLayout.for_pose_indices(pose_indices).frame_dim


class PreprocessCase(_StrictModel):
    """One sequence: features (num_frames x FRAME_DIM, float32) are the preprocessed coordinates
    then presence; model_input is the same after the float16 storage step training data took."""

    name: str
    description: str
    rows: GoldenRows
    info: GoldenInfo
    features: list[list[float]] | None
    presence: list[list[float]] | None
    model_input: list[list[float]] | None

    @model_validator(mode="after")
    def _shapes(self) -> Self:
        arrays = (self.features, self.presence, self.model_input)
        if self.info.excluded != all(array is None for array in arrays):
            raise ValueError("excluded cases have no arrays; kept cases have all three")
        if self.info.excluded != (self.info.reason is not None):
            raise ValueError("reason must be set exactly when the case is excluded")
        _check_matrix("presence", self.presence, 2)
        return self

    def check_width(self, width: int) -> None:
        _check_matrix(f"{self.name} features", self.features, width)
        _check_matrix(f"{self.name} model_input", self.model_input, width)


class GoldenPreprocessConfig(_StrictModel):
    num_frames: int
    use_z: Literal[False]
    pose_indices: list[int]
    pose_swap_pairs: list[tuple[int, int]]
    min_hand_frames: int


class SnapCase(_StrictModel):
    x: float
    snapped: float


class PreprocessGolden(_StrictModel):
    schema_version: Literal[2]
    description: str
    float_encoding: str
    config: GoldenPreprocessConfig
    snap_x_cases: list[SnapCase]
    cases: list[PreprocessCase] = Field(min_length=12)

    @model_validator(mode="after")
    def _widths(self) -> Self:
        width = frame_dim(self.config.pose_indices)
        for case in self.cases:
            case.check_width(width)
        return self


class ModelCase(_StrictModel):
    sequence_id: int
    label: int
    input: list[list[float]]
    logits: list[float]
    embedding: list[float] | None = None


class ModelGolden(_StrictModel):
    schema_version: Literal[2]
    description: str
    float_encoding: str
    source: DataSource
    split: Literal["val"]
    variant: WebVariant
    model_sha256: str = Field(pattern=SHA256_PATTERN)
    num_classes: int
    frame_dim: int
    cases: list[ModelCase] = Field(min_length=MODEL_CASES, max_length=MODEL_CASES)

    @model_validator(mode="after")
    def _widths(self) -> Self:
        if any(len(case.logits) != self.num_classes for case in self.cases):
            raise ValueError(f"every case needs {self.num_classes} logits")
        for case in self.cases:
            _check_matrix(f"sequence {case.sequence_id} input", case.input, self.frame_dim)
        return self


def float32_lists(array: npt.ArrayLike) -> Any:
    """Nested lists of the shortest decimals that round back to the same float32 values."""
    values = np.asarray(array, dtype=np.float32)
    short = np.array([float(str(value)) for value in values.ravel()], dtype=np.float64)
    # Decimal -> float64 -> float32 can in principle round differently from decimal -> float32;
    # any such value is written as its exact float64 instead.
    mismatch = short.astype(np.float32) != values.ravel()
    short[mismatch] = values.ravel()[mismatch].astype(np.float64)
    return short.reshape(values.shape).tolist()


def _nullable(values: npt.NDArray[np.float64]) -> list[float | None]:
    return [None if np.isnan(value) else float(value) for value in values.tolist()]


def rows_to_json(df: pd.DataFrame) -> GoldenRows:
    return GoldenRows(
        frame=[int(value) for value in df["frame"]],
        type=[str(value) for value in df["type"]],
        landmark_index=[int(value) for value in df["landmark_index"]],
        x=_nullable(df["x"].to_numpy(dtype=np.float64)),
        y=_nullable(df["y"].to_numpy(dtype=np.float64)),
        z=_nullable(df["z"].to_numpy(dtype=np.float64)),
    )


def rows_from_json(rows: GoldenRows) -> pd.DataFrame:
    def column(values: list[float | None]) -> FloatArray:
        return np.array([np.nan if value is None else value for value in values], dtype=np.float64)

    return pd.DataFrame(
        {
            "frame": np.asarray(rows.frame, dtype=np.int64),
            "type": rows.type,
            "landmark_index": np.asarray(rows.landmark_index, dtype=np.int64),
            "x": column(rows.x),
            "y": column(rows.y),
            "z": column(rows.z),
        }
    )


BlockBuilder = Callable[[int, np.random.Generator, str], dict[str, FloatArray]]
FrameBuilder = Callable[[Mapping[str, FloatArray]], pd.DataFrame]


class RawCase(NamedTuple):
    name: str
    description: str
    rows: pd.DataFrame


def synthetic_cases(
    landmark_blocks: BlockBuilder, long_frame: FrameBuilder, seed: int = GOLDEN_SEED
) -> list[RawCase]:
    """Long-format sequences covering every preprocessing branch.

    ``landmark_blocks`` and ``long_frame`` are the test suite's synthetic fixture builders
    (tests/conftest.py). Face rows are dropped except in one case, to keep the file small;
    coordinates are rounded to INPUT_DECIMALS so the JSON stores them exactly.
    """
    rng = np.random.default_rng(seed)
    nan = np.nan

    def blocks(n_frames: int, case: str = "both_hands", face: bool = False) -> dict[str, Any]:
        built = dict(landmark_blocks(n_frames, rng, case))
        if not face:
            del built["face"]
        return built

    cases: list[tuple[str, str, Mapping[str, FloatArray]]] = []

    b = blocks(45, "right_hand_only")
    cases.append(("right_only_longer", "Right hand only for 45 frames, downsampled to 32.", b))

    b = blocks(12, "both_hands")
    cases.append(
        (
            "both_hands_tie_shorter",
            "Both hands in all 12 frames: a presence tie, so no mirroring (ties go to the right "
            "hand); upsampled to 32.",
            b,
        )
    )

    b = blocks(30, "left_hand_only")
    cases.append(("left_only_mirrored", "Left hand only: mirrored to the right.", b))

    b = blocks(36, "both_hands")
    b["right_hand"][5:17] = nan
    cases.append(
        (
            "left_dominant_both_hands",
            "Both hands, right hand missing in frames 5-16 (left 36 vs right 24 frames): mirrored, "
            "hand blocks, pose pairs and presence columns swapped.",
            b,
        )
    )

    b = blocks(30, "both_hands")
    b["left_hand"][20:] = nan
    b["right_hand"][:10] = nan
    cases.append(
        (
            "tie_with_edge_gaps",
            "Left hand in frames 0-19, right hand in frames 10-29: a 20-20 tie, not mirrored; "
            "each hand is NaN (then 0) outside its own span, without extrapolation.",
            b,
        )
    )

    b = blocks(25, "pose_missing")
    cases.append(
        (
            "pose_nan_hand_fallback",
            "Pose NaN in every frame: no shoulders, so the hands' bounding box normalises.",
            b,
        )
    )

    b = blocks(20, "left_hand_only")
    del b["pose"]
    cases.append(
        (
            "pose_rows_absent_mirrored_fallback",
            "No pose rows at all and only the left hand: mirrored, hand_fallback normalisation.",
            b,
        )
    )

    b = blocks(40, "right_hand_only")
    b["right_hand"][12:19] = nan
    cases.append(
        (
            "hand_less_frames_dropped",
            "Right hand missing in frames 12-18: those frames are dropped before resampling "
            "(t_raw 40, t_kept 33).",
            b,
        )
    )

    b = blocks(28, "both_hands")
    b["right_hand"][8:14, :5] = nan
    b["left_hand"][20:, 10] = nan
    cases.append(
        (
            "partial_hand_landmarks",
            "Right hand has 5 NaN landmarks in frames 8-13 and the left hand 1 in frames 20-27: "
            "those hands count as absent there, but their finite points are kept and resampled.",
            b,
        )
    )

    b = blocks(33, "partial_hands")
    cases.append(
        (
            "fixture_partial_hands",
            "The fixture's partial_hands case: no left hand, right hand missing in the first "
            "third of the frames.",
            b,
        )
    )

    b = blocks(30, "right_hand_only")
    b["pose"][:6, 11] = nan
    b["pose"][24:, 12] = nan
    b["pose"][10:12, [11, 12], 1] = nan
    cases.append(
        (
            "shoulders_partly_missing",
            "Left shoulder NaN in frames 0-5, right shoulder in frames 24-29 and both shoulder y "
            "values in frames 10-11: medians use only finite values, per axis.",
            b,
        )
    )

    b = blocks(22, "right_hand_only")
    b["pose"][:, 12] = nan
    cases.append(
        (
            "one_shoulder_never_visible",
            "Right shoulder NaN in every frame: no frame has both shoulders, so hand_fallback.",
            b,
        )
    )

    b = blocks(18, "right_hand_only")
    keep = np.zeros(18, dtype=bool)
    keep[[2, 9, 15]] = True
    b["right_hand"][~keep] = nan
    cases.append(
        (
            "exactly_min_hand_frames",
            "A hand in exactly 3 frames (the minimum): kept and upsampled from 3 to 32 frames.",
            b,
        )
    )

    b = blocks(20, "right_hand_only")
    keep = np.zeros(20, dtype=bool)
    keep[[4, 11]] = True
    b["right_hand"][~keep] = nan
    cases.append(
        (
            "excluded_insufficient_hand_frames",
            "A hand in only 2 frames: excluded as insufficient_hand_frames.",
            b,
        )
    )

    b = blocks(16, "coincident_shoulders")
    cases.append(
        (
            "excluded_degenerate_shoulders",
            "Both shoulders at the same point: shoulder distance 0, excluded as degenerate_scale.",
            b,
        )
    )

    b = blocks(10, "right_hand_only")
    del b["pose"]
    b["right_hand"][..., 0] = 0.4
    b["right_hand"][..., 1] = 0.5
    cases.append(
        (
            "excluded_degenerate_hand_fallback",
            "No pose and every hand point at one position: zero hand spread, excluded as "
            "degenerate_scale under hand_fallback.",
            b,
        )
    )

    b = blocks(8, "both_hands", face=True)
    b["right_hand"][[1, 6]] = nan
    cases.append(
        (
            "face_rows_shuffled_frame_gaps",
            "Includes face rows (dropped), rows in random order and non-consecutive frame "
            "numbers; left 8 vs right 6 frames, so mirrored.",
            b,
        )
    )

    b = blocks(24, "right_hand_only")
    b["right_hand"][..., 0] += 0.75
    b["pose"][:, [15, 16], 1] -= 0.8
    cases.append(
        (
            "coordinates_outside_unit_square",
            "Hand x beyond 1.0 and wrist y below 0.0, as when a hand leaves the frame: such "
            "values are used as they are.",
            b,
        )
    )

    raw = []
    for name, description, case_blocks in cases:
        df = long_frame(case_blocks).loc[:, ["frame", "type", "landmark_index", "x", "y", "z"]]
        if name == "face_rows_shuffled_frame_gaps":
            frames = np.sort(df["frame"].unique())
            remap = dict(zip(frames, 100 + np.cumsum(rng.integers(1, 5, frames.size)), strict=True))
            df = df.assign(frame=df["frame"].map(remap)).sample(frac=1.0, random_state=seed)
        df = df.assign(
            frame=df["frame"].astype(np.int64),
            landmark_index=df["landmark_index"].astype(np.int64),
            x=df["x"].round(INPUT_DECIMALS),
            y=df["y"].round(INPUT_DECIMALS),
            z=df["z"].round(INPUT_DECIMALS),
        ).reset_index(drop=True)
        raw.append(RawCase(name, description, df))
    return raw


def pack_features(features: Float32Array, presence: Float32Array) -> Float32Array:
    """(T, L, 2) coordinates and (T, 2) presence -> (T, 2L + 2) in the model layout."""
    coords = features.shape[1] * features.shape[2]
    packed = np.empty((features.shape[0], coords + 2), dtype=np.float32)
    packed[:, :coords] = features.reshape(features.shape[0], coords)
    packed[:, coords:] = presence
    return packed


def golden_case(case: RawCase, settings: GoldenPreprocessConfig) -> PreprocessCase:
    result = preprocess_sequence(
        case.rows,
        num_frames=settings.num_frames,
        use_z=False,
        pose_indices=settings.pose_indices,
        min_hand_frames=settings.min_hand_frames,
    )
    info = GoldenInfo.model_validate(dict(result.info))
    if result.features is None or result.hand_presence is None:
        return PreprocessCase(
            name=case.name,
            description=case.description,
            rows=rows_to_json(case.rows),
            info=info,
            features=None,
            presence=None,
            model_input=None,
        )
    stored = to_storage(0, result)
    if stored.features is None or stored.presence is None:
        raise ValueError(f"{case.name}: excluded by the float16 storage step")
    model_input = to_model_input(
        stored.features[None], stored.presence[None], num_landmarks=stored.features.shape[1]
    )[0]
    return PreprocessCase(
        name=case.name,
        description=case.description,
        rows=rows_to_json(case.rows),
        info=info,
        features=float32_lists(pack_features(result.features, result.hand_presence)),
        presence=float32_lists(result.hand_presence),
        model_input=float32_lists(model_input),
    )


SNAP_X_INPUTS: tuple[float, ...] = (
    0.5 + 0.5 / X_GRID,  # exactly half-way: rounds to even (down)
    0.25 + 1.5 / X_GRID,  # exactly half-way: rounds to even (up)
    0.7 + 2.5 / X_GRID,
    0.123456789,
    1.0 - 2.0**-40,
    1e-12,
    -0.0,
    -0.3 + 0.5 / X_GRID,
    1.25,
)


def build_preprocess_golden(
    landmark_blocks: BlockBuilder,
    long_frame: FrameBuilder,
    min_hand_frames: int,
    pose_indices: Sequence[int] = POSE_INDICES,
) -> PreprocessGolden:
    settings = GoldenPreprocessConfig(
        num_frames=NUM_FRAMES,
        use_z=False,
        pose_indices=list(pose_indices),
        pose_swap_pairs=list(pose_swap_pairs(pose_indices)),
        min_hand_frames=min_hand_frames,
    )
    raw = synthetic_cases(landmark_blocks, long_frame)
    cases = [golden_case(case, settings) for case in raw]
    for case, built in zip(raw, cases, strict=True):
        if not rows_from_json(built.rows).equals(case.rows):
            raise AssertionError(f"{case.name}: rows do not survive the JSON encoding exactly")
    snapped = snap_x(np.asarray(SNAP_X_INPUTS, dtype=np.float64))
    return PreprocessGolden(
        schema_version=GOLDEN_SCHEMA_VERSION,
        description=(
            "Python preprocess_sequence outputs for synthetic long-format sequences built with "
            "the test suite's fixture logic; written by scripts/export_web.py."
        ),
        float_encoding=FLOAT_ENCODING,
        config=settings,
        snap_x_cases=[
            SnapCase(x=float(x), snapped=float(s))
            for x, s in zip(SNAP_X_INPUTS, snapped, strict=True)
        ],
        cases=cases,
    )


def model_case_rows(n_val: int, n_cases: int = MODEL_CASES) -> npt.NDArray[np.intp]:
    """Evenly spaced val rows, so the cases span the split rather than its first sequences."""
    if n_val < n_cases:
        raise ValueError(f"val has {n_val} sequences, {n_cases} model cases are needed")
    return np.unique(np.linspace(0, n_val - 1, n_cases).round().astype(np.intp))


def build_model_golden(
    val: SplitArrays,
    logits: Float32Array,
    *,
    source: DataSource,
    variant: WebVariant,
    model_sha256: str,
    embeddings: Float32Array | None = None,
) -> ModelGolden:
    rows = model_case_rows(len(val.y))
    return ModelGolden(
        schema_version=GOLDEN_SCHEMA_VERSION,
        description=(
            "Processed val inputs (num_frames x FRAME_DIM) and the logits Python onnxruntime "
            "gives for them with the exported model.onnx; written by scripts/export_web.py."
        ),
        float_encoding=FLOAT_ENCODING,
        source=source,
        split="val",
        variant=variant,
        model_sha256=model_sha256,
        num_classes=int(logits.shape[1]),
        frame_dim=int(val.X.shape[2]),
        cases=[
            ModelCase(
                sequence_id=int(val.sequence_id[row]),
                label=int(val.y[row]),
                input=float32_lists(val.X[row]),
                logits=float32_lists(logits[row]),
                embedding=None if embeddings is None else float32_lists(embeddings[row]),
            )
            for row in rows
        ],
    )


def write_golden(path: Path, golden: BaseModel) -> None:
    """Compact JSON with one case per line, written through a temporary name."""
    payload = golden.model_dump(mode="json")
    cases = payload.pop("cases")
    header = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    lines = [json.dumps(case, separators=(",", ":"), allow_nan=False) for case in cases]
    text = header[:-1] + ',"cases":[\n' + ",\n".join(lines) + "\n]}\n"
    _write_text(path, text)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    temporary.replace(path)


# ---------------------------------------------------------------------------------------------
# Export


DEFAULT_WORK_DIR = Path("results/web_export")


class ExportResult(NamedTuple):
    meta: WebMeta
    model_path: Path
    meta_path: Path
    golden_paths: dict[str, Path]


WasmVerifier = Callable[["ExportResult"], WasmCheck]


def read_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    return payload


class RunInfo(NamedTuple):
    directory: Path
    summary: dict[str, Any]
    seed: int
    checkpoint: Path
    sha256: str

    @property
    def key(self) -> str:
        """<experiment>/seed_<seed>, as quant and results directories name runs."""
        return f"{self.directory.parent.name}/{self.directory.name}"


def run_info(directory: Path) -> RunInfo:
    summary = read_json(directory / SUMMARY_FILE)
    checkpoint = directory / CHECKPOINT_FILE
    return RunInfo(directory, summary, int(summary["seed"]), checkpoint, sha256_file(checkpoint))


def ensemble_members(run_directory: Path) -> list[RunInfo]:
    """Every seed of the run's experiment that has a summary and a checkpoint, by seed; they
    must share one split file, so val is the same data for every member."""
    members = [
        run_info(directory)
        for directory in sorted(run_directory.parent.glob("seed_*"))
        if (directory / SUMMARY_FILE).is_file() and (directory / CHECKPOINT_FILE).is_file()
    ]
    members.sort(key=lambda member: member.seed)
    if len(members) < 2:
        raise ValueError(
            f"{run_directory.parent} has {len(members)} seed(s) with {CHECKPOINT_FILE}; an "
            "ensemble needs at least two"
        )
    splits = {sha256_file(m.directory / m.summary["splits_file"]) for m in members}
    if len(splits) != 1:
        raise ValueError(f"the seeds of {run_directory.parent} use different split files")
    return members


def ensemble_sha256(shas: Sequence[str]) -> str:
    """One digest for an ensemble's member checkpoints, in member order."""
    return hashlib.sha256("\n".join(shas).encode("ascii")).hexdigest()


def source_sha256(members: Sequence[RunInfo]) -> str:
    if len(members) == 1:
        return members[0].sha256
    return ensemble_sha256([member.sha256 for member in members])


def model_metadata(
    variant: str, members: Sequence[RunInfo], layout: InputLayout, label_map: Mapping[int, str]
) -> dict[str, str]:
    metadata = {
        META_VARIANT: variant,
        META_SCHEMA_VERSION: str(VARIANT_SCHEMA_VERSIONS[variant]),
        META_SOURCE_SHA256: source_sha256(members),
        META_LABEL_MAP: json.dumps({str(index): sign for index, sign in label_map.items()}),
        META_INPUT_LAYOUT: json.dumps(layout.to_dict()),
    }
    if len(members) > 1:
        metadata[META_ENSEMBLE_MEMBERS] = json.dumps(
            [{"run": member.key, "sha256": member.sha256} for member in members]
        )
    return metadata


def build_fp32(
    members: Sequence[RunInfo], path: Path, layout: InputLayout, label_map: Mapping[int, str]
) -> None:
    """fp32 ONNX (opset 17) of one checkpoint, or of the seed ensemble, with metadata."""
    scratch = path.with_name(f".{path.stem}.export.onnx")
    try:
        if len(members) == 1:
            model = load_checkpoint(members[0].checkpoint).model
            export_onnx(model, scratch, layout=layout, embedding=supports_embedding(model))
        else:
            export_ensemble_onnx([member.checkpoint for member in members], scratch)
        save_with_metadata(
            onnx.load(str(scratch)), path, model_metadata("fp32", members, layout, label_map)
        )
    finally:
        scratch.unlink(missing_ok=True)


def _scored(path: Path, val: SplitArrays) -> CandidateInput:
    logits = OrtModel(path, intra_op_threads=1).run(val.X)
    if not np.isfinite(logits).all():
        return CandidateInput(None, None, None, "non-finite val outputs")
    return CandidateInput(path, path.stat().st_size, top_k_accuracy(logits, val.y, 1), None)


SINGLE_RUN_NOTE = (
    "not built: no quant_report.json for this run (scripts/quantize.py builds int8_static for "
    "v1 runs and skips feature set v2)"
)
ENSEMBLE_INT8_STATIC_NOTE = (
    "not built: int8_static keeps the FeatureBlock in float32 by its '/features/' node scope, "
    "which ensemble members do not have ('/members.<n>/features/')"
)


def build_candidates(
    members: Sequence[RunInfo],
    work_dir: Path,
    layout: InputLayout,
    label_map: Mapping[int, str],
    val: SplitArrays,
) -> dict[WebVariant, CandidateInput]:
    """fp32, fp16 and int8_dynamic of a single run without a quant report (feature set v2) or
    of an ensemble, built with the scripts/quantize.py builders into ``work_dir`` and scored on
    val. int8_static needs calibration data and the quant report's checks, so only
    scripts/quantize.py builds it."""
    work_dir.mkdir(parents=True, exist_ok=True)
    fp32 = work_dir / variant_file("fp32")
    build_fp32(members, fp32, layout, label_map)
    inputs: dict[WebVariant, CandidateInput] = {"fp32": _scored(fp32, val)}
    builders: dict[WebVariant, Callable[[Path], onnx.ModelProto]] = {
        "fp16": lambda _: convert_fp16(fp32, keep_feature_block_fp32=False),
        "int8_dynamic": lambda path: quantize_int8_dynamic(fp32, path),
    }
    for variant, build in builders.items():
        path = work_dir / variant_file(variant)
        try:
            save_with_metadata(
                build(path), path, model_metadata(variant, members, layout, label_map)
            )
            inputs[variant] = _scored(path, val)
        except Exception as error:  # Recorded as the candidate's note, never raised.
            inputs[variant] = CandidateInput(
                None, None, None, f"build failed: {type(error).__name__}: {error}"
            )
    inputs["int8_static"] = CandidateInput(
        None, None, None, ENSEMBLE_INT8_STATIC_NOTE if len(members) > 1 else SINGLE_RUN_NOTE
    )
    return inputs


def make_selection(
    candidates: Sequence[VariantCandidate],
    order: Sequence[WebVariant],
    fp32_val_top1: float,
    source: Literal["quant_report", "export_build"],
    *,
    chosen: WebVariant,
    forced: bool,
    fallbacks: Sequence[Fallback],
    checks: Mapping[str, WasmCheck],
) -> VariantSelection:
    failed = {fallback.variant for fallback in fallbacks}
    expected = next((variant for variant in order if variant not in failed), order[0])
    return VariantSelection(
        rule=VARIANT_RULE,
        max_drop_pp=VARIANT_RULE_MAX_DROP_PP,
        source=source,
        fp32_val_top1=fp32_val_top1,
        candidates=[
            c.model_copy(update={"wasm": checks.get(c.variant, c.wasm)}) for c in candidates
        ],
        order=list(order),
        recommended=order[0],
        chosen=chosen,
        forced=forced,
        fallbacks=list(fallbacks),
        follows_rule=chosen == (order[0] if forced else expected),
    )


def write_static_goldens(
    golden_dir: Path,
    landmark_blocks: BlockBuilder,
    long_frame: FrameBuilder,
    min_hand_frames: int,
) -> dict[str, Path]:
    """The golden files that do not depend on the exported model: preprocessing for the v1 and
    v2 selections, and the aspect cases."""
    paths = {}
    for name, pose_indices in (
        (PREPROCESS_CASES_FILE, POSE_INDICES),
        (PREPROCESS_CASES_V2_FILE, V2_GOLDEN_POSE_INDICES),
    ):
        golden = build_preprocess_golden(landmark_blocks, long_frame, min_hand_frames, pose_indices)
        write_golden(golden_dir / name, golden)
        paths[name] = golden_dir / name
    aspect = build_aspect_golden(landmark_blocks)
    _write_text(
        golden_dir / ASPECT_CASES_FILE,
        json.dumps(aspect.model_dump(mode="json"), separators=(",", ":")) + "\n",
    )
    paths[ASPECT_CASES_FILE] = golden_dir / ASPECT_CASES_FILE
    return paths


def recordings_app_block(meta: WebMeta) -> dict[str, Any]:
    return {
        "model_name": meta.model_name,
        "variant": meta.variant,
        "run": meta.run,
        "model_sha256": meta.model_sha256,
        "feature_set": meta.feature_set,
        "pose_indices": meta.pose_indices,
        "pose_swap_pairs": meta.preprocessing.pose_swap_pairs,
        "num_frames": meta.num_frames,
        "min_hand_frames": meta.preprocessing.min_hand_frames,
        "threshold": meta.threshold.threshold,
        "num_classes": meta.num_classes,
        "r_train": None if meta.aspect is None else meta.aspect.r_train,
    }


def _write_meta(path: Path, meta: WebMeta) -> None:
    _write_text(path, json.dumps(meta.model_dump(mode="json"), indent=2) + "\n")


def export_web(
    run_directory: Path,
    quant_root: Path,
    processed_dir: Path,
    *,
    out_dir: Path,
    golden_dir: Path,
    landmark_blocks: BlockBuilder,
    long_frame: FrameBuilder,
    variant: WebVariant | None = None,
    target_precision: float = DEFAULT_TARGET_PRECISION,
    data_source: DataSource = "asl-signs",
    ensemble: bool = False,
    work_dir: Path = DEFAULT_WORK_DIR,
    aspect_report: Path | None = None,
    wasm_verifier: WasmVerifier | None = None,
) -> ExportResult:
    """Export the variant VARIANT_RULE picks (or ``variant``) of one run, or of the seed
    ensemble of its experiment, to ``out_dir``, and the golden vectors to ``golden_dir``.

    Variants come from the run's quant_report.json when there is one (they must have been built
    from the run's current best.pt); otherwise they are built into ``work_dir``. With
    ``wasm_verifier`` each exported variant is checked in onnxruntime-web; a failure is
    recorded and the next eligible variant is exported instead.
    """
    if variant is not None and variant not in WEB_VARIANTS:
        raise ValueError(f"variant must be one of {WEB_VARIANTS}, got {variant!r}")
    primary = run_info(run_directory)
    feature_set: FeatureSetName = primary.summary.get("feature_set", "v1")
    layout = checkpoint_layout(primary.checkpoint)
    members = ensemble_members(run_directory) if ensemble else [primary]
    experiment = run_directory.parent.name
    run_key = f"{experiment}/ensemble" if ensemble else primary.key
    label_map = {int(k): v for k, v in read_json(run_directory / "label_map.json").items()}
    labels = [label_map[index] for index in range(len(label_map))]
    settings = preprocessing_settings(processed_dir, layout)
    aspect = aspect_settings(processed_dir, aspect_report, data_source)
    val = load_eval_split(processed_dir, run_directory / primary.summary["splits_file"])

    quant_dir = quant_root / primary.key  # <quant root>/<experiment>/seed_<seed>.
    source: Literal["quant_report", "export_build"]
    if not ensemble and (quant_dir / QUANT_REPORT_FILE).is_file():
        report = read_json(quant_dir / QUANT_REPORT_FILE)
        inputs = candidates_from_report(report, quant_dir, primary.sha256)
        source = "quant_report"
    else:
        inputs = build_candidates(members, work_dir / run_key, layout, label_map, val)
        source = "export_build"
    candidates, order, fp32_top1 = rank_candidates(inputs)
    if variant is not None and inputs[variant].path is None:
        raise ValueError(f"variant {variant} is not available: {inputs[variant].note}")
    attempts: list[WebVariant] = [variant] if variant is not None else list(order)

    golden_paths = write_static_goldens(
        golden_dir, landmark_blocks, long_frame, settings.min_hand_frames
    )
    fallbacks: list[Fallback] = []
    checks: dict[str, WasmCheck] = {}
    for attempt in attempts:
        selection = make_selection(
            candidates,
            order,
            fp32_top1,
            source,
            chosen=attempt,
            forced=variant is not None,
            fallbacks=fallbacks,
            checks=checks,
        )
        path = inputs[attempt].path
        assert path is not None  # attempts hold available variants only.
        result = _export_file(
            path,
            out_dir=out_dir,
            golden_dir=golden_dir,
            golden_paths=golden_paths,
            members=members,
            layout=layout,
            label_map=label_map,
            val=val,
            target_precision=target_precision,
            landmark_blocks=landmark_blocks,
            fields={
                "model_name": primary.summary["model"],
                "variant": attempt,
                "data_source": data_source,
                "run": run_key,
                "seed": None if ensemble else primary.seed,
                "seeds": [member.seed for member in members],
                "ensemble": ensemble,
                "feature_set": feature_set,
                "labels": labels,
                "preprocessing": settings,
                "aspect": aspect,
                "variant_selection": selection,
            },
        )
        if wasm_verifier is None:
            return result
        check = wasm_verifier(result)
        checks[attempt] = check
        if check.status != "failed":
            meta = result.meta.model_copy(
                update={
                    "variant_selection": make_selection(
                        candidates,
                        order,
                        fp32_top1,
                        source,
                        chosen=attempt,
                        forced=variant is not None,
                        fallbacks=fallbacks,
                        checks=checks,
                    )
                }
            )
            _write_meta(result.meta_path, meta)
            return result._replace(meta=meta)
        fallbacks.append(Fallback(variant=attempt, reason=check.detail or "failed"))
    raise RuntimeError(
        "onnxruntime-web could not run any eligible variant: "
        + "; ".join(f"{f.variant}: {f.reason}" for f in fallbacks)
    )


def _export_file(
    source: Path,
    *,
    out_dir: Path,
    golden_dir: Path,
    golden_paths: dict[str, Path],
    members: Sequence[RunInfo],
    layout: InputLayout,
    label_map: Mapping[int, str],
    val: SplitArrays,
    target_precision: float,
    landmark_blocks: BlockBuilder,
    fields: dict[str, Any],
) -> ExportResult:
    """Copy one variant file to ``out_dir`` and write meta.json, the model golden and the
    recording example for it."""
    variant = fields["variant"]
    metadata = OrtModel(source, intra_op_threads=1).metadata
    if metadata.get(META_SOURCE_SHA256) != source_sha256(members):
        raise ValueError(f"{source} was not built from {members[0].checkpoint}")
    if metadata.get(META_VARIANT) != variant:
        raise ValueError(f"{source} records variant {metadata.get(META_VARIANT)!r}")
    if label_map_from_metadata(metadata, source, layout.to_dict()) != dict(label_map):
        raise ValueError(f"{source} label map differs from the run's label_map.json")

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / MODEL_FILE
    temporary = model_path.with_name(f"{model_path.name}.tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(model_path)
    model_sha256 = sha256_file(model_path)
    model = OrtModel(model_path, intra_op_threads=1)
    logits = model.run(val.X)
    embedding_dim = model.embedding_dim
    embedding_block = None
    golden_embeddings = None
    if embedding_dim is not None:
        embedding_block = EmbeddingBlock(
            output_name=EMBEDDING_NAME,
            dim=embedding_dim,
            source="the stages' output averaged over time, the vector the classifier head reads",
        )
        rows = model_case_rows(len(val.y))
        _, case_embeddings = model.run_with_embedding(val.X[rows])
        golden_embeddings = np.zeros((len(val.y), embedding_dim), dtype=np.float32)
        golden_embeddings[rows] = case_embeddings

    meta = WebMeta(
        schema_version=WEB_META_SCHEMA_VERSION,
        num_classes=len(fields["labels"]),
        num_frames=layout.num_frames,
        landmark_order=landmark_names(layout.pose_indices),
        pose_indices=list(layout.pose_indices),
        input_layout=InputLayoutBlock.model_validate(layout.to_dict()),
        opset=onnx_opset(model_path),
        model_file=MODEL_FILE,
        model_bytes=model_path.stat().st_size,
        model_sha256=model_sha256,
        source_checkpoints=[
            SourceCheckpoint(run=member.key, sha256=member.sha256) for member in members
        ],
        threshold=threshold_from_logits(logits, val.y, target_precision),
        calibration=calibration_from_logits(logits, val.y),
        features_spec_version=spec.SUPPORTED_VERSION,
        features_spec_sha256=sha256_file(spec.spec_path()),
        tool_versions=tool_versions(),
        embedding=embedding_block,
        **fields,
    )
    meta_path = out_dir / META_FILE
    _write_meta(meta_path, meta)

    paths = dict(golden_paths)
    model_golden = build_model_golden(
        val,
        logits,
        source=meta.data_source,
        variant=variant,
        model_sha256=model_sha256,
        embeddings=golden_embeddings,
    )
    write_golden(golden_dir / MODEL_CASES_FILE, model_golden)
    paths[MODEL_CASES_FILE] = golden_dir / MODEL_CASES_FILE
    example = example_recordings(
        landmark_blocks, recordings_app_block(meta), meta.labels, model.run
    )
    _write_text(
        golden_dir / RECORDING_EXAMPLE_FILE,
        json.dumps(example.model_dump(mode="json"), separators=(",", ":")) + "\n",
    )
    paths[RECORDING_EXAMPLE_FILE] = golden_dir / RECORDING_EXAMPLE_FILE
    return ExportResult(meta, model_path, meta_path, paths)
