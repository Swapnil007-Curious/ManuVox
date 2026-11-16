"""Golden files for the web test suite's aspect and recorder tests, written by
scripts/export_web.py: aspect_cases.json (calibration, visibility and per-frame correction cases
with the preprocessing that follows, for the v1 and v2 landmark selections) and
recording_example.json (a recordings file in the "Download recordings" schema)."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from edgevision.data.landmarks import DEFAULT_POSE_INDICES, N_HAND, N_POSE, pose_swap_pairs
from edgevision.data.preprocess import PreprocessResult
from edgevision.live.aspect import (
    DEFAULT_Y_MID,
    K_MAX,
    K_MIN,
    MIN_VISIBILITY,
    RawFrames,
    calibrate,
    calibration_frames,
    correct_frames,
    frame_ratios,
)
from edgevision.live.recordings import (
    RECORDINGS_FORMAT,
    RECORDINGS_SCHEMA_VERSION,
    RecordingsFile,
    replay,
)

FloatArray = npt.NDArray[np.float64]
BlockBuilder = Callable[[int, np.random.Generator, str], dict[str, FloatArray]]

ASPECT_SCHEMA_VERSION: Literal[1] = 1
V2_POSE_INDICES: tuple[int, ...] = tuple(range(17))
GOLDEN_SEED = 2027
DECIMALS = 6


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


Values = list[float | None]


class FrameBlocks(_Strict):
    """Per-frame flat [x0, y0, z0, ...] lists (null for a missing block or value)."""

    left_hand: list[Values | None]
    right_hand: list[Values | None]
    pose: list[Values | None]


class SelectionResult(_Strict):
    pose_indices: list[int]
    pose_swap_pairs: list[tuple[int, int]]
    info: dict[str, Any]
    features: list[list[float]] | None


class FrameCase(_Strict):
    name: str
    description: str
    k: float
    frames: FrameBlocks
    corrected: FrameBlocks
    results: list[SelectionResult]


class CalibrationCase(_Strict):
    name: str
    ratios: list[float | None]
    r_train: float
    k: float
    r_live: float | None
    frames: int
    status: Literal["calibrated", "no_shoulders"]


class VisibilityCase(_Strict):
    name: str
    pose: Values
    visibility: Values | None
    usable: bool
    ratio: float | None


class AspectGolden(_Strict):
    schema_version: Literal[1]
    description: str
    k_min: float
    k_max: float
    default_y_mid: float
    min_visibility: float
    calibration_cases: list[CalibrationCase]
    visibility_cases: list[VisibilityCase]
    frame_cases: list[FrameCase]


def _nullable(values: npt.ArrayLike) -> list[float | None]:
    return [None if np.isnan(v) else float(v) for v in np.asarray(values, dtype=np.float64).ravel()]


def _blocks_json(block: FloatArray) -> list[Values | None]:
    """(T, n, 3) -> per-frame flat lists; an all-NaN frame becomes null."""
    return [None if np.isnan(frame).all() else _nullable(frame) for frame in block]


def frame_blocks(frames: RawFrames) -> FrameBlocks:
    return FrameBlocks(
        left_hand=_blocks_json(frames.left_hand),
        right_hand=_blocks_json(frames.right_hand),
        pose=_blocks_json(frames.pose),
    )


def _float32_lists(array: npt.ArrayLike) -> list[list[float]]:
    values = np.asarray(array, dtype=np.float32)
    return [[float(str(v)) for v in row] for row in values]


def _packed(result: PreprocessResult) -> list[list[float]] | None:
    if result.features is None or result.hand_presence is None:
        return None
    flat = result.features.reshape(result.features.shape[0], -1)
    return _float32_lists(np.concatenate([flat, result.hand_presence], axis=1))


def _rounded(block: FloatArray) -> FloatArray:
    return np.asarray(np.round(block, DECIMALS), dtype=np.float64)


def _raw(blocks: Mapping[str, FloatArray]) -> RawFrames:
    n = next(iter(blocks.values())).shape[0]
    empty_hand = np.full((n, N_HAND, 3), np.nan)
    return RawFrames(
        _rounded(blocks.get("left_hand", empty_hand)),
        _rounded(blocks.get("right_hand", empty_hand)),
        _rounded(blocks.get("pose", np.full((n, N_POSE, 3), np.nan))),
    )


def _frame_case(name: str, description: str, frames: RawFrames, k: float) -> FrameCase:
    results = []
    for pose_indices in (DEFAULT_POSE_INDICES, V2_POSE_INDICES):
        result = replay(
            frames, k, pose_indices=list(pose_indices), num_frames=32, min_hand_frames=3
        )
        results.append(
            SelectionResult(
                pose_indices=list(pose_indices),
                pose_swap_pairs=list(pose_swap_pairs(pose_indices)),
                info=dict(result.info),
                features=_packed(result),
            )
        )
    return FrameCase(
        name=name,
        description=description,
        k=k,
        frames=frame_blocks(frames),
        corrected=frame_blocks(correct_frames(frames, k)),
        results=results,
    )


def build_aspect_golden(landmark_blocks: BlockBuilder, seed: int = GOLDEN_SEED) -> AspectGolden:
    rng = np.random.default_rng(seed)

    def blocks(n: int, case: str) -> dict[str, FloatArray]:
        built = dict(landmark_blocks(n, rng, case))
        del built["face"]
        built["pose"][:, 0, 1] -= 0.25  # Nose above the shoulders, as in a real frame.
        return built

    frame_cases = []
    b = blocks(20, "right_hand_only")
    frame_cases.append(
        _frame_case("landscape_k_0_4", "k 0.4 compresses y about y_mid.", _raw(b), 0.4)
    )
    b = blocks(18, "both_hands")
    b["right_hand"][3:9] = np.nan
    frame_cases.append(
        _frame_case(
            "left_dominant_k_2_5",
            "k 2.5 stretches y; the left hand dominates, so mirroring swaps the selected pose "
            "pairs (3 for v1, 8 for v2).",
            _raw(b),
            2.5,
        )
    )
    b = blocks(16, "right_hand_only")
    b["pose"][:5, 11] = np.nan
    b["pose"][12:, 12, 1] = np.nan
    frame_cases.append(
        _frame_case(
            "shoulders_missing_some_frames",
            "Frames without both shoulder y values scale about y = 0.5 instead.",
            _raw(b),
            0.6,
        )
    )
    b = blocks(14, "right_hand_only")
    del b["pose"]
    frame_cases.append(
        _frame_case("no_pose", "No pose at all: every frame scales about 0.5.", _raw(b), 0.8)
    )
    b = blocks(15, "right_hand_only")
    frame_cases.append(
        _frame_case("identity_k_1", "k 1 leaves every value unchanged.", _raw(b), 1.0)
    )

    calibration_cases = [
        CalibrationCase(
            name=name,
            ratios=_nullable(ratios),
            r_train=r_train,
            **calibrate(ratios, r_train)._asdict(),
        )
        for name, ratios, r_train in (
            ("odd_count", np.array([0.9, 1.1, 1.0]), 0.4),
            ("even_count_with_nan", np.array([0.8, np.nan, 1.2, 1.0, 0.9]), 0.4),
            ("clamped_low", np.array([3.0, 3.2]), 0.4),
            ("clamped_high", np.array([0.02, 0.03]), 0.4),
            ("no_shoulders", np.array([np.nan, np.nan]), 0.4),
            ("empty", np.empty(0), 0.4),
        )
    ]

    visibility_cases = []
    base = blocks(1, "both_hands")["pose"]
    seen = np.full((1, N_POSE), 0.9)
    for name, pose, visibility in (
        ("visible", base, seen),
        ("no_visibility_scores", base, None),
        ("nose_low_visibility", base, np.where(np.arange(N_POSE) == 0, 0.3, 0.9)[None]),
        ("shoulder_outside_frame", _with(base, 11, 0, 1.05), seen),
        ("shoulder_missing", _with(base, 12, 1, np.nan), seen),
    ):
        pose = _rounded(pose)
        ratio = float(frame_ratios(pose)[0])
        visibility_cases.append(
            VisibilityCase(
                name=name,
                pose=_nullable(pose[0]),
                visibility=None if visibility is None else _nullable(visibility[0]),
                usable=bool(calibration_frames(pose, visibility)[0]),
                ratio=None if np.isnan(ratio) else ratio,
            )
        )

    return AspectGolden(
        schema_version=ASPECT_SCHEMA_VERSION,
        description=(
            "Python edgevision.live.aspect outputs (calibration, visibility rule, per-frame y "
            "correction) and preprocess_sequence on the corrected frames for the v1 and v2 "
            "selections; written by scripts/export_web.py."
        ),
        k_min=K_MIN,
        k_max=K_MAX,
        default_y_mid=DEFAULT_Y_MID,
        min_visibility=MIN_VISIBILITY,
        calibration_cases=calibration_cases,
        visibility_cases=visibility_cases,
        frame_cases=frame_cases,
    )


def _with(pose: FloatArray, point: int, axis: int, value: float) -> FloatArray:
    changed = pose.copy()
    changed[:, point, axis] = value
    return changed


# ---------------------------------------------------------------------------------------------
# Recording example


def example_recordings(
    landmark_blocks: BlockBuilder,
    app: Mapping[str, Any],
    labels: Sequence[str],
    classify: Callable[[npt.NDArray[np.float32]], npt.NDArray[np.float32]],
    seed: int = GOLDEN_SEED,
) -> RecordingsFile:
    """Two recordings in the download schema: one classified segment and one discarded for
    too few hand frames. ``app`` is the file's app block; ``classify`` maps (1, T, frame_dim)
    model input to logits."""
    rng = np.random.default_rng(seed)
    pose_indices = list(app["pose_indices"])
    recordings = []
    for index, (case, n_frames, hand_frames) in enumerate(
        (("right_hand_only", 24, None), ("right_hand_only", 12, 2))
    ):
        built = dict(landmark_blocks(n_frames, rng, case))
        del built["face"]
        if hand_frames is not None:
            built["right_hand"][hand_frames:] = np.nan
        frames = _raw(built)
        k = 0.5
        result = replay(
            frames,
            k,
            pose_indices=pose_indices,
            num_frames=int(app["num_frames"]),
            min_hand_frames=int(app["min_hand_frames"]),
        )
        t_ms = [round(1000.0 + 33.333 * t, 3) for t in range(n_frames)]
        visibility = [_nullable(np.full(N_POSE, 0.99)) for _ in range(n_frames)]
        hand = np.isfinite(frames.right_hand[..., :2]).all(axis=(1, 2))
        classified = result.features is not None and result.hand_presence is not None
        app_result: dict[str, Any] = {
            "status": "discarded",
            "reason": "insufficient_hand_frames",
            "info": None,
            "features": None,
            "model_input": None,
            "top5": [],
            "prediction": None,
            "max_probability": None,
            "accepted": None,
            "threshold": float(app["threshold"]),
        }
        if classified and result.features is not None and result.hand_presence is not None:
            flat = result.features.reshape(result.features.shape[0], -1)
            features = np.concatenate([flat, result.hand_presence], axis=1).astype(np.float32)
            stored = features.copy()
            stored[:, : flat.shape[1]] = flat.astype(np.float16).astype(np.float32)
            logits = classify(stored[None])[0].astype(np.float64)
            probabilities = np.exp(logits - logits.max())
            probabilities /= probabilities.sum()
            top = np.argsort(-probabilities, kind="stable")[:5]
            best = float(probabilities[top[0]])
            app_result.update(
                status="classified",
                reason=None,
                info=dict(result.info),
                features=_float32_lists(features),
                model_input=_float32_lists(stored),
                top5=[
                    {
                        "index": int(i),
                        "label": labels[int(i)],
                        "probability": float(probabilities[i]),
                    }
                    for i in top
                ],
                prediction=labels[int(top[0])],
                max_probability=best,
                accepted=best >= float(app["threshold"]),
            )
        recordings.append(
            {
                "id": index,
                "session": 0,
                "intended_label": labels[index % len(labels)],
                "intended_index": index % len(labels),
                "recorded_at": f"2026-11-07T10:00:{index:02d}.000Z",
                "camera": {"width": 640, "height": 480, "frame_rate": 30.0},
                "framing": "full",
                "crop": {"sx": 0, "sy": 0, "sw": 640, "sh": 480},
                "swap_hands": False,
                "pose_every": 2,
                "min_hand_frames": int(app["min_hand_frames"]),
                "calibration": {
                    "mode": "auto",
                    "k": k,
                    "r_live": 0.8,
                    "r_train": 0.4,
                    "frames": 60,
                    "status": "calibrated",
                },
                "segment": {
                    "capture_mode": "auto",
                    "reason": "hands_lowered",
                    "start_ms": t_ms[0],
                    "end_ms": t_ms[-1],
                    "frames": n_frames,
                    "hand_frames": int(hand.sum()),
                },
                "frames": {
                    "t_ms": t_ms,
                    "left_hand": _blocks_json(frames.left_hand),
                    "right_hand": _blocks_json(frames.right_hand),
                    "pose": _blocks_json(frames.pose),
                    "pose_visibility": visibility,
                    "pose_ran": [t % 2 == 0 for t in range(n_frames)],
                },
                "app_result": app_result,
            }
        )
    return RecordingsFile.model_validate(
        {
            "format": RECORDINGS_FORMAT,
            "schema_version": RECORDINGS_SCHEMA_VERSION,
            "created_at": "2026-11-07T10:01:00.000Z",
            "user_agent": "synthetic example written by scripts/export_web.py",
            "app": dict(app),
            "recordings": recordings,
        }
    )
