"""The "Download recordings" file of the web demo's Record test signs mode, and its replay
through Python preprocessing.

One file holds the segments of one page session. Each recording stores the raw landmarks the
app saw for every frame of a segment (after hand assignment, before any aspect correction, in
unmirrored normalised coordinates of the processed frame, null for missing), the settings in
force, and what the app computed: its preprocessed features, model input and top-5. No video or
images are stored. web/src/recorder.ts writes and validates the same schema.
"""

import json
from pathlib import Path
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from edgevision.data.landmarks import N_HAND, N_POSE
from edgevision.data.preprocess import PreprocessResult, preprocess_sequence
from edgevision.live.aspect import RawFrames, correct_frames

FloatArray = npt.NDArray[np.float64]

RECORDINGS_FORMAT: Literal["edge-vision-live-recordings"] = "edge-vision-live-recordings"
RECORDINGS_SCHEMA_VERSION: Literal[1] = 1
COORDS_PER_POINT = 3  # x, y, z as MediaPipe reports them.
HAND_VALUES = N_HAND * COORDS_PER_POINT
POSE_VALUES = N_POSE * COORDS_PER_POINT

Framing = Literal["full", "portrait"]
AspectMode = Literal["frame", "auto", "manual", "off"]
CalibrationStatus = Literal[
    "frame", "calibrated", "no_shoulders", "manual", "off", "pending", "unavailable"
]
EndReason = Literal["hands_lowered", "max_length", "manual_release", "hands_lost", "still", "flush"]
AppStatus = Literal["classified", "excluded", "discarded"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AppInfo(_Strict):
    model_name: str
    variant: str
    run: str
    model_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    feature_set: Literal["v1", "v2"]
    pose_indices: list[int]
    pose_swap_pairs: list[tuple[int, int]]
    num_frames: int = Field(ge=1)
    min_hand_frames: int = Field(ge=1)
    threshold: float = Field(ge=0.0, le=1.0)
    num_classes: int = Field(ge=2)
    r_train: float | None


class Camera(_Strict):
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    frame_rate: float | None


class Crop(_Strict):
    sx: int
    sy: int
    sw: int = Field(ge=0)
    sh: int = Field(ge=0)


class CalibrationRecord(_Strict):
    mode: AspectMode
    k: float = Field(gt=0.0)
    r_live: float | None
    r_train: float | None
    frames: int = Field(ge=0)
    status: CalibrationStatus


class SegmentRecord(_Strict):
    capture_mode: Literal["auto", "manual"]
    reason: EndReason
    start_ms: float
    end_ms: float
    frames: int = Field(ge=0)
    hand_frames: int = Field(ge=0)


Values = list[float | None]


class RecordedFrames(_Strict):
    """Column-wise per-frame data; each landmark entry is a flat [x0, y0, z0, x1, ...] list, or
    null when the block was not detected in that frame."""

    t_ms: list[float]
    left_hand: list[Values | None]
    right_hand: list[Values | None]
    pose: list[Values | None]
    pose_visibility: list[Values | None]
    # True when the pose landmarker ran on this frame (it may run every Nth frame only);
    # other frames hold a pose interpolated between runs, or null.
    pose_ran: list[bool]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        n = len(self.t_ms)
        columns = (self.left_hand, self.right_hand, self.pose, self.pose_visibility, self.pose_ran)
        if any(len(column) != n for column in columns):
            raise ValueError("frame columns differ in length")
        sizes = (
            (self.left_hand, HAND_VALUES),
            (self.right_hand, HAND_VALUES),
            (self.pose, POSE_VALUES),
            (self.pose_visibility, N_POSE),
        )
        for column, size in sizes:
            if any(entry is not None and len(entry) != size for entry in column):
                raise ValueError(f"a landmark entry does not have {size} values")
        return self


class Candidate(_Strict):
    index: int = Field(ge=0)
    label: str
    probability: float = Field(ge=0.0, le=1.0)


class SequenceInfoRecord(_Strict):
    t_raw: int
    t_kept: int
    mirrored: bool
    norm_method: Literal["shoulders", "hand_fallback"] | None
    excluded: bool
    reason: Literal["insufficient_hand_frames", "degenerate_scale"] | None


class AppResult(_Strict):
    status: AppStatus
    reason: str | None
    info: SequenceInfoRecord | None
    # (num_frames, frame_dim) float32 before the float16 storage step, and the model input.
    features: list[list[float]] | None
    model_input: list[list[float]] | None
    top5: list[Candidate]
    prediction: str | None
    max_probability: float | None
    accepted: bool | None
    threshold: float

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        classified = self.status == "classified"
        if classified != (self.features is not None and self.model_input is not None):
            raise ValueError("features and model_input are set exactly for classified segments")
        if classified != (len(self.top5) > 0) or classified != (self.prediction is not None):
            raise ValueError("top5 and prediction are set exactly for classified segments")
        return self


class Recording(_Strict):
    id: int = Field(ge=0)
    session: int = Field(ge=0)
    intended_label: str
    intended_index: int = Field(ge=0)
    recorded_at: str
    camera: Camera
    framing: Framing
    crop: Crop
    swap_hands: bool
    pose_every: int = Field(ge=1, le=3)
    # The min_hand_frames in force (a settings override or the model's value).
    min_hand_frames: int = Field(ge=1)
    calibration: CalibrationRecord
    segment: SegmentRecord
    frames: RecordedFrames
    app_result: AppResult


class RecordingsFile(_Strict):
    format: Literal["edge-vision-live-recordings"]
    schema_version: Literal[1]
    created_at: str
    user_agent: str
    app: AppInfo
    recordings: list[Recording]


def load_recordings(path: Path) -> RecordingsFile:
    with path.open(encoding="utf-8") as handle:
        return RecordingsFile.model_validate(json.load(handle))


# ---------------------------------------------------------------------------------------------
# Replay


def _block(entries: list[Values | None], points: int) -> FloatArray:
    """(T, points, 3) from per-frame flat lists; null entries and values become NaN."""
    out = np.full((len(entries), points, COORDS_PER_POINT), np.nan)
    for t, entry in enumerate(entries):
        if entry is not None:
            values = np.array([np.nan if v is None else v for v in entry], dtype=np.float64)
            out[t] = values.reshape(points, COORDS_PER_POINT)
    return out


def raw_frames(frames: RecordedFrames) -> RawFrames:
    return RawFrames(
        _block(frames.left_hand, N_HAND),
        _block(frames.right_hand, N_HAND),
        _block(frames.pose, N_POSE),
    )


def pose_visibility(frames: RecordedFrames) -> FloatArray:
    out = np.full((len(frames.t_ms), N_POSE), np.nan)
    for t, entry in enumerate(frames.pose_visibility):
        if entry is not None:
            out[t] = [np.nan if v is None else v for v in entry]
    return out


def long_rows(frames: RawFrames) -> pd.DataFrame:
    """Long-format rows (frame, type, landmark_index, x, y, z) with every landmark of every
    frame, NaN where missing, so frames without any landmark still count in t_raw as they do in
    the app."""
    n_frames = frames.pose.shape[0]
    parts = []
    for kind, block in (
        ("left_hand", frames.left_hand),
        ("right_hand", frames.right_hand),
        ("pose", frames.pose),
    ):
        points = block.shape[1]
        parts.append(
            pd.DataFrame(
                {
                    "frame": np.repeat(np.arange(n_frames, dtype=np.int64), points),
                    "type": kind,
                    "landmark_index": np.tile(np.arange(points, dtype=np.int64), n_frames),
                    "x": block[..., 0].ravel(),
                    "y": block[..., 1].ravel(),
                    "z": block[..., 2].ravel(),
                }
            )
        )
    return pd.concat(parts, ignore_index=True)


def replay(
    frames: RawFrames,
    k: float,
    *,
    pose_indices: list[int],
    num_frames: int,
    min_hand_frames: int,
) -> PreprocessResult:
    """The app's segment pipeline: aspect correction with ``k``, then preprocess_sequence."""
    return preprocess_sequence(
        long_rows(correct_frames(frames, k)),
        num_frames=num_frames,
        use_z=False,
        pose_indices=pose_indices,
        min_hand_frames=min_hand_frames,
    )
