"""Aspect calibration for live landmarks; web/src/aspect.ts implements the same functions.

MediaPipe coordinates are fractions of the frame's width (x) and height (y), so one body gives
different y/x proportions in a portrait phone video and in a landscape webcam frame, and
shoulder-width normalisation keeps the difference. The ratio

    r = |y_nose - y_shoulder_mid| / |x_shoulder_left - x_shoulder_right|

(the definition behind preprocess_report.json's aspect_stats) measures it. Scaling every y about
the frame's shoulder midpoint by k = r_train / r_live, before preprocessing, gives live frames the
training ratio; x is unchanged.

Raw landmark blocks are (T, n, C) float64 arrays in unmirrored normalised image coordinates with
C >= 2 (x, y[, z]) and NaN where a point is missing.
"""

from typing import Literal, NamedTuple

import numpy as np
import numpy.typing as npt

from edgevision import spec

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]

K_MIN = float(spec.ASPECT["k_min"])
K_MAX = float(spec.ASPECT["k_max"])
NOSE = spec.POSE_POINTS["nose"]
LEFT_SHOULDER = spec.POSE_POINTS["left_shoulder"]
RIGHT_SHOULDER = spec.POSE_POINTS["right_shoulder"]
CALIBRATION_POINTS = (NOSE, LEFT_SHOULDER, RIGHT_SHOULDER)
# y_mid for frames without both shoulders: the frame's vertical centre.
DEFAULT_Y_MID = float(spec.ASPECT["default_y_mid"])
# A pose point counts as visible when it lies inside the frame with at least this visibility.
MIN_VISIBILITY = float(spec.ASPECT["calibration"]["min_visibility"])
# The training videos' effective width / height, and the per-frame factor it gives.
TRAIN_WIDTH_OVER_HEIGHT = float(spec.ASPECT["train_width_over_height"])


def frame_k(width: float, height: float) -> float:
    """k converting a width x height frame to the training aspect convention, clamped."""
    if not (width > 0.0 and height > 0.0):
        return 1.0
    return clamp_k(TRAIN_WIDTH_OVER_HEIGHT / (width / height))


CalibrationStatus = Literal["calibrated", "no_shoulders"]


def frame_ratios(pose: FloatArray) -> FloatArray:
    """Per-frame r for (T, 33, C) pose; NaN where the nose or a shoulder is missing or the
    shoulders' x coincide (the same rule as preprocessing's aspect measurement)."""
    nose, left, right = (pose[:, index, :2] for index in CALIBRATION_POINTS)
    finite = np.isfinite(nose).all(axis=1) & np.isfinite(left).all(axis=1)
    finite &= np.isfinite(right).all(axis=1)
    width = np.abs(left[:, 0] - right[:, 0])
    usable = finite & (width > 0.0)
    ratios = np.full(pose.shape[0], np.nan)
    mid = 0.5 * (left[usable, 1] + right[usable, 1])
    ratios[usable] = np.abs(nose[usable, 1] - mid) / width[usable]
    return ratios


def visible_frames(
    pose: FloatArray, visibility: FloatArray | None, points: tuple[int, ...]
) -> BoolArray:
    """Frames whose ``points`` are all visible: finite, inside the frame and, when
    ``visibility`` (T, 33) is given, with visibility >= MIN_VISIBILITY."""
    selected: FloatArray = np.asarray(pose[:, list(points), :2], dtype=np.float64)
    inside: BoolArray = np.all(
        np.isfinite(selected) & (selected >= 0.0) & (selected <= 1.0), axis=2
    )
    if visibility is not None:
        seen: FloatArray = np.asarray(visibility[:, list(points)], dtype=np.float64)
        inside &= np.isfinite(seen) & (seen >= MIN_VISIBILITY)
    usable: BoolArray = np.all(inside, axis=1)
    return usable


def calibration_frames(pose: FloatArray, visibility: FloatArray | None = None) -> BoolArray:
    """Frames whose nose and both shoulders are visible; only these calibrate."""
    return visible_frames(pose, visibility, CALIBRATION_POINTS)


def shoulder_frames(pose: FloatArray, visibility: FloatArray | None = None) -> BoolArray:
    """Frames whose two shoulders are visible."""
    return visible_frames(pose, visibility, (LEFT_SHOULDER, RIGHT_SHOULDER))


class Calibration(NamedTuple):
    """k = r_train / r_live clamped to [K_MIN, K_MAX]; k = 1 when nothing was measurable."""

    k: float
    r_live: float | None
    frames: int
    status: CalibrationStatus


def calibrate(ratios: npt.ArrayLike, r_train: float) -> Calibration:
    """Median of the finite ``ratios`` as r_live, and the clamped correction factor."""
    values = np.asarray(ratios, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return Calibration(1.0, None, 0, "no_shoulders")
    r_live = float(np.median(values))
    k = r_train / r_live if r_live > 0.0 else K_MAX
    return Calibration(clamp_k(k), r_live, int(values.size), "calibrated")


def clamp_k(k: float) -> float:
    return float(min(max(k, K_MIN), K_MAX))


def shoulder_mid_y(pose: FloatArray | None, n_frames: int) -> FloatArray:
    """(T,) shoulder-midpoint y per frame; DEFAULT_Y_MID where either shoulder's y is missing."""
    if pose is None:
        return np.full(n_frames, DEFAULT_Y_MID)
    left, right = pose[:, LEFT_SHOULDER, 1], pose[:, RIGHT_SHOULDER, 1]
    both = np.isfinite(left) & np.isfinite(right)
    return np.where(both, 0.5 * (left + right), DEFAULT_Y_MID)


def correct_y(y: FloatArray, y_mid: FloatArray, k: float) -> FloatArray:
    """y' = y_mid + (y - y_mid) * k, with y_mid broadcast per frame."""
    corrected: FloatArray = y_mid + (y - y_mid) * k
    return corrected


class RawFrames(NamedTuple):
    """Raw landmark blocks of one segment: (T, 21, C), (T, 21, C) and (T, 33, C), NaN where a
    point or a whole block is missing."""

    left_hand: FloatArray
    right_hand: FloatArray
    pose: FloatArray


def correct_frames(frames: RawFrames, k: float) -> RawFrames:
    """Every block's y scaled by k about each frame's shoulder midpoint (taken before the
    correction, which leaves it unchanged); x and z untouched.

    k == 1 is the identity and returns exact copies: y_mid + (y - y_mid) * 1 can differ from y in
    the last bit, so "no correction" must skip the arithmetic (aspect.ts does the same).
    """
    if k == 1.0:
        return RawFrames(*(block.copy() for block in frames))
    n_frames = frames.pose.shape[0]
    y_mid = shoulder_mid_y(frames.pose, n_frames)[:, None]

    def corrected(block: FloatArray) -> FloatArray:
        out = block.copy()
        out[..., 1] = correct_y(block[..., 1], y_mid, k)
        return out

    return RawFrames(*(corrected(block) for block in frames))
