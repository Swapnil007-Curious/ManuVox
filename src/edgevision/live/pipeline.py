"""The live recognition pipeline, from per-frame landmarks to a decision per sign segment.

This is the reference implementation: the offline evaluation (Holistic landmarks of recorded
clips) runs it directly, and web/src/livePipeline.ts is a step-by-step port checked against the
golden fixtures this module writes (edgevision.live.parity). Every constant comes from
features_spec.json (``live`` and ``aspect``).

Per frame:
  1. hand assignment: detected hands go to the Holistic blocks (left_hand is the hand at pose
     landmark 15); the legacy rule of the first demo is kept for the ablation;
  2. aspect correction: every y scaled by k about the frame's shoulder midpoint, with
     k = train_width_over_height / (width / height) by default;
  3. motion energy: the speed of the wrist and fingertips over a 100 ms window, in shoulder
     widths per second;
  4. segmentation: IDLE -> SIGNING when a hand moves faster than start_energy; SIGNING ends when
     the hands are gone for hand_lost_ms, stay below still_energy for still_ms, or after max_ms.
Per segment:
  5. preprocessing as in training, except that the 32 output frames are placed on the
     kept-frame clock (each kept frame advances it by its own capture interval) rather than the
     kept-frame index, then the float16 storage step;
  6. the model, then softmax(logits / temperature) and the confidence threshold.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, Generic, Literal, NamedTuple, TypeVar

import numpy as np
import numpy.typing as npt

from edgevision import spec
from edgevision.data.landmarks import N_HAND, N_POSE
from edgevision.data.preprocess import (
    MIN_SCALE,
    PreprocessResult,
    SequenceInfo,
    fill_missing,
    hand_presence,
    left_is_dominant,
    mirror_sequence,
    normalization_params,
    normalize_coords,
    pivot_sequence,
    resample_presence,
)
from edgevision.live.aspect import RawFrames, correct_frames, frame_k
from edgevision.live.recordings import long_rows, replay

FloatArray = npt.NDArray[np.float64]
Float32Array = npt.NDArray[np.float32]

HandRule = Literal["nearest_pose_wrist", "label", "label_inverted"]
Resampling = Literal["time", "index"]
HAND_RULES: tuple[HandRule, ...] = ("nearest_pose_wrist", "label", "label_inverted")

LEFT_WRIST = spec.POSE_POINTS["left_wrist"]
RIGHT_WRIST = spec.POSE_POINTS["right_wrist"]
LEFT_SHOULDER = spec.POSE_POINTS["left_shoulder"]
RIGHT_SHOULDER = spec.POSE_POINTS["right_shoulder"]
MOTION_POINTS: tuple[int, ...] = (spec.WRIST, *spec.FINGERTIPS)
MOTION = spec.LIVE["motion"]
SEGMENTER = spec.LIVE["segmenter"]


# ---------------------------------------------------------------------------------------------
# 1. Hand assignment


class DetectedHand(NamedTuple):
    """One HandLandmarker result: (21, 3) landmarks in unmirrored normalised coordinates and the
    top handedness category name ("Left" or "Right")."""

    landmarks: FloatArray
    label: str


def _wrist_distance(hand: FloatArray, pose: FloatArray, wrist: int) -> float:
    return float(np.hypot(hand[0, 0] - pose[wrist, 0], hand[0, 1] - pose[wrist, 1]))


def _by_label(
    hands: Sequence[DetectedHand], inverted: bool
) -> tuple[FloatArray | None, FloatArray | None]:
    """Label rule; two hands with the same (or an unknown) label are split by position: the
    hand further left in the unmirrored image (smaller wrist x) is the signer's right hand."""
    left: FloatArray | None = None
    right: FloatArray | None = None

    def block(label: str) -> str | None:
        own = {"Left": "left", "Right": "right"}.get(label)
        if own is None or not inverted:
            return own
        return "right" if own == "left" else "left"

    blocks = [block(hand.label) for hand in hands]
    if len(hands) == 2 and (blocks[0] == blocks[1] or None in blocks):
        first, second = hands
        if first.landmarks[0, 0] <= second.landmarks[0, 0]:
            image_left, image_right = first, second
        else:
            image_left, image_right = second, first
        return image_right.landmarks, image_left.landmarks
    for hand, name in zip(hands, blocks, strict=True):
        if name == "left":
            left = hand.landmarks
        elif name == "right":
            right = hand.landmarks
    return left, right


def assign_hands(
    hands: Sequence[DetectedHand], pose: FloatArray | None, rule: HandRule = "nearest_pose_wrist"
) -> tuple[FloatArray | None, FloatArray | None]:
    """(left_hand, right_hand) blocks for up to two detected hands.

    nearest_pose_wrist: Holistic's rule. One hand takes the block of the nearer pose wrist (15 is
    the signer's left); two hands take the pairing with the smaller total wrist distance. Without
    finite pose wrists it falls back to ``label``. label: Tasks HandLandmarker's true handedness
    on an unmirrored frame. label_inverted: the first demo's rule, which assumed a mirrored
    frame and so swapped every hand.
    """
    usable = [hand for hand in hands if hand.landmarks.shape[0] >= N_HAND][:2]
    if not usable:
        return None, None
    if rule == "label_inverted":
        return _by_label(usable, inverted=True)
    wrists_known = (
        pose is not None
        and bool(np.isfinite(pose[[LEFT_WRIST, RIGHT_WRIST], :2]).all())
        and all(bool(np.isfinite(hand.landmarks[0, :2]).all()) for hand in usable)
    )
    if rule == "label" or not wrists_known:
        return _by_label(usable, inverted=False)
    assert pose is not None
    if len(usable) == 1:
        hand = usable[0].landmarks
        to_left = _wrist_distance(hand, pose, LEFT_WRIST)
        to_right = _wrist_distance(hand, pose, RIGHT_WRIST)
        if to_left == to_right:
            return _by_label(usable, inverted=False)
        return (hand, None) if to_left < to_right else (None, hand)
    first, second = (hand.landmarks for hand in usable)
    straight = _wrist_distance(first, pose, LEFT_WRIST) + _wrist_distance(second, pose, RIGHT_WRIST)
    crossed = _wrist_distance(first, pose, RIGHT_WRIST) + _wrist_distance(second, pose, LEFT_WRIST)
    if straight == crossed:
        return _by_label(usable, inverted=False)
    return (first, second) if straight < crossed else (second, first)


# ---------------------------------------------------------------------------------------------
# 2. Aspect correction of one frame


def shoulder_mid_y(pose: FloatArray | None) -> float:
    """The frame's shoulder-midpoint y, or default_y_mid without both shoulders."""
    if pose is None:
        return float(spec.ASPECT["default_y_mid"])
    left, right = pose[LEFT_SHOULDER, 1], pose[RIGHT_SHOULDER, 1]
    if np.isfinite(left) and np.isfinite(right):
        return float(0.5 * (left + right))
    return float(spec.ASPECT["default_y_mid"])


def correct_block(block: FloatArray | None, y_mid: float, k: float) -> FloatArray | None:
    if block is None:
        return None
    out = block.copy()
    if k != 1.0:
        out[:, 1] = y_mid + (block[:, 1] - y_mid) * k
    return out


# ---------------------------------------------------------------------------------------------
# 3. Motion energy


@dataclass
class MotionTracker:
    """Speed of the wrist and fingertips over a fixed time window, in shoulder widths per
    second.

    Per present hand: the mean displacement of its six motion points between this frame and the
    latest earlier frame at least window_ms older (the oldest kept frame while the hand is newer
    than that), divided by the scale (the frame's shoulder distance, else the last known one,
    else hand_scale_factor x the wrist-middle MCP distance) and by the time between the two
    frames. A fixed window keeps the units and the tracker-noise floor independent of the frame
    rate. update() returns the largest speed among present hands (0 for a hand seen in one frame
    only), or NaN when no hand is present.
    """

    window_ms: float = float(MOTION["window_ms"])
    hand_scale_factor: float = float(MOTION["hand_scale_factor"])
    last_scale: float | None = None
    history: dict[str, list[tuple[float, FloatArray]]] = field(default_factory=dict)

    def reset(self) -> None:
        self.last_scale = None
        self.history = {}

    def _scale(
        self, left: FloatArray | None, right: FloatArray | None, pose: FloatArray | None
    ) -> float | None:
        if pose is not None:
            a, b = pose[LEFT_SHOULDER, :2], pose[RIGHT_SHOULDER, :2]
            if np.isfinite(a).all() and np.isfinite(b).all():
                distance = float(np.hypot(a[0] - b[0], a[1] - b[1]))
                if distance > MIN_SCALE:
                    self.last_scale = distance
                    return distance
        if self.last_scale is not None:
            return self.last_scale
        for hand in (right, left):
            if hand is not None:
                size = float(
                    np.hypot(
                        hand[spec.MIDDLE_MCP, 0] - hand[spec.WRIST, 0],
                        hand[spec.MIDDLE_MCP, 1] - hand[spec.WRIST, 1],
                    )
                )
                if size > MIN_SCALE:
                    return self.hand_scale_factor * size
        return None

    def update(
        self,
        time_ms: float,
        left: FloatArray | None,
        right: FloatArray | None,
        pose: FloatArray | None,
    ) -> float:
        """Feed one frame of aspect-corrected blocks (None for a hand that is not present)."""
        scale = self._scale(left, right, pose)
        energy = float("nan")
        for name, hand in (("left", left), ("right", right)):
            if hand is None:
                self.history.pop(name, None)
                continue
            points = hand[list(MOTION_POINTS), :2].astype(np.float64)
            history = [item for item in self.history.get(name, []) if item[0] < time_ms]
            speed = 0.0
            if history and scale is not None:
                older = [item for item in history if item[0] <= time_ms - self.window_ms]
                reference_time, reference = older[-1] if older else history[0]
                step = float(np.hypot(*(points - reference).T).mean())
                speed = step / scale / ((time_ms - reference_time) / 1000.0)
            # Keep the latest frame at least window_ms old and everything newer.
            older_count = sum(1 for item in history if item[0] <= time_ms - self.window_ms)
            history = history[max(older_count - 1, 0) :]
            history.append((time_ms, points))
            self.history[name] = history
            energy = speed if np.isnan(energy) else max(energy, speed)
        return energy


# ---------------------------------------------------------------------------------------------
# 4. Segmentation

T = TypeVar("T")


class SegmenterSettings(NamedTuple):
    start_energy: float = float(SEGMENTER["start_energy"])
    still_energy: float = float(SEGMENTER["still_energy"])
    still_ms: float = float(SEGMENTER["still_ms"])
    hand_lost_ms: float = float(SEGMENTER["hand_lost_ms"])
    max_ms: float = float(SEGMENTER["max_ms"])
    min_ms: float = float(SEGMENTER["min_ms"])
    preroll_ms: float = float(SEGMENTER["preroll_ms"])
    tail_ms: float = float(SEGMENTER["tail_ms"])
    min_hand_frames: int = spec.MIN_HAND_FRAMES


class TimedFrame(NamedTuple, Generic[T]):
    frame: T
    time: float
    has_hand: bool
    energy: float


EndReason = Literal["hands_lost", "still", "max_length", "flush", "hands_lowered", "manual_release"]


class SegmentEvent(NamedTuple, Generic[T]):
    """A finished segment; ``kind`` is "segment" or "discarded" (too short or too few hands)."""

    kind: Literal["segment", "discarded"]
    frames: list[TimedFrame[T]]
    reason: EndReason
    start: float
    end: float
    hand_frames: int


class MotionSegmenter(Generic[T]):
    """IDLE -> SIGNING -> END state machine on motion energy (see the module docstring).

    A segment starts with the frames of the last preroll_ms before the trigger. It ends when no
    hand has been present for hand_lost_ms (frames from the loss on are dropped), when the
    energy has stayed below still_energy for still_ms (frames later than tail_ms into the still
    period are dropped), or when it reaches max_ms. Segments shorter than min_ms or with fewer
    than min_hand_frames hand frames are discarded.
    """

    def __init__(self, settings: SegmenterSettings | None = None) -> None:
        self.settings = settings or SegmenterSettings()
        self.reset()

    def reset(self) -> None:
        self.recent: list[TimedFrame[T]] = []
        self.frames: list[TimedFrame[T]] = []
        self.signing = False
        self.start = 0.0
        self.still_since: float | None = None
        self.lost_since: float | None = None

    @property
    def state(self) -> Literal["idle", "signing"]:
        return "signing" if self.signing else "idle"

    def push(self, frame: T, time: float, has_hand: bool, energy: float) -> SegmentEvent[T] | None:
        settings = self.settings
        timed = TimedFrame(frame, time, has_hand, energy)
        if not self.signing:
            self.recent = [item for item in self.recent if item.time >= time - settings.preroll_ms]
            self.recent.append(timed)
            if not (has_hand and energy >= settings.start_energy):
                return None
            self.signing = True
            self.frames = self.recent
            self.recent = []
            self.start = self.frames[0].time
            self.still_since = None
            self.lost_since = None
            return self._check_length(time)
        self.frames.append(timed)
        if not has_hand:
            if self.lost_since is None:
                self.lost_since = time
            if time - self.lost_since >= settings.hand_lost_ms:
                cut = self.lost_since
                return self._finish([f for f in self.frames if f.time < cut], "hands_lost")
            return self._check_length(time)
        self.lost_since = None
        if energy < settings.still_energy:
            if self.still_since is None:
                self.still_since = time
            if time - self.still_since >= settings.still_ms:
                keep_until = self.still_since + settings.tail_ms
                return self._finish([f for f in self.frames if f.time <= keep_until], "still")
        else:
            self.still_since = None
        return self._check_length(time)

    def flush(self) -> SegmentEvent[T] | None:
        """End the segment in progress, if any (the stream stopped)."""
        if not self.signing:
            self.reset()
            return None
        return self._finish(list(self.frames), "flush")

    def _check_length(self, time: float) -> SegmentEvent[T] | None:
        if time - self.start < self.settings.max_ms:
            return None
        return self._finish(list(self.frames), "max_length")

    def _finish(self, frames: list[TimedFrame[T]], reason: EndReason) -> SegmentEvent[T]:
        start = self.start
        self.reset()
        hand_frames = sum(1 for item in frames if item.has_hand)
        end = frames[-1].time if frames else start
        too_short = end - start < self.settings.min_ms
        kind: Literal["segment", "discarded"] = (
            "discarded" if too_short or hand_frames < self.settings.min_hand_frames else "segment"
        )
        return SegmentEvent(kind, frames, reason, start, end, hand_frames)


class PresenceSegmenterSettings(NamedTuple):
    """The first demo's automatic capture (web/src/segmenter.ts, auto mode)."""

    start_frames: int = 3
    end_frames: int = 8
    max_ms: float = 3000.0
    min_hand_frames: int = spec.MIN_HAND_FRAMES


class PresenceSegmenter(Generic[T]):
    """A segment starts after start_frames consecutive frames with a hand and ends after
    end_frames frames without one (those are dropped) or at max_ms."""

    def __init__(self, settings: PresenceSegmenterSettings | None = None) -> None:
        self.settings = settings or PresenceSegmenterSettings()
        self.reset()

    def reset(self) -> None:
        self.pending: list[TimedFrame[T]] = []
        self.frames: list[TimedFrame[T]] = []
        self.recording = False
        self.hand_frames = 0
        self.no_hand_run = 0
        self.start = 0.0

    def push(
        self, frame: T, time: float, has_hand: bool, energy: float = 0.0
    ) -> SegmentEvent[T] | None:
        timed = TimedFrame(frame, time, has_hand, energy)
        settings = self.settings
        if not self.recording:
            if not has_hand:
                self.pending = []
                return None
            self.pending.append(timed)
            if len(self.pending) < settings.start_frames:
                return None
            self.recording = True
            self.frames = self.pending
            self.pending = []
            self.hand_frames = len(self.frames)
            self.no_hand_run = 0
            self.start = self.frames[0].time
            return self._check_length(time)
        self.frames.append(timed)
        if has_hand:
            self.hand_frames += 1
            self.no_hand_run = 0
        else:
            self.no_hand_run += 1
            if self.no_hand_run >= settings.end_frames:
                kept = self.frames[: len(self.frames) - self.no_hand_run]
                return self._finish(kept, "hands_lowered", kept[-1].time if kept else time)
        return self._check_length(time)

    def flush(self) -> SegmentEvent[T] | None:
        if not self.recording:
            self.reset()
            return None
        return self._finish(list(self.frames), "flush", self.frames[-1].time)

    def _check_length(self, time: float) -> SegmentEvent[T] | None:
        if time - self.start < self.settings.max_ms:
            return None
        return self._finish(list(self.frames), "max_length", time)

    def _finish(
        self, frames: list[TimedFrame[T]], reason: EndReason, end: float
    ) -> SegmentEvent[T]:
        hand_frames, start = self.hand_frames, self.start
        self.reset()
        kind: Literal["segment", "discarded"] = (
            "discarded" if hand_frames < self.settings.min_hand_frames else "segment"
        )
        return SegmentEvent(kind, frames, reason, start, end, hand_frames)


# ---------------------------------------------------------------------------------------------
# 5. Segment preprocessing


class LiveFrame(NamedTuple):
    """One frame after hand assignment and before aspect correction: (21, 3), (21, 3) and
    (33, 3) blocks in unmirrored normalised coordinates, NaN where missing."""

    time: float
    left_hand: FloatArray
    right_hand: FloatArray
    pose: FloatArray


def stack_frames(frames: Sequence[LiveFrame]) -> tuple[FloatArray, RawFrames]:
    times = np.array([frame.time for frame in frames], dtype=np.float64)
    raw = RawFrames(
        np.stack([frame.left_hand for frame in frames]).astype(np.float64),
        np.stack([frame.right_hand for frame in frames]).astype(np.float64),
        np.stack([frame.pose for frame in frames]).astype(np.float64),
    )
    return times, raw


def kept_clock(times: FloatArray, keep: npt.NDArray[np.bool_], max_gap_ms: float) -> FloatArray:
    """Clock positions of the kept frames: 0 for the first, then each kept frame advances by the
    time since the frame captured just before it (kept or not), capped at max_gap_ms."""
    clock: list[float] = []
    for index in np.flatnonzero(keep):
        if not clock:
            clock.append(0.0)
            continue
        interval = float(times[index] - times[index - 1]) if index > 0 else 0.0
        clock.append(clock[-1] + min(max(interval, 0.0), max_gap_ms))
    return np.asarray(clock, dtype=np.float64)


def resample_on_clock(coords: FloatArray, clock: FloatArray, num_frames: int) -> FloatArray:
    """resample_coords with the kept-frame clock as the source axis; a constant frame rate
    gives exactly linspace(0, t_kept - 1, num_frames) in index units."""
    t_kept, n_points, n_coords = coords.shape
    if t_kept == 1 or clock[-1] <= 0.0:
        positions = np.linspace(0.0, t_kept - 1, num_frames)
        source = np.arange(t_kept, dtype=np.float64)
    else:
        positions = np.linspace(0.0, clock[-1], num_frames)
        source = clock
    out = np.full((num_frames, n_points, n_coords), np.nan)
    for point in range(n_points):
        for axis in range(n_coords):
            series = coords[:, point, axis]
            finite = np.isfinite(series)
            if finite.any():
                out[:, point, axis] = np.interp(
                    positions, source[finite], series[finite], left=np.nan, right=np.nan
                )
    return out


def nearest_on_clock(
    presence: npt.NDArray[np.bool_], clock: FloatArray, num_frames: int
) -> npt.NDArray[np.bool_]:
    """Presence at the nearest kept frame on the clock; a tie goes to the later frame, as
    floor(position + 0.5) does on the index."""
    t_kept = presence.shape[0]
    if t_kept == 1 or clock[-1] <= 0.0:
        return resample_presence(presence, num_frames)
    positions = np.linspace(0.0, clock[-1], num_frames)
    rows = []
    for position in positions:
        later = int(np.searchsorted(clock, position, side="left"))
        later = min(later, t_kept - 1)
        earlier = max(later - 1, 0)
        chosen = later if clock[later] - position <= position - clock[earlier] else earlier
        rows.append(chosen)
    return presence[np.asarray(rows, dtype=np.intp)]


def preprocess_on_clock(
    times: FloatArray,
    raw: RawFrames,
    *,
    pose_indices: Sequence[int],
    num_frames: int,
    min_hand_frames: int,
    max_gap_ms: float = float(SEGMENTER["max_gap_ms"]),
) -> PreprocessResult:
    """preprocess_sequence on already aspect-corrected frames, resampled on the kept-frame
    clock instead of the kept-frame index; every other step is training's."""
    coords = pivot_sequence(long_rows(raw), pose_indices)
    presence = hand_presence(coords)
    keep: npt.NDArray[np.bool_] = np.asarray(presence.any(axis=1), dtype=bool)
    info = SequenceInfo(
        t_raw=int(keep.size),
        t_kept=int(keep.sum()),
        mirrored=False,
        norm_method=None,
        excluded=False,
        reason=None,
    )
    if info["t_kept"] < min_hand_frames:
        info["excluded"], info["reason"] = True, "insufficient_hand_frames"
        return PreprocessResult(None, None, info)
    clock = kept_clock(times, keep, max_gap_ms)
    coords, presence = coords[keep], presence[keep]
    if left_is_dominant(presence):
        coords = mirror_sequence(coords, pose_indices)
        presence = presence[:, ::-1]
        info["mirrored"] = True
    center, scale, info["norm_method"] = normalization_params(coords, pose_indices)
    if not np.isfinite(scale) or scale < MIN_SCALE:
        info["excluded"], info["reason"] = True, "degenerate_scale"
        return PreprocessResult(None, None, info)
    normalized = normalize_coords(coords, center, scale, use_z=False)
    features = fill_missing(resample_on_clock(normalized, clock, num_frames))
    resampled = nearest_on_clock(presence, clock, num_frames).astype(np.float32)
    return PreprocessResult(features, resampled, info)


def to_model_input(result: PreprocessResult) -> Float32Array | None:
    """The float16 storage step training data took, then the (num_frames, 2L + 2) packing;
    None when excluded or when a coordinate overflows float16."""
    if result.features is None or result.hand_presence is None:
        return None
    with np.errstate(over="ignore"):
        stored = result.features.astype(np.float16)
    if not np.isfinite(stored).all():
        return None
    frames = stored.shape[0]
    packed = np.empty((frames, stored.shape[1] * 2 + 2), dtype=np.float32)
    packed[:, :-2] = stored.reshape(frames, -1)
    packed[:, -2:] = result.hand_presence
    return packed


def preprocess_segment(
    frames: Sequence[LiveFrame],
    k: float,
    *,
    pose_indices: Sequence[int],
    num_frames: int,
    min_hand_frames: int,
    resampling: Resampling = "time",
) -> tuple[PreprocessResult, Float32Array | None]:
    """Aspect correction with ``k``, preprocessing and the storage step for one segment."""
    times, raw = stack_frames(frames)
    if resampling == "index":
        result = replay(
            raw,
            k,
            pose_indices=list(pose_indices),
            num_frames=num_frames,
            min_hand_frames=min_hand_frames,
        )
    else:
        result = preprocess_on_clock(
            times,
            correct_frames(raw, k),
            pose_indices=pose_indices,
            num_frames=num_frames,
            min_hand_frames=min_hand_frames,
        )
    return result, to_model_input(result)


# ---------------------------------------------------------------------------------------------
# 6. Decision


def softmax(logits: npt.ArrayLike, temperature: float = 1.0) -> FloatArray:
    values = np.asarray(logits, dtype=np.float64) / temperature
    values = values - values.max(axis=-1, keepdims=True)
    exp = np.exp(values)
    out: FloatArray = exp / exp.sum(axis=-1, keepdims=True)
    return out


class Decision(NamedTuple):
    accepted: bool
    top: list[int]
    probabilities: FloatArray


def decide(logits: npt.ArrayLike, temperature: float, threshold: float, k: int = 5) -> Decision:
    probabilities = softmax(logits, temperature)
    top = [int(i) for i in np.argsort(-probabilities, kind="stable")[:k]]
    return Decision(bool(probabilities[top[0]] >= threshold), top, probabilities)


# ---------------------------------------------------------------------------------------------
# The whole stream


class PipelineConfig(NamedTuple):
    pose_indices: tuple[int, ...]
    num_frames: int
    min_hand_frames: int
    width: float
    height: float
    hand_rule: HandRule = "nearest_pose_wrist"
    aspect: Literal["frame", "calibrated", "off"] = "frame"
    calibrated_k: float = 1.0
    resampling: Resampling = "time"
    segmenter: Literal["motion", "presence"] = "motion"


@dataclass
class SegmentOutcome:
    event: SegmentEvent[LiveFrame]
    result: PreprocessResult | None
    model_input: Float32Array | None
    logits: Float32Array | None = None


def frame_correction_k(config: PipelineConfig) -> float:
    if config.aspect == "off":
        return 1.0
    if config.aspect == "calibrated":
        return config.calibrated_k
    return frame_k(config.width, config.height)


class LivePipeline:
    """Frames in (already assigned to blocks), segment outcomes out. ``classify`` maps one
    model input to logits."""

    def __init__(
        self, config: PipelineConfig, classify: Callable[[Float32Array], Float32Array]
    ) -> None:
        self.config = config
        self.classify = classify
        self.k = frame_correction_k(config)
        self.motion = MotionTracker()
        self.last_energy = float("nan")
        self.segmenter: MotionSegmenter[LiveFrame] | PresenceSegmenter[LiveFrame] = (
            MotionSegmenter() if config.segmenter == "motion" else PresenceSegmenter()
        )

    def push(self, frame: LiveFrame) -> SegmentOutcome | None:
        left = frame.left_hand if np.isfinite(frame.left_hand[:, :2]).all() else None
        right = frame.right_hand if np.isfinite(frame.right_hand[:, :2]).all() else None
        pose = frame.pose if np.isfinite(frame.pose[:, :2]).any() else None
        y_mid = shoulder_mid_y(pose)
        energy = self.motion.update(
            frame.time,
            correct_block(left, y_mid, self.k),
            correct_block(right, y_mid, self.k),
            correct_block(pose, y_mid, self.k),
        )
        self.last_energy = energy
        has_hand = left is not None or right is not None
        event = self.segmenter.push(frame, frame.time, has_hand, energy)
        return self._outcome(event)

    def flush(self) -> SegmentOutcome | None:
        return self._outcome(self.segmenter.flush())

    def _outcome(self, event: SegmentEvent[LiveFrame] | None) -> SegmentOutcome | None:
        if event is None:
            return None
        if event.kind == "discarded":
            return SegmentOutcome(event, None, None)
        config = self.config
        result, model_input = preprocess_segment(
            [item.frame for item in event.frames],
            self.k,
            pose_indices=config.pose_indices,
            num_frames=config.num_frames,
            min_hand_frames=config.min_hand_frames,
            resampling=config.resampling,
        )
        logits = None if model_input is None else self.classify(model_input)
        return SegmentOutcome(event, result, model_input, logits)


def empty_block(points: int) -> FloatArray:
    return np.full((points, 3), np.nan)


def live_frame(
    time: float,
    left: FloatArray | None,
    right: FloatArray | None,
    pose: FloatArray | None,
) -> LiveFrame:
    return LiveFrame(
        time,
        empty_block(N_HAND) if left is None else np.asarray(left, dtype=np.float64)[:, :3],
        empty_block(N_HAND) if right is None else np.asarray(right, dtype=np.float64)[:, :3],
        empty_block(N_POSE) if pose is None else np.asarray(pose, dtype=np.float64)[:, :3],
    )


def summary(outcome: SegmentOutcome) -> dict[str, Any]:
    event = outcome.event
    return {
        "kind": event.kind,
        "reason": event.reason,
        "start_ms": event.start,
        "end_ms": event.end,
        "frames": len(event.frames),
        "hand_frames": event.hand_frames,
        "excluded": None if outcome.result is None else outcome.result.info["reason"],
    }
