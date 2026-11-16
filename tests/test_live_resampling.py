"""Resampling by time (features_spec.json resampling.live_axis): a sign gives the same model input
whatever frame rate the camera delivers, and dropped frames do not distort it, while resampling
by kept-frame index (training's rule, which assumes a constant rate) does.

The sign is a smooth synthetic trajectory: the right hand's 21 points move and turn over 2 s in
front of a still body. Sampled at 8, 15 and 30 fps it must give the 30 fps model input within the
error of linear interpolation: for x(t) = A sin(w t) sampled every dt that error is at most
dt^2 / 8 * A w^2, which for the fastest component (A = 0.4 shoulder widths, w = pi rad/s) at 8 fps
(dt = 0.125 s) is 0.0077; TOLERANCE adds the 30 fps reference's own error and float16 storage.
"""

import numpy as np
import pytest

from edgevision.live.pipeline import live_frame, preprocess_segment
from edgevision.spec import MIN_HAND_FRAMES, NUM_FRAMES, POSE_SETS

DURATION_S = 2.0
POSE_INDICES = POSE_SETS["v2"]
SHOULDER_WIDTH = 0.3
TOLERANCE = 0.02

rng = np.random.default_rng(7)
HAND_SHAPE = rng.normal(0.0, 0.03, size=(21, 2))
POSE = np.column_stack([rng.uniform(0.35, 0.65, 33), rng.uniform(0.2, 0.9, 33), np.zeros(33)])
POSE[11, :2] = (0.5 + SHOULDER_WIDTH / 2, 0.55)  # The signer's left shoulder: image right.
POSE[12, :2] = (0.5 - SHOULDER_WIDTH / 2, 0.55)


def hand_at(t: float) -> np.ndarray:
    """The right hand at time t (s): centre on a slow curve, shape turning about it."""
    centre = np.array(
        [
            0.5 + 0.4 * SHOULDER_WIDTH * np.sin(np.pi * t),
            0.6 - 0.3 * SHOULDER_WIDTH * np.sin(0.5 * np.pi * t),
        ]
    )
    angle = 0.6 * np.sin(0.5 * np.pi * t)
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    points = centre + HAND_SHAPE @ rotation.T
    return np.column_stack([points, np.zeros(21)])


def model_input(times_s: np.ndarray, resampling: str) -> np.ndarray:
    frames = [live_frame(float(t) * 1000.0, None, hand_at(float(t)), POSE) for t in times_s]
    result, packed = preprocess_segment(
        frames,
        1.0,
        pose_indices=POSE_INDICES,
        num_frames=NUM_FRAMES,
        min_hand_frames=MIN_HAND_FRAMES,
        resampling=resampling,  # type: ignore[arg-type]
    )
    assert packed is not None and result.info["norm_method"] == "shoulders"
    return packed


def at_rate(fps: float) -> np.ndarray:
    return np.arange(round(DURATION_S * fps) + 1) / fps


@pytest.fixture(scope="module")
def reference() -> np.ndarray:
    return model_input(at_rate(30.0), "time")


@pytest.mark.parametrize("fps", [8.0, 10.0, 15.0, 24.0])
def test_time_resampling_gives_the_same_input_at_any_frame_rate(reference, fps):
    error = np.abs(model_input(at_rate(fps), "time") - reference).max()
    assert error < TOLERANCE, f"{fps} fps differs from 30 fps by {error:.4f}"


def test_at_a_constant_rate_time_and_index_resampling_agree():
    times = at_rate(15.0)
    np.testing.assert_allclose(model_input(times, "time"), model_input(times, "index"), atol=1e-3)


def test_dropped_frames_do_not_distort_time_resampling(reference):
    """A camera that delivers 30 fps but a tracker that skips frames unevenly (here a slow
    stretch at the start and a fast one at the end): the kept frames' capture times keep the
    trajectory in place, kept-frame indices stretch the slow stretch."""
    times = at_rate(30.0)
    keep = np.ones(times.size, dtype=bool)
    keep[1:30] = np.arange(1, 30) % 4 == 0  # 7.5 fps for the first second, 30 fps after it.
    by_time = np.abs(model_input(times[keep], "time") - reference).max()
    by_index = np.abs(model_input(times[keep], "index") - reference).max()
    assert by_time < TOLERANCE
    assert by_index > 5 * by_time, (by_time, by_index)
