"""Aspect calibration and correction (edgevision.live.aspect), the recordings schema and the
replay of recorded frames; web/src/aspect.ts is compared against the same functions through
web/tests/golden/aspect_cases.json."""

import numpy as np
import pytest

import conftest
from edgevision.data.landmarks import N_HAND, N_POSE
from edgevision.data.preprocess import aspect_measurement, hand_presence, pivot_sequence
from edgevision.live.aspect import (
    K_MAX,
    K_MIN,
    RawFrames,
    calibrate,
    calibration_frames,
    clamp_k,
    correct_frames,
    frame_ratios,
    shoulder_frames,
    shoulder_mid_y,
)
from edgevision.live.golden import build_aspect_golden
from edgevision.live.recordings import long_rows, replay
from live_support import motion_blocks, raw_from_blocks

V1 = [0, 11, 12, 13, 14, 15, 16]


def blocks(n_frames=12, seed=0, label=0):
    return raw_from_blocks(motion_blocks(label, n_frames, np.random.default_rng(seed)))


def test_ratio_uses_the_preprocessing_definition():
    frames = blocks(20)
    frames.pose[3, 0] = np.nan  # no nose
    frames.pose[5, 12, 0] = frames.pose[5, 11, 0]  # shoulders' x coincide
    ratios = frame_ratios(frames.pose)
    assert np.isnan(ratios[3]) and np.isnan(ratios[5])
    coords = pivot_sequence(long_rows(frames), V1)
    kept = coords[hand_presence(coords).any(axis=1)]
    measured = aspect_measurement(kept, V1)
    np.testing.assert_allclose(np.sort(ratios[np.isfinite(ratios)]), np.sort(measured.ratios))
    assert np.nanmedian(ratios) == pytest.approx(0.4, rel=0.05)


def test_calibration_median_clamp_and_no_shoulder_case():
    assert calibrate([0.9, 1.1, 1.0], 0.4) == (pytest.approx(0.4), 1.0, 3, "calibrated")
    assert calibrate([0.8, np.nan, 1.2, 1.0, 0.9], 0.4).r_live == pytest.approx(0.95)
    assert calibrate([3.0], 0.4).k == K_MIN
    assert calibrate([0.01], 0.4).k == K_MAX
    assert calibrate([0.0], 0.4).k == K_MAX
    assert calibrate([np.nan], 0.4) == (1.0, None, 0, "no_shoulders")
    assert calibrate([], 0.4) == (1.0, None, 0, "no_shoulders")
    assert [clamp_k(k) for k in (0.1, 0.25, 2.0, 4.0, 5.0)] == [0.25, 0.25, 2.0, 4.0, 4.0]


def test_visibility_rule():
    pose = blocks(4).pose
    visibility = np.full((4, N_POSE), 0.9)
    visibility[1, 0] = 0.4  # nose barely visible
    pose[2, 11, 0] = 1.2  # left shoulder outside the frame
    pose[3, 12] = np.nan
    np.testing.assert_array_equal(calibration_frames(pose, visibility), [True, False, False, False])
    np.testing.assert_array_equal(shoulder_frames(pose, visibility), [True, True, False, False])
    np.testing.assert_array_equal(calibration_frames(pose), [True, True, False, False])


def test_correction_scales_y_about_the_shoulder_midpoint():
    frames = blocks(6)
    frames.pose[2, 12, 1] = np.nan  # this frame scales about 0.5
    corrected = correct_frames(frames, 0.5)
    mid = shoulder_mid_y(frames.pose, 6)
    assert mid[2] == 0.5 and mid[0] == pytest.approx(
        0.5 * (frames.pose[0, 11, 1] + frames.pose[0, 12, 1])
    )
    for before, after in zip(frames, corrected, strict=True):
        np.testing.assert_array_equal(after[..., 0], before[..., 0])
        np.testing.assert_array_equal(after[..., 2], before[..., 2])
        np.testing.assert_allclose(
            after[..., 1], mid[:, None] + (before[..., 1] - mid[:, None]) * 0.5
        )
    # The shoulder midpoint is a fixed point, so the ratio scales by exactly k.
    np.testing.assert_allclose(frame_ratios(corrected.pose), 0.5 * frame_ratios(frames.pose))
    np.testing.assert_array_equal(shoulder_mid_y(corrected.pose, 6)[[0, 1]], mid[[0, 1]])


def test_k_1_is_the_exact_identity():
    frames = blocks(5)
    corrected = correct_frames(frames, 1.0)
    for before, after in zip(frames, corrected, strict=True):
        assert after is not before
        np.testing.assert_array_equal(after, before)


def test_replay_counts_landmark_free_frames_like_the_app():
    frames = blocks(10)
    for block in frames:
        block[4] = np.nan  # a frame where nothing was detected
    result = replay(frames, 1.0, pose_indices=V1, num_frames=32, min_hand_frames=3)
    assert (result.info["t_raw"], result.info["t_kept"]) == (10, 9)
    empty = RawFrames(
        np.full((3, N_HAND, 3), np.nan),
        np.full((3, N_HAND, 3), np.nan),
        np.full((3, N_POSE, 3), np.nan),
    )
    excluded = replay(empty, 0.5, pose_indices=V1, num_frames=32, min_hand_frames=3)
    assert excluded.info["reason"] == "insufficient_hand_frames"


def test_aspect_golden_is_deterministic_and_covers_the_cases():
    golden = build_aspect_golden(conftest.landmark_blocks)
    assert golden == build_aspect_golden(conftest.landmark_blocks)
    statuses = {case.status for case in golden.calibration_cases}
    assert statuses == {"calibrated", "no_shoulders"}
    assert {case.k for case in golden.calibration_cases} >= {K_MIN, K_MAX, 1.0}
    usable = {case.name: case.usable for case in golden.visibility_cases}
    assert (
        usable["visible"] and not usable["nose_low_visibility"] and not usable["shoulder_missing"]
    )
    selections = {
        len(result.pose_indices) for case in golden.frame_cases for result in case.results
    }
    assert selections == {7, 17}
    mirrored = [r for case in golden.frame_cases for r in case.results if r.info["mirrored"]]
    assert {len(r.pose_swap_pairs) for r in mirrored} == {3, 8}
    identity = next(case for case in golden.frame_cases if case.k == 1.0)
    assert identity.corrected == identity.frames
