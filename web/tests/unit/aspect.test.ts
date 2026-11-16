// Aspect correction and calibration against Python (aspect_cases.json, written by
// scripts/export_web.py from src/edgevision/live/aspect.py), for the v1 and v2 selections.

import { describe, expect, it } from "vitest";
import {
  AspectCalibrator,
  DEFAULT_Y_MID,
  K_MAX,
  K_MIN,
  MIN_VISIBILITY,
  calibrate,
  calibrationUsable,
  clampK,
  correctFrame,
  frameRatio,
  shoulderMidY,
  shouldersVisible,
  type RawFrame,
} from "../../src/aspect";
import { preprocessSegment } from "../../src/live";
import { DEFAULT_CONFIG, type PreprocessConfig } from "../../src/preprocess";
import type { FrameSample } from "../../src/recorder";
import { aspectGolden, block, type FrameBlocks } from "./golden";

const golden = aspectGolden();
const FEATURE_TOLERANCE = 1e-5;

function frames(blocks: FrameBlocks): FrameSample[] {
  return blocks.pose.map((pose, t) => ({
    t: t * 33,
    left: block(blocks.left_hand[t] ?? null),
    right: block(blocks.right_hand[t] ?? null),
    pose: block(pose),
    visibility: null,
    poseRan: true,
  }));
}

function same(actual: Float64Array | null, expected: (number | null)[] | null): boolean {
  if (actual === null || expected === null) return actual === null && expected === null;
  return actual.length === expected.length && expected.every((value, i) => (value === null ? Number.isNaN(actual[i]) : actual[i] === value));
}

describe("constants match Python", () => {
  it("uses the same clamp range, default midpoint and visibility rule", () => {
    expect([K_MIN, K_MAX, DEFAULT_Y_MID, MIN_VISIBILITY]).toEqual([golden.k_min, golden.k_max, golden.default_y_mid, golden.min_visibility]);
  });
});

describe("calibration", () => {
  for (const item of golden.calibration_cases) {
    it(`matches Python: ${item.name}`, () => {
      const ratios = item.ratios.map((value) => (value === null ? Number.NaN : value));
      const result = calibrate(ratios, item.r_train);
      expect(result).toEqual({ k: item.k, rLive: item.r_live, frames: item.frames, status: item.status });
    });
  }

  it("clamps k to [0.25, 4]", () => {
    expect([clampK(0.01), clampK(0.25), clampK(1.3), clampK(4), clampK(9)]).toEqual([0.25, 0.25, 1.3, 4, 4]);
    expect(calibrate([10], 0.4).k).toBe(0.25);
    expect(calibrate([0.01], 0.4).k).toBe(4);
    expect(calibrate([0], 0.4).k).toBe(4);
  });

  it("keeps k = 1 when no frame had the nose and both shoulders visible", () => {
    const calibrator = new AspectCalibrator(2000);
    calibrator.start(0);
    const hidden = new Float64Array(99).fill(Number.NaN);
    calibrator.add(hidden, null);
    calibrator.add(null, null);
    expect(calibrator.finish(1999, 0.4)).toBeNull();
    expect(calibrator.finish(2000, 0.4)).toEqual({ k: 1, rLive: null, frames: 0, status: "no_shoulders" });
    expect(calibrator.running).toBe(false);
  });

  it("collects only visible frames within its window", () => {
    const visible = golden.visibility_cases.find((item) => item.name === "visible");
    const outside = golden.visibility_cases.find((item) => item.name === "shoulder_outside_frame");
    if (!visible || !outside) throw new Error("visibility cases missing");
    const calibrator = new AspectCalibrator(2000);
    calibrator.start(100);
    calibrator.add(block(visible.pose), block(visible.visibility));
    calibrator.add(block(outside.pose), block(outside.visibility));
    calibrator.add(block(visible.pose), Float64Array.from({ length: 33 }, () => 0.1));
    expect(calibrator.measured).toBe(1);
    expect(calibrator.remaining(1100)).toBeCloseTo(1, 10);
    const result = calibrator.finish(2100, 0.4);
    expect(result?.frames).toBe(1);
    expect(result?.k).toBe(clampK(0.4 / (visible.ratio as number)));
  });
});

describe("visibility and ratio", () => {
  for (const item of golden.visibility_cases) {
    it(`matches Python: ${item.name}`, () => {
      const pose = block(item.pose);
      const visibility = block(item.visibility);
      expect(calibrationUsable(pose, visibility)).toBe(item.usable);
      const ratio = frameRatio(pose);
      if (item.ratio === null) expect(ratio).toBeNaN();
      else expect(ratio).toBe(item.ratio);
    });
  }

  it("counts shoulders alone for the segment debug line", () => {
    const item = golden.visibility_cases.find((candidate) => candidate.name === "nose_low_visibility");
    if (!item) throw new Error("case missing");
    expect(calibrationUsable(block(item.pose), block(item.visibility))).toBe(false);
    expect(shouldersVisible(block(item.pose), block(item.visibility))).toBe(true);
  });
});

describe("per-frame correction and the preprocessing that follows", () => {
  for (const item of golden.frame_cases) {
    it(`matches Python exactly: ${item.name}`, () => {
      const input = frames(item.frames);
      input.forEach((frame, t) => {
        const corrected: RawFrame = correctFrame(frame, item.k);
        expect(same(corrected.left, item.corrected.left_hand[t] ?? null)).toBe(true);
        expect(same(corrected.right, item.corrected.right_hand[t] ?? null)).toBe(true);
        expect(same(corrected.pose, item.corrected.pose[t] ?? null)).toBe(true);
      });
    });

    for (const expected of item.results) {
      it(`preprocesses like Python: ${item.name}, ${expected.pose_indices.length} pose points`, () => {
        const config: PreprocessConfig = { ...DEFAULT_CONFIG, poseIndices: expected.pose_indices, poseSwapPairs: expected.pose_swap_pairs };
        const { result } = preprocessSegment(frames(item.frames), item.k, config);
        expect(result.info).toEqual(expected.info);
        if (expected.features === null) {
          expect(result.features).toBeNull();
          return;
        }
        const want = expected.features.flat();
        const got = result.features as Float32Array;
        expect(got.length).toBe(want.length);
        let maxDiff = 0;
        want.forEach((value, i) => {
          maxDiff = Math.max(maxDiff, Math.abs((got[i] as number) - value));
        });
        expect(maxDiff).toBeLessThanOrEqual(FEATURE_TOLERANCE);
      });
    }
  }

  it("k = 1 is the identity and returns the same frame", () => {
    const frame = frames(golden.frame_cases[0]?.frames as FrameBlocks)[0] as FrameSample;
    expect(correctFrame(frame, 1)).toBe(frame);
  });

  it("scales about 0.5 without both shoulders", () => {
    expect(shoulderMidY(null)).toBe(0.5);
    const pose = new Float64Array(99).fill(Number.NaN);
    pose[11 * 3 + 1] = 0.6;
    expect(shoulderMidY(pose)).toBe(0.5);
    pose[12 * 3 + 1] = 0.4;
    expect(shoulderMidY(pose)).toBe(0.5 * (0.6 + 0.4));
    const corrected = correctFrame({ left: Float64Array.from({ length: 63 }, () => 0.9), right: null, pose: null }, 0.5);
    expect(corrected.left?.[1]).toBe(0.5 + (0.9 - 0.5) * 0.5);
    expect(corrected.left?.[0]).toBe(0.9);
    expect(corrected.left?.[2]).toBe(0.9);
  });
});
