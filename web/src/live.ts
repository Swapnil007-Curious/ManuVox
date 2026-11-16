// Live frames and the segment pipeline: raw landmarks per frame, then at segment end the aspect
// correction, selection, preprocessing and the float16 storage step. Pure. Python replays the
// same steps (src/edgevision/live/recordings.py replay) on recorded raw frames.

import { correctFrame, shouldersVisible } from "./aspect";
import {
  preprocessCoords,
  preprocessCoordsOnClock,
  rawFrameCoords,
  stackFrames,
  toModelInput,
  type PreprocessConfig,
  type PreprocessResult,
} from "./preprocess";
import type { FrameSample } from "./recorder";
import { N_HAND, N_POSE } from "./spec";

export const HAND_POINTS = N_HAND;
export const POSE_POINTS = N_POSE;
export type Resampling = "time" | "index";

/** One processed camera frame: raw blocks (unmirrored, before aspect correction). */
export interface LiveFrame extends FrameSample {
  hasHand: boolean;
}

interface Landmark {
  x: number;
  y: number;
  z?: number;
  visibility?: number;
}

/** Flat [x0, y0, z0, ...] of `count` points, or null without a detection. */
export function flatPoints(points: readonly Landmark[] | null, count: number): Float64Array | null {
  if (!points || points.length < count) return null;
  const out = new Float64Array(count * 3);
  for (let i = 0; i < count; i += 1) {
    const point = points[i] as Landmark;
    out[i * 3] = point.x;
    out[i * 3 + 1] = point.y;
    out[i * 3 + 2] = point.z ?? Number.NaN;
  }
  return out;
}

/** Per-point visibility of a pose, NaN where the landmarker gives none. */
export function flatVisibility(points: readonly Landmark[] | null): Float64Array | null {
  if (!points || points.length < POSE_POINTS) return null;
  return Float64Array.from(points.slice(0, POSE_POINTS), (point) => point.visibility ?? Number.NaN);
}

function fullHand(block: Float64Array | null): boolean {
  if (!block) return false;
  for (let i = 0; i < HAND_POINTS; i += 1) {
    if (!Number.isFinite(block[i * 3] as number) || !Number.isFinite(block[i * 3 + 1] as number)) return false;
  }
  return true;
}

/** A frame has a hand when all 21 x, y of either hand are finite (preprocessing's rule). */
export function rawHasHand(frame: Pick<FrameSample, "left" | "right">): boolean {
  return fullHand(frame.left) || fullHand(frame.right);
}

export interface SegmentOutcome {
  result: PreprocessResult;
  /** The model input after the float16 storage step, or null when excluded. */
  input: Float32Array | null;
}

/**
 * Aspect correction with `k`, then preprocessing (on the kept-frame clock of the frames' times,
 * or on the kept-frame index) and the storage step, for one segment.
 */
export function preprocessSegment(
  frames: readonly FrameSample[],
  k: number,
  cfg: PreprocessConfig,
  resampling: Resampling = "time",
): SegmentOutcome {
  const coords = frames.map((frame) => {
    const corrected = correctFrame(frame, k);
    return rawFrameCoords(corrected.left, corrected.right, corrected.pose, cfg.poseIndices, cfg.xGrid);
  });
  const stacked = stackFrames(coords, cfg.poseIndices);
  const result =
    resampling === "time"
      ? preprocessCoordsOnClock(stacked, frames.map((frame) => frame.t), cfg)
      : preprocessCoords(stacked, cfg);
  const input = result.features ? toModelInput(result.features, cfg.poseIndices) : null;
  return { result, input };
}

export interface SegmentStats {
  frames: number;
  shouldersVisible: number;
  handsPresent: number;
}

/** Fractions of a segment's frames with both shoulders visible and with a hand. */
export function segmentStats(frames: readonly FrameSample[]): SegmentStats {
  const n = frames.length;
  const shoulders = frames.filter((frame) => shouldersVisible(frame.pose, frame.visibility)).length;
  const hands = frames.filter((frame) => rawHasHand(frame)).length;
  return { frames: n, shouldersVisible: n ? shoulders / n : 0, handsPresent: n ? hands / n : 0 };
}
