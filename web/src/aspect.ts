// Aspect calibration, a port of src/edgevision/live/aspect.py (the source of truth). Pure.
//
// MediaPipe coordinates are fractions of the frame's width (x) and height (y), so a landscape
// webcam and the portrait phone videos the model was trained on give one body different y/x
// proportions, and shoulder-width normalisation keeps the difference. The ratio
//   r = |y_nose - y_shoulder_mid| / |x_shoulder_left - x_shoulder_right|
// measures it; scaling every y about the frame's shoulder midpoint by k = r_train / r_live, before
// preprocessing, gives live frames the training ratio. x and z are unchanged.
//
// Raw blocks are flat Float64Arrays [x0, y0, z0, x1, ...] in unmirrored normalised coordinates:
// 21 points per hand, 33 for the pose, null when the block was not detected.

import { nanMedian } from "./preprocess";
import { ASPECT, POSE } from "./spec";

export const K_MIN = ASPECT.k_min;
export const K_MAX = ASPECT.k_max;
export const NOSE = POSE.nose;
export const LEFT_SHOULDER = POSE.left_shoulder;
export const RIGHT_SHOULDER = POSE.right_shoulder;
export const DEFAULT_Y_MID = ASPECT.default_y_mid;
export const MIN_VISIBILITY = ASPECT.calibration.min_visibility;
export const CALIBRATION_MS = ASPECT.calibration.seconds * 1000;
/** The training videos' effective width / height (features_spec.json aspect). */
export const TRAIN_WIDTH_OVER_HEIGHT = ASPECT.train_width_over_height;
const STRIDE = 3;

/** k converting a width x height frame to the training aspect convention, clamped (aspect.py frame_k). */
export function frameK(width: number, height: number): number {
  if (!(width > 0 && height > 0)) return 1;
  return clampK(TRAIN_WIDTH_OVER_HEIGHT / (width / height));
}

export interface RawFrame {
  left: Float64Array | null;
  right: Float64Array | null;
  pose: Float64Array | null;
}

export type CalibrationStatus = "calibrated" | "no_shoulders";

export interface Calibration {
  k: number;
  rLive: number | null;
  frames: number;
  status: CalibrationStatus;
}

function x(block: Float64Array, point: number): number {
  return block[point * STRIDE] as number;
}

function y(block: Float64Array, point: number): number {
  return block[point * STRIDE + 1] as number;
}

/** r for one pose, or NaN when the nose or a shoulder is missing or the shoulders' x coincide. */
export function frameRatio(pose: Float64Array | null): number {
  if (!pose) return Number.NaN;
  const values = [x(pose, NOSE), y(pose, NOSE), x(pose, LEFT_SHOULDER), y(pose, LEFT_SHOULDER), x(pose, RIGHT_SHOULDER), y(pose, RIGHT_SHOULDER)];
  if (!values.every(Number.isFinite)) return Number.NaN;
  const width = Math.abs(x(pose, LEFT_SHOULDER) - x(pose, RIGHT_SHOULDER));
  if (!(width > 0)) return Number.NaN;
  const mid = 0.5 * (y(pose, LEFT_SHOULDER) + y(pose, RIGHT_SHOULDER));
  return Math.abs(y(pose, NOSE) - mid) / width;
}

function visiblePoints(pose: Float64Array | null, visibility: Float64Array | null, points: readonly number[]): boolean {
  if (!pose) return false;
  return points.every((point) => {
    const px = x(pose, point);
    const py = y(pose, point);
    const inside = Number.isFinite(px) && Number.isFinite(py) && px >= 0 && px <= 1 && py >= 0 && py <= 1;
    if (!inside) return false;
    if (!visibility) return true;
    const seen = visibility[point] as number;
    return Number.isFinite(seen) && seen >= MIN_VISIBILITY;
  });
}

/** Nose and both shoulders inside the frame with visibility >= MIN_VISIBILITY (if given). */
export function calibrationUsable(pose: Float64Array | null, visibility: Float64Array | null): boolean {
  return visiblePoints(pose, visibility, [NOSE, LEFT_SHOULDER, RIGHT_SHOULDER]);
}

/** Both shoulders inside the frame with visibility >= MIN_VISIBILITY (if given). */
export function shouldersVisible(pose: Float64Array | null, visibility: Float64Array | null): boolean {
  return visiblePoints(pose, visibility, [LEFT_SHOULDER, RIGHT_SHOULDER]);
}

export function clampK(k: number): number {
  return Math.min(Math.max(k, K_MIN), K_MAX);
}

/** Median of the finite ratios as r_live and k = r_train / r_live, clamped; k = 1 without any. */
export function calibrate(ratios: readonly number[], rTrain: number): Calibration {
  const finite = ratios.filter(Number.isFinite);
  if (finite.length === 0) return { k: 1, rLive: null, frames: 0, status: "no_shoulders" };
  const rLive = nanMedian(finite);
  const k = rLive > 0 ? rTrain / rLive : K_MAX;
  return { k: clampK(k), rLive, frames: finite.length, status: "calibrated" };
}

/** Shoulder-midpoint y, or DEFAULT_Y_MID when either shoulder's y is missing. */
export function shoulderMidY(pose: Float64Array | null): number {
  if (!pose) return DEFAULT_Y_MID;
  const left = y(pose, LEFT_SHOULDER);
  const right = y(pose, RIGHT_SHOULDER);
  return Number.isFinite(left) && Number.isFinite(right) ? 0.5 * (left + right) : DEFAULT_Y_MID;
}

function correctBlock(block: Float64Array | null, yMid: number, k: number): Float64Array | null {
  if (!block) return null;
  const out = Float64Array.from(block);
  for (let i = 1; i < out.length; i += STRIDE) out[i] = yMid + ((block[i] as number) - yMid) * k;
  return out;
}

/**
 * y' = y_mid + (y - y_mid) * k for every point of the frame, about its shoulder midpoint (taken
 * before the correction, which leaves it unchanged). k === 1 returns the frame itself: the
 * arithmetic can change the last bit of y, so "no correction" must skip it, as aspect.py does.
 */
export function correctFrame<T extends RawFrame>(frame: T, k: number): T {
  if (k === 1) return frame;
  const yMid = shoulderMidY(frame.pose);
  return { ...frame, left: correctBlock(frame.left, yMid, k), right: correctBlock(frame.right, yMid, k), pose: correctBlock(frame.pose, yMid, k) };
}

/** Collects r over a fixed window from its first frame (the 2-second calibration). */
export class AspectCalibrator {
  private ratios: number[] = [];
  private startTime: number | null = null;
  private finished = false;

  constructor(private readonly durationMs = CALIBRATION_MS) {}

  start(now: number): void {
    this.ratios = [];
    this.startTime = now;
    this.finished = false;
  }

  get running(): boolean {
    return this.startTime !== null && !this.finished;
  }

  /** Seconds left in the window. */
  remaining(now: number): number {
    if (this.startTime === null) return 0;
    return Math.max(0, (this.durationMs - (now - this.startTime)) / 1000);
  }

  /** Add one frame on which the pose landmarker ran; only visible frames count. */
  add(pose: Float64Array | null, visibility: Float64Array | null): void {
    if (!this.running || !calibrationUsable(pose, visibility)) return;
    const ratio = frameRatio(pose);
    if (Number.isFinite(ratio)) this.ratios.push(ratio);
  }

  /** The calibration once the window is over, else null. Ends the window. */
  finish(now: number, rTrain: number): Calibration | null {
    if (!this.running || this.startTime === null || now - this.startTime < this.durationMs) return null;
    this.finished = true;
    return calibrate(this.ratios, rTrain);
  }

  get measured(): number {
    return this.ratios.length;
  }
}
