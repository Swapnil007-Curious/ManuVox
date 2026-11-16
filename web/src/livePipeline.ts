// The live pipeline's per-frame steps, a port of src/edgevision/live/pipeline.py (the source of
// truth): hand assignment, aspect correction of one frame, motion energy and the motion
// segmenter. Pure: no DOM, no timers. web/tests/unit/liveParity.test.ts checks it against the
// golden fixtures Python writes.

import { DEFAULT_Y_MID } from "./aspect";
import type { DetectedHand } from "./hands";
import type { Point2 } from "./preprocess";
import { FINGERTIPS, MIDDLE_MCP, MIN_HAND_FRAMES, MIN_SCALE, MOTION, N_HAND, POSE, SEGMENTER, WRIST } from "./spec";

export type HandRule = "nearest_pose_wrist" | "label" | "label_inverted";
export const HAND_RULES: readonly HandRule[] = ["nearest_pose_wrist", "label", "label_inverted"];
const MOTION_POINTS: readonly number[] = [WRIST, ...FINGERTIPS];
const STRIDE = 3;

export interface Assigned {
  left: readonly Point2[] | null;
  right: readonly Point2[] | null;
}

// ---------------------------------------------------------------------------------------------
// 1. Hand assignment

/** Label rule; two hands with the same or an unknown label: the image-left one is the right hand. */
function byLabel(hands: readonly DetectedHand[], inverted: boolean): Assigned {
  const blocks = hands.map((hand) => {
    const own = hand.label === "Left" ? "left" : hand.label === "Right" ? "right" : null;
    if (own === null || !inverted) return own;
    return own === "left" ? "right" : "left";
  });
  const [first, second] = hands;
  if (first && second && (blocks[0] === blocks[1] || blocks[0] === null || blocks[1] === null)) {
    const firstX = (first.landmarks[0] as Point2).x;
    const secondX = (second.landmarks[0] as Point2).x;
    const [imageLeft, imageRight] = firstX <= secondX ? [first, second] : [second, first];
    return { left: imageRight.landmarks, right: imageLeft.landmarks };
  }
  const out: { left: readonly Point2[] | null; right: readonly Point2[] | null } = { left: null, right: null };
  hands.forEach((hand, i) => {
    if (blocks[i] === "left") out.left = hand.landmarks;
    else if (blocks[i] === "right") out.right = hand.landmarks;
  });
  return out;
}

function distance(a: Point2, b: Point2): number {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function finitePoint(point: Point2 | undefined): point is Point2 {
  return point !== undefined && Number.isFinite(point.x) && Number.isFinite(point.y);
}

/**
 * assign_hands: nearest_pose_wrist is Holistic's rule (one hand takes the block of the nearer
 * pose wrist, 15 being the signer's left; two hands take the pairing with the smaller total
 * distance; without pose wrists the label decides). label: Tasks' true handedness on an
 * unmirrored frame. label_inverted: the first demo's rule (hands.ts blockForLabel).
 */
export function assignHands(hands: readonly DetectedHand[], pose: readonly Point2[] | null, rule: HandRule): Assigned {
  const usable = hands.filter((hand) => hand.landmarks.length >= N_HAND).slice(0, 2);
  if (usable.length === 0) return { left: null, right: null };
  // The first demo's mapping (hands.ts blockForLabel): "Left" -> right_hand.
  if (rule === "label_inverted") return byLabel(usable, true);
  const leftWrist = pose?.[POSE.left_wrist];
  const rightWrist = pose?.[POSE.right_wrist];
  const wristsKnown =
    finitePoint(leftWrist) && finitePoint(rightWrist) && usable.every((hand) => finitePoint(hand.landmarks[0]));
  if (rule === "label" || !wristsKnown) return byLabel(usable, false);
  const l = leftWrist as Point2;
  const r = rightWrist as Point2;
  if (usable.length === 1) {
    const hand = usable[0] as DetectedHand;
    const wrist = hand.landmarks[0] as Point2;
    const toLeft = distance(wrist, l);
    const toRight = distance(wrist, r);
    if (toLeft === toRight) return byLabel(usable, false);
    return toLeft < toRight ? { left: hand.landmarks, right: null } : { left: null, right: hand.landmarks };
  }
  const [first, second] = usable as [DetectedHand, DetectedHand];
  const a = first.landmarks[0] as Point2;
  const b = second.landmarks[0] as Point2;
  const straight = distance(a, l) + distance(b, r);
  const crossed = distance(a, r) + distance(b, l);
  if (straight === crossed) return byLabel(usable, false);
  return straight < crossed ? { left: first.landmarks, right: second.landmarks } : { left: second.landmarks, right: first.landmarks };
}

// ---------------------------------------------------------------------------------------------
// 2. Aspect correction of one frame (flat [x, y, z, ...] blocks)

export function shoulderMidY(pose: Float64Array | null): number {
  if (!pose) return DEFAULT_Y_MID;
  const left = pose[POSE.left_shoulder * STRIDE + 1] as number;
  const right = pose[POSE.right_shoulder * STRIDE + 1] as number;
  return Number.isFinite(left) && Number.isFinite(right) ? 0.5 * (left + right) : DEFAULT_Y_MID;
}

export function correctBlock(block: Float64Array | null, yMid: number, k: number): Float64Array | null {
  if (!block) return null;
  const out = Float64Array.from(block);
  if (k !== 1) for (let i = 1; i < out.length; i += STRIDE) out[i] = yMid + ((block[i] as number) - yMid) * k;
  return out;
}

// ---------------------------------------------------------------------------------------------
// 3. Motion energy

function point(block: Float64Array, index: number): [number, number] {
  return [block[index * STRIDE] as number, block[index * STRIDE + 1] as number];
}

/** MotionTracker: speed of the wrist and fingertips over window_ms, in shoulder widths per second. */
export class MotionTracker {
  private lastScale: number | null = null;
  private history = new Map<"left" | "right", { time: number; points: Float64Array }[]>();

  constructor(
    private readonly windowMs = MOTION.window_ms,
    private readonly handScaleFactor = MOTION.hand_scale_factor,
  ) {}

  reset(): void {
    this.lastScale = null;
    this.history.clear();
  }

  private scale(left: Float64Array | null, right: Float64Array | null, pose: Float64Array | null): number | null {
    if (pose) {
      const [ax, ay] = point(pose, POSE.left_shoulder);
      const [bx, by] = point(pose, POSE.right_shoulder);
      if ([ax, ay, bx, by].every(Number.isFinite)) {
        const d = Math.hypot(ax - bx, ay - by);
        if (d > MIN_SCALE) {
          this.lastScale = d;
          return d;
        }
      }
    }
    if (this.lastScale !== null) return this.lastScale;
    for (const hand of [right, left]) {
      if (!hand) continue;
      const [wx, wy] = point(hand, WRIST);
      const [mx, my] = point(hand, MIDDLE_MCP);
      const size = Math.hypot(mx - wx, my - wy);
      if (size > MIN_SCALE) return this.handScaleFactor * size;
    }
    return null;
  }

  /** One frame of aspect-corrected blocks (null for an absent hand); NaN without a hand. */
  update(timeMs: number, left: Float64Array | null, right: Float64Array | null, pose: Float64Array | null): number {
    const scale = this.scale(left, right, pose);
    let energy = Number.NaN;
    for (const [name, hand] of [
      ["left", left],
      ["right", right],
    ] as const) {
      if (!hand) {
        this.history.delete(name);
        continue;
      }
      const points = new Float64Array(MOTION_POINTS.length * 2);
      MOTION_POINTS.forEach((index, i) => {
        const [x, y] = point(hand, index);
        points[i * 2] = x;
        points[i * 2 + 1] = y;
      });
      let history = (this.history.get(name) ?? []).filter((item) => item.time < timeMs);
      let speed = 0;
      if (history.length > 0 && scale !== null) {
        const older = history.filter((item) => item.time <= timeMs - this.windowMs);
        const reference = older.length > 0 ? (older[older.length - 1] as (typeof history)[number]) : (history[0] as (typeof history)[number]);
        let sum = 0;
        for (let i = 0; i < MOTION_POINTS.length; i += 1) {
          sum += Math.hypot((points[i * 2] as number) - (reference.points[i * 2] as number), (points[i * 2 + 1] as number) - (reference.points[i * 2 + 1] as number));
        }
        speed = sum / MOTION_POINTS.length / scale / ((timeMs - reference.time) / 1000);
      }
      const olderCount = history.filter((item) => item.time <= timeMs - this.windowMs).length;
      history = history.slice(Math.max(olderCount - 1, 0));
      history.push({ time: timeMs, points });
      this.history.set(name, history);
      energy = Number.isNaN(energy) ? speed : Math.max(energy, speed);
    }
    return energy;
  }
}

// ---------------------------------------------------------------------------------------------
// 4. The motion segmenter

export interface MotionSegmenterSettings {
  startEnergy: number;
  stillEnergy: number;
  stillMs: number;
  handLostMs: number;
  maxMs: number;
  minMs: number;
  prerollMs: number;
  tailMs: number;
  minHandFrames: number;
}

export const DEFAULT_MOTION_SETTINGS: MotionSegmenterSettings = {
  startEnergy: SEGMENTER.start_energy,
  stillEnergy: SEGMENTER.still_energy,
  stillMs: SEGMENTER.still_ms,
  handLostMs: SEGMENTER.hand_lost_ms,
  maxMs: SEGMENTER.max_ms,
  minMs: SEGMENTER.min_ms,
  prerollMs: SEGMENTER.preroll_ms,
  tailMs: SEGMENTER.tail_ms,
  minHandFrames: MIN_HAND_FRAMES,
};

export interface MotionFrame<T> {
  frame: T;
  time: number;
  hasHand: boolean;
  energy: number;
}

export type MotionEndReason = "hands_lost" | "still" | "max_length" | "flush";

export interface MotionEvent<T> {
  kind: "segment" | "discarded";
  frames: MotionFrame<T>[];
  reason: MotionEndReason;
  start: number;
  end: number;
  handFrames: number;
}

/** MotionSegmenter: IDLE -> SIGNING -> END on motion energy. */
export class MotionSegmenter<T> {
  private recent: MotionFrame<T>[] = [];
  private frames: MotionFrame<T>[] = [];
  private signing = false;
  private start = 0;
  private stillSince: number | null = null;
  private lostSince: number | null = null;

  constructor(private settings: MotionSegmenterSettings = DEFAULT_MOTION_SETTINGS) {}

  get state(): "idle" | "signing" {
    return this.signing ? "signing" : "idle";
  }

  /** The frames of the segment in progress (for early guesses). */
  get current(): readonly MotionFrame<T>[] {
    return this.frames;
  }

  get startTime(): number {
    return this.start;
  }

  configure(settings: MotionSegmenterSettings): void {
    this.settings = { ...settings };
    this.reset();
  }

  reset(): void {
    this.recent = [];
    this.frames = [];
    this.signing = false;
    this.start = 0;
    this.stillSince = null;
    this.lostSince = null;
  }

  push(frame: T, time: number, hasHand: boolean, energy: number): MotionEvent<T> | null {
    const s = this.settings;
    const timed: MotionFrame<T> = { frame, time, hasHand, energy };
    if (!this.signing) {
      this.recent = this.recent.filter((item) => item.time >= time - s.prerollMs);
      this.recent.push(timed);
      if (!(hasHand && energy >= s.startEnergy)) return null;
      this.signing = true;
      this.frames = this.recent;
      this.recent = [];
      this.start = (this.frames[0] as MotionFrame<T>).time;
      this.stillSince = null;
      this.lostSince = null;
      return this.checkLength(time);
    }
    this.frames.push(timed);
    if (!hasHand) {
      if (this.lostSince === null) this.lostSince = time;
      if (time - this.lostSince >= s.handLostMs) {
        const cut = this.lostSince;
        return this.finish(this.frames.filter((item) => item.time < cut), "hands_lost");
      }
      return this.checkLength(time);
    }
    this.lostSince = null;
    if (energy < s.stillEnergy) {
      if (this.stillSince === null) this.stillSince = time;
      if (time - this.stillSince >= s.stillMs) {
        const keepUntil = this.stillSince + s.tailMs;
        return this.finish(this.frames.filter((item) => item.time <= keepUntil), "still");
      }
    } else {
      this.stillSince = null;
    }
    return this.checkLength(time);
  }

  flush(): MotionEvent<T> | null {
    if (!this.signing) {
      this.reset();
      return null;
    }
    return this.finish([...this.frames], "flush");
  }

  private checkLength(time: number): MotionEvent<T> | null {
    if (time - this.start < this.settings.maxMs) return null;
    return this.finish([...this.frames], "max_length");
  }

  private finish(frames: MotionFrame<T>[], reason: MotionEndReason): MotionEvent<T> {
    const start = this.start;
    this.reset();
    const handFrames = frames.filter((item) => item.hasHand).length;
    const end = frames.length > 0 ? (frames[frames.length - 1] as MotionFrame<T>).time : start;
    const kind = end - start < this.settings.minMs || handFrames < this.settings.minHandFrames ? "discarded" : "segment";
    return { kind, frames, reason, start, end, handFrames };
  }
}

// ---------------------------------------------------------------------------------------------
// The stream: assigned frames in, segments out (edgevision.live.pipeline.LivePipeline)

export interface SessionFrame {
  t: number;
  left: Float64Array | null;
  right: Float64Array | null;
  pose: Float64Array | null;
}

export interface SessionEvent<F extends SessionFrame> {
  kind: "segment" | "discarded";
  frames: F[];
  reason: string;
  start: number;
  end: number;
  handFrames: number;
}

function fullHand(block: Float64Array | null): Float64Array | null {
  if (!block) return null;
  for (let i = 0; i < N_HAND; i += 1) {
    if (!Number.isFinite(block[i * STRIDE] as number) || !Number.isFinite(block[i * STRIDE + 1] as number)) return null;
  }
  return block;
}

/** Motion energy and the motion segmenter over a stream of frames, with aspect factor k. */
export class LiveSession<F extends SessionFrame> {
  readonly motion = new MotionTracker();
  readonly segmenter: MotionSegmenter<F>;
  lastEnergy = Number.NaN;

  constructor(
    public k: number,
    settings: MotionSegmenterSettings = DEFAULT_MOTION_SETTINGS,
  ) {
    this.segmenter = new MotionSegmenter<F>(settings);
  }

  reset(): void {
    this.motion.reset();
    this.segmenter.reset();
    this.lastEnergy = Number.NaN;
  }

  /**
   * Feed one frame (hands already assigned, before aspect correction). `motionPose` is the pose
   * motion energy uses for the shoulder midpoint and scale: the frame's own by default; the app
   * passes the latest detected pose on frames where the pose landmarker was skipped, since a
   * missing pose would move y_mid to the frame centre and show the shift as motion.
   */
  push(frame: F, motionPose: Float64Array | null = frame.pose): SessionEvent<F> | null {
    const left = fullHand(frame.left);
    const right = fullHand(frame.right);
    const yMid = shoulderMidY(motionPose);
    this.lastEnergy = this.motion.update(
      frame.t,
      correctBlock(left, yMid, this.k),
      correctBlock(right, yMid, this.k),
      correctBlock(motionPose, yMid, this.k),
    );
    return this.toEvent(this.segmenter.push(frame, frame.t, left !== null || right !== null, this.lastEnergy));
  }

  flush(): SessionEvent<F> | null {
    return this.toEvent(this.segmenter.flush());
  }

  private toEvent(event: MotionEvent<F> | null): SessionEvent<F> | null {
    if (!event) return null;
    return {
      kind: event.kind,
      frames: event.frames.map((item) => item.frame),
      reason: event.reason,
      start: event.start,
      end: event.end,
      handFrames: event.handFrames,
    };
  }
}
