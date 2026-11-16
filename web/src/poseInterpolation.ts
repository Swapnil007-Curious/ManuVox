// Pose landmarker every Nth frame, with pose points for the frames in between. Pure.
//
// The hand landmarker runs on every frame. The pose landmarker runs on frames whose index is a
// multiple of N (setting 1-3); a frame between two runs gets each pose value interpolated
// linearly in time between them. Such a frame is pending until the next run; frames still pending
// when a segment ends (flush) hold the last run's pose, as nothing later is known yet. When
// either run found no pose, a pending frame takes the pose of the run nearer in time (the
// earlier one on ties).

export interface PoseFrame {
  t: number;
  pose: Float64Array | null;
  visibility: Float64Array | null;
  poseRan: boolean;
}

interface Run {
  t: number;
  pose: Float64Array | null;
  visibility: Float64Array | null;
}

export const MIN_POSE_EVERY = 1;
export const MAX_POSE_EVERY = 3;
export const DEFAULT_POSE_EVERY = 2;

export function shouldRunPose(frameIndex: number, every: number): boolean {
  return frameIndex % every === 0;
}

function lerp(a: Float64Array, b: Float64Array, weight: number): Float64Array {
  const out = new Float64Array(a.length);
  for (let i = 0; i < a.length; i += 1) {
    const start = a[i] as number;
    out[i] = start + ((b[i] as number) - start) * weight;
  }
  return out;
}

export class PoseInterpolator<F extends PoseFrame> {
  private previous: Run | null = null;
  private pending: F[] = [];

  /** The last run's pose, for drawing and guidance. */
  get latest(): Run | null {
    return this.previous;
  }

  get pendingCount(): number {
    return this.pending.length;
  }

  /** A frame on which the pose landmarker ran; resolves the frames since the previous run. */
  ran(frame: F): void {
    frame.poseRan = true;
    const current: Run = { t: frame.t, pose: frame.pose, visibility: frame.visibility };
    const previous = this.previous;
    for (const item of this.pending) {
      if (!previous) {
        item.pose = current.pose;
        item.visibility = current.visibility;
        continue;
      }
      const span = current.t - previous.t;
      const weight = span > 0 ? Math.min(Math.max((item.t - previous.t) / span, 0), 1) : 0;
      if (previous.pose && current.pose) {
        item.pose = lerp(previous.pose, current.pose, weight);
        item.visibility = previous.visibility && current.visibility ? lerp(previous.visibility, current.visibility, weight) : null;
      } else {
        const nearer = weight <= 0.5 ? previous : current;
        item.pose = nearer.pose;
        item.visibility = nearer.visibility;
      }
    }
    this.pending = [];
    this.previous = current;
  }

  /** A frame without a pose run; its pose is filled in by the next run or by flush(). */
  skipped(frame: F): void {
    frame.poseRan = false;
    frame.pose = null;
    frame.visibility = null;
    this.pending.push(frame);
  }

  /** Hold the last run's pose on every pending frame (a segment is ending). */
  flush(): void {
    for (const item of this.pending) {
      item.pose = this.previous?.pose ?? null;
      item.visibility = this.previous?.visibility ?? null;
    }
    this.pending = [];
  }

  reset(): void {
    this.previous = null;
    this.pending = [];
  }
}
