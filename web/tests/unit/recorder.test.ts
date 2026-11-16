// Recordings schema (shared with src/edgevision/live/recordings.py), label search and pose
// interpolation.

import { describe, expect, it } from "vitest";
import { PoseInterpolator, shouldRunPose, type PoseFrame } from "../../src/poseInterpolation";
import { DEFAULT_CONFIG, preprocessCoords, stackFrames } from "../../src/preprocess";
import {
  RECORDINGS_FORMAT,
  RecordingStore,
  appResult,
  filterLabels,
  matrix,
  nullable,
  recordFrames,
  recordingsFileName,
  validateRecordingsFile,
  type AppInfo,
  type FrameSample,
  type Recording,
} from "../../src/recorder";
import { preprocessSegment, rawHasHand, segmentStats } from "../../src/live";
import { recordingExample } from "./golden";

const APP: AppInfo = {
  model_name: "cnn",
  variant: "fp16",
  run: "cnn_signer_independent/seed_0",
  model_sha256: "a".repeat(64),
  feature_set: "v1",
  pose_indices: [0, 11, 12, 13, 14, 15, 16],
  pose_swap_pairs: [
    [11, 12],
    [13, 14],
    [15, 16],
  ],
  num_frames: 32,
  min_hand_frames: 3,
  threshold: 0.41,
  num_classes: 3,
  r_train: 0.398,
};

function hand(offset: number): Float64Array {
  return Float64Array.from({ length: 63 }, (_, i) => 0.3 + ((i * 7 + offset) % 40) / 100);
}

function pose(t: number): Float64Array {
  const values = Float64Array.from({ length: 99 }, (_, i) => 0.2 + ((i * 3 + t) % 50) / 100);
  values[0 * 3 + 1] = 0.3;
  values[11 * 3] = 0.65;
  values[11 * 3 + 1] = 0.6;
  values[12 * 3] = 0.35;
  values[12 * 3 + 1] = 0.6;
  return values;
}

function sample(t: number, withHand: boolean): FrameSample {
  return { t: 1000 + 33 * t, left: null, right: withHand ? hand(t) : null, pose: pose(t), visibility: new Float64Array(33).fill(0.9), poseRan: t % 2 === 0 };
}

function recordingFor(frames: FrameSample[], k: number): Omit<Recording, "id"> {
  const { result, input } = preprocessSegment(frames, k, DEFAULT_CONFIG);
  const outcome = input
    ? appResult(
        {
          result,
          modelInput: input,
          frameDim: 100,
          top5: [
            { index: 1, label: "bird", probability: 0.6 },
            { index: 0, label: "apple", probability: 0.3 },
            { index: 2, label: "cat", probability: 0.1 },
          ],
          accepted: true,
        },
        APP.threshold,
      )
    : appResult({ result: null, reason: "insufficient_hand_frames" }, APP.threshold);
  return {
    session: 0,
    intended_label: "bird",
    intended_index: 1,
    recorded_at: "2026-11-07T10:00:00.000Z",
    camera: { width: 640, height: 480, frame_rate: 30 },
    framing: "full",
    crop: { sx: 0, sy: 0, sw: 640, sh: 480 },
    swap_hands: false,
    pose_every: 2,
    min_hand_frames: 3,
    calibration: { mode: "auto", k, r_live: 0.8, r_train: 0.398, frames: 40, status: "calibrated" },
    segment: { capture_mode: "auto", reason: "hands_lowered", start_ms: frames[0]?.t ?? 0, end_ms: frames.at(-1)?.t ?? 0, frames: frames.length, hand_frames: frames.filter(rawHasHand).length },
    frames: recordFrames(frames),
    app_result: outcome,
  };
}

describe("recordings file", () => {
  it("accepts the example Python wrote (recording_example.json)", () => {
    expect(validateRecordingsFile(recordingExample())).toEqual([]);
  });

  it("builds files that pass its own validation, classified and discarded", () => {
    const store = new RecordingStore();
    store.add(recordingFor(Array.from({ length: 20 }, (_, t) => sample(t, true)), 0.5));
    store.add(recordingFor(Array.from({ length: 8 }, (_, t) => sample(t, t < 2)), 1));
    const file = store.toFile(APP, "vitest", new Date("2026-11-07T10:00:00Z"));
    expect(file.format).toBe(RECORDINGS_FORMAT);
    expect(validateRecordingsFile(JSON.parse(JSON.stringify(file)))).toEqual([]);
    expect(file.recordings.map((recording) => recording.id)).toEqual([0, 1]);
    expect(file.recordings.map((recording) => recording.app_result.status)).toEqual(["classified", "discarded"]);
    const classified = file.recordings[0] as Recording;
    expect(classified.app_result.features).toHaveLength(32);
    expect(classified.app_result.features?.[0]).toHaveLength(100);
    expect(classified.frames.left_hand.every((entry) => entry === null)).toBe(true);
    expect(classified.frames.right_hand[0]).toHaveLength(63);
    expect(classified.frames.pose_ran.slice(0, 4)).toEqual([true, false, true, false]);
    store.clear();
    expect(store.count).toBe(0);
  });

  it("stores NaN as null and rejects malformed files", () => {
    expect(nullable(Float64Array.from([1, Number.NaN, Number.POSITIVE_INFINITY]))).toEqual([1, null, null]);
    const store = new RecordingStore();
    store.add(recordingFor(Array.from({ length: 12 }, (_, t) => sample(t, true)), 1));
    const good = JSON.parse(JSON.stringify(store.toFile(APP, "vitest")));
    const broken = (change: (file: { recordings: Record<string, unknown>[]; [key: string]: unknown }) => void): string[] => {
      const copy = structuredClone(good);
      change(copy);
      return validateRecordingsFile(copy);
    };
    expect(broken((file) => (file.extra = 1))).toContain("file: unexpected extra");
    expect(broken((file) => ((file.recordings[0] as { frames: { pose: unknown[] } }).frames.pose = []))).not.toEqual([]);
    expect(broken((file) => (((file.recordings[0] as { frames: { right_hand: unknown[][] } }).frames.right_hand[0] as unknown[]).length = 10))).not.toEqual([]);
    expect(broken((file) => ((file.recordings[0] as { app_result: { top5: unknown[] } }).app_result.top5 = []))).not.toEqual([]);
    expect(broken((file) => ((file.recordings[0] as { pose_every: number }).pose_every = 4))).not.toEqual([]);
    expect(broken((file) => (file.schema_version = 2))).not.toEqual([]);
    expect(validateRecordingsFile(null)).toEqual(["not an object"]);
  });

  it("names the download with the local time", () => {
    expect(recordingsFileName(new Date(2026, 10, 7, 9, 5, 3))).toBe("edge-vision-recordings-20261107-090503.json");
  });

  it("packs matrices row by row", () => {
    expect(matrix(Float32Array.from([1, 2, 3, 4, 5, 6]), 3)).toEqual([
      [1, 2, 3],
      [4, 5, 6],
    ]);
  });
});

describe("segment pipeline", () => {
  it("k = 1 equals preprocessing the raw frames directly", () => {
    const frames = Array.from({ length: 16 }, (_, t) => sample(t, t !== 3));
    const { result } = preprocessSegment(frames, 1, DEFAULT_CONFIG);
    expect(result.info.t_raw).toBe(16);
    expect(result.info.t_kept).toBe(15);
    expect(result.info.norm_method).toBe("shoulders");
    const stats = segmentStats(frames);
    expect(stats.handsPresent).toBeCloseTo(15 / 16, 12);
    expect(stats.shouldersVisible).toBe(1);
    expect(preprocessCoords(stackFrames([]), DEFAULT_CONFIG).info.excluded).toBe(true);
  });
});

describe("label search", () => {
  const labels = ["TV", "bird", "birthday", "blue", "thankyou"];

  it("filters case-insensitively on every word, keeping label order and indices", () => {
    expect(filterLabels(labels, "bir")).toEqual([
      { index: 1, label: "bird" },
      { index: 2, label: "birthday" },
    ]);
    expect(filterLabels(labels, "tv")).toEqual([{ index: 0, label: "TV" }]);
    expect(filterLabels(labels, "  ")).toHaveLength(5);
    expect(filterLabels(labels, "th you")).toEqual([{ index: 4, label: "thankyou" }]);
  });
});

describe("pose interpolation", () => {
  function frame(t: number, value: number | null): PoseFrame {
    return { t, pose: value === null ? null : Float64Array.from([value, value * 2]), visibility: value === null ? null : Float64Array.from([1]), poseRan: false };
  }

  it("runs the pose landmarker on every Nth frame", () => {
    expect([0, 1, 2, 3, 4, 5].map((i) => shouldRunPose(i, 2))).toEqual([true, false, true, false, true, false]);
    expect([0, 1, 2, 3].map((i) => shouldRunPose(i, 3))).toEqual([true, false, false, true]);
    expect([0, 1].map((i) => shouldRunPose(i, 1))).toEqual([true, true]);
  });

  it("interpolates linearly in time between two runs", () => {
    const interpolator = new PoseInterpolator<PoseFrame>();
    const first = frame(0, 0.2);
    interpolator.ran(first);
    const middle = frame(25, null);
    const late = frame(75, null);
    interpolator.skipped(middle);
    interpolator.skipped(late);
    expect(middle.pose).toBeNull();
    expect(interpolator.pendingCount).toBe(2);
    interpolator.ran(frame(100, 0.6));
    expect(Array.from(middle.pose as Float64Array)).toEqual([0.2 + (0.6 - 0.2) * 0.25, 0.4 + (1.2 - 0.4) * 0.25]);
    expect(Array.from(late.pose as Float64Array)).toEqual([0.2 + (0.6 - 0.2) * 0.75, 0.4 + (1.2 - 0.4) * 0.75]);
    expect(middle.poseRan).toBe(false);
    expect(interpolator.pendingCount).toBe(0);
  });

  it("takes the nearer run when one found no pose, and holds the last pose on flush", () => {
    const interpolator = new PoseInterpolator<PoseFrame>();
    interpolator.ran(frame(0, 0.5));
    const near = frame(10, null);
    const far = frame(90, null);
    interpolator.skipped(near);
    interpolator.skipped(far);
    interpolator.ran(frame(100, null));
    expect(Array.from(near.pose as Float64Array)).toEqual([0.5, 1]);
    expect(far.pose).toBeNull();
    interpolator.ran(frame(130, 0.9));
    const pending = frame(140, null);
    interpolator.skipped(pending);
    interpolator.flush();
    expect(Array.from(pending.pose as Float64Array)).toEqual([0.9, 1.8]);
    interpolator.reset();
    expect(interpolator.latest).toBeNull();
  });
});
