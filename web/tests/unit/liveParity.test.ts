// Python / browser parity on 20 real val sequences (web/tests/golden/live_parity.json, written by
// scripts/export_live_golden.py from the Python reference pipeline). Every model input the web
// pipeline computes must match Python's within the fixture's tolerance (1e-4), and the exported
// model must give the same top-1; any drift in preprocessing, hand assignment, aspect
// correction, resampling, motion energy or segmentation fails here.

import { readFileSync } from "node:fs";
import * as ort from "onnxruntime-web";
import { beforeAll, describe, expect, it } from "vitest";
import { frameK } from "../../src/aspect";
import type { DetectedHand } from "../../src/hands";
import { flatPoints, preprocessSegment, type LiveFrame } from "../../src/live";
import { LiveSession, assignHands } from "../../src/livePipeline";
import { validateMeta } from "../../src/meta";
import { DEFAULT_CONFIG, preprocessRows, toModelInput, type LongRows, type Point2, type PreprocessConfig } from "../../src/preprocess";
import { SPEC } from "../../src/spec";

interface GoldenFrame {
  t: number;
  hands: { label: string; landmarks: (number | null)[][] }[];
  pose: (number | null)[][];
  visibility: number[];
}

interface Info {
  t_raw: number;
  t_kept: number;
  mirrored: boolean;
  norm_method: string | null;
  excluded: boolean;
  reason: string | null;
}

interface Golden {
  tolerance: number;
  model_sha256: string;
  camera: { width: number; height: number; k: number };
  preprocessing: { pose_indices: number[]; pose_swap_pairs: [number, number][]; num_frames: number; min_hand_frames: number };
  dataset_cases: {
    sequence_id: number;
    label: number;
    blocks: { left_hand: (number[] | null)[]; right_hand: (number[] | null)[]; pose: (number[] | null)[] };
    info: Info;
    model_input: number[][];
    logits: number[];
  }[];
  live_cases: { sequence_id: number; label: number; frames: GoldenFrame[]; info: Info; model_input: number[][]; logits: number[] }[];
  stream: {
    frames: GoldenFrame[];
    energies: (number | null)[];
    segments: {
      kind: string;
      reason: string;
      first_frame: number | null;
      last_frame: number | null;
      frames: number;
      hand_frames: number;
      model_input: number[][] | null;
      logits: number[] | null;
    }[];
  };
}

const read = (path: string): unknown => JSON.parse(readFileSync(new URL(`../../${path}`, import.meta.url), "utf8"));
const golden = read("tests/golden/live_parity.json") as Golden;
const meta = validateMeta(read("public/model/meta.json"));

const cfg: PreprocessConfig = {
  ...DEFAULT_CONFIG,
  poseIndices: golden.preprocessing.pose_indices,
  poseSwapPairs: golden.preprocessing.pose_swap_pairs,
  numFrames: golden.preprocessing.num_frames,
  minHandFrames: golden.preprocessing.min_hand_frames,
};

// The aspect factor the app derives from the camera size and features_spec.json.
const k = frameK(golden.camera.width, golden.camera.height);

function argmax(values: ArrayLike<number>): number {
  let best = 0;
  for (let i = 1; i < values.length; i += 1) if ((values[i] as number) > (values[best] as number)) best = i;
  return best;
}

function maxAbsDiff(a: ArrayLike<number>, b: readonly number[]): number {
  expect(a.length).toBe(b.length);
  let max = 0;
  for (let i = 0; i < b.length; i += 1) max = Math.max(max, Math.abs((a[i] as number) - (b[i] as number)));
  return max;
}

/** The long rows Python built: kind by kind, frame by frame, point by point; z = 0 where x is finite. */
function rowsFromBlocks(blocks: Golden["dataset_cases"][number]["blocks"]): LongRows {
  const rows: LongRows = { frame: [], type: [], landmark_index: [], x: [], y: [], z: [] };
  const push = (frame: number, type: string, index: number, x: number | null, y: number | null): void => {
    (rows.frame as number[]).push(frame);
    (rows.type as string[]).push(type);
    (rows.landmark_index as number[]).push(index);
    (rows.x as (number | null)[]).push(x);
    (rows.y as (number | null)[]).push(y);
    (rows.z as (number | null)[]).push(x === null ? null : 0);
  };
  const kinds: [string, (number[] | null)[], readonly number[]][] = [
    ["left_hand", blocks.left_hand, Array.from({ length: 21 }, (_, i) => i)],
    ["right_hand", blocks.right_hand, Array.from({ length: 21 }, (_, i) => i)],
    ["pose", blocks.pose, golden.preprocessing.pose_indices],
  ];
  for (const [type, frames, points] of kinds) {
    frames.forEach((values, frame) => {
      points.forEach((index, p) => {
        if (values === null) push(frame, type, index, null, null);
        else push(frame, type, index, values[p * 2] as number, values[p * 2 + 1] as number);
      });
    });
  }
  return rows;
}

const points = (values: (number | null)[][]): Point2[] => values.map(([x, y]) => ({ x: x ?? Number.NaN, y: y ?? Number.NaN, z: 0 }) as Point2);

/** One golden frame through hand assignment, as the app's frame loop does it. */
function liveFrame(frame: GoldenFrame): LiveFrame {
  const pose = points(frame.pose);
  const hands: DetectedHand[] = frame.hands.map((hand) => ({ landmarks: points(hand.landmarks), label: hand.label, score: 1 }));
  const assigned = assignHands(hands, pose, "nearest_pose_wrist");
  const left = flatPoints(assigned.left, 21);
  const right = flatPoints(assigned.right, 21);
  return {
    t: frame.t,
    left,
    right,
    pose: flatPoints(pose, 33),
    visibility: Float64Array.from(frame.visibility),
    poseRan: true,
    hasHand: false,
  };
}

let session: ort.InferenceSession;
async function logitsOf(input: ArrayLike<number>): Promise<Float32Array> {
  const tensor = new ort.Tensor("float32", Float32Array.from(input), [1, cfg.numFrames, input.length / cfg.numFrames]);
  const out = await session.run({ [meta.input_layout.input_name]: tensor });
  return out[meta.input_layout.output_name]?.data as Float32Array;
}

beforeAll(async () => {
  ort.env.wasm.numThreads = 1;
  session = await ort.InferenceSession.create(readFileSync(new URL("../../public/model/model.onnx", import.meta.url)), {
    executionProviders: ["wasm"],
  });
}, 60_000);

describe("live parity fixtures", () => {
  it("come from the committed model and the shared features spec", () => {
    expect(golden.model_sha256).toBe(meta.model_sha256);
    expect(golden.dataset_cases).toHaveLength(20);
    expect(golden.live_cases).toHaveLength(20);
    expect(golden.tolerance).toBe(1e-4);
    expect(golden.preprocessing.pose_indices).toEqual(meta.preprocessing.pose_indices);
    expect(k).toBeCloseTo(SPEC.aspect.train_width_over_height / (golden.camera.width / golden.camera.height), 12);
  });
});

describe("dataset rows: 20 val sequences, raw long rows -> features -> logits", () => {
  it.each(golden.dataset_cases.map((c, i) => [i, c] as const))("case %i", async (_, item) => {
    const result = preprocessRows(rowsFromBlocks(item.blocks), cfg);
    expect(result.info).toEqual(item.info);
    const input = toModelInput(result.features as Float32Array, cfg.poseIndices) as Float32Array;
    expect(maxAbsDiff(input, item.model_input.flat())).toBeLessThanOrEqual(golden.tolerance);
    expect(argmax(await logitsOf(input))).toBe(argmax(item.logits));
  });
});

describe("live frames: 20 val sequences as MediaPipe Tasks reports them -> features -> logits", () => {
  it.each(golden.live_cases.map((c, i) => [i, c] as const))("case %i", async (_, item) => {
    const frames = item.frames.map(liveFrame);
    const { result, input } = preprocessSegment(frames, k, cfg, "time");
    expect(result.info).toEqual(item.info);
    expect(input).not.toBeNull();
    expect(maxAbsDiff(input as Float32Array, item.model_input.flat())).toBeLessThanOrEqual(golden.tolerance);
    expect(argmax(await logitsOf(input as Float32Array))).toBe(argmax(item.logits));
  });
});

describe("stream: motion energy, segmentation and each segment's input", () => {
  it("matches Python frame by frame", async () => {
    const live = new LiveSession<LiveFrame>(k);
    const frames = golden.stream.frames.map(liveFrame);
    const index = new Map(frames.map((frame, i) => [frame, i]));
    const events = [];
    frames.forEach((frame, i) => {
      const event = live.push(frame);
      const expected = golden.stream.energies[i] ?? null;
      if (expected === null) expect(Number.isNaN(live.lastEnergy), `energy of frame ${i}`).toBe(true);
      else expect(Math.abs(live.lastEnergy - expected), `energy of frame ${i}`).toBeLessThanOrEqual(1e-9 * Math.max(1, expected));
      if (event) events.push(event);
    });
    const last = live.flush();
    if (last) events.push(last);
    expect(events.map((e) => e.kind)).toEqual(golden.stream.segments.map((s) => s.kind));
    for (const [i, event] of events.entries()) {
      const expected = golden.stream.segments[i] as Golden["stream"]["segments"][number];
      expect(event.reason).toBe(expected.reason);
      expect(event.handFrames).toBe(expected.hand_frames);
      expect(event.frames.length).toBe(expected.frames);
      expect(index.get(event.frames[0] as LiveFrame)).toBe(expected.first_frame);
      expect(index.get(event.frames[event.frames.length - 1] as LiveFrame)).toBe(expected.last_frame);
      if (expected.model_input === null) continue;
      const { input } = preprocessSegment(event.frames, k, cfg, "time");
      expect(maxAbsDiff(input as Float32Array, expected.model_input.flat())).toBeLessThanOrEqual(golden.tolerance);
      expect(argmax(await logitsOf(input as Float32Array))).toBe(argmax(expected.logits as number[]));
    }
  });
});
