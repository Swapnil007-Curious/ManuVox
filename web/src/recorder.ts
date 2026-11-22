// Record test signs: completed segments kept in memory with their raw landmarks, the settings in
// force and what the app computed, exported as one JSON file for scripts/live_diagnose.py.
// No video or images are stored. Pure: no DOM access. The schema matches
// src/edgevision/live/recordings.py; validateRecordingsFile checks it field by field.

import type { PreprocessResult, SequenceInfo } from "./preprocess";

export const RECORDINGS_FORMAT = "edge-vision-live-recordings";
export const RECORDINGS_SCHEMA_VERSION = 1;
export const HAND_VALUES = 21 * 3;
export const POSE_VALUES = 33 * 3;
export const POSE_POINTS = 33;

export type Values = (number | null)[];
export type AspectMode = "frame" | "auto" | "manual" | "off";
export type CalibrationRecordStatus = "frame" | "calibrated" | "no_shoulders" | "manual" | "off" | "pending" | "unavailable";
export type AppStatus = "classified" | "excluded" | "discarded";

export interface AppInfo {
  model_name: string;
  variant: string;
  run: string;
  model_sha256: string;
  feature_set: "v1" | "v2";
  pose_indices: number[];
  pose_swap_pairs: [number, number][];
  num_frames: number;
  min_hand_frames: number;
  threshold: number;
  num_classes: number;
  r_train: number | null;
}

export interface CalibrationRecord {
  mode: AspectMode;
  k: number;
  r_live: number | null;
  r_train: number | null;
  frames: number;
  status: CalibrationRecordStatus;
}

export interface CandidateRecord {
  index: number;
  label: string;
  probability: number;
}

export interface AppResult {
  status: AppStatus;
  reason: string | null;
  info: SequenceInfo | null;
  features: number[][] | null;
  model_input: number[][] | null;
  top5: CandidateRecord[];
  prediction: string | null;
  max_probability: number | null;
  accepted: boolean | null;
  threshold: number;
}

export interface RecordedFrames {
  t_ms: number[];
  left_hand: (Values | null)[];
  right_hand: (Values | null)[];
  pose: (Values | null)[];
  pose_visibility: (Values | null)[];
  pose_ran: boolean[];
}

export interface Recording {
  id: number;
  session: number;
  intended_label: string;
  intended_index: number;
  recorded_at: string;
  camera: { width: number; height: number; frame_rate: number | null };
  framing: "full" | "portrait";
  crop: { sx: number; sy: number; sw: number; sh: number };
  swap_hands: boolean;
  pose_every: number;
  min_hand_frames: number;
  calibration: CalibrationRecord;
  segment: {
    capture_mode: "auto" | "manual";
    reason: "hands_lowered" | "max_length" | "manual_release" | "hands_lost" | "still" | "flush";
    start_ms: number;
    end_ms: number;
    frames: number;
    hand_frames: number;
  };
  frames: RecordedFrames;
  app_result: AppResult;
}

export interface RecordingsFile {
  format: typeof RECORDINGS_FORMAT;
  schema_version: typeof RECORDINGS_SCHEMA_VERSION;
  created_at: string;
  user_agent: string;
  app: AppInfo;
  recordings: Recording[];
}

/** The per-frame data a recording keeps: raw blocks before any aspect correction. */
export interface FrameSample {
  t: number;
  left: Float64Array | null;
  right: Float64Array | null;
  pose: Float64Array | null;
  visibility: Float64Array | null;
  poseRan: boolean;
}

/** NaN (and infinities) become null, so the JSON stays valid. */
export function nullable(values: ArrayLike<number> | null): Values | null {
  if (!values) return null;
  return Array.from(values, (value) => (Number.isFinite(value) ? value : null));
}

export function recordFrames(frames: readonly FrameSample[]): RecordedFrames {
  return {
    t_ms: frames.map((frame) => frame.t),
    left_hand: frames.map((frame) => nullable(frame.left)),
    right_hand: frames.map((frame) => nullable(frame.right)),
    pose: frames.map((frame) => nullable(frame.pose)),
    pose_visibility: frames.map((frame) => nullable(frame.visibility)),
    pose_ran: frames.map((frame) => frame.poseRan),
  };
}

/** (rows x width) float32 values as nested lists. */
export function matrix(values: Float32Array, width: number): number[][] {
  const rows: number[][] = [];
  for (let start = 0; start < values.length; start += width) rows.push(Array.from(values.subarray(start, start + width)));
  return rows;
}

export interface ClassifiedOutcome {
  result: PreprocessResult;
  modelInput: Float32Array;
  frameDim: number;
  top5: CandidateRecord[];
  accepted: boolean;
}

/** app_result for a classified segment, a preprocessing exclusion, or a segmenter discard. */
export function appResult(
  outcome: ClassifiedOutcome | { result: PreprocessResult | null; reason: string },
  threshold: number,
): AppResult {
  if ("modelInput" in outcome) {
    const best = outcome.top5[0] as CandidateRecord;
    return {
      status: "classified",
      reason: null,
      info: outcome.result.info,
      features: matrix(outcome.result.features as Float32Array, outcome.frameDim),
      model_input: matrix(outcome.modelInput, outcome.frameDim),
      top5: outcome.top5,
      prediction: best.label,
      max_probability: best.probability,
      accepted: outcome.accepted,
      threshold,
    };
  }
  return {
    status: outcome.result ? "excluded" : "discarded",
    reason: outcome.reason,
    info: outcome.result?.info ?? null,
    features: null,
    model_input: null,
    top5: [],
    prediction: null,
    max_probability: null,
    accepted: null,
    threshold,
  };
}

export class RecordingStore {
  private items: Recording[] = [];

  get count(): number {
    return this.items.length;
  }

  get recordings(): readonly Recording[] {
    return this.items;
  }

  /** Adds a recording; its id is its position in this store. */
  add(recording: Omit<Recording, "id">): Recording {
    const stored = { id: this.items.length, ...recording } as Recording;
    this.items.push(stored);
    return stored;
  }

  clear(): void {
    this.items = [];
  }

  toFile(app: AppInfo, userAgent: string, now: Date = new Date()): RecordingsFile {
    return {
      format: RECORDINGS_FORMAT,
      schema_version: RECORDINGS_SCHEMA_VERSION,
      created_at: now.toISOString(),
      user_agent: userAgent,
      app,
      recordings: [...this.items],
    };
  }
}

export function recordingsFileName(date = new Date()): string {
  const pad = (value: number): string => String(value).padStart(2, "0");
  return `manuvox-recordings-${date.getFullYear()}${pad(date.getMonth() + 1)}${pad(date.getDate())}-${pad(date.getHours())}${pad(date.getMinutes())}${pad(date.getSeconds())}.json`;
}

/** Labels containing every word of the query (case-insensitive), in label order. */
export function filterLabels(labels: readonly string[], query: string): { index: number; label: string }[] {
  const words = query.toLowerCase().split(/\s+/).filter(Boolean);
  return labels
    .map((label, index) => ({ index, label }))
    .filter(({ label }) => words.every((word) => label.toLowerCase().includes(word)));
}

// ---------------------------------------------------------------------------------------------
// Validation, as strict as the Python schema: exact key sets, types and per-frame shapes.

type Problems = string[];

function isObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function keys(value: Record<string, unknown>, expected: readonly string[], path: string, problems: Problems): void {
  const actual = Object.keys(value);
  const missing = expected.filter((key) => !actual.includes(key));
  const extra = actual.filter((key) => !expected.includes(key));
  if (missing.length) problems.push(`${path}: missing ${missing.join(", ")}`);
  if (extra.length) problems.push(`${path}: unexpected ${extra.join(", ")}`);
}

const isNumber = (value: unknown): value is number => typeof value === "number" && Number.isFinite(value);
const isInteger = (value: unknown): value is number => Number.isInteger(value);
const isNullableNumber = (value: unknown): boolean => value === null || isNumber(value);
const isString = (value: unknown): value is string => typeof value === "string";
const isBoolean = (value: unknown): value is boolean => typeof value === "boolean";

function check(condition: boolean, problems: Problems, message: string): void {
  if (!condition) problems.push(message);
}

function checkValues(entries: unknown, size: number, frames: number, path: string, problems: Problems): void {
  if (!Array.isArray(entries) || entries.length !== frames) {
    problems.push(`${path}: expected ${frames} entries`);
    return;
  }
  entries.forEach((entry, t) => {
    if (entry === null) return;
    if (!Array.isArray(entry) || entry.length !== size || !entry.every(isNullableNumber)) {
      problems.push(`${path}[${t}]: expected null or ${size} numbers or nulls`);
    }
  });
}

function checkMatrix(value: unknown, path: string, problems: Problems): void {
  if (!Array.isArray(value) || !value.every((row) => Array.isArray(row) && row.every(isNumber))) {
    problems.push(`${path}: expected a matrix of numbers`);
  }
}

function checkApp(app: unknown, problems: Problems): void {
  if (!isObject(app)) {
    problems.push("app: expected an object");
    return;
  }
  keys(app, ["model_name", "variant", "run", "model_sha256", "feature_set", "pose_indices", "pose_swap_pairs", "num_frames", "min_hand_frames", "threshold", "num_classes", "r_train"], "app", problems);
  check(isString(app.model_name) && isString(app.variant) && isString(app.run), problems, "app: names must be strings");
  check(isString(app.model_sha256) && /^[0-9a-f]{64}$/.test(app.model_sha256), problems, "app.model_sha256: not a sha256");
  check(app.feature_set === "v1" || app.feature_set === "v2", problems, "app.feature_set: v1 or v2");
  check(Array.isArray(app.pose_indices) && app.pose_indices.every(isInteger), problems, "app.pose_indices: integers");
  check(
    Array.isArray(app.pose_swap_pairs) && app.pose_swap_pairs.every((pair) => Array.isArray(pair) && pair.length === 2 && pair.every(isInteger)),
    problems,
    "app.pose_swap_pairs: integer pairs",
  );
  check(isInteger(app.num_frames) && isInteger(app.min_hand_frames) && isInteger(app.num_classes), problems, "app: counts must be integers");
  check(isNumber(app.threshold) && app.threshold >= 0 && app.threshold <= 1, problems, "app.threshold: in [0, 1]");
  check(isNullableNumber(app.r_train), problems, "app.r_train: number or null");
}

const SEQUENCE_INFO_KEYS = ["t_raw", "t_kept", "mirrored", "norm_method", "excluded", "reason"] as const;

function checkAppResult(value: unknown, path: string, problems: Problems): void {
  if (!isObject(value)) {
    problems.push(`${path}: expected an object`);
    return;
  }
  keys(value, ["status", "reason", "info", "features", "model_input", "top5", "prediction", "max_probability", "accepted", "threshold"], path, problems);
  const classified = value.status === "classified";
  check(["classified", "excluded", "discarded"].includes(value.status as string), problems, `${path}.status: invalid`);
  check(value.reason === null || isString(value.reason), problems, `${path}.reason: string or null`);
  if (value.info !== null) {
    if (!isObject(value.info)) problems.push(`${path}.info: object or null`);
    else keys(value.info, SEQUENCE_INFO_KEYS, `${path}.info`, problems);
  }
  if (classified) {
    checkMatrix(value.features, `${path}.features`, problems);
    checkMatrix(value.model_input, `${path}.model_input`, problems);
    check(isString(value.prediction), problems, `${path}.prediction: string for a classified segment`);
  } else {
    check(value.features === null && value.model_input === null, problems, `${path}: features only for classified segments`);
    check(value.prediction === null, problems, `${path}.prediction: null unless classified`);
  }
  const top5 = value.top5;
  if (!Array.isArray(top5) || (classified ? top5.length === 0 : top5.length !== 0)) {
    problems.push(`${path}.top5: candidates exactly for classified segments`);
  } else {
    top5.forEach((candidate, i) => {
      if (!isObject(candidate)) {
        problems.push(`${path}.top5[${i}]: object`);
        return;
      }
      keys(candidate, ["index", "label", "probability"], `${path}.top5[${i}]`, problems);
      check(isInteger(candidate.index) && isString(candidate.label) && isNumber(candidate.probability), problems, `${path}.top5[${i}]: types`);
    });
  }
  check(isNullableNumber(value.max_probability), problems, `${path}.max_probability: number or null`);
  check(value.accepted === null || isBoolean(value.accepted), problems, `${path}.accepted: boolean or null`);
  check(isNumber(value.threshold), problems, `${path}.threshold: number`);
}

function checkRecording(value: unknown, index: number, problems: Problems): void {
  const path = `recordings[${index}]`;
  if (!isObject(value)) {
    problems.push(`${path}: expected an object`);
    return;
  }
  keys(value, ["id", "session", "intended_label", "intended_index", "recorded_at", "camera", "framing", "crop", "swap_hands", "pose_every", "min_hand_frames", "calibration", "segment", "frames", "app_result"], path, problems);
  check(isInteger(value.id) && isInteger(value.session) && isInteger(value.intended_index), problems, `${path}: ids must be integers`);
  check(isString(value.intended_label) && isString(value.recorded_at), problems, `${path}: label and time must be strings`);
  check(value.framing === "full" || value.framing === "portrait", problems, `${path}.framing: full or portrait`);
  check(isBoolean(value.swap_hands), problems, `${path}.swap_hands: boolean`);
  check(isInteger(value.pose_every) && (value.pose_every as number) >= 1 && (value.pose_every as number) <= 3, problems, `${path}.pose_every: 1-3`);
  check(isInteger(value.min_hand_frames) && (value.min_hand_frames as number) >= 1, problems, `${path}.min_hand_frames: >= 1`);
  const camera = value.camera;
  if (!isObject(camera)) problems.push(`${path}.camera: object`);
  else {
    keys(camera, ["width", "height", "frame_rate"], `${path}.camera`, problems);
    check(isInteger(camera.width) && isInteger(camera.height) && isNullableNumber(camera.frame_rate), problems, `${path}.camera: types`);
  }
  const crop = value.crop;
  if (!isObject(crop)) problems.push(`${path}.crop: object`);
  else {
    keys(crop, ["sx", "sy", "sw", "sh"], `${path}.crop`, problems);
    check([crop.sx, crop.sy, crop.sw, crop.sh].every(isInteger), problems, `${path}.crop: integers`);
  }
  const calibration = value.calibration;
  if (!isObject(calibration)) problems.push(`${path}.calibration: object`);
  else {
    keys(calibration, ["mode", "k", "r_live", "r_train", "frames", "status"], `${path}.calibration`, problems);
    check(["frame", "auto", "manual", "off"].includes(calibration.mode as string), problems, `${path}.calibration.mode: invalid`);
    check(isNumber(calibration.k) && calibration.k > 0, problems, `${path}.calibration.k: positive number`);
    check(isNullableNumber(calibration.r_live) && isNullableNumber(calibration.r_train), problems, `${path}.calibration: r values`);
    check(isInteger(calibration.frames), problems, `${path}.calibration.frames: integer`);
    check(["frame", "calibrated", "no_shoulders", "manual", "off", "pending", "unavailable"].includes(calibration.status as string), problems, `${path}.calibration.status: invalid`);
  }
  const segment = value.segment;
  if (!isObject(segment)) problems.push(`${path}.segment: object`);
  else {
    keys(segment, ["capture_mode", "reason", "start_ms", "end_ms", "frames", "hand_frames"], `${path}.segment`, problems);
    check(segment.capture_mode === "auto" || segment.capture_mode === "manual", problems, `${path}.segment.capture_mode: invalid`);
    check(["hands_lowered", "max_length", "manual_release", "hands_lost", "still", "flush"].includes(segment.reason as string), problems, `${path}.segment.reason: invalid`);
    check(isNumber(segment.start_ms) && isNumber(segment.end_ms) && isInteger(segment.frames) && isInteger(segment.hand_frames), problems, `${path}.segment: types`);
  }
  const frames = value.frames;
  if (!isObject(frames)) problems.push(`${path}.frames: object`);
  else {
    keys(frames, ["t_ms", "left_hand", "right_hand", "pose", "pose_visibility", "pose_ran"], `${path}.frames`, problems);
    const t = frames.t_ms;
    if (!Array.isArray(t) || !t.every(isNumber)) problems.push(`${path}.frames.t_ms: numbers`);
    else {
      checkValues(frames.left_hand, HAND_VALUES, t.length, `${path}.frames.left_hand`, problems);
      checkValues(frames.right_hand, HAND_VALUES, t.length, `${path}.frames.right_hand`, problems);
      checkValues(frames.pose, POSE_VALUES, t.length, `${path}.frames.pose`, problems);
      checkValues(frames.pose_visibility, POSE_POINTS, t.length, `${path}.frames.pose_visibility`, problems);
      check(Array.isArray(frames.pose_ran) && frames.pose_ran.length === t.length && frames.pose_ran.every(isBoolean), problems, `${path}.frames.pose_ran: booleans`);
    }
  }
  checkAppResult(value.app_result, `${path}.app_result`, problems);
}

/** Every way `raw` departs from the recordings schema; empty when it is valid. */
export function validateRecordingsFile(raw: unknown): string[] {
  const problems: Problems = [];
  if (!isObject(raw)) return ["not an object"];
  keys(raw, ["format", "schema_version", "created_at", "user_agent", "app", "recordings"], "file", problems);
  check(raw.format === RECORDINGS_FORMAT, problems, `format: expected ${RECORDINGS_FORMAT}`);
  check(raw.schema_version === RECORDINGS_SCHEMA_VERSION, problems, `schema_version: expected ${RECORDINGS_SCHEMA_VERSION}`);
  check(isString(raw.created_at) && isString(raw.user_agent), problems, "created_at and user_agent: strings");
  checkApp(raw.app, problems);
  if (!Array.isArray(raw.recordings)) problems.push("recordings: expected a list");
  else raw.recordings.forEach((recording, index) => checkRecording(recording, index, problems));
  return problems;
}
