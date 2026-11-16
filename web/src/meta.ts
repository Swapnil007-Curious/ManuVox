// meta.json written by scripts/export_web.py, with the runtime checks the app relies on.

import { frameDim, validatePoseIndices, type PosePair, type PreprocessConfig } from "./preprocess";
import { SUPPORTED_SPEC_VERSION, poseSetName } from "./spec";

export type Variant = "fp32" | "fp16" | "int8_dynamic" | "int8_static";

export interface ThresholdBlock {
  computed_on: "val";
  target_precision: number;
  threshold: number;
  coverage: number;
  accuracy_above_threshold: number;
  overall_val_top1: number;
  val_sequences: number;
  method: "target_precision" | "fallback_max_accuracy";
  fallback_min_coverage: number;
  warning: string | null;
}

/** softmax(logits / temperature) and the "not sure" threshold, both fitted on val. */
export interface CalibrationBlock {
  computed_on: "val";
  method: "temperature_scaling";
  temperature: number;
  nll_before: number;
  nll_after: number;
  ece_before: number;
  ece_after: number;
  criterion: string;
  threshold: number;
  coverage: number;
  selective_accuracy: number;
  net_score: number;
  val_top1: number;
  val_top3: number;
  val_sequences: number;
}

export interface InputLayout {
  num_frames: number;
  num_landmarks: number;
  coords_per_landmark: number;
  frame_dim: number;
  presence_left_index: number;
  presence_right_index: number;
  input_name: string;
  output_name: string;
  pose_indices: number[];
}

export interface AspectBlock {
  r_train: number;
  stats: Record<string, unknown>;
  source: string;
  k_min: number;
  k_max: number;
  calibration_seconds: number;
  min_visibility: number;
}

export interface VariantCandidate {
  variant: Variant;
  available: boolean;
  file_bytes: number | null;
  val_top1: number | null;
  drop_pp: number | null;
  eligible: boolean;
  note: string | null;
  wasm: { status: "passed" | "failed" | "not_run"; detail: string | null };
}

export interface WebMeta {
  schema_version: 2;
  model_name: string;
  variant: Variant;
  data_source: "asl-signs" | "synthetic";
  run: string;
  seed: number | null;
  seeds: number[];
  ensemble: boolean;
  feature_set: "v1" | "v2";
  num_classes: number;
  labels: string[];
  num_frames: number;
  landmark_order: string[];
  pose_indices: number[];
  input_layout: InputLayout;
  preprocessing: {
    num_frames: number;
    use_z: false;
    pose_indices: number[];
    pose_swap_pairs: [number, number][];
    min_hand_frames: number;
    min_scale: number;
    x_grid: number;
    storage_dtype: "float16";
    source: string;
  };
  aspect: AspectBlock | null;
  opset: number;
  model_file: string;
  model_bytes: number;
  model_sha256: string;
  threshold: ThresholdBlock;
  calibration: CalibrationBlock;
  features_spec_version: number;
  features_spec_sha256: string;
  variant_selection: {
    rule: string;
    chosen: Variant;
    recommended: Variant;
    candidates: VariantCandidate[];
    fallbacks: { variant: Variant; reason: string }[];
  };
}

function fail(message: string): never {
  throw new Error(`meta.json: ${message}`);
}

/** Check the fields the app depends on; throws with a readable message otherwise. */
export function validateMeta(raw: unknown): WebMeta {
  if (raw === null || typeof raw !== "object") fail("not an object");
  const meta = raw as WebMeta;
  if (meta.schema_version !== 2) fail(`unsupported schema_version ${String(meta.schema_version)} (re-export with scripts/export_web.py)`);
  if (!Array.isArray(meta.labels) || meta.labels.length !== meta.num_classes || meta.num_classes < 2) {
    fail("labels must list num_classes (>= 2) signs");
  }
  if (!/^[0-9a-f]{64}$/.test(meta.model_sha256)) fail("model_sha256 is not a sha256 hex digest");
  if (meta.feature_set !== "v1" && meta.feature_set !== "v2") fail("feature_set must be v1 or v2");
  const layout = meta.input_layout;
  const pre = meta.preprocessing;
  if (!layout || !pre) fail("input_layout and preprocessing are required");
  if (pre.use_z !== false) fail("use_z must be false");
  if (!Array.isArray(pre.pose_swap_pairs)) fail("preprocessing.pose_swap_pairs is required");
  try {
    validatePoseIndices(pre.pose_indices, pre.pose_swap_pairs);
  } catch (error) {
    fail(error instanceof Error ? error.message : String(error));
  }
  if (layout.num_frames !== pre.num_frames || meta.num_frames !== pre.num_frames) fail("num_frames differs between fields");
  if (layout.frame_dim !== frameDim(pre.pose_indices)) fail("frame_dim does not match pose_indices");
  if (layout.coords_per_landmark !== 2) fail("coords_per_landmark must be 2");
  const threshold = meta.threshold?.threshold;
  if (typeof threshold !== "number" || threshold < 0 || threshold > 1) fail("threshold must be in [0, 1]");
  const calibration = meta.calibration;
  if (!calibration || !(calibration.temperature > 0) || !(calibration.threshold >= 0 && calibration.threshold <= 1)) {
    fail("calibration must hold a temperature > 0 and a threshold in [0, 1] (re-export with scripts/export_web.py)");
  }
  if (meta.features_spec_version !== SUPPORTED_SPEC_VERSION) {
    fail(`features_spec_version ${String(meta.features_spec_version)} differs from this build's features_spec.json (${SUPPORTED_SPEC_VERSION})`);
  }
  if (poseSetName(pre.pose_indices) === null) fail("pose_indices are not a pose set of features_spec.json");
  if (meta.aspect !== null) {
    const aspect = meta.aspect;
    if (!aspect || !(aspect.r_train > 0) || !(aspect.k_min > 0) || !(aspect.k_max >= aspect.k_min)) {
      fail("aspect must be null or hold r_train > 0 and k_min <= k_max");
    }
  }
  return meta;
}

export function preprocessConfig(meta: WebMeta): PreprocessConfig {
  return {
    numFrames: meta.preprocessing.num_frames,
    poseIndices: meta.preprocessing.pose_indices,
    poseSwapPairs: meta.preprocessing.pose_swap_pairs.map((pair) => [pair[0], pair[1]] as PosePair),
    minHandFrames: meta.preprocessing.min_hand_frames,
    minScale: meta.preprocessing.min_scale,
    xGrid: meta.preprocessing.x_grid,
  };
}
