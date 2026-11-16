// features_spec.json, the single source of truth for the landmark features, shared with Python
// (src/edgevision/spec.py). It is bundled at build time; modules take their constants from here.

import raw from "../../features_spec.json";

export interface FeaturesSpec {
  spec_version: number;
  landmark_types: { order: string[]; counts: Record<string, number> };
  selection: { order: string[]; hand_points: number; pose_sets: Record<string, number[]>; default_pose_set: string };
  pose_points: Record<string, number>;
  pose_swap_pairs: [number, number][];
  hand_points: { wrist: number; middle_mcp: number; fingertips: number[] };
  coordinates: { axes: string[]; use_z: boolean; x_snap_grid: number };
  min_hand_frames: number;
  normalization: { min_scale: number };
  resampling: { num_frames: number };
  model_input: { input_name: string; output_name: string };
  aspect: {
    train_width_over_height: number;
    default_y_mid: number;
    k_min: number;
    k_max: number;
    calibration: { seconds: number; min_visibility: number };
  };
  live: {
    hand_assignment: { rule: string };
    motion: { hand_scale_factor: number; window_ms: number };
    segmenter: {
      start_energy: number;
      still_energy: number;
      still_ms: number;
      hand_lost_ms: number;
      max_ms: number;
      min_ms: number;
      preroll_ms: number;
      tail_ms: number;
      max_gap_ms: number;
      early_guess_ms: number;
      early_guess_every_ms: number;
    };
  };
}

export const SPEC = raw as unknown as FeaturesSpec;
export const SUPPORTED_SPEC_VERSION = 1;
if (SPEC.spec_version !== SUPPORTED_SPEC_VERSION) {
  throw new Error(`features_spec.json spec_version ${SPEC.spec_version} is not supported`);
}

export const LANDMARK_COUNTS = SPEC.landmark_types.counts as { face: number; left_hand: number; pose: number; right_hand: number };
export const N_HAND = SPEC.selection.hand_points;
export const N_POSE = LANDMARK_COUNTS.pose;
export const POSE_SETS = SPEC.selection.pose_sets;
export const DEFAULT_POSE_INDICES: readonly number[] = POSE_SETS[SPEC.selection.default_pose_set] as number[];
export const POSE = SPEC.pose_points as {
  nose: number;
  mouth_left: number;
  mouth_right: number;
  left_shoulder: number;
  right_shoulder: number;
  left_wrist: number;
  right_wrist: number;
};
export const POSE_SWAP_PAIRS: readonly (readonly [number, number])[] = SPEC.pose_swap_pairs;
export const WRIST = SPEC.hand_points.wrist;
export const MIDDLE_MCP = SPEC.hand_points.middle_mcp;
export const FINGERTIPS: readonly number[] = SPEC.hand_points.fingertips;
export const X_SNAP_GRID = SPEC.coordinates.x_snap_grid;
export const MIN_HAND_FRAMES = SPEC.min_hand_frames;
export const MIN_SCALE = SPEC.normalization.min_scale;
export const NUM_FRAMES = SPEC.resampling.num_frames;
export const ASPECT = SPEC.aspect;
export const MOTION = SPEC.live.motion;
export const SEGMENTER = SPEC.live.segmenter;

/** The spec pose set equal to `poseIndices`, or null. */
export function poseSetName(poseIndices: readonly number[]): string | null {
  for (const [name, indices] of Object.entries(POSE_SETS)) {
    if (indices.length === poseIndices.length && indices.every((value, i) => value === poseIndices[i])) return name;
  }
  return null;
}
