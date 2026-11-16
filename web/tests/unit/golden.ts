// Loaders for the golden files written by scripts/export_web.py.

import { readFileSync } from "node:fs";
import type { LongRows, SequenceInfo } from "../../src/preprocess";

export interface PreprocessCase {
  name: string;
  description: string;
  rows: LongRows;
  info: SequenceInfo;
  features: number[][] | null;
  presence: number[][] | null;
  model_input: number[][] | null;
}

export interface PreprocessGolden {
  config: { num_frames: number; use_z: false; pose_indices: number[]; pose_swap_pairs: [number, number][]; min_hand_frames: number };
  snap_x_cases: { x: number; snapped: number }[];
  cases: PreprocessCase[];
}

export interface ModelGolden {
  source: string;
  variant: string;
  model_sha256: string;
  num_classes: number;
  frame_dim: number;
  cases: { sequence_id: number; label: number; input: number[][]; logits: number[] }[];
}

type Values = (number | null)[];

export interface FrameBlocks {
  left_hand: (Values | null)[];
  right_hand: (Values | null)[];
  pose: (Values | null)[];
}

export interface AspectGolden {
  k_min: number;
  k_max: number;
  default_y_mid: number;
  min_visibility: number;
  calibration_cases: { name: string; ratios: (number | null)[]; r_train: number; k: number; r_live: number | null; frames: number; status: string }[];
  visibility_cases: { name: string; pose: Values; visibility: Values | null; usable: boolean; ratio: number | null }[];
  frame_cases: {
    name: string;
    description: string;
    k: number;
    frames: FrameBlocks;
    corrected: FrameBlocks;
    results: { pose_indices: number[]; pose_swap_pairs: [number, number][]; info: SequenceInfo; features: number[][] | null }[];
  }[];
}

function load<T>(name: string): T {
  return JSON.parse(readFileSync(new URL(`../golden/${name}`, import.meta.url), "utf8")) as T;
}

export const PREPROCESS_GOLDEN_FILES = ["preprocess_cases.json", "preprocess_cases_v2.json"] as const;
export const preprocessGolden = (name: (typeof PREPROCESS_GOLDEN_FILES)[number] = "preprocess_cases.json"): PreprocessGolden =>
  load<PreprocessGolden>(name);
export const modelGolden = (): ModelGolden => load<ModelGolden>("model_cases.json");
export const aspectGolden = (): AspectGolden => load<AspectGolden>("aspect_cases.json");
export const recordingExample = (): unknown => load<unknown>("recording_example.json");

/** Flat per-frame values (null -> NaN) as a Float64Array, or null for a missing block. */
export function block(values: Values | null): Float64Array | null {
  return values ? Float64Array.from(values, (value) => (value === null ? Number.NaN : value)) : null;
}
