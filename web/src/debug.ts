// Test hook, installed only when the page URL has ?debug=1.

import type { SegmentResult } from "./engine";
import type { WebMeta } from "./meta";
import type { LongRows, SequenceInfo } from "./preprocess";
import type { RecordingsFile } from "./recorder";

export interface DebugPreprocessResult {
  features: number[] | null;
  presence: number[] | null;
  modelInput: number[] | null;
  info: SequenceInfo;
}

/** One finished segment, as the live-gap harness (web/scripts/live-gap.mjs) reads it. */
export interface DebugSegment {
  kind: "segment" | "discarded";
  reason: string;
  start_ms: number;
  end_ms: number;
  frames: number;
  hand_frames: number;
  k: number;
  excluded: string | null;
  /** The distribution behind the decision shown (Teach Dex's blend when it applied); null when not classified. */
  probabilities: number[] | null;
  /** softmax(logits / temperature) of the model alone. */
  base_probabilities?: number[] | null;
  /** True when the user's examples changed the decision shown ("Personalized for you"). */
  personalized?: boolean;
  logits?: number[] | null;
  /** The model's penultimate embedding, when it has that output. */
  embedding?: number[] | null;
}

export interface DebugStatus {
  running: boolean;
  fps: number;
  frames: number;
  handMsP50: number;
  poseMsP50: number;
  classifierMsP50: number;
  classifier: object | null;
  tracker: object | null;
  crossOriginIsolated: boolean;
}

export interface DebugHook {
  /** Resolves once the classifier is loaded; rejects if loading failed. */
  ready: Promise<void>;
  meta(): WebMeta | null;
  classify(input: number[]): Promise<number[]>;
  /** Logits and the embedding output (null for models without one). */
  classifyWithEmbedding(input: number[]): Promise<{ logits: number[]; embedding: number[] | null }>;
  preprocess(rows: LongRows): DebugPreprocessResult;
  /** The "Download recordings" file for the recordings held now. */
  recordings(): RecordingsFile | null;
  serviceWorkerControlled(): boolean;
  /** Finished segments since the page loaded (or the last clearSegments). */
  segments(): DebugSegment[];
  clearSegments(): void;
  status(): DebugStatus;
  /** Stop the camera; a sign in progress is still recognised. */
  stop(): void;
  /** Start the 60-second benchmark (the camera must be running); false when it cannot start. */
  startBenchmark(): boolean;
  /** Show a result as if a segment had just been classified (screenshots and UI tests). */
  simulate(result: SegmentResult): void;
  /** Drop real segments' results and guesses (simulate still shows its own). */
  muteResults(on: boolean): void;
}

declare global {
  interface Window {
    __edgevision?: DebugHook;
  }
}

export function debugEnabled(search: string = location.search): boolean {
  return new URLSearchParams(search).get("debug") === "1";
}

export function installDebugHook(hook: DebugHook): void {
  window.__edgevision = hook;
}
