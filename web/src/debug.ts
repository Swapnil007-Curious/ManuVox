// Test hook, installed only when the page URL has ?debug=1.

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
  /** softmax(logits / temperature) of the decision rule in force; null when not classified. */
  probabilities: number[] | null;
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
