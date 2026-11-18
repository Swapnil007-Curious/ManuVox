// User settings and their persistence. Every storage access is wrapped: if localStorage is
// missing, blocked or throws, the app keeps working with in-memory settings.

import { DEFAULT_GUIDANCE_SETTINGS } from "./guidance";
import { HAND_RULES, type HandRule } from "./livePipeline";
import type { BackendPreference } from "./ortBackend";
import { DEFAULT_POSE_EVERY, MAX_POSE_EVERY, MIN_POSE_EVERY } from "./poseInterpolation";
import { DEFAULT_SEGMENTER_SETTINGS, type SegmenterMode } from "./segmenter";

export type Framing = "full" | "portrait";
export type Delegate = "CPU" | "GPU";
/** frame: k from the frame's width / height (features_spec.json aspect); auto: body-ratio calibration. */
export type AspectMode = "frame" | "auto" | "manual" | "off";
/** motion: the motion-energy state machine; presence: the first demo's hand-count rule. */
export type AutoSegmenter = "motion" | "presence";
export type Resampling = "time" | "index";
/** calibrated: temperature-scaled softmax and meta.json's calibration threshold; raw: softmax and meta.json's threshold. */
export type DecisionMode = "calibrated" | "raw";

export interface Settings {
  mode: SegmenterMode;
  startFrames: number;
  endFrames: number;
  maxDurationS: number;
  /** null means the model's preprocessing min_hand_frames from meta.json. */
  minHandFrames: number | null;
  /** null means the threshold calibrated on val and stored in meta.json. */
  confidenceThreshold: number | null;
  framing: Framing;
  swapHands: boolean;
  speak: boolean;
  /** Requested MediaPipe delegate; GPU falls back to CPU when it fails. */
  delegate: Delegate;
  /** Pose landmarker every Nth frame (1-3); poses in between are interpolated. */
  poseEvery: number;
  /** Aspect correction: calibrated at camera start, a manual k, or off (k = 1). */
  aspectMode: AspectMode;
  /** k used in manual mode. */
  aspectK: number;
  /** How detected hands map to the model's left/right blocks (features_spec.json live.hand_assignment). */
  handRule: HandRule;
  /** Detect on the horizontally flipped frame (an ablation of the live-gap study; off for use). */
  detectMirrored: boolean;
  /** Automatic capture: the motion segmenter or the first demo's presence rule. */
  autoSegmenter: AutoSegmenter;
  /** Place the 32 model frames by capture time (time) or by frame count (index). */
  resampling: Resampling;
  decision: DecisionMode;
  /** onnxruntime-web backend; auto tries WebGPU and keeps it only when it is faster. */
  classifierBackend: BackendPreference;
  /** wasm threads for the classifier; 0 picks them from the device (1 without cross-origin isolation). */
  classifierThreads: number;
  lowLightThreshold: number;
  minShoulderDistance: number;
  maxShoulderDistance: number;
  /** Teach Dex: blend the model with the user's own examples of the signs they taught. */
  personalization: boolean;
}

export const DEFAULT_SETTINGS: Settings = {
  mode: DEFAULT_SEGMENTER_SETTINGS.mode,
  startFrames: DEFAULT_SEGMENTER_SETTINGS.startFrames,
  endFrames: DEFAULT_SEGMENTER_SETTINGS.endFrames,
  maxDurationS: DEFAULT_SEGMENTER_SETTINGS.maxDurationMs / 1000,
  minHandFrames: null,
  confidenceThreshold: null,
  framing: "full",
  swapHands: false,
  speak: false,
  delegate: "GPU",
  poseEvery: DEFAULT_POSE_EVERY,
  aspectMode: "frame",
  aspectK: 1,
  handRule: "nearest_pose_wrist",
  detectMirrored: false,
  autoSegmenter: "motion",
  resampling: "time",
  decision: "calibrated",
  classifierBackend: "auto",
  classifierThreads: 0,
  lowLightThreshold: DEFAULT_GUIDANCE_SETTINGS.lowLightThreshold,
  minShoulderDistance: DEFAULT_GUIDANCE_SETTINGS.minShoulderDistance,
  maxShoulderDistance: DEFAULT_GUIDANCE_SETTINGS.maxShoulderDistance,
  personalization: true,
};

export const STORAGE_KEY = "edge-vision.settings.v3";
/** Settings stored before the GPU default; read once, without their delegate. */
export const LEGACY_STORAGE_KEY = "edge-vision.settings.v1";
/** Settings stored before the live-pipeline fixes; read once, without the settings whose defaults changed. */
export const V2_STORAGE_KEY = "edge-vision.settings.v2";
const V2_CHANGED: readonly string[] = ["aspectMode", "swapHands"];

interface NumberRule {
  min: number;
  max: number;
  integer: boolean;
}

export const NUMBER_RULES: Record<
  | "startFrames"
  | "endFrames"
  | "maxDurationS"
  | "minHandFrames"
  | "confidenceThreshold"
  | "poseEvery"
  | "aspectK"
  | "classifierThreads"
  | "lowLightThreshold"
  | "minShoulderDistance"
  | "maxShoulderDistance",
  NumberRule
> = {
  startFrames: { min: 1, max: 30, integer: true },
  endFrames: { min: 1, max: 60, integer: true },
  maxDurationS: { min: 0.5, max: 10, integer: false },
  minHandFrames: { min: 1, max: 60, integer: true },
  confidenceThreshold: { min: 0, max: 1, integer: false },
  poseEvery: { min: MIN_POSE_EVERY, max: MAX_POSE_EVERY, integer: true },
  aspectK: { min: 0.25, max: 4, integer: false },
  classifierThreads: { min: 0, max: 8, integer: true },
  lowLightThreshold: { min: 0, max: 255, integer: false },
  minShoulderDistance: { min: 0, max: 2, integer: false },
  maxShoulderDistance: { min: 0, max: 2, integer: false },
};

function validNumber(value: unknown, rule: NumberRule): value is number {
  return (
    typeof value === "number" &&
    Number.isFinite(value) &&
    value >= rule.min &&
    value <= rule.max &&
    (!rule.integer || Number.isInteger(value))
  );
}

function pick<T>(value: unknown, allowed: readonly T[], fallback: T): T {
  return allowed.includes(value as T) ? (value as T) : fallback;
}

/** Keep each valid field of `raw`; anything missing or invalid takes its default. */
export function sanitizeSettings(raw: unknown): Settings {
  const source = (raw !== null && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const number = (key: keyof typeof NUMBER_RULES, fallback: number): number =>
    validNumber(source[key], NUMBER_RULES[key]) ? (source[key] as number) : fallback;
  const nullableNumber = (key: "minHandFrames" | "confidenceThreshold"): number | null =>
    validNumber(source[key], NUMBER_RULES[key]) ? (source[key] as number) : null;
  const settings: Settings = {
    mode: pick(source.mode, ["auto", "manual"] as const, DEFAULT_SETTINGS.mode),
    startFrames: number("startFrames", DEFAULT_SETTINGS.startFrames),
    endFrames: number("endFrames", DEFAULT_SETTINGS.endFrames),
    maxDurationS: number("maxDurationS", DEFAULT_SETTINGS.maxDurationS),
    minHandFrames: nullableNumber("minHandFrames"),
    confidenceThreshold: nullableNumber("confidenceThreshold"),
    framing: pick(source.framing, ["full", "portrait"] as const, DEFAULT_SETTINGS.framing),
    swapHands: typeof source.swapHands === "boolean" ? source.swapHands : DEFAULT_SETTINGS.swapHands,
    speak: typeof source.speak === "boolean" ? source.speak : DEFAULT_SETTINGS.speak,
    delegate: pick(source.delegate, ["CPU", "GPU"] as const, DEFAULT_SETTINGS.delegate),
    poseEvery: number("poseEvery", DEFAULT_SETTINGS.poseEvery),
    aspectMode: pick(source.aspectMode, ["frame", "auto", "manual", "off"] as const, DEFAULT_SETTINGS.aspectMode),
    aspectK: number("aspectK", DEFAULT_SETTINGS.aspectK),
    handRule: pick(source.handRule, HAND_RULES, DEFAULT_SETTINGS.handRule),
    detectMirrored: typeof source.detectMirrored === "boolean" ? source.detectMirrored : DEFAULT_SETTINGS.detectMirrored,
    autoSegmenter: pick(source.autoSegmenter, ["motion", "presence"] as const, DEFAULT_SETTINGS.autoSegmenter),
    resampling: pick(source.resampling, ["time", "index"] as const, DEFAULT_SETTINGS.resampling),
    decision: pick(source.decision, ["calibrated", "raw"] as const, DEFAULT_SETTINGS.decision),
    classifierBackend: pick(source.classifierBackend, ["auto", "wasm", "webgpu"] as const, DEFAULT_SETTINGS.classifierBackend),
    classifierThreads: number("classifierThreads", DEFAULT_SETTINGS.classifierThreads),
    lowLightThreshold: number("lowLightThreshold", DEFAULT_SETTINGS.lowLightThreshold),
    minShoulderDistance: number("minShoulderDistance", DEFAULT_SETTINGS.minShoulderDistance),
    maxShoulderDistance: number("maxShoulderDistance", DEFAULT_SETTINGS.maxShoulderDistance),
    personalization: typeof source.personalization === "boolean" ? source.personalization : DEFAULT_SETTINGS.personalization,
  };
  if (settings.minShoulderDistance >= settings.maxShoulderDistance) {
    settings.minShoulderDistance = DEFAULT_SETTINGS.minShoulderDistance;
    settings.maxShoulderDistance = DEFAULT_SETTINGS.maxShoulderDistance;
  }
  return settings;
}

/** The subset of the Storage interface used here, so tests can pass a stub. */
export interface KeyValueStore {
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
}

/** window.localStorage, or null when even reading the property throws (e.g. blocked storage). */
export function browserStorage(): KeyValueStore | null {
  try {
    return typeof localStorage === "undefined" ? null : localStorage;
  } catch {
    return null;
  }
}

export function loadSettings(store: KeyValueStore | null): Settings {
  try {
    const text = store?.getItem(STORAGE_KEY);
    if (text) return sanitizeSettings(JSON.parse(text));
    const v2 = store?.getItem(V2_STORAGE_KEY);
    if (v2) {
      const parsed: unknown = JSON.parse(v2);
      if (parsed !== null && typeof parsed === "object") for (const key of V2_CHANGED) delete (parsed as Record<string, unknown>)[key];
      return sanitizeSettings(parsed);
    }
    // Earlier settings keep everything except the delegate, whose default changed to GPU.
    const legacy = store?.getItem(LEGACY_STORAGE_KEY);
    const parsed: unknown = legacy ? JSON.parse(legacy) : null;
    if (parsed !== null && typeof parsed === "object") delete (parsed as Record<string, unknown>).delegate;
    return sanitizeSettings(parsed);
  } catch {
    return { ...DEFAULT_SETTINGS };
  }
}

/** Returns false when the settings could not be stored; they still apply for this session. */
export function saveSettings(store: KeyValueStore | null, settings: Settings): boolean {
  try {
    if (!store) return false;
    store.setItem(STORAGE_KEY, JSON.stringify(settings));
    return true;
  } catch {
    return false;
  }
}
