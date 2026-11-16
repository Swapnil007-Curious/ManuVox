import "./styles.css";

import { AspectCalibrator, frameK, TRAIN_WIDTH_OVER_HEIGHT, type CalibrationStatus } from "./aspect";
import { BenchmarkRecorder, benchmarkFileName, downloadJson, type BenchmarkContext } from "./benchmark";
import { REQUESTED, cropRect, openCamera, stopCamera, type CameraInfo, type CropRect } from "./camera";
import { sha256Hex } from "./classifier";
import { debugEnabled, installDebugHook, type DebugSegment } from "./debug";
import { MESSAGES, guidance, meanLuminance, shoulderDistance, type GuidanceMessage, type GuidanceSettings } from "./guidance";
import { wristConsistency, type DetectedHand } from "./hands";
import { GpuDetectError, InferenceClient, type DetectResult } from "./inferenceClient";
import { HAND_CONNECTIONS, trackerAssets } from "./landmarker";
import { HAND_POINTS, POSE_POINTS, flatPoints, flatVisibility, preprocessSegment, rawHasHand, segmentStats, type LiveFrame } from "./live";
import { LiveSession, assignHands, type Assigned, type SessionEvent } from "./livePipeline";
import { preprocessConfig, validateMeta, type WebMeta } from "./meta";
import { defaultThreads, type BackendChoice } from "./ortBackend";
import { drawOverlay } from "./overlay";
import { PoseInterpolator, shouldRunPose } from "./poseInterpolation";
import { DEFAULT_CONFIG, preprocessRows, toModelInput, type Point2, type PreprocessConfig, type PreprocessResult } from "./preprocess";
import {
  RecordingStore,
  appResult,
  filterLabels,
  recordFrames,
  recordingsFileName,
  type AppInfo,
  type AppResult,
  type CalibrationRecord,
  type RecordingsFile,
} from "./recorder";
import { decide, softmax, type Decision } from "./scoring";
import { Segmenter, type SegmenterEvent, type SegmenterSettings } from "./segmenter";
import { SEGMENTER } from "./spec";
import {
  DEFAULT_SETTINGS,
  NUMBER_RULES,
  browserStorage,
  loadSettings,
  sanitizeSettings,
  saveSettings,
  type Settings,
} from "./settings";
import { FpsMeter, RollingWindow } from "./stats";

type Delegate = Settings["delegate"];

interface TrackerState {
  requested: Delegate;
  delegate: Delegate;
  fallbackReason: string | null;
  loadMs: number;
  warmupMs: number;
}

interface ClassifierState extends BackendChoice {
  crossOriginIsolated: boolean;
}

const BASE = import.meta.env.BASE_URL;
// The model directory under BASE; a debug page may load another one (the live-gap harness
// evaluates the first demo's model this way).
const MODEL_DIR = (debugEnabled() && new URLSearchParams(location.search).get("model")) || "model";
const HISTORY_LENGTH = 10;
const PERF_REFRESH_MS = 500;
const LUMINANCE_EVERY = 10;
const WRIST_WINDOW = 90;
const FIXED_INPUT_RUNS = 50;
const SHOWN_CANDIDATES = 3;
const RECORDED_CANDIDATES = 5;
const RECORDER_LIST_LENGTH = 10;
const CLEAR_CONFIRM_MS = 4000;
const SEGMENT_LOG_LENGTH = 500;
const BACKEND_CACHE_KEY = "edge-vision.classifier-backend.v1";

function element<T extends HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`missing #${id}`);
  return found as T;
}

const ui = {
  status: element<HTMLParagraphElement>("status"),
  syntheticWarning: element<HTMLParagraphElement>("synthetic-warning"),
  noticeScope: element<HTMLLIElement>("notice-scope"),
  video: element<HTMLVideoElement>("video"),
  preview: element<HTMLCanvasElement>("preview"),
  placeholder: element<HTMLParagraphElement>("preview-placeholder"),
  guidance: element<HTMLUListElement>("guidance"),
  start: element<HTMLButtonElement>("start"),
  stop: element<HTMLButtonElement>("stop"),
  recalibrate: element<HTMLButtonElement>("recalibrate"),
  calibrationStatus: element<HTMLParagraphElement>("calibration-status"),
  record: element<HTMLButtonElement>("record"),
  recordHelp: element<HTMLParagraphElement>("record-help"),
  segmentState: element<HTMLParagraphElement>("segment-state"),
  resultSign: element<HTMLParagraphElement>("result-sign"),
  resultDetail: element<HTMLParagraphElement>("result-detail"),
  candidates: element<HTMLOListElement>("candidates"),
  segmentDebugText: element<HTMLParagraphElement>("segment-debug-text"),
  announcer: element<HTMLDivElement>("announcer"),
  history: element<HTMLOListElement>("history"),
  settingsForm: element<HTMLFormElement>("settings"),
  resetSettings: element<HTMLButtonElement>("reset-settings"),
  settingsNote: element<HTMLParagraphElement>("settings-note"),
  aspectKValue: element<HTMLOutputElement>("aspectK-value"),
  fps: element<HTMLOutputElement>("fps"),
  perfBody: element<HTMLTableSectionElement>("perf-body"),
  diagCamera: element<HTMLElement>("diag-camera"),
  diagShoulders: element<HTMLElement>("diag-shoulders"),
  diagLight: element<HTMLElement>("diag-light"),
  diagWrists: element<HTMLElement>("diag-wrists"),
  diagAspect: element<HTMLElement>("diag-aspect"),
  diagMediapipe: element<HTMLElement>("diag-mediapipe"),
  benchmark: element<HTMLButtonElement>("benchmark"),
  benchmarkStatus: element<HTMLParagraphElement>("benchmark-status"),
  recorderOn: element<HTMLInputElement>("recorder-on"),
  labelSearch: element<HTMLInputElement>("label-search"),
  labelSelect: element<HTMLSelectElement>("label-select"),
  recorderStatus: element<HTMLParagraphElement>("recorder-status"),
  recorderDownload: element<HTMLButtonElement>("recorder-download"),
  recorderClear: element<HTMLButtonElement>("recorder-clear"),
  recorderList: element<HTMLOListElement>("recorder-list"),
  modelInfo: element<HTMLParagraphElement>("model-info"),
  offlineStatus: element<HTMLParagraphElement>("offline-status"),
};

const storage = browserStorage();
let settings: Settings = loadSettings(storage);
let meta: WebMeta | null = null;
// Two instances of the inference worker (src/inference.worker.ts): one runs the landmarkers on
// every frame, the other the classifier, so a segment's classification (or an early guess) never
// holds up frame tracking, and tracking never delays a result.
const trackerClient = new InferenceClient("edge-vision-tracker");
const classifierClient = new InferenceClient("edge-vision-classifier");
let classifierState: ClassifierState | null = null;
let tracker: TrackerState | null = null;
let trackerLoading: Promise<TrackerState> | null = null;
// A GPU failure seen while detecting; later loads go straight to the CPU and report it.
let gpuFailure: string | null = null;
let stream: MediaStream | null = null;
let camera: CameraInfo | null = null;
let crop: CropRect = { sx: 0, sy: 0, sw: 0, sh: 0 };
let running = false;
// Incremented on every start and stop, so a frame callback from an earlier run cannot resume.
let loopGeneration = 0;
// A recording session: one camera run with one framing; offline calibration groups by it.
let session = -1;
let lastTimestamp = 0;
let frameIndex = 0;
let framesWithoutHands = 0;
let luminance: number | null = null;
let lastShoulderDistance: number | null = null;
let lastPosePoints: Point2[] | null = null;
let lastGuidance = "";
let lastPerfRefresh = 0;
let lastSegmentInput: Float32Array | null = null;
const wristWindow: { checked: number; consistent: number }[] = [];
// Detection runs on one frame at a time. A video frame that arrives meanwhile updates the preview
// and is remembered (the newest one only); it is sent as soon as the running detection finishes,
// rather than waiting for the next frame callback.
let detectBusy = false;
let pendingCapture: number | null = null;
let lastAssigned: Assigned = { left: null, right: null };
let lastPoseFlat: Float64Array | null = null;
let guessInFlight = false;
let lastGuessAt = 0;
const segmentLog: DebugSegment[] = [];

interface CalibrationState {
  status: CalibrationStatus | "pending" | "unavailable";
  k: number;
  rLive: number | null;
  frames: number;
}
let calibration: CalibrationState = { status: "pending", k: 1, rLive: null, frames: 0 };
const calibrator = new AspectCalibrator();

// Automatic capture with the motion segmenter, or the first demo's presence rule / manual mode.
const liveSession = new LiveSession<LiveFrame>(1);
const segmenter = new Segmenter<LiveFrame>();
const interpolator = new PoseInterpolator<LiveFrame>();
const recordings = new RecordingStore();
let recorderOn = false;
let intendedIndex: number | null = null;
let clearArmedUntil = 0;
const fpsMeter = new FpsMeter();
const benchmark = new BenchmarkRecorder();
const windows = {
  hand: new RollingWindow(300),
  pose: new RollingWindow(300),
  preprocess: new RollingWindow(300),
  classifier: new RollingWindow(300),
  total: new RollingWindow(300),
};
const PERF_ROWS: [keyof typeof windows, string][] = [
  ["hand", "Hand landmarker (per frame)"],
  ["pose", "Pose landmarker (per run)"],
  ["preprocess", "Preprocess (per segment)"],
  ["classifier", "Classifier (per segment)"],
  ["total", "Total per frame"],
];

interface HistoryEntry {
  time: Date;
  text: string;
}
const history: HistoryEntry[] = [];

const lumaCanvas = document.createElement("canvas");
lumaCanvas.width = 32;
lumaCanvas.height = 24;
const lumaContext = lumaCanvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
const previewContext = ui.preview.getContext("2d") as CanvasRenderingContext2D;
ui.preview.hidden = true;

let resolveReady: () => void = () => undefined;
let rejectReady: (error: unknown) => void = () => undefined;
const modelReady = new Promise<void>((resolve, reject) => {
  resolveReady = resolve;
  rejectReady = reject;
});
modelReady.catch(() => undefined);

// ---------------------------------------------------------------------------------------------
// Helpers

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

function setStatus(text: string, kind: "info" | "error" = "info"): void {
  ui.status.textContent = text;
  ui.status.dataset.kind = kind;
}

function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

function config(): PreprocessConfig {
  const base = meta ? preprocessConfig(meta) : DEFAULT_CONFIG;
  return { ...base, minHandFrames: settings.minHandFrames ?? base.minHandFrames };
}

/** The confidence threshold: a manual setting, else the decision rule's value from meta.json. */
function threshold(): number {
  if (settings.confidenceThreshold !== null) return settings.confidenceThreshold;
  if (!meta) return 1;
  return settings.decision === "calibrated" ? meta.calibration.threshold : meta.threshold.threshold;
}

function temperature(): number {
  return settings.decision === "calibrated" && meta ? meta.calibration.temperature : 1;
}

/** Automatic capture through the motion segmenter (the default). */
function motionCapture(): boolean {
  return settings.mode === "auto" && settings.autoSegmenter === "motion";
}

function rTrain(): number | null {
  return meta?.aspect?.r_train ?? null;
}

/** The size of the frame the landmarkers see: the crop, or the camera frame before the first one. */
function frameSize(): { width: number; height: number } {
  if (crop.sw > 0 && crop.sh > 0) return { width: crop.sw, height: crop.sh };
  return { width: camera?.width ?? 0, height: camera?.height ?? 0 };
}

/** The aspect factor applied to the next segment. */
function currentK(): number {
  if (settings.aspectMode === "off") return 1;
  if (settings.aspectMode === "manual") return settings.aspectK;
  if (settings.aspectMode === "frame") {
    const { width, height } = frameSize();
    return frameK(width, height);
  }
  return calibration.k;
}

function calibrationRecord(k: number): CalibrationRecord {
  const auto = settings.aspectMode === "auto";
  const fixed = { frame: "frame", manual: "manual", off: "off" } as const;
  return {
    mode: settings.aspectMode,
    k,
    r_live: auto ? calibration.rLive : null,
    r_train: rTrain(),
    frames: auto ? calibration.frames : 0,
    status: auto ? calibration.status : fixed[settings.aspectMode as keyof typeof fixed],
  };
}

function segmenterSettings(): SegmenterSettings {
  return {
    mode: settings.mode,
    startFrames: settings.startFrames,
    endFrames: settings.endFrames,
    maxDurationMs: settings.maxDurationS * 1000,
    minHandFrames: config().minHandFrames,
  };
}

function guidanceSettings(): GuidanceSettings {
  return {
    lowLightThreshold: settings.lowLightThreshold,
    minShoulderDistance: settings.minShoulderDistance,
    maxShoulderDistance: settings.maxShoulderDistance,
    noHandFrames: 15,
  };
}

/** The classifier worker runs requests one at a time, so classifier calls never overlap. */
async function runClassifier(input: Float32Array): Promise<Float32Array> {
  if (!classifierState) throw new Error("classifier not loaded");
  return (await classifierClient.classify(input)).logits;
}

// ---------------------------------------------------------------------------------------------
// Settings form

const NUMBER_FIELDS = Object.keys(NUMBER_RULES) as (keyof typeof NUMBER_RULES)[];

function fieldElement<T extends HTMLElement>(name: string): T {
  return element<T>(name);
}

function renderSettings(): void {
  for (const name of NUMBER_FIELDS) {
    const input = fieldElement<HTMLInputElement>(name);
    const value = settings[name];
    input.value = value === null ? "" : String(value);
    input.removeAttribute("aria-invalid");
  }
  const placeholderHandFrames = meta ? String(meta.preprocessing.min_hand_frames) : "";
  fieldElement<HTMLInputElement>("minHandFrames").placeholder = placeholderHandFrames;
  fieldElement<HTMLInputElement>("confidenceThreshold").placeholder = meta ? meta.threshold.threshold.toFixed(3) : "";
  fieldElement<HTMLSelectElement>("framing").value = settings.framing;
  fieldElement<HTMLSelectElement>("delegate").value = settings.delegate;
  fieldElement<HTMLSelectElement>("classifierBackend").value = settings.classifierBackend;
  fieldElement<HTMLInputElement>("swapHands").checked = settings.swapHands;
  fieldElement<HTMLInputElement>("speak").checked = settings.speak;
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="mode"]')) {
    radio.checked = radio.value === settings.mode;
  }
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="aspectMode"]')) {
    radio.checked = radio.value === settings.aspectMode;
  }
  const slider = fieldElement<HTMLInputElement>("aspectK");
  slider.disabled = settings.aspectMode !== "manual";
  ui.aspectKValue.textContent = settings.aspectK.toFixed(2);
  const manual = settings.mode === "manual";
  ui.record.hidden = !manual;
  ui.recordHelp.hidden = !manual;
  ui.recalibrate.disabled = !running || settings.aspectMode !== "auto";
}

function persist(): void {
  const saved = saveSettings(storage, settings);
  ui.settingsNote.textContent = saved ? "" : "Settings could not be stored in this browser; they apply until the page is closed.";
}

function applySettings(next: Settings, changed: keyof Settings | "all"): void {
  const previous = settings;
  settings = next;
  persist();
  renderSettings();
  segmenter.configure(segmenterSettings());
  liveSession.reset();
  interpolator.reset();
  setRecordingUi(false);
  if (previous.delegate !== settings.delegate && (changed === "delegate" || changed === "all")) {
    void reloadLandmarkers();
  }
  if (previous.classifierBackend !== settings.classifierBackend || previous.classifierThreads !== settings.classifierThreads) {
    void loadClassifier().catch((error: unknown) => setStatus(`Could not load the classifier: ${message(error)}`, "error"));
  }
  if (previous.framing !== settings.framing && running) session += 1;
  const geometryChanged = previous.framing !== settings.framing;
  const nowAuto = settings.aspectMode === "auto" && previous.aspectMode !== "auto";
  if (running && settings.aspectMode === "auto" && (geometryChanged || nowAuto)) startCalibration(performance.now());
  renderCalibration(performance.now());
}

function readNumberField(name: keyof typeof NUMBER_RULES): number | null | undefined {
  const input = fieldElement<HTMLInputElement>(name);
  const text = input.value.trim();
  const nullable = name === "minHandFrames" || name === "confidenceThreshold";
  if (text === "" && nullable) return null;
  const value = Number(text);
  const rule = NUMBER_RULES[name];
  const valid = text !== "" && Number.isFinite(value) && value >= rule.min && value <= rule.max && (!rule.integer || Number.isInteger(value));
  input.setAttribute("aria-invalid", valid ? "false" : "true");
  return valid ? value : undefined;
}

function bindSettings(): void {
  ui.settingsForm.addEventListener("change", (event) => {
    const target = event.target as HTMLInputElement | HTMLSelectElement;
    const name = target.name as keyof Settings;
    const raw: Record<string, unknown> = { ...settings };
    if ((NUMBER_FIELDS as string[]).includes(name)) {
      const value = readNumberField(name as keyof typeof NUMBER_RULES);
      if (value === undefined) {
        ui.settingsNote.textContent = `${target.labels?.[0]?.textContent ?? name}: value out of range; previous value kept.`;
        return;
      }
      raw[name] = value;
    } else if (target instanceof HTMLInputElement && target.type === "checkbox") {
      raw[name] = target.checked;
    } else {
      raw[name] = target.value;
    }
    const next = sanitizeSettings(raw);
    if (next.minShoulderDistance !== raw.minShoulderDistance || next.maxShoulderDistance !== raw.maxShoulderDistance) {
      ui.settingsNote.textContent = '"Too far" distance must be below "too close" distance; both were reset.';
    }
    applySettings(next, name);
  });
  fieldElement<HTMLInputElement>("aspectK").addEventListener("input", (event) => {
    ui.aspectKValue.textContent = Number((event.target as HTMLInputElement).value).toFixed(2);
  });
  ui.settingsForm.addEventListener("submit", (event) => event.preventDefault());
  ui.resetSettings.addEventListener("click", () => {
    applySettings({ ...DEFAULT_SETTINGS }, "all");
    ui.settingsNote.textContent = "Settings reset to defaults.";
  });
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="mode"]')) {
    radio.addEventListener("change", () => {
      if (radio.checked) applySettings(sanitizeSettings({ ...settings, mode: radio.value }), "mode");
    });
  }
}

// ---------------------------------------------------------------------------------------------
// Aspect calibration

function startCalibration(now: number): void {
  if (rTrain() === null) {
    calibration = { status: "unavailable", k: 1, rLive: null, frames: 0 };
  } else {
    // The previous k stays in use until the new calibration is done.
    calibration = { ...calibration, status: "pending" };
    calibrator.start(now);
  }
  renderCalibration(now);
}

function updateCalibration(now: number): void {
  const r = rTrain();
  if (r === null || !calibrator.running) return;
  const result = calibrator.finish(now, r);
  if (!result) return;
  calibration = { status: result.status, k: result.k, rLive: result.rLive, frames: result.frames };
  renderCalibration(now);
}

function renderCalibration(now: number): void {
  let text: string;
  if (settings.aspectMode === "off") text = "Aspect correction is off (k = 1).";
  else if (settings.aspectMode === "manual") text = `Aspect correction: manual k ${settings.aspectK.toFixed(2)}.`;
  else if (settings.aspectMode === "frame") {
    const { width, height } = frameSize();
    text = width > 0
      ? `Aspect correction from the frame size: k ${currentK().toFixed(2)} (${width} × ${height}; training width / height ${TRAIN_WIDTH_OVER_HEIGHT.toFixed(3)}).`
      : `Aspect correction from the frame size (training width / height ${TRAIN_WIDTH_OVER_HEIGHT.toFixed(3)}).`;
  }
  else if (calibration.status === "unavailable") text = "This model has no training aspect statistics, so k stays 1.";
  else if (calibration.status === "pending") {
    text = running
      ? `Calibrating the aspect ratio: ${calibrator.remaining(now).toFixed(1)} s left. Face the camera with your head and shoulders in view.`
      : "Aspect calibration runs for 2 seconds when the camera starts.";
  } else if (calibration.status === "no_shoulders") {
    text = `${MESSAGES.calibration_no_shoulders}, then press Recalibrate. k stays 1.`;
  } else {
    text = `Aspect calibrated: k ${calibration.k.toFixed(2)} (your ratio ${calibration.rLive?.toFixed(3)}, training ${rTrain()?.toFixed(3)}, ${calibration.frames} frames).`;
  }
  if (ui.calibrationStatus.textContent !== text) ui.calibrationStatus.textContent = text;
  ui.recalibrate.disabled = !running || settings.aspectMode !== "auto";
}

// ---------------------------------------------------------------------------------------------
// Results

function speak(text: string): void {
  if (!settings.speak || typeof speechSynthesis === "undefined") return;
  // Network voices would send the text off the device, so only local voices are used.
  const voices = speechSynthesis.getVoices().filter((voice) => voice.localService);
  const voice = voices.find((candidate) => candidate.lang.toLowerCase().startsWith("en")) ?? voices[0];
  if (!voice) {
    ui.settingsNote.textContent = "No on-device voice is available, so results are not spoken.";
    return;
  }
  const utterance = new SpeechSynthesisUtterance(text);
  utterance.voice = voice;
  utterance.lang = voice.lang;
  speechSynthesis.cancel();
  speechSynthesis.speak(utterance);
}

function renderHistory(): void {
  ui.history.replaceChildren(
    ...history.map((entry) => {
      const item = document.createElement("li");
      item.textContent = `${entry.time.toLocaleTimeString()}: ${entry.text}`;
      return item;
    }),
  );
}

/** Top-3 always; the top-1 large when it reaches the threshold, otherwise "Not sure". */
function showDecision(decision: Decision): void {
  const shown = decision.top.slice(0, SHOWN_CANDIDATES);
  const candidates = shown.map((candidate) => `${candidate.label} ${percent(candidate.probability)}`);
  ui.candidates.replaceChildren(
    ...shown.map((candidate) => {
      const item = document.createElement("li");
      item.textContent = `${candidate.label}: ${percent(candidate.probability)}`;
      return item;
    }),
  );
  ui.candidates.hidden = false;
  if (decision.accepted && decision.label !== null) {
    ui.resultSign.textContent = decision.label;
    ui.resultSign.dataset.state = "sure";
    ui.candidates.dataset.state = "sure";
    ui.resultDetail.textContent = `${percent(decision.probability)} confident`;
    ui.announcer.textContent = `Sign: ${decision.label}, ${percent(decision.probability)}.`;
    speak(decision.label);
  } else {
    ui.resultSign.textContent = "Not sure";
    ui.resultSign.dataset.state = "unsure";
    ui.candidates.dataset.state = "unsure";
    ui.resultDetail.textContent = `Top probability ${percent(decision.probability)} is below the threshold ${percent(decision.threshold)}.`;
    ui.announcer.textContent = `Not sure. Candidates: ${candidates.join(", ")}.`;
  }
  history.unshift({
    time: new Date(),
    text: decision.accepted && decision.label !== null ? `${decision.label} (${percent(decision.probability)})` : `Not sure (${candidates.join(", ")})`,
  });
  history.length = Math.min(history.length, HISTORY_LENGTH);
  renderHistory();
}

const REASON_TEXT: Record<string, string> = {
  insufficient_hand_frames: "too few frames with a hand",
  degenerate_scale: "body or hand too small to normalise",
  float16_overflow: "coordinates out of range",
};

function renderSegmentDebug(frames: readonly LiveFrame[], k: number, result: PreprocessResult | null, outcome: string): void {
  const stats = segmentStats(frames);
  ui.segmentDebugText.textContent =
    `${stats.frames} frames, norm ${result?.info.norm_method ?? "–"}, k ${k.toFixed(2)}, ` +
    `shoulders visible ${percent(stats.shouldersVisible)}, hands present ${percent(stats.handsPresent)}` +
    `${result?.info.mirrored ? ", mirrored" : ""}; ${outcome}.`;
}

/** A finished segment from either segmenter. */
type CaptureEvent = SessionEvent<LiveFrame>;

function fromSegmenter(event: SegmenterEvent<LiveFrame>): CaptureEvent {
  if (event.type === "segment") {
    const { segment } = event;
    return { kind: "segment", frames: segment.frames.map((item) => item.frame), reason: segment.reason, start: segment.startTime, end: segment.endTime, handFrames: segment.handFrames };
  }
  return { kind: "discarded", frames: event.frames.map((item) => item.frame), reason: event.reason, start: event.startTime, end: event.endTime, handFrames: event.handFrames };
}

function logSegment(entry: DebugSegment): void {
  if (!debugEnabled()) return;
  segmentLog.push(entry);
  if (segmentLog.length > SEGMENT_LOG_LENGTH) segmentLog.shift();
}

async function handleSegmentEvent(event: CaptureEvent, completedAt: number): Promise<void> {
  if (!meta) return;
  const frames = event.frames;
  const k = currentK();
  const reason = event.reason as Parameters<typeof recordSegment>[2];
  const record = (result: AppResult): void => recordSegment(frames, k, reason, event.start, event.end, event.handFrames, result);
  const logBase = { kind: event.kind, reason: event.reason, start_ms: event.start, end_ms: event.end, frames: frames.length, hand_frames: event.handFrames, k };
  if (event.kind === "discarded") {
    ui.segmentState.textContent = `Segment discarded: ${event.handFrames} frames with a hand (${config().minHandFrames} needed), ${((event.end - event.start) / 1000).toFixed(2)} s.`;
    renderSegmentDebug(frames, k, null, "discarded");
    record(appResult({ result: null, reason: "insufficient_hand_frames" }, threshold()));
    logSegment({ ...logBase, excluded: null, probabilities: null });
    return;
  }
  const p0 = performance.now();
  const cfg = config();
  const { result, input } = preprocessSegment(frames, k, cfg, settings.resampling);
  const p1 = performance.now();
  if (!input) {
    const why = result.info.reason ?? "float16_overflow";
    ui.segmentState.textContent = `No sign recognised: ${REASON_TEXT[why] ?? why}.`;
    windows.preprocess.push(p1 - p0);
    renderSegmentDebug(frames, k, result, `excluded (${why})`);
    record(appResult({ result, reason: why }, threshold()));
    logSegment({ ...logBase, excluded: why, probabilities: null });
    return;
  }
  lastSegmentInput = input;
  const logits = await runClassifier(input);
  const p2 = performance.now();
  const decision = decide(logits, meta.labels, threshold(), RECORDED_CANDIDATES, temperature());
  showDecision(decision);
  const p3 = performance.now();
  windows.preprocess.push(p1 - p0);
  windows.classifier.push(p2 - p1);
  ui.segmentState.textContent = `Recognised a ${((event.end - event.start) / 1000).toFixed(1)} s segment.`;
  renderSegmentDebug(frames, k, result, decision.accepted ? "accepted" : "below threshold");
  record(
    appResult(
      { result, modelInput: input, frameDim: meta.input_layout.frame_dim, top5: decision.top, accepted: decision.accepted },
      threshold(),
    ),
  );
  logSegment({ ...logBase, excluded: null, probabilities: Array.from(softmax(logits, temperature())) });
  benchmark.addSegment(
    {
      frames: frames.length,
      hand_frames: event.handFrames,
      preprocess_ms: p1 - p0,
      classifier_ms: p2 - p1,
      end_to_end_ms: p3 - completedAt,
      accepted: decision.accepted,
    },
    p3,
  );
}

/** While a sign is in progress, classify the frames so far now and then and show the guess. */
function earlyGuess(now: number): void {
  if (!meta || !motionCapture() || liveSession.segmenter.state !== "signing" || guessInFlight) return;
  const elapsed = now - liveSession.segmenter.startTime;
  if (elapsed < SEGMENTER.early_guess_ms || now - lastGuessAt < SEGMENTER.early_guess_every_ms) return;
  const frames = liveSession.segmenter.current.map((item) => item.frame);
  const { input } = preprocessSegment(frames, currentK(), config(), settings.resampling);
  if (!input) return;
  guessInFlight = true;
  lastGuessAt = now;
  const active = meta;
  runClassifier(input)
    .then((logits) => {
      if (liveSession.segmenter.state !== "signing") return;
      const guess = decide(logits, active.labels, threshold(), 1, temperature());
      ui.segmentState.textContent = guess.accepted
        ? `Signing… looks like "${guess.top[0]?.label}" (${percent(guess.probability)}).`
        : "Signing… not sure yet.";
    })
    .catch(() => undefined)
    .finally(() => {
      guessInFlight = false;
    });
}

// ---------------------------------------------------------------------------------------------
// Record test signs

function appInfo(active: WebMeta): AppInfo {
  return {
    model_name: active.model_name,
    variant: active.variant,
    run: active.run,
    model_sha256: active.model_sha256,
    feature_set: active.feature_set,
    pose_indices: [...active.preprocessing.pose_indices],
    pose_swap_pairs: active.preprocessing.pose_swap_pairs.map((pair) => [pair[0], pair[1]]),
    num_frames: active.num_frames,
    min_hand_frames: active.preprocessing.min_hand_frames,
    threshold: active.threshold.threshold,
    num_classes: active.num_classes,
    r_train: active.aspect?.r_train ?? null,
  };
}

function recordSegment(
  frames: readonly LiveFrame[],
  k: number,
  reason: "hands_lowered" | "max_length" | "manual_release" | "hands_lost" | "still" | "flush",
  startTime: number,
  endTime: number,
  handFrames: number,
  result: AppResult,
): void {
  if (!recorderOn || !meta) return;
  if (intendedIndex === null) {
    ui.recorderStatus.textContent = "Pick the intended sign first: this segment was not recorded.";
    return;
  }
  const label = meta.labels[intendedIndex] as string;
  recordings.add({
    session: Math.max(session, 0),
    intended_label: label,
    intended_index: intendedIndex,
    recorded_at: new Date().toISOString(),
    camera: { width: camera?.width ?? 0, height: camera?.height ?? 0, frame_rate: camera?.frameRate ?? null },
    framing: settings.framing,
    crop: { ...crop },
    swap_hands: settings.swapHands,
    pose_every: settings.poseEvery,
    min_hand_frames: config().minHandFrames,
    calibration: calibrationRecord(k),
    segment: { capture_mode: settings.mode, reason, start_ms: startTime, end_ms: endTime, frames: frames.length, hand_frames: handFrames },
    frames: recordFrames(frames),
    app_result: result,
  });
  renderRecorder(`Recorded "${label}" (${result.prediction ? `predicted ${result.prediction}` : result.status}).`);
}

function renderRecorder(note = ""): void {
  const count = recordings.count;
  const summary = count === 0 ? "No recordings." : `${count} recording${count === 1 ? "" : "s"} in memory.`;
  ui.recorderStatus.textContent = note ? `${note} ${summary}` : summary;
  ui.recorderDownload.disabled = count === 0;
  ui.recorderClear.disabled = count === 0;
  ui.recorderClear.textContent = performance.now() < clearArmedUntil ? "Press again to clear" : "Clear recordings";
  ui.recorderList.replaceChildren(
    ...recordings.recordings
      .slice(-RECORDER_LIST_LENGTH)
      .reverse()
      .map((recording) => {
        const item = document.createElement("li");
        const outcome = recording.app_result;
        const predicted = outcome.prediction
          ? `predicted ${outcome.prediction} (${percent(outcome.max_probability ?? 0)})`
          : outcome.status;
        item.textContent = `#${recording.id + 1} ${recording.intended_label}: ${predicted}, ${recording.frames.t_ms.length} frames`;
        return item;
      }),
  );
}

function renderLabelOptions(): void {
  if (!meta) return;
  const matches = filterLabels(meta.labels, ui.labelSearch.value);
  ui.labelSelect.replaceChildren(
    ...matches.map(({ index, label }) => {
      const option = document.createElement("option");
      option.value = String(index);
      option.textContent = label;
      option.selected = index === intendedIndex;
      return option;
    }),
  );
}

function bindRecorder(): void {
  ui.recorderOn.addEventListener("change", () => {
    recorderOn = ui.recorderOn.checked;
    renderRecorder(recorderOn ? (intendedIndex === null ? "Pick the intended sign." : "") : "Recording paused.");
  });
  ui.labelSearch.addEventListener("input", () => renderLabelOptions());
  ui.labelSearch.addEventListener("keydown", (event) => {
    if (event.key === "ArrowDown" && ui.labelSelect.options.length > 0) {
      event.preventDefault();
      if (ui.labelSelect.selectedIndex < 0) ui.labelSelect.selectedIndex = 0;
      ui.labelSelect.dispatchEvent(new Event("change"));
      ui.labelSelect.focus();
    }
  });
  ui.labelSelect.addEventListener("change", () => {
    const value = ui.labelSelect.value;
    intendedIndex = value === "" ? null : Number(value);
    if (meta && intendedIndex !== null) renderRecorder(`Intended sign: ${meta.labels[intendedIndex]}.`);
  });
  ui.recorderDownload.addEventListener("click", () => {
    if (!meta || recordings.count === 0) return;
    downloadJson(recordings.toFile(appInfo(meta), navigator.userAgent), recordingsFileName());
    renderRecorder("Downloaded.");
  });
  ui.recorderClear.addEventListener("click", () => {
    const now = performance.now();
    if (now >= clearArmedUntil) {
      clearArmedUntil = now + CLEAR_CONFIRM_MS;
      renderRecorder();
      setTimeout(() => renderRecorder(), CLEAR_CONFIRM_MS + 50);
      return;
    }
    clearArmedUntil = 0;
    recordings.clear();
    renderRecorder("Recordings cleared.");
  });
}

// ---------------------------------------------------------------------------------------------
// Frame loop

function renderGuidance(messages: GuidanceMessage[]): void {
  const key = messages.map((item) => item.code).join(",");
  if (key === lastGuidance) return;
  lastGuidance = key;
  ui.guidance.replaceChildren(
    ...messages.map((item) => {
      const li = document.createElement("li");
      li.textContent = item.text;
      return li;
    }),
  );
}

/** Draw the current video frame (cropped) and the latest landmarks on the preview canvas. */
function drawPreview(): void {
  const { videoWidth: width, videoHeight: height } = ui.video;
  if (width === 0 || height === 0) return;
  const rect = cropRect(width, height, settings.framing);
  if (ui.preview.width !== rect.sw || ui.preview.height !== rect.sh) {
    ui.preview.width = rect.sw;
    ui.preview.height = rect.sh;
  }
  previewContext.drawImage(ui.video, rect.sx, rect.sy, rect.sw, rect.sh, 0, 0, rect.sw, rect.sh);
  // Landmarks from a mirrored frame (an ablation setting) are flipped back for drawing.
  const unmirror = (points: readonly Point2[] | null): readonly Point2[] | null =>
    points && settings.detectMirrored ? points.map((point) => ({ ...point, x: 1 - point.x })) : points;
  drawOverlay(previewContext, {
    left: unmirror(lastAssigned.left),
    right: unmirror(lastAssigned.right),
    pose: unmirror(lastPosePoints),
    connections: HAND_CONNECTIONS,
  });
}

function measureLuminance(): number {
  lumaContext.drawImage(ui.video, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, lumaCanvas.width, lumaCanvas.height);
  return meanLuminance(lumaContext.getImageData(0, 0, lumaCanvas.width, lumaCanvas.height).data);
}

function updateSegmentState(): void {
  if (segmenter.state === "recording") ui.segmentState.textContent = "Capturing a sign…";
}

/** A GPU delegate that fails while detecting: switch to the CPU for the rest of the session. */
async function fallBackToCpu(error: unknown): Promise<void> {
  if (!tracker || tracker.delegate !== "GPU") throw error;
  gpuFailure = `GPU delegate failed while detecting: ${message(error)}`;
  tracker = null;
  setStatus("The GPU landmark backend failed; continuing on the CPU.");
  await getLandmarkers();
  interpolator.reset();
  segmenter.reset();
  liveSession.reset();
}

function toDetectedHands(result: DetectResult): DetectedHand[] {
  return result.hands.map((hand) => ({ landmarks: hand.landmarks, label: hand.label, score: hand.score }));
}

async function processFrame(t0: number): Promise<void> {
  if (!tracker || !meta) return;
  const { videoWidth: width, videoHeight: height } = ui.video;
  if (width === 0 || height === 0) return;
  const rect = cropRect(width, height, settings.framing);
  crop = rect;
  const bitmap = await createImageBitmap(ui.video, rect.sx, rect.sy, rect.sw, rect.sh);
  const timestamp = Math.max(lastTimestamp + 1, Math.round(t0));
  lastTimestamp = timestamp;
  const runPose = shouldRunPose(frameIndex, settings.poseEvery);
  let detected: DetectResult;
  try {
    detected = await trackerClient.detect(bitmap, timestamp, runPose, settings.detectMirrored);
  } catch (error) {
    if (error instanceof GpuDetectError) {
      await fallBackToCpu(error);
      return;
    }
    throw error;
  }
  if (!running) return;
  if (runPose) {
    lastPosePoints = detected.pose;
    lastPoseFlat = flatPoints(detected.pose, POSE_POINTS);
  }
  const hands = toDetectedHands(detected);
  let assigned = assignHands(hands, lastPosePoints, settings.handRule);
  if (settings.swapHands && settings.handRule !== "nearest_pose_wrist") assigned = { left: assigned.right, right: assigned.left };
  lastAssigned = assigned;
  const frame: LiveFrame = {
    t: t0,
    left: flatPoints(assigned.left, HAND_POINTS),
    right: flatPoints(assigned.right, HAND_POINTS),
    pose: null,
    visibility: null,
    poseRan: runPose,
    hasHand: false,
  };
  frame.hasHand = rawHasHand(frame);
  if (runPose) {
    frame.pose = lastPoseFlat;
    frame.visibility = flatVisibility(detected.pose);
    interpolator.ran(frame);
    calibrator.add(frame.pose, frame.visibility);
  } else {
    interpolator.skipped(frame);
  }
  wristWindow.push(wristConsistency(assigned, lastPosePoints));
  if (wristWindow.length > WRIST_WINDOW) wristWindow.shift();

  frameIndex += 1;
  framesWithoutHands = hands.length > 0 ? 0 : framesWithoutHands + 1;
  if (frameIndex % LUMINANCE_EVERY === 1) luminance = measureLuminance();
  updateCalibration(t0);

  const pose = lastPosePoints;
  const shoulders = pose && pose[11] && pose[12] ? ([pose[11], pose[12]] as const) : null;
  lastShoulderDistance = shoulders ? shoulderDistance(shoulders) : null;
  const messages = guidance({ hands: hands.map((hand) => hand.landmarks), shoulders, luminance, framesWithoutHands }, guidanceSettings());
  if (settings.aspectMode === "auto" && calibration.status === "no_shoulders") {
    messages.push({ code: "calibration_no_shoulders", text: MESSAGES.calibration_no_shoulders });
  }
  renderGuidance(messages);

  let event: CaptureEvent | null = null;
  if (motionCapture()) {
    liveSession.k = currentK();
    event = liveSession.push(frame, frame.pose ?? lastPoseFlat);
    if (liveSession.segmenter.state === "signing" && !event) ui.segmentState.textContent ||= "Capturing a sign…";
  } else {
    const legacy = segmenter.push(frame, frame.hasHand, t0);
    event = legacy ? fromSegmenter(legacy) : null;
  }
  updateSegmentState();
  if (event) {
    interpolator.flush();
    void handleSegmentEvent(event, t0).catch((error: unknown) => setStatus(`Recognition failed: ${message(error)}`, "error"));
  } else {
    earlyGuess(t0);
  }

  const end = performance.now();
  windows.hand.push(detected.handMs);
  if (runPose) windows.pose.push(detected.poseMs);
  windows.total.push(end - t0);
  fpsMeter.tick(end);
  benchmark.addFrame({ hand_ms: detected.handMs, pose_ms: detected.poseMs, pose_ran: runPose, total_ms: end - t0 }, end);
  if (end - lastPerfRefresh >= PERF_REFRESH_MS) {
    lastPerfRefresh = end;
    renderPerformance(end);
    renderCalibration(end);
  }
  if (benchmark.done(end)) void finishBenchmark();
}

type FrameMetadata = { captureTime?: number; expectedDisplayTime?: number };

function scheduleFrame(generation: number): void {
  if (!running || generation !== loopGeneration) return;
  const video = ui.video as HTMLVideoElement & {
    requestVideoFrameCallback?: (callback: (now: number, metadata: FrameMetadata) => void) => number;
  };
  if (typeof video.requestVideoFrameCallback === "function") {
    video.requestVideoFrameCallback((now, metadata) => onVideoFrame(generation, now, metadata));
  } else {
    requestAnimationFrame((now) => onVideoFrame(generation, now, {}));
  }
}

/**
 * Every new video frame updates the preview; when the worker is free it also gets the frame,
 * timestamped with its capture time (requestVideoFrameCallback's captureTime when the camera
 * reports one), which is the clock motion energy and resampling use.
 */
function onVideoFrame(generation: number, now: number, metadata: FrameMetadata): void {
  if (!running || generation !== loopGeneration) return;
  drawPreview();
  const captured = metadata.captureTime ?? now;
  if (detectBusy) pendingCapture = captured;
  else if (tracker) detectFrame(generation, captured);
  scheduleFrame(generation);
}

/**
 * Detect on the video's current frame, captured at `captured`. When it finishes, a frame that
 * arrived meanwhile (still the video's current frame, since `pendingCapture` keeps the newest)
 * goes straight to the worker, so the worker never idles until the next frame callback.
 */
function detectFrame(generation: number, captured: number): void {
  detectBusy = true;
  pendingCapture = null;
  processFrame(captured)
    .catch((error: unknown) => setStatus(`Frame processing failed: ${message(error)}`, "error"))
    .finally(() => {
      detectBusy = false;
      const next = pendingCapture;
      if (next !== null && running && generation === loopGeneration && tracker) detectFrame(generation, next);
    });
}

// ---------------------------------------------------------------------------------------------
// Performance panel and benchmark

function format(value: number): string {
  return Number.isFinite(value) ? value.toFixed(1) : "–";
}

function renderPerformance(now: number): void {
  ui.fps.textContent = running ? fpsMeter.fps(now).toFixed(1) : "–";
  ui.perfBody.replaceChildren(
    ...PERF_ROWS.map(([key, label]) => {
      const summary = windows[key].summary();
      const row = document.createElement("tr");
      for (const [i, text] of [label, format(summary.p50), format(summary.p95), String(summary.n)].entries()) {
        const cell = document.createElement(i === 0 ? "th" : "td");
        if (i === 0) cell.setAttribute("scope", "row");
        cell.textContent = text;
        row.append(cell);
      }
      return row;
    }),
  );
  ui.diagCamera.textContent = camera
    ? `${camera.width} × ${camera.height}${camera.frameRate ? ` at ${camera.frameRate.toFixed(0)} fps` : ""}${settings.framing === "portrait" ? ", portrait crop" : ""}`
    : "–";
  ui.diagShoulders.textContent = lastShoulderDistance === null
    ? "no pose"
    : `${lastShoulderDistance.toFixed(2)} (range ${settings.minShoulderDistance}–${settings.maxShoulderDistance})`;
  ui.diagLight.textContent = luminance === null ? "–" : `${luminance.toFixed(0)} / 255 (warning below ${settings.lowLightThreshold})`;
  const checked = wristWindow.reduce((sum, item) => sum + item.checked, 0);
  const consistent = wristWindow.reduce((sum, item) => sum + item.consistent, 0);
  ui.diagWrists.textContent = checked === 0
    ? "no hand and pose seen together yet"
    : `${percent(consistent / checked)} of recent hands${consistent / checked < 0.5 ? ' (try "Swap hands")' : ""}`;
  const r = rTrain();
  const mode = settings.aspectMode === "auto" ? `automatic, ${calibration.status.replace("_", " ")}` : settings.aspectMode;
  ui.diagAspect.textContent =
    `k ${currentK().toFixed(3)} (${mode}); r_live ${calibration.rLive === null ? "–" : calibration.rLive.toFixed(3)}, ` +
    `r_train ${r === null ? "–" : r.toFixed(3)}`;
  const poseEvery = `pose every ${settings.poseEvery === 1 ? "frame" : `${settings.poseEvery} frames`}`;
  const classifierText = classifierState
    ? `; classifier ${classifierState.backend}${classifierState.backend === "wasm" ? ` × ${classifierState.threads} thread${classifierState.threads === 1 ? "" : "s"}` : ""} (${classifierState.reason}); in a worker`
    : "";
  ui.diagMediapipe.textContent = tracker
    ? `${tracker.delegate}${tracker.fallbackReason ? ` (${tracker.fallbackReason})` : ""}; ${poseEvery}${classifierText}`
    : `${settings.delegate} requested; ${poseEvery}${classifierText}`;
  if (benchmark.isRunning) {
    ui.benchmarkStatus.textContent = `Benchmark running: ${Math.ceil(benchmark.remaining(now))} s left. Sign a few times.`;
  }
}

function benchmarkContext(): BenchmarkContext {
  const activeMeta = meta as WebMeta;
  return {
    camera: {
      width: camera?.width ?? 0,
      height: camera?.height ?? 0,
      frame_rate: camera?.frameRate ?? null,
      requested: { ...REQUESTED },
    },
    settings: { ...settings, effective_threshold: threshold(), effective_min_hand_frames: config().minHandFrames },
    model: {
      name: activeMeta.model_name,
      variant: activeMeta.variant,
      sha256: activeMeta.model_sha256,
      data_source: activeMeta.data_source,
      threshold: activeMeta.threshold.threshold,
      feature_set: activeMeta.feature_set,
    },
    ort_wasm: {
      execution_provider: classifierState?.backend ?? "wasm",
      num_threads: classifierState?.threads ?? 1,
      simd: true,
      in_worker: true,
      cross_origin_isolated: classifierState?.crossOriginIsolated ?? false,
      choice: classifierState?.reason ?? null,
      timings_ms: classifierState?.timings ?? {},
    },
    mediapipe: {
      delegate_requested: settings.delegate,
      delegate: tracker?.delegate ?? settings.delegate,
      fallback_reason: tracker?.fallbackReason ?? null,
      pose_every_n: settings.poseEvery,
      hand: "hand_landmarker.task (numHands 2)",
      pose: "pose_landmarker_lite.task",
    },
    aspect: { mode: settings.aspectMode, k: currentK(), r_live: calibration.rLive, r_train: rTrain(), status: calibrationRecord(currentK()).status },
  };
}

async function finishBenchmark(): Promise<void> {
  if (!benchmark.isRunning || !meta) return;
  benchmark.stop(performance.now());
  const size = meta.input_layout.num_frames * meta.input_layout.frame_dim;
  const input = lastSegmentInput ?? new Float32Array(size);
  const fixed: number[] = [];
  for (let i = 0; i < FIXED_INPUT_RUNS; i += 1) {
    const start = performance.now();
    await runClassifier(input);
    fixed.push(performance.now() - start);
  }
  const final = benchmark.report(benchmarkContext(), fixed);
  downloadJson(final, benchmarkFileName());
  ui.benchmark.disabled = !running;
  ui.benchmarkStatus.textContent = `Benchmark done: ${final.frames.count} frames at ${final.frames.end_to_end_fps.toFixed(1)} fps, ${final.segments.count} segments. The JSON file was downloaded.`;
}

// ---------------------------------------------------------------------------------------------
// Camera controls

async function getLandmarkers(): Promise<TrackerState> {
  if (tracker && tracker.requested === settings.delegate) return tracker;
  if (!trackerLoading) {
    const requested = settings.delegate;
    trackerLoading = trackerClient
      .initTracker(trackerAssets(), requested, gpuFailure)
      .then((created) => {
        tracker = { requested, ...created };
        return tracker;
      })
      .finally(() => {
        trackerLoading = null;
      });
  }
  return trackerLoading;
}

/** The classifier backend cached for this model and browser, so "auto" measures WebGPU once. */
function cachedBackend(sha: string): "wasm" | "webgpu" | null {
  try {
    const raw = storage?.getItem(BACKEND_CACHE_KEY);
    const cached = raw ? (JSON.parse(raw) as { sha?: string; ua?: string; backend?: string }) : null;
    if (cached && cached.sha === sha && cached.ua === navigator.userAgent && (cached.backend === "wasm" || cached.backend === "webgpu")) return cached.backend;
  } catch {
    // A missing or unreadable cache only means measuring again.
  }
  return null;
}

async function loadClassifier(): Promise<void> {
  if (!meta) return;
  const active = meta;
  const bytes = await (await fetchOk(`${BASE}${MODEL_DIR}/${active.model_file}`)).arrayBuffer();
  const digest = await sha256Hex(bytes);
  if (digest !== active.model_sha256) throw new Error("model.onnx does not match the sha256 in meta.json");
  const threads = settings.classifierThreads > 0 ? settings.classifierThreads : defaultThreads();
  const cached = settings.classifierBackend === "auto" ? cachedBackend(active.model_sha256) : null;
  const layout = {
    inputName: active.input_layout.input_name,
    outputName: active.input_layout.output_name,
    frames: active.input_layout.num_frames,
    width: active.input_layout.frame_dim,
  };
  const loaded = await classifierClient.initClassifier(bytes, layout, cached ?? settings.classifierBackend, threads);
  classifierState = { ...loaded.choice, crossOriginIsolated: loaded.crossOriginIsolated };
  if (cached) classifierState.reason = `${cached}, measured faster on this device earlier`;
  else if (settings.classifierBackend === "auto" && Object.keys(loaded.choice.timings).length > 0) {
    try {
      storage?.setItem(BACKEND_CACHE_KEY, JSON.stringify({ sha: active.model_sha256, ua: navigator.userAgent, backend: loaded.choice.backend }));
    } catch {
      // Not cached: auto mode measures again next time.
    }
  }
}

async function reloadLandmarkers(): Promise<void> {
  const wasRunning = running;
  if (wasRunning) stopCapture();
  setStatus(`Loading landmark models (${settings.delegate})…`);
  try {
    await getLandmarkers();
    setStatus("Landmark models ready.");
    if (wasRunning) await startCapture();
  } catch (error) {
    setStatus(`Could not load landmark models: ${message(error)}`, "error");
  }
}

async function startCapture(): Promise<void> {
  if (running || !meta) return;
  ui.start.disabled = true;
  try {
    setStatus("Starting camera…");
    const opened = await openCamera(ui.video);
    stream = opened.stream;
    camera = opened.info;
    setStatus("Loading landmark models…");
    await getLandmarkers();
    running = true;
    session += 1;
    frameIndex = 0;
    framesWithoutHands = 0;
    lastPosePoints = null;
    lastPoseFlat = null;
    lastAssigned = { left: null, right: null };
    crop = { sx: 0, sy: 0, sw: 0, sh: 0 };
    interpolator.reset();
    segmenter.configure(segmenterSettings());
    liveSession.reset();
    fpsMeter.clear();
    ui.preview.hidden = false;
    ui.placeholder.hidden = true;
    ui.stop.disabled = false;
    ui.benchmark.disabled = false;
    ui.benchmarkStatus.textContent = "Ready. Sign a few times during the run so whole segments are timed.";
    ui.segmentState.textContent = settings.mode === "auto" ? "Waiting for hands." : "Hold to record a sign.";
    const fallback = tracker?.fallbackReason ? ` Landmarks run on the CPU (${tracker.fallbackReason}).` : "";
    setStatus(`Running. Video stays on this device.${fallback}`);
    startCalibration(performance.now());
    loopGeneration += 1;
    scheduleFrame(loopGeneration);
  } catch (error) {
    stopCapture();
    const text = error instanceof DOMException && error.name === "NotAllowedError" ? "Camera permission was denied." : message(error);
    setStatus(`Could not start: ${text}`, "error");
  }
}

function stopCapture(): void {
  running = false;
  loopGeneration += 1;
  stopCamera(stream, ui.video);
  stream = null;
  // A sign in progress when the camera stops is still recognised.
  const pending = motionCapture() ? liveSession.flush() : null;
  if (pending) {
    interpolator.flush();
    void handleSegmentEvent(pending, performance.now()).catch(() => undefined);
  }
  segmenter.reset();
  liveSession.reset();
  interpolator.reset();
  setRecordingUi(false);
  if (benchmark.isRunning) {
    benchmark.cancel();
    ui.benchmarkStatus.textContent = "Benchmark cancelled because the camera stopped.";
  }
  ui.preview.hidden = true;
  ui.placeholder.hidden = false;
  ui.stop.disabled = true;
  ui.start.disabled = !meta;
  ui.benchmark.disabled = true;
  ui.segmentState.textContent = "Not capturing.";
  renderGuidance([]);
  renderPerformance(performance.now());
  renderCalibration(performance.now());
}

function setRecordingUi(on: boolean): void {
  ui.record.setAttribute("aria-pressed", String(on));
  ui.record.textContent = on ? "Recording… release to recognise" : "Hold to record a sign";
}

function startManual(): void {
  if (!running || settings.mode !== "manual" || segmenter.state === "recording") return;
  segmenter.setRecording(true, performance.now());
  setRecordingUi(true);
  ui.segmentState.textContent = "Recording…";
}

function stopManual(): void {
  if (settings.mode !== "manual" || segmenter.state !== "recording") return;
  const now = performance.now();
  const event = segmenter.setRecording(false, now);
  setRecordingUi(false);
  if (event) {
    interpolator.flush();
    void handleSegmentEvent(fromSegmenter(event), now).catch((error) => setStatus(message(error), "error"));
  }
}

function isOtherControl(target: EventTarget | null): boolean {
  if (!(target instanceof Element) || target === ui.record) return false;
  return target.closest("input, select, textarea, button, summary, a, [contenteditable]") !== null;
}

function bindControls(): void {
  ui.start.addEventListener("click", () => void startCapture());
  ui.stop.addEventListener("click", () => stopCapture());
  ui.recalibrate.addEventListener("click", () => {
    if (running && settings.aspectMode === "auto") startCalibration(performance.now());
  });
  ui.benchmark.addEventListener("click", () => {
    if (!running || benchmark.isRunning) return;
    benchmark.start(performance.now());
    ui.benchmark.disabled = true;
    ui.benchmarkStatus.textContent = "Benchmark running: 60 s left. Sign a few times.";
  });
  ui.record.addEventListener("pointerdown", (event) => {
    ui.record.setPointerCapture(event.pointerId);
    startManual();
  });
  for (const type of ["pointerup", "pointercancel", "lostpointercapture"] as const) {
    ui.record.addEventListener(type, () => stopManual());
  }
  ui.record.addEventListener("click", (event) => event.preventDefault());
  ui.record.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.repeat) {
      event.preventDefault();
      startManual();
    }
  });
  ui.record.addEventListener("keyup", (event) => {
    if (event.key === "Enter") stopManual();
  });
  document.addEventListener("keydown", (event) => {
    if (event.code !== "Space" || settings.mode !== "manual" || !running || isOtherControl(event.target)) return;
    event.preventDefault();
    if (!event.repeat) startManual();
  });
  document.addEventListener("keyup", (event) => {
    if (event.code !== "Space" || settings.mode !== "manual" || isOtherControl(event.target)) return;
    event.preventDefault();
    stopManual();
  });
  window.addEventListener("blur", () => stopManual());
  document.addEventListener("visibilitychange", () => {
    if (document.hidden && running) stopCapture();
  });
}

// ---------------------------------------------------------------------------------------------
// Service worker and boot

function updateOfflineStatus(): void {
  const controlled = "serviceWorker" in navigator && navigator.serviceWorker.controller !== null;
  ui.offlineStatus.textContent = controlled
    ? "Offline use: available. The app, model and landmark files are stored on this device."
    : "Offline use: not yet available (stored after the first complete load).";
}

const ISOLATION_RELOAD_KEY = "edge-vision.isolation-reload";

/**
 * The service worker adds the COOP/COEP headers that hosts like GitHub Pages cannot send, but a
 * page loaded before it took control is not cross-origin isolated (so the classifier has no wasm
 * threads). Reload such a page once per tab session when the worker takes over, unless the camera
 * is already running; otherwise isolation starts with the next load.
 */
function reloadForIsolation(): void {
  if (self.crossOriginIsolated || running || navigator.serviceWorker.controller === null) return;
  try {
    if (sessionStorage.getItem(ISOLATION_RELOAD_KEY)) return;
    sessionStorage.setItem(ISOLATION_RELOAD_KEY, "1");
  } catch {
    return; // Without session storage a reload could repeat; keep the page as it is.
  }
  location.reload();
}

function registerServiceWorker(): void {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) {
    updateOfflineStatus();
    return;
  }
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    updateOfflineStatus();
    reloadForIsolation();
  });
  navigator.serviceWorker.register(`${BASE}sw.js`, { scope: BASE }).then(
    () => updateOfflineStatus(),
    (error: unknown) => {
      ui.offlineStatus.textContent = `Offline use: unavailable (${message(error)}).`;
    },
  );
}

function renderModelInfo(active: WebMeta): void {
  ui.noticeScope.textContent = `Recognises single isolated signs from a ${active.num_classes}-sign ASL vocabulary. It does not translate sentences.`;
  ui.syntheticWarning.hidden = active.data_source !== "synthetic";
  const source = active.ensemble ? `ensemble of seeds ${active.seeds.join(", ")}` : `seed ${active.seed}`;
  ui.modelInfo.textContent = `Model: ${active.model_name} (${active.variant}, features ${active.feature_set}, ${source}, ${(active.model_bytes / 1e6).toFixed(2)} MB, sha256 ${active.model_sha256.slice(0, 12)}…), confidence threshold ${threshold().toFixed(3)}${temperature() !== 1 ? ` at temperature ${temperature().toFixed(3)}` : ""}, both set on validation data.`;
}

async function fetchOk(url: string): Promise<Response> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  return response;
}

async function boot(): Promise<void> {
  renderSettings();
  bindSettings();
  bindControls();
  bindRecorder();
  registerServiceWorker();
  renderPerformance(performance.now());
  renderCalibration(performance.now());
  renderRecorder();
  try {
    setStatus("Loading model…");
    meta = validateMeta(await (await fetchOk(`${BASE}${MODEL_DIR}/meta.json`)).json());
    renderModelInfo(meta);
    renderSettings();
    renderLabelOptions();
    await loadClassifier();
    segmenter.configure(segmenterSettings());
    ui.start.disabled = false;
    document.body.dataset.modelReady = "true";
    setStatus("Model ready. Press Start camera.");
    resolveReady();
    void getLandmarkers().catch(() => undefined);
  } catch (error) {
    setStatus(`Could not load the model: ${message(error)}`, "error");
    rejectReady(error);
  }
}

if (debugEnabled()) {
  installDebugHook({
    ready: modelReady,
    meta: () => meta,
    async classify(input: number[]): Promise<number[]> {
      await modelReady;
      return Array.from(await runClassifier(Float32Array.from(input)));
    },
    preprocess(rows) {
      const cfg = config();
      const result = preprocessRows(rows, cfg);
      const modelInput = result.features ? toModelInput(result.features, cfg.poseIndices) : null;
      return {
        features: result.features ? Array.from(result.features) : null,
        presence: result.presence ? Array.from(result.presence) : null,
        modelInput: modelInput ? Array.from(modelInput) : null,
        info: result.info,
      };
    },
    recordings: (): RecordingsFile | null => (meta ? recordings.toFile(appInfo(meta), navigator.userAgent) : null),
    serviceWorkerControlled: () => "serviceWorker" in navigator && navigator.serviceWorker.controller !== null,
    segments: () => segmentLog.map((entry) => ({ ...entry })),
    clearSegments: () => {
      segmentLog.length = 0;
    },
    status: () => ({
      running,
      fps: fpsMeter.fps(performance.now()),
      frames: windows.hand.summary().n,
      handMsP50: windows.hand.summary().p50,
      poseMsP50: windows.pose.summary().p50,
      classifierMsP50: windows.classifier.summary().p50,
      classifier: classifierState ? { ...classifierState } : null,
      tracker: tracker ? { ...tracker } : null,
      crossOriginIsolated: self.crossOriginIsolated === true,
    }),
    stop: () => stopCapture(),
  });
}

void boot();
