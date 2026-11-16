// The recognition engine: camera, the two inference workers, the frame loop, hand assignment,
// aspect correction, segmentation, preprocessing, classification and the decision rule. It owns
// no UI: the app listens to its events (state, frame, guess, result, ...) and draws them.
//
// This is the pipeline of the live-gap study (features_spec.json, src/livePipeline.ts,
// src/live.ts, the motion segmenter and the calibrated decision from meta.json), moved out of the
// old main.ts unchanged so that the interface can be redesigned without touching it.

import { AspectCalibrator, frameK, TRAIN_WIDTH_OVER_HEIGHT, type CalibrationStatus } from "./aspect";
import { BenchmarkRecorder, benchmarkFileName, downloadJson, type BenchmarkContext } from "./benchmark";
import { REQUESTED, cropRect, openCamera, stopCamera, type CameraInfo, type CropRect } from "./camera";
import { sha256Hex } from "./classifier";
import type { DebugHook, DebugSegment } from "./debug";
import { MESSAGES, guidance, meanLuminance, shoulderDistance, type GuidanceMessage, type GuidanceSettings } from "./guidance";
import { wristConsistency, type DetectedHand } from "./hands";
import { GpuDetectError, InferenceClient, type DetectResult } from "./inferenceClient";
import { assetUrl, trackerAssets } from "./landmarker";
import { HAND_POINTS, POSE_POINTS, flatPoints, flatVisibility, preprocessSegment, rawHasHand, segmentStats, type LiveFrame } from "./live";
import { LiveSession, assignHands, type SessionEvent } from "./livePipeline";
import { preprocessConfig, validateMeta, type WebMeta } from "./meta";
import { defaultThreads, type BackendChoice } from "./ortBackend";
import { PoseInterpolator, shouldRunPose } from "./poseInterpolation";
import { DEFAULT_CONFIG, preprocessRows, toModelInput, type Point2, type PreprocessConfig, type PreprocessResult } from "./preprocess";
import { RecordingStore, appResult, recordFrames, type AppInfo, type AppResult, type CalibrationRecord } from "./recorder";
import { decide, softmax, type Decision } from "./scoring";
import { Segmenter, type SegmenterEvent, type SegmenterSettings } from "./segmenter";
import { DEFAULT_SETTINGS, type Settings } from "./settings";
import { SEGMENTER } from "./spec";
import { FpsMeter, RollingWindow, type Summary } from "./stats";

type Delegate = Settings["delegate"];

export interface TrackerState {
  requested: Delegate;
  delegate: Delegate;
  fallbackReason: string | null;
  loadMs: number;
  warmupMs: number;
}

export interface ClassifierState extends BackendChoice {
  crossOriginIsolated: boolean;
}

/** off: camera stopped; idle: watching for a sign; signing: a segment is open; reading: classifying it. */
export type EngineState = "off" | "idle" | "signing" | "reading";

export type Hand = "left" | "right";

export interface FrameInfo {
  t: number;
  /** Assigned hands and the latest pose, in unmirrored coordinates of the cropped frame. */
  left: readonly Point2[] | null;
  right: readonly Point2[] | null;
  pose: readonly Point2[] | null;
  crop: CropRect;
  handsDetected: number;
  guidance: GuidanceMessage[];
  /** For the framing guide: a body (nose and both shoulders) and at least one hand are in view. */
  bodyInView: boolean;
  handInView: boolean;
  /** Seconds without any detected hand. */
  secondsWithoutHands: number;
}

export interface SegmentResult {
  kind: "classified" | "excluded" | "discarded";
  start: number;
  end: number;
  frames: number;
  handFrames: number;
  reason: string;
  /** The signing hand: the side preprocessing keeps as dominant (left when it mirrored). */
  dominant: Hand | null;
  /** Top five with the decision rule in force (temperature, threshold); null unless classified. */
  decision: Decision | null;
  excluded: string | null;
  /** Classifier time in ms; null unless classified. */
  classifierMs: number | null;
}

export type CameraError = "denied" | "no_camera" | "busy" | "insecure" | "other";

export interface LoadProgress {
  stage: "model" | "landmarks" | "warmup" | "ready";
  /** 0..1 over everything that is downloaded and initialised. */
  fraction: number;
  detail: string;
}

export interface PerfSnapshot {
  fps: number;
  hand: Summary;
  pose: Summary;
  preprocess: Summary;
  classifier: Summary;
  total: Summary;
  camera: CameraInfo | null;
  framing: Settings["framing"];
  luminance: number | null;
  shoulderDistance: number | null;
  wrists: { checked: number; consistent: number };
  tracker: TrackerState | null;
  classifier_backend: ClassifierState | null;
  delegateRequested: Delegate;
  poseEvery: number;
  aspect: { mode: Settings["aspectMode"]; k: number; rLive: number | null; rTrain: number | null; status: string };
  benchmark: { running: boolean; remaining: number };
}

export interface EngineEvents {
  progress: LoadProgress;
  ready: WebMeta;
  state: EngineState;
  frame: FrameInfo;
  guess: Decision;
  result: SegmentResult;
  status: { text: string; kind: "info" | "error" };
  cameraError: { kind: CameraError; message: string };
  calibration: string;
  benchmark: string;
  recorder: string;
}

type Listener<T> = (value: T) => void;

const MODEL_FETCH_SHARE = 0.4;
const ASSET_FETCH_SHARE = 0.45;
const LUMINANCE_EVERY = 10;
const WRIST_WINDOW = 90;
const FIXED_INPUT_RUNS = 50;
const RECORDED_CANDIDATES = 5;
const SEGMENT_LOG_LENGTH = 500;
const BACKEND_CACHE_KEY = "edge-vision.classifier-backend.v1";

export interface EngineOptions {
  video: HTMLVideoElement;
  settings: Settings;
  storage: { getItem(key: string): string | null; setItem(key: string, value: string): void } | null;
  /** The model directory under the base path (the live-gap harness may load another one). */
  modelDir: string;
  /** Keep a log of segments for the debug hook. */
  logSegments: boolean;
}

/** fetch with byte progress; falls back to one step when the body cannot be streamed. */
async function fetchBytes(url: string, onBytes: (loaded: number) => void): Promise<ArrayBuffer> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
  if (!response.body) {
    const buffer = await response.arrayBuffer();
    onBytes(buffer.byteLength);
    return buffer;
  }
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onBytes(loaded);
  }
  const out = new Uint8Array(loaded);
  let offset = 0;
  for (const chunk of chunks) {
    out.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return out.buffer;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

/** A camera failure in the words of the state the app shows. */
export function cameraErrorKind(error: unknown): CameraError {
  const name = error instanceof DOMException ? error.name : "";
  if (name === "NotAllowedError" || name === "SecurityError" || name === "PermissionDeniedError") return "denied";
  if (name === "NotFoundError" || name === "OverconstrainedError" || name === "DevicesNotFoundError") return "no_camera";
  if (name === "NotReadableError" || name === "TrackStartError" || name === "AbortError") return "busy";
  if (error instanceof Error && /secure context/i.test(error.message)) return "insecure";
  return "other";
}

export class Engine {
  readonly video: HTMLVideoElement;
  settings: Settings;
  meta: WebMeta | null = null;
  readonly ready: Promise<WebMeta>;

  private readonly storage: EngineOptions["storage"];
  private readonly modelDir: string;
  private readonly logSegments: boolean;
  private readonly listeners = new Map<keyof EngineEvents, Set<Listener<never>>>();
  private resolveReady: (meta: WebMeta) => void = () => undefined;
  private rejectReady: (error: unknown) => void = () => undefined;

  // Two instances of the inference worker: landmarkers on every frame, and the classifier, so
  // classifying never holds up tracking.
  private readonly trackerClient = new InferenceClient("edge-vision-tracker");
  private readonly classifierClient = new InferenceClient("edge-vision-classifier");
  classifierState: ClassifierState | null = null;
  tracker: TrackerState | null = null;
  private trackerLoading: Promise<TrackerState> | null = null;
  private gpuFailure: string | null = null;
  private stream: MediaStream | null = null;
  camera: CameraInfo | null = null;
  private crop: CropRect = { sx: 0, sy: 0, sw: 0, sh: 0 };
  running = false;
  private loopGeneration = 0;
  private session = -1;
  private lastTimestamp = 0;
  private frameIndex = 0;
  private framesWithoutHands = 0;
  private lastHandAt = 0;
  private luminance: number | null = null;
  private lastShoulderDistance: number | null = null;
  private lastPosePoints: Point2[] | null = null;
  private lastSegmentInput: Float32Array | null = null;
  private readonly wristWindow: { checked: number; consistent: number }[] = [];
  private detectBusy = false;
  private pendingCapture: number | null = null;
  private lastPoseFlat: Float64Array | null = null;
  private guessInFlight = false;
  private lastGuessAt = 0;
  private reading = 0;
  private lastState: EngineState = "off";
  private readonly segmentLog: DebugSegment[] = [];

  private calibration: { status: CalibrationStatus | "pending" | "unavailable"; k: number; rLive: number | null; frames: number } = {
    status: "pending",
    k: 1,
    rLive: null,
    frames: 0,
  };
  private readonly calibrator = new AspectCalibrator();
  private readonly liveSession = new LiveSession<LiveFrame>(1);
  private readonly segmenter = new Segmenter<LiveFrame>();
  private readonly interpolator = new PoseInterpolator<LiveFrame>();
  readonly recordings = new RecordingStore();
  recorderOn = false;
  intendedIndex: number | null = null;
  private readonly fpsMeter = new FpsMeter();
  readonly benchmark = new BenchmarkRecorder();
  private readonly windows = {
    hand: new RollingWindow(300),
    pose: new RollingWindow(300),
    preprocess: new RollingWindow(300),
    classifier: new RollingWindow(300),
    total: new RollingWindow(300),
  };

  private readonly lumaCanvas: HTMLCanvasElement;
  private readonly lumaContext: CanvasRenderingContext2D;

  constructor(options: EngineOptions) {
    this.video = options.video;
    this.settings = options.settings;
    this.storage = options.storage;
    this.modelDir = options.modelDir;
    this.logSegments = options.logSegments;
    this.ready = new Promise<WebMeta>((resolve, reject) => {
      this.resolveReady = resolve;
      this.rejectReady = reject;
    });
    this.ready.catch(() => undefined);
    this.lumaCanvas = document.createElement("canvas");
    this.lumaCanvas.width = 32;
    this.lumaCanvas.height = 24;
    this.lumaContext = this.lumaCanvas.getContext("2d", { willReadFrequently: true }) as CanvasRenderingContext2D;
  }

  // -------------------------------------------------------------------------------------------
  // Events

  on<K extends keyof EngineEvents>(type: K, listener: Listener<EngineEvents[K]>): () => void {
    let set = this.listeners.get(type);
    if (!set) {
      set = new Set();
      this.listeners.set(type, set);
    }
    set.add(listener as Listener<never>);
    return () => set.delete(listener as Listener<never>);
  }

  /** Debug hook only: drop the results and guesses of real segments (screenshots stage their own). */
  private muted = false;

  private emit<K extends keyof EngineEvents>(type: K, value: EngineEvents[K]): void {
    if (this.muted && (type === "result" || type === "guess")) return;
    for (const listener of this.listeners.get(type) ?? []) (listener as Listener<EngineEvents[K]>)(value);
  }

  private setStatus(text: string, kind: "info" | "error" = "info"): void {
    this.emit("status", { text, kind });
  }

  private updateState(): void {
    let state: EngineState = "off";
    if (this.running) {
      if (this.reading > 0) state = "reading";
      else if (this.motionCapture()) state = this.liveSession.segmenter.state === "signing" ? "signing" : "idle";
      else state = this.segmenter.state === "recording" ? "signing" : "idle";
    } else if (this.reading > 0) {
      state = "reading";
    }
    if (state !== this.lastState) {
      this.lastState = state;
      this.emit("state", state);
    }
  }

  get state(): EngineState {
    return this.lastState;
  }

  // -------------------------------------------------------------------------------------------
  // Settings and the decision rule

  config(): PreprocessConfig {
    const base = this.meta ? preprocessConfig(this.meta) : DEFAULT_CONFIG;
    return { ...base, minHandFrames: this.settings.minHandFrames ?? base.minHandFrames };
  }

  /** The confidence threshold: a manual setting, else the decision rule's value from meta.json. */
  threshold(): number {
    if (this.settings.confidenceThreshold !== null) return this.settings.confidenceThreshold;
    if (!this.meta) return 1;
    return this.settings.decision === "calibrated" ? this.meta.calibration.threshold : this.meta.threshold.threshold;
  }

  temperature(): number {
    return this.settings.decision === "calibrated" && this.meta ? this.meta.calibration.temperature : 1;
  }

  /** Automatic capture through the motion segmenter (the default). */
  motionCapture(): boolean {
    return this.settings.mode === "auto" && this.settings.autoSegmenter === "motion";
  }

  rTrain(): number | null {
    return this.meta?.aspect?.r_train ?? null;
  }

  /** The size of the frame the landmarkers see: the crop, or the camera frame before the first one. */
  frameSize(): { width: number; height: number } {
    if (this.crop.sw > 0 && this.crop.sh > 0) return { width: this.crop.sw, height: this.crop.sh };
    return { width: this.camera?.width ?? 0, height: this.camera?.height ?? 0 };
  }

  /** The aspect factor applied to the next segment. */
  currentK(): number {
    const settings = this.settings;
    if (settings.aspectMode === "off") return 1;
    if (settings.aspectMode === "manual") return settings.aspectK;
    if (settings.aspectMode === "frame") {
      const { width, height } = this.frameSize();
      return frameK(width, height);
    }
    return this.calibration.k;
  }

  private calibrationRecord(k: number): CalibrationRecord {
    const auto = this.settings.aspectMode === "auto";
    const fixed = { frame: "frame", manual: "manual", off: "off" } as const;
    return {
      mode: this.settings.aspectMode,
      k,
      r_live: auto ? this.calibration.rLive : null,
      r_train: this.rTrain(),
      frames: auto ? this.calibration.frames : 0,
      status: auto ? this.calibration.status : fixed[this.settings.aspectMode as keyof typeof fixed],
    };
  }

  private segmenterSettings(): SegmenterSettings {
    return {
      mode: this.settings.mode,
      startFrames: this.settings.startFrames,
      endFrames: this.settings.endFrames,
      maxDurationMs: this.settings.maxDurationS * 1000,
      minHandFrames: this.config().minHandFrames,
    };
  }

  private guidanceSettings(): GuidanceSettings {
    return {
      lowLightThreshold: this.settings.lowLightThreshold,
      minShoulderDistance: this.settings.minShoulderDistance,
      maxShoulderDistance: this.settings.maxShoulderDistance,
      noHandFrames: 15,
    };
  }

  /** The classifier worker runs requests one at a time, so classifier calls never overlap. */
  async runClassifier(input: Float32Array): Promise<Float32Array> {
    if (!this.classifierState) throw new Error("classifier not loaded");
    return (await this.classifierClient.classify(input)).logits;
  }

  /** Apply new pipeline settings (from the settings form). */
  applySettings(next: Settings): void {
    const previous = this.settings;
    this.settings = next;
    this.segmenter.configure(this.segmenterSettings());
    this.liveSession.reset();
    this.interpolator.reset();
    if (previous.delegate !== next.delegate) void this.reloadLandmarkers();
    if (previous.classifierBackend !== next.classifierBackend || previous.classifierThreads !== next.classifierThreads) {
      void this.loadClassifier().catch((error: unknown) => this.setStatus(`Could not load the classifier: ${message(error)}`, "error"));
    }
    if (previous.framing !== next.framing && this.running) this.session += 1;
    const geometryChanged = previous.framing !== next.framing;
    const nowAuto = next.aspectMode === "auto" && previous.aspectMode !== "auto";
    if (this.running && next.aspectMode === "auto" && (geometryChanged || nowAuto)) this.startCalibration(performance.now());
    this.emit("calibration", this.calibrationText(performance.now()));
    this.updateState();
  }

  // -------------------------------------------------------------------------------------------
  // Aspect calibration (the body-ratio mode)

  private startCalibration(now: number): void {
    if (this.rTrain() === null) {
      this.calibration = { status: "unavailable", k: 1, rLive: null, frames: 0 };
    } else {
      // The previous k stays in use until the new calibration is done.
      this.calibration = { ...this.calibration, status: "pending" };
      this.calibrator.start(now);
    }
    this.emit("calibration", this.calibrationText(now));
  }

  recalibrate(): void {
    if (this.running && this.settings.aspectMode === "auto") this.startCalibration(performance.now());
  }

  private updateCalibration(now: number): void {
    const r = this.rTrain();
    if (r === null || !this.calibrator.running) return;
    const result = this.calibrator.finish(now, r);
    if (!result) return;
    this.calibration = { status: result.status, k: result.k, rLive: result.rLive, frames: result.frames };
    this.emit("calibration", this.calibrationText(now));
  }

  calibrationText(now: number): string {
    const settings = this.settings;
    const calibration = this.calibration;
    if (settings.aspectMode === "off") return "Aspect correction is off (k = 1).";
    if (settings.aspectMode === "manual") return `Aspect correction: manual k ${settings.aspectK.toFixed(2)}.`;
    if (settings.aspectMode === "frame") {
      const { width, height } = this.frameSize();
      return width > 0
        ? `Aspect correction from the frame size: k ${this.currentK().toFixed(2)} (${width} × ${height}; training width / height ${TRAIN_WIDTH_OVER_HEIGHT.toFixed(3)}).`
        : `Aspect correction from the frame size (training width / height ${TRAIN_WIDTH_OVER_HEIGHT.toFixed(3)}).`;
    }
    if (calibration.status === "unavailable") return "This model has no training aspect statistics, so k stays 1.";
    if (calibration.status === "pending") {
      return this.running
        ? `Calibrating the aspect ratio: ${this.calibrator.remaining(now).toFixed(1)} s left. Face the camera with your head and shoulders in view.`
        : "Aspect calibration runs for 2 seconds when the camera starts.";
    }
    if (calibration.status === "no_shoulders") return `${MESSAGES.calibration_no_shoulders}, then press Recalibrate. k stays 1.`;
    return `Aspect calibrated: k ${calibration.k.toFixed(2)} (your ratio ${calibration.rLive?.toFixed(3)}, training ${this.rTrain()?.toFixed(3)}, ${calibration.frames} frames).`;
  }

  // -------------------------------------------------------------------------------------------
  // Segments

  private logSegment(entry: DebugSegment): void {
    if (!this.logSegments) return;
    this.segmentLog.push(entry);
    if (this.segmentLog.length > SEGMENT_LOG_LENGTH) this.segmentLog.shift();
  }

  private fromSegmenter(event: SegmenterEvent<LiveFrame>): SessionEvent<LiveFrame> {
    if (event.type === "segment") {
      const { segment } = event;
      return { kind: "segment", frames: segment.frames.map((item) => item.frame), reason: segment.reason, start: segment.startTime, end: segment.endTime, handFrames: segment.handFrames };
    }
    return { kind: "discarded", frames: event.frames.map((item) => item.frame), reason: event.reason, start: event.startTime, end: event.endTime, handFrames: event.handFrames };
  }

  /** The side with more hand frames, for segments that preprocessing did not reach. */
  private static handCountSide(frames: readonly LiveFrame[]): Hand | null {
    let left = 0;
    let right = 0;
    for (const frame of frames) {
      if (frame.left) left += 1;
      if (frame.right) right += 1;
    }
    if (left === 0 && right === 0) return null;
    return left > right ? "left" : "right";
  }

  private async handleSegmentEvent(event: SessionEvent<LiveFrame>, completedAt: number): Promise<void> {
    const meta = this.meta;
    if (!meta) return;
    const frames = event.frames;
    const k = this.currentK();
    const reason = event.reason as Parameters<Engine["recordSegment"]>[2];
    const record = (result: AppResult): void => this.recordSegment(frames, k, reason, event.start, event.end, event.handFrames, result);
    const logBase = { kind: event.kind, reason: event.reason, start_ms: event.start, end_ms: event.end, frames: frames.length, hand_frames: event.handFrames, k };
    const base = { start: event.start, end: event.end, frames: frames.length, handFrames: event.handFrames, reason: event.reason };
    if (event.kind === "discarded") {
      record(appResult({ result: null, reason: "insufficient_hand_frames" }, this.threshold()));
      this.logSegment({ ...logBase, excluded: null, probabilities: null });
      this.emit("result", { ...base, kind: "discarded", dominant: Engine.handCountSide(frames), decision: null, excluded: "insufficient_hand_frames", classifierMs: null });
      return;
    }
    this.reading += 1;
    this.updateState();
    let released = false;
    const release = (): void => {
      if (released) return;
      released = true;
      this.reading = Math.max(0, this.reading - 1);
      this.updateState();
    };
    try {
      const p0 = performance.now();
      const cfg = this.config();
      const { result, input } = preprocessSegment(frames, k, cfg, this.settings.resampling);
      const p1 = performance.now();
      if (!input) {
        const why = result.info.reason ?? "float16_overflow";
        this.windows.preprocess.push(p1 - p0);
        record(appResult({ result, reason: why }, this.threshold()));
        this.logSegment({ ...logBase, excluded: why, probabilities: null });
        release();
        this.emit("result", { ...base, kind: "excluded", dominant: Engine.handCountSide(frames), decision: null, excluded: why, classifierMs: null });
        return;
      }
      this.lastSegmentInput = input;
      const logits = await this.runClassifier(input);
      const p2 = performance.now();
      const decision = decide(logits, meta.labels, this.threshold(), RECORDED_CANDIDATES, this.temperature());
      this.windows.preprocess.push(p1 - p0);
      this.windows.classifier.push(p2 - p1);
      record(
        appResult(
          { result, modelInput: input, frameDim: meta.input_layout.frame_dim, top5: decision.top, accepted: decision.accepted },
          this.threshold(),
        ),
      );
      this.logSegment({ ...logBase, excluded: null, probabilities: Array.from(softmax(logits, this.temperature())) });
      release();
      this.emit("result", { ...base, kind: "classified", dominant: result.info.mirrored ? "left" : "right", decision, excluded: null, classifierMs: p2 - p1 });
      const p3 = performance.now();
      this.benchmark.addSegment(
        { frames: frames.length, hand_frames: event.handFrames, preprocess_ms: p1 - p0, classifier_ms: p2 - p1, end_to_end_ms: p3 - completedAt, accepted: decision.accepted },
        p3,
      );
    } finally {
      release();
    }
  }

  /** Inject a finished result as if it had been classified (debug hook; screenshots and tests). */
  simulateResult(result: SegmentResult): void {
    const muted = this.muted;
    this.muted = false;
    this.emit("result", result);
    this.muted = muted;
  }

  /** While a sign is in progress, classify the frames so far now and then and emit a guess. */
  private earlyGuess(now: number): void {
    const meta = this.meta;
    if (!meta || !this.motionCapture() || this.liveSession.segmenter.state !== "signing" || this.guessInFlight) return;
    const elapsed = now - this.liveSession.segmenter.startTime;
    if (elapsed < SEGMENTER.early_guess_ms || now - this.lastGuessAt < SEGMENTER.early_guess_every_ms) return;
    const frames = this.liveSession.segmenter.current.map((item) => item.frame);
    const { input } = preprocessSegment(frames, this.currentK(), this.config(), this.settings.resampling);
    if (!input) return;
    this.guessInFlight = true;
    this.lastGuessAt = now;
    this.runClassifier(input)
      .then((logits) => {
        if (this.liveSession.segmenter.state !== "signing") return;
        this.emit("guess", decide(logits, meta.labels, this.threshold(), 3, this.temperature()));
      })
      .catch(() => undefined)
      .finally(() => {
        this.guessInFlight = false;
      });
  }

  // -------------------------------------------------------------------------------------------
  // Record test signs

  appInfo(active: WebMeta): AppInfo {
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

  private recordSegment(
    frames: readonly LiveFrame[],
    k: number,
    reason: "hands_lowered" | "max_length" | "manual_release" | "hands_lost" | "still" | "flush",
    startTime: number,
    endTime: number,
    handFrames: number,
    result: AppResult,
  ): void {
    const meta = this.meta;
    if (!this.recorderOn || !meta) return;
    if (this.intendedIndex === null) {
      this.emit("recorder", "Pick the intended sign first: this segment was not recorded.");
      return;
    }
    const label = meta.labels[this.intendedIndex] as string;
    this.recordings.add({
      session: Math.max(this.session, 0),
      intended_label: label,
      intended_index: this.intendedIndex,
      recorded_at: new Date().toISOString(),
      camera: { width: this.camera?.width ?? 0, height: this.camera?.height ?? 0, frame_rate: this.camera?.frameRate ?? null },
      framing: this.settings.framing,
      crop: { ...this.crop },
      swap_hands: this.settings.swapHands,
      pose_every: this.settings.poseEvery,
      min_hand_frames: this.config().minHandFrames,
      calibration: this.calibrationRecord(k),
      segment: { capture_mode: this.settings.mode, reason, start_ms: startTime, end_ms: endTime, frames: frames.length, hand_frames: handFrames },
      frames: recordFrames(frames),
      app_result: result,
    });
    this.emit("recorder", `Recorded "${label}" (${result.prediction ? `predicted ${result.prediction}` : result.status}).`);
  }

  // -------------------------------------------------------------------------------------------
  // Frame loop

  private measureLuminance(): number {
    const crop = this.crop;
    this.lumaContext.drawImage(this.video, crop.sx, crop.sy, crop.sw, crop.sh, 0, 0, this.lumaCanvas.width, this.lumaCanvas.height);
    return meanLuminance(this.lumaContext.getImageData(0, 0, this.lumaCanvas.width, this.lumaCanvas.height).data);
  }

  /** A GPU delegate that fails while detecting: switch to the CPU for the rest of the session. */
  private async fallBackToCpu(error: unknown): Promise<void> {
    if (!this.tracker || this.tracker.delegate !== "GPU") throw error;
    this.gpuFailure = `GPU delegate failed while detecting: ${message(error)}`;
    this.tracker = null;
    this.setStatus("The GPU landmark backend failed; continuing on the CPU.");
    await this.getLandmarkers();
    this.interpolator.reset();
    this.segmenter.reset();
    this.liveSession.reset();
  }

  private async processFrame(t0: number): Promise<void> {
    const meta = this.meta;
    if (!this.tracker || !meta) return;
    const { videoWidth: width, videoHeight: height } = this.video;
    if (width === 0 || height === 0) return;
    const settings = this.settings;
    const rect = cropRect(width, height, settings.framing);
    this.crop = rect;
    const bitmap = await createImageBitmap(this.video, rect.sx, rect.sy, rect.sw, rect.sh);
    const timestamp = Math.max(this.lastTimestamp + 1, Math.round(t0));
    this.lastTimestamp = timestamp;
    const runPose = shouldRunPose(this.frameIndex, settings.poseEvery);
    let detected: DetectResult;
    try {
      detected = await this.trackerClient.detect(bitmap, timestamp, runPose, settings.detectMirrored);
    } catch (error) {
      if (error instanceof GpuDetectError) {
        await this.fallBackToCpu(error);
        return;
      }
      throw error;
    }
    if (!this.running) return;
    if (runPose) {
      this.lastPosePoints = detected.pose;
      this.lastPoseFlat = flatPoints(detected.pose, POSE_POINTS);
    }
    const hands: DetectedHand[] = detected.hands.map((hand) => ({ landmarks: hand.landmarks, label: hand.label, score: hand.score }));
    let assigned = assignHands(hands, this.lastPosePoints, settings.handRule);
    if (settings.swapHands && settings.handRule !== "nearest_pose_wrist") assigned = { left: assigned.right, right: assigned.left };
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
      frame.pose = this.lastPoseFlat;
      frame.visibility = flatVisibility(detected.pose);
      this.interpolator.ran(frame);
      this.calibrator.add(frame.pose, frame.visibility);
    } else {
      this.interpolator.skipped(frame);
    }
    this.wristWindow.push(wristConsistency(assigned, this.lastPosePoints));
    if (this.wristWindow.length > WRIST_WINDOW) this.wristWindow.shift();

    this.frameIndex += 1;
    this.framesWithoutHands = hands.length > 0 ? 0 : this.framesWithoutHands + 1;
    if (hands.length > 0) this.lastHandAt = t0;
    if (this.frameIndex % LUMINANCE_EVERY === 1) this.luminance = this.measureLuminance();
    this.updateCalibration(t0);

    const pose = this.lastPosePoints;
    const shoulders = pose && pose[11] && pose[12] ? ([pose[11], pose[12]] as const) : null;
    this.lastShoulderDistance = shoulders ? shoulderDistance(shoulders) : null;
    const messages = guidance(
      { hands: hands.map((hand) => hand.landmarks), shoulders, luminance: this.luminance, framesWithoutHands: this.framesWithoutHands },
      this.guidanceSettings(),
    );
    if (settings.aspectMode === "auto" && this.calibration.status === "no_shoulders") {
      messages.push({ code: "calibration_no_shoulders", text: MESSAGES.calibration_no_shoulders });
    }

    let event: SessionEvent<LiveFrame> | null = null;
    if (this.motionCapture()) {
      this.liveSession.k = this.currentK();
      event = this.liveSession.push(frame, frame.pose ?? this.lastPoseFlat);
    } else {
      const legacy = this.segmenter.push(frame, frame.hasHand, t0);
      event = legacy ? this.fromSegmenter(legacy) : null;
    }
    const nose = pose?.[0];
    const inside = (p: Point2 | undefined): boolean => p !== undefined && p.x >= 0 && p.x <= 1 && p.y >= 0 && p.y <= 1;
    this.emit("frame", {
      t: t0,
      left: assigned.left,
      right: assigned.right,
      pose,
      crop: rect,
      handsDetected: hands.length,
      guidance: messages,
      bodyInView: inside(nose) && inside(pose?.[11]) && inside(pose?.[12]),
      handInView: hands.length > 0,
      secondsWithoutHands: this.lastHandAt > 0 ? (t0 - this.lastHandAt) / 1000 : 0,
    });
    if (event) {
      this.interpolator.flush();
      void this.handleSegmentEvent(event, t0).catch((error: unknown) => this.setStatus(`Recognition failed: ${message(error)}`, "error"));
    } else {
      this.earlyGuess(t0);
    }
    this.updateState();

    const end = performance.now();
    this.windows.hand.push(detected.handMs);
    if (runPose) this.windows.pose.push(detected.poseMs);
    this.windows.total.push(end - t0);
    this.fpsMeter.tick(end);
    this.benchmark.addFrame({ hand_ms: detected.handMs, pose_ms: detected.poseMs, pose_ran: runPose, total_ms: end - t0 }, end);
    if (this.benchmark.isRunning) this.emit("benchmark", `Benchmark running: ${Math.ceil(this.benchmark.remaining(end))} s left. Sign a few times.`);
    if (this.benchmark.done(end)) void this.finishBenchmark();
  }

  private scheduleFrame(generation: number): void {
    if (!this.running || generation !== this.loopGeneration) return;
    type FrameMetadata = { captureTime?: number; expectedDisplayTime?: number };
    const video = this.video as HTMLVideoElement & {
      requestVideoFrameCallback?: (callback: (now: number, metadata: FrameMetadata) => void) => number;
    };
    if (typeof video.requestVideoFrameCallback === "function") {
      video.requestVideoFrameCallback((now, metadata) => this.onVideoFrame(generation, now, metadata.captureTime ?? now));
    } else {
      requestAnimationFrame((now) => this.onVideoFrame(generation, now, now));
    }
  }

  /**
   * Every new video frame goes to the tracker worker when it is free, timestamped with its capture
   * time (requestVideoFrameCallback's captureTime when the camera reports one), the clock motion
   * energy and resampling use. A frame that arrives while the worker is busy is remembered (the
   * newest only) and sent the moment the running detection finishes.
   */
  private onVideoFrame(generation: number, _now: number, captured: number): void {
    if (!this.running || generation !== this.loopGeneration) return;
    if (this.detectBusy) this.pendingCapture = captured;
    else if (this.tracker) this.detectFrame(generation, captured);
    this.scheduleFrame(generation);
  }

  private detectFrame(generation: number, captured: number): void {
    this.detectBusy = true;
    this.pendingCapture = null;
    this.processFrame(captured)
      .catch((error: unknown) => this.setStatus(`Frame processing failed: ${message(error)}`, "error"))
      .finally(() => {
        this.detectBusy = false;
        const next = this.pendingCapture;
        if (next !== null && this.running && generation === this.loopGeneration && this.tracker) this.detectFrame(generation, next);
      });
  }

  // -------------------------------------------------------------------------------------------
  // Performance and benchmark

  perf(now = performance.now()): PerfSnapshot {
    const wrists = this.wristWindow.reduce((sum, item) => ({ checked: sum.checked + item.checked, consistent: sum.consistent + item.consistent }), { checked: 0, consistent: 0 });
    return {
      fps: this.running ? this.fpsMeter.fps(now) : 0,
      hand: this.windows.hand.summary(),
      pose: this.windows.pose.summary(),
      preprocess: this.windows.preprocess.summary(),
      classifier: this.windows.classifier.summary(),
      total: this.windows.total.summary(),
      camera: this.camera,
      framing: this.settings.framing,
      luminance: this.luminance,
      shoulderDistance: this.lastShoulderDistance,
      wrists,
      tracker: this.tracker,
      classifier_backend: this.classifierState,
      delegateRequested: this.settings.delegate,
      poseEvery: this.settings.poseEvery,
      aspect: { mode: this.settings.aspectMode, k: this.currentK(), rLive: this.calibration.rLive, rTrain: this.rTrain(), status: this.calibration.status },
      benchmark: { running: this.benchmark.isRunning, remaining: this.benchmark.remaining(now) },
    };
  }

  private benchmarkContext(context: Record<string, unknown> = {}): BenchmarkContext {
    const activeMeta = this.meta as WebMeta;
    const settings = this.settings;
    return {
      camera: { width: this.camera?.width ?? 0, height: this.camera?.height ?? 0, frame_rate: this.camera?.frameRate ?? null, requested: { ...REQUESTED } },
      settings: { ...settings, effective_threshold: this.threshold(), effective_min_hand_frames: this.config().minHandFrames, ...context },
      model: {
        name: activeMeta.model_name,
        variant: activeMeta.variant,
        sha256: activeMeta.model_sha256,
        data_source: activeMeta.data_source,
        threshold: activeMeta.threshold.threshold,
        feature_set: activeMeta.feature_set,
      },
      ort_wasm: {
        execution_provider: this.classifierState?.backend ?? "wasm",
        num_threads: this.classifierState?.threads ?? 1,
        simd: true,
        in_worker: true,
        cross_origin_isolated: this.classifierState?.crossOriginIsolated ?? false,
        choice: this.classifierState?.reason ?? null,
        timings_ms: this.classifierState?.timings ?? {},
      },
      mediapipe: {
        delegate_requested: settings.delegate,
        delegate: this.tracker?.delegate ?? settings.delegate,
        fallback_reason: this.tracker?.fallbackReason ?? null,
        pose_every_n: settings.poseEvery,
        hand: "hand_landmarker.task (numHands 2)",
        pose: "pose_landmarker_lite.task",
      },
      aspect: { mode: settings.aspectMode, k: this.currentK(), r_live: this.calibration.rLive, r_train: this.rTrain(), status: this.calibrationRecord(this.currentK()).status },
    };
  }

  /** Extra context recorded in the benchmark file (the app adds whether the HUD is drawn). */
  benchmarkExtras: () => Record<string, unknown> = () => ({});

  startBenchmark(): boolean {
    if (!this.running || this.benchmark.isRunning) return false;
    this.benchmark.start(performance.now());
    this.emit("benchmark", "Benchmark running: 60 s left. Sign a few times.");
    return true;
  }

  private async finishBenchmark(): Promise<void> {
    const meta = this.meta;
    if (!this.benchmark.isRunning || !meta) return;
    this.benchmark.stop(performance.now());
    const size = meta.input_layout.num_frames * meta.input_layout.frame_dim;
    const input = this.lastSegmentInput ?? new Float32Array(size);
    const fixed: number[] = [];
    for (let i = 0; i < FIXED_INPUT_RUNS; i += 1) {
      const start = performance.now();
      await this.runClassifier(input);
      fixed.push(performance.now() - start);
    }
    const final = this.benchmark.report(this.benchmarkContext(this.benchmarkExtras()), fixed);
    downloadJson(final, benchmarkFileName());
    this.emit("benchmark", `Benchmark done: ${final.frames.count} frames at ${final.frames.end_to_end_fps.toFixed(1)} fps, ${final.segments.count} segments. The JSON file was downloaded.`);
  }

  // -------------------------------------------------------------------------------------------
  // Loading

  async getLandmarkers(): Promise<TrackerState> {
    if (this.tracker && this.tracker.requested === this.settings.delegate) return this.tracker;
    if (!this.trackerLoading) {
      const requested = this.settings.delegate;
      this.trackerLoading = this.trackerClient
        .initTracker(trackerAssets(), requested, this.gpuFailure)
        .then((created) => {
          this.tracker = { requested, ...created };
          return this.tracker;
        })
        .finally(() => {
          this.trackerLoading = null;
        });
    }
    return this.trackerLoading;
  }

  /** The classifier backend cached for this model and browser, so "auto" measures WebGPU once. */
  private cachedBackend(sha: string): "wasm" | "webgpu" | null {
    try {
      const raw = this.storage?.getItem(BACKEND_CACHE_KEY);
      const cached = raw ? (JSON.parse(raw) as { sha?: string; ua?: string; backend?: string }) : null;
      if (cached && cached.sha === sha && cached.ua === navigator.userAgent && (cached.backend === "wasm" || cached.backend === "webgpu")) return cached.backend;
    } catch {
      // A missing or unreadable cache only means measuring again.
    }
    return null;
  }

  private async loadClassifier(bytes?: ArrayBuffer): Promise<void> {
    const active = this.meta;
    if (!active) return;
    const modelBytes = bytes ?? (await fetchBytes(assetUrl(`${this.modelDir}/${active.model_file}`), () => undefined));
    const digest = await sha256Hex(modelBytes);
    if (digest !== active.model_sha256) throw new Error("model.onnx does not match the sha256 in meta.json");
    const settings = this.settings;
    const threads = settings.classifierThreads > 0 ? settings.classifierThreads : defaultThreads();
    const cached = settings.classifierBackend === "auto" ? this.cachedBackend(active.model_sha256) : null;
    const layout = {
      inputName: active.input_layout.input_name,
      outputName: active.input_layout.output_name,
      frames: active.input_layout.num_frames,
      width: active.input_layout.frame_dim,
    };
    const loaded = await this.classifierClient.initClassifier(modelBytes, layout, cached ?? settings.classifierBackend, threads);
    this.classifierState = { ...loaded.choice, crossOriginIsolated: loaded.crossOriginIsolated };
    if (cached) this.classifierState.reason = `${cached}, measured faster on this device earlier`;
    else if (settings.classifierBackend === "auto" && Object.keys(loaded.choice.timings).length > 0) {
      try {
        this.storage?.setItem(BACKEND_CACHE_KEY, JSON.stringify({ sha: active.model_sha256, ua: navigator.userAgent, backend: loaded.choice.backend }));
      } catch {
        // Not cached: auto mode measures again next time.
      }
    }
  }

  private async reloadLandmarkers(): Promise<void> {
    const wasRunning = this.running;
    if (wasRunning) this.stop();
    this.setStatus(`Loading landmark models (${this.settings.delegate})…`);
    try {
      await this.getLandmarkers();
      this.setStatus("Landmark models ready.");
      if (wasRunning) await this.start();
    } catch (error) {
      this.setStatus(`Could not load landmark models: ${message(error)}`, "error");
    }
  }

  /**
   * Downloads the model and the landmark files with byte progress, starts the classifier, then
   * the landmarkers. Resolves `ready` once the classifier can run; the landmarkers keep loading
   * (start() waits for them).
   */
  async load(): Promise<void> {
    const progress = (stage: LoadProgress["stage"], fraction: number, detail: string): void =>
      this.emit("progress", { stage, fraction: Math.min(1, Math.max(0, fraction)), detail });
    try {
      progress("model", 0, "Loading the sign model…");
      const metaResponse = await fetch(assetUrl(`${this.modelDir}/meta.json`));
      if (!metaResponse.ok) throw new Error(`meta.json: HTTP ${metaResponse.status}`);
      const meta = validateMeta(await metaResponse.json());
      this.meta = meta;
      const modelBytes = await fetchBytes(assetUrl(`${this.modelDir}/${meta.model_file}`), (loaded) =>
        progress("model", (MODEL_FETCH_SHARE * loaded) / meta.model_bytes, "Loading the sign model…"),
      );
      await this.loadClassifier(modelBytes);
      this.segmenter.configure(this.segmenterSettings());
      progress("landmarks", MODEL_FETCH_SHARE, "Loading hand and body tracking…");
      this.resolveReady(meta);
      this.emit("ready", meta);
      // Fetch the landmark files here, with progress; the worker then reads them from the cache.
      const assets = trackerAssets();
      const sizes = await this.assetSizes();
      const files = [assets.loader, assets.binary, assets.hand, assets.pose];
      const total = files.reduce((sum, url) => sum + (sizes.get(url.split("/").pop() ?? "") ?? 1), 0);
      let done = 0;
      for (const url of files) {
        const expected = sizes.get(url.split("/").pop() ?? "") ?? 1;
        await fetchBytes(url, (loaded) =>
          progress("landmarks", MODEL_FETCH_SHARE + (ASSET_FETCH_SHARE * (done + Math.min(loaded, expected))) / total, "Loading hand and body tracking…"),
        ).catch(() => undefined);
        done += expected;
      }
      progress("warmup", MODEL_FETCH_SHARE + ASSET_FETCH_SHARE, "Warming up the tracker…");
      await this.getLandmarkers();
      progress("ready", 1, "Ready");
    } catch (error) {
      this.setStatus(`Could not load the model: ${message(error)}`, "error");
      this.rejectReady(error);
      throw error;
    }
  }

  private async assetSizes(): Promise<Map<string, number>> {
    const sizes = new Map<string, number>();
    try {
      const response = await fetch(assetUrl("mediapipe/ASSETS.json"));
      const manifest = (await response.json()) as { assets?: { file: string; bytes: number }[] };
      for (const asset of manifest.assets ?? []) sizes.set(asset.file, asset.bytes);
    } catch {
      // Without the manifest each file counts as one step.
    }
    return sizes;
  }

  // -------------------------------------------------------------------------------------------
  // Camera

  async start(): Promise<boolean> {
    if (this.running || !this.meta) return this.running;
    try {
      this.setStatus("Starting camera…");
      const opened = await openCamera(this.video);
      this.stream = opened.stream;
      this.camera = opened.info;
      this.setStatus("Loading landmark models…");
      await this.getLandmarkers();
      this.running = true;
      this.session += 1;
      this.frameIndex = 0;
      this.framesWithoutHands = 0;
      this.lastHandAt = performance.now();
      this.lastPosePoints = null;
      this.lastPoseFlat = null;
      this.crop = { sx: 0, sy: 0, sw: 0, sh: 0 };
      this.interpolator.reset();
      this.segmenter.configure(this.segmenterSettings());
      this.liveSession.reset();
      this.fpsMeter.clear();
      const fallback = this.tracker?.fallbackReason ? ` Landmarks run on the CPU (${this.tracker.fallbackReason}).` : "";
      this.setStatus(`Running. Video stays on this device.${fallback}`);
      this.startCalibration(performance.now());
      this.loopGeneration += 1;
      this.scheduleFrame(this.loopGeneration);
      this.updateState();
      return true;
    } catch (error) {
      this.stop();
      const kind = cameraErrorKind(error);
      this.emit("cameraError", { kind, message: message(error) });
      this.setStatus(kind === "denied" ? "Camera permission was denied." : `Could not start: ${message(error)}`, "error");
      return false;
    }
  }

  stop(): void {
    this.running = false;
    this.loopGeneration += 1;
    stopCamera(this.stream, this.video);
    this.stream = null;
    // A sign in progress when the camera stops is still recognised.
    const pending = this.motionCapture() ? this.liveSession.flush() : null;
    if (pending) {
      this.interpolator.flush();
      void this.handleSegmentEvent(pending, performance.now()).catch(() => undefined);
    }
    this.segmenter.reset();
    this.liveSession.reset();
    this.interpolator.reset();
    if (this.benchmark.isRunning) {
      this.benchmark.cancel();
      this.emit("benchmark", "Benchmark cancelled because the camera stopped.");
    }
    this.emit("calibration", this.calibrationText(performance.now()));
    this.updateState();
  }

  /** Hold-to-record mode: start (true) or stop (false) a manual segment. */
  setManualRecording(on: boolean): void {
    if (!this.running || this.settings.mode !== "manual") return;
    const now = performance.now();
    if (on) {
      if (this.segmenter.state !== "recording") this.segmenter.setRecording(true, now);
    } else if (this.segmenter.state === "recording") {
      const event = this.segmenter.setRecording(false, now);
      if (event) {
        this.interpolator.flush();
        void this.handleSegmentEvent(this.fromSegmenter(event), now).catch((error: unknown) => this.setStatus(message(error), "error"));
      }
    }
    this.updateState();
  }

  /** Stats of a segment's frames, for the diagnostics drawer. */
  static stats(frames: readonly LiveFrame[]): ReturnType<typeof segmentStats> {
    return segmentStats(frames);
  }

  // -------------------------------------------------------------------------------------------
  // Debug hook (?debug=1): the parity tests, the live-gap harness and the benchmark script.

  debugHook(): DebugHook {
    return {
      ready: this.ready.then(() => undefined),
      meta: () => this.meta,
      classify: async (input: number[]): Promise<number[]> => {
        await this.ready;
        return Array.from(await this.runClassifier(Float32Array.from(input)));
      },
      preprocess: (rows) => {
        const cfg = this.config();
        const result: PreprocessResult = preprocessRows(rows, cfg);
        const modelInput = result.features ? toModelInput(result.features, cfg.poseIndices) : null;
        return {
          features: result.features ? Array.from(result.features) : null,
          presence: result.presence ? Array.from(result.presence) : null,
          modelInput: modelInput ? Array.from(modelInput) : null,
          info: result.info,
        };
      },
      recordings: () => (this.meta ? this.recordings.toFile(this.appInfo(this.meta), navigator.userAgent) : null),
      serviceWorkerControlled: () => "serviceWorker" in navigator && navigator.serviceWorker.controller !== null,
      segments: () => this.segmentLog.map((entry) => ({ ...entry })),
      clearSegments: () => {
        this.segmentLog.length = 0;
      },
      status: () => ({
        running: this.running,
        fps: this.fpsMeter.fps(performance.now()),
        frames: this.windows.hand.summary().n,
        handMsP50: this.windows.hand.summary().p50,
        poseMsP50: this.windows.pose.summary().p50,
        classifierMsP50: this.windows.classifier.summary().p50,
        classifier: this.classifierState ? { ...this.classifierState } : null,
        tracker: this.tracker ? { ...this.tracker } : null,
        crossOriginIsolated: self.crossOriginIsolated === true,
      }),
      stop: () => this.stop(),
      startBenchmark: () => this.startBenchmark(),
      simulate: (result) => this.simulateResult(result),
      muteResults: (on) => {
        this.muted = on;
      },
    };
  }
}

export { DEFAULT_SETTINGS };
