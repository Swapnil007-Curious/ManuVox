/// <reference lib="webworker" />
// The inference worker: MediaPipe Tasks HandLandmarker and PoseLandmarker on transferred
// ImageBitmaps, and the onnxruntime-web classifier, off the page's main thread. The page keeps
// the cheap per-frame logic (hand assignment, motion energy, segmentation, preprocessing) and
// the UI; src/inferenceClient.ts is its side of this protocol.
//
// MediaPipe's wasm loader is a classic script (it relies on sloppy-mode rules, so it cannot be
// loaded as a module). In a module worker importScripts() throws, and tasks-vision then calls
// self.import(url): the hook below runs the file as a classic script in the worker's global
// scope, which is what importScripts() would do. The file is the app's own same-origin asset,
// pinned by sha256 in scripts/fetch-assets.mjs; the page itself keeps eval forbidden by its CSP.

import { HandLandmarker, PoseLandmarker, type NormalizedLandmark } from "@mediapipe/tasks-vision";
import { chooseRunner, type BackendChoice, type BackendPreference, type Runner, type TensorLayout } from "./ortBackend";

type Delegate = "GPU" | "CPU";

interface TrackerAssets {
  loader: string;
  binary: string;
  hand: string;
  pose: string;
}

export type WorkerRequest =
  | { type: "init-tracker"; id: number; assets: TrackerAssets; delegate: Delegate; knownGpuFailure: string | null }
  | { type: "init-classifier"; id: number; model: ArrayBuffer; layout: TensorLayout; backend: BackendPreference; threads: number }
  | { type: "detect"; id: number; bitmap: ImageBitmap; timestamp: number; runPose: boolean; mirror: boolean }
  | { type: "classify"; id: number; input: Float32Array }
  | { type: "close-tracker"; id: number };

export interface WireHand {
  landmarks: { x: number; y: number; z: number }[];
  label: string;
  score: number;
}

export interface DetectResult {
  hands: WireHand[];
  pose: { x: number; y: number; z: number; visibility: number }[] | null;
  handMs: number;
  poseMs: number;
  width: number;
  height: number;
}

export type WorkerResponse =
  | { id: number; ok: true; type: "init-tracker"; delegate: Delegate; fallbackReason: string | null; loadMs: number; warmupMs: number }
  | { id: number; ok: true; type: "init-classifier"; choice: BackendChoice; crossOriginIsolated: boolean }
  | { id: number; ok: true; type: "detect"; result: DetectResult }
  | { id: number; ok: true; type: "classify"; logits: Float32Array; embedding: Float32Array | null; ms: number }
  | { id: number; ok: true; type: "close-tracker" }
  | { id: number; ok: false; error: string; gpuFailure?: boolean };

declare const self: DedicatedWorkerGlobalScope & {
  import?: (url: string) => Promise<void>;
  ModuleFactory?: unknown;
};

const sources = new Map<string, Promise<string>>();
self.import = async (url: string): Promise<void> => {
  let source = sources.get(url);
  if (!source) {
    source = fetch(url).then((response) => {
      if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
      return response.text();
    });
    sources.set(url, source);
  }
  // Indirect eval: global scope, so the script's `var ModuleFactory` becomes self.ModuleFactory.
  (0, eval)(`${await source}
//# sourceURL=${url}`);
};

let hand: HandLandmarker | null = null;
let pose: PoseLandmarker | null = null;
let delegate: Delegate = "GPU";
let runner: Runner | null = null;
let mirrorCanvas: OffscreenCanvas | null = null;
// VIDEO-mode timestamps must increase; warm-up frames use 1, 2, ... and real frames are later.
let warmupTimestamp = 0;
const WARMUP_FRAMES = 2;
// A GPU delegate slower than this per frame (hand + pose, after warm-up) is replaced by the CPU
// delegate. Where WebGL is software-rendered (SwiftShader: a blocklisted GPU, headless Chromium)
// the GPU delegate took ~1.5 s per frame on the target laptop, against ~90 ms on the CPU delegate
// and ~30 ms on its real GPU.
const GPU_SLOW_MS = 250;
const PROBE_FRAMES = 2;

/**
 * The GPU delegate compiles its shaders on the first inference, which takes many seconds on an
 * integrated GPU; running both landmarkers on blank frames right after loading moves that cost
 * from the first camera frame to page load.
 */
function blankFrame(): OffscreenCanvas {
  const canvas = new OffscreenCanvas(256, 192);
  const context = canvas.getContext("2d") as OffscreenCanvasRenderingContext2D;
  context.fillStyle = "#808080";
  context.fillRect(0, 0, canvas.width, canvas.height);
  return canvas;
}

function warmUp(): void {
  if (!hand || !pose) return;
  const canvas = blankFrame();
  for (let i = 0; i < WARMUP_FRAMES; i += 1) {
    warmupTimestamp += 1;
    hand.detectForVideo(canvas, warmupTimestamp);
    pose.detectForVideo(canvas, warmupTimestamp);
  }
}

/** Median ms per frame (hand + pose) on a blank frame, after warm-up. */
function probeMs(): number {
  if (!hand || !pose) return Number.POSITIVE_INFINITY;
  const canvas = blankFrame();
  const times: number[] = [];
  for (let i = 0; i < PROBE_FRAMES; i += 1) {
    warmupTimestamp += 1;
    const start = performance.now();
    hand.detectForVideo(canvas, warmupTimestamp);
    pose.detectForVideo(canvas, warmupTimestamp);
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  return times[times.length >> 1] as number;
}

function message(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

async function createLandmarkers(assets: TrackerAssets, requested: Delegate): Promise<void> {
  const fileset = { wasmLoaderPath: assets.loader, wasmBinaryPath: assets.binary };
  const createdHand = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: assets.hand, delegate: requested },
    runningMode: "VIDEO",
    numHands: 2,
  });
  try {
    pose = await PoseLandmarker.createFromOptions(fileset, {
      baseOptions: { modelAssetPath: assets.pose, delegate: requested },
      runningMode: "VIDEO",
      numPoses: 1,
      outputSegmentationMasks: false,
    });
  } catch (error) {
    createdHand.close();
    throw error;
  }
  hand = createdHand;
  delegate = requested;
}

function closeTracker(): void {
  hand?.close();
  pose?.close();
  hand = null;
  pose = null;
}

async function initTracker(request: Extract<WorkerRequest, { type: "init-tracker" }>): Promise<WorkerResponse> {
  closeTracker();
  const t0 = performance.now();
  let fallbackReason: string | null = null;
  if (request.delegate === "GPU" && request.knownGpuFailure === null) {
    try {
      await createLandmarkers(request.assets, "GPU");
      const t1 = performance.now();
      warmUp();
      const warmupMs = performance.now() - t1;
      const perFrame = probeMs();
      if (perFrame <= GPU_SLOW_MS) {
        return { id: request.id, ok: true, type: "init-tracker", delegate: "GPU", fallbackReason: null, loadMs: t1 - t0, warmupMs };
      }
      closeTracker();
      fallbackReason = `GPU delegate too slow (${perFrame.toFixed(0)} ms per frame after warm-up, likely software rendering)`;
    } catch (error) {
      closeTracker();
      fallbackReason = `GPU delegate failed: ${message(error)}`;
    }
  } else if (request.delegate === "GPU") {
    fallbackReason = request.knownGpuFailure;
  }
  await createLandmarkers(request.assets, "CPU");
  const t1 = performance.now();
  warmUp();
  return { id: request.id, ok: true, type: "init-tracker", delegate: "CPU", fallbackReason, loadMs: t1 - t0, warmupMs: performance.now() - t1 };
}

function plain(points: readonly NormalizedLandmark[]): { x: number; y: number; z: number; visibility: number }[] {
  return points.map((point) => ({ x: point.x, y: point.y, z: point.z, visibility: point.visibility ?? Number.NaN }));
}

function detect(request: Extract<WorkerRequest, { type: "detect" }>): WorkerResponse {
  const { bitmap } = request;
  try {
    if (!hand || !pose) throw new Error("landmarkers not loaded");
    let source: ImageBitmap | OffscreenCanvas = bitmap;
    if (request.mirror) {
      // The ablation's naive selfie pipeline: landmarks of the horizontally flipped frame.
      if (!mirrorCanvas || mirrorCanvas.width !== bitmap.width || mirrorCanvas.height !== bitmap.height) {
        mirrorCanvas = new OffscreenCanvas(bitmap.width, bitmap.height);
      }
      const context = mirrorCanvas.getContext("2d") as OffscreenCanvasRenderingContext2D;
      context.setTransform(-1, 0, 0, 1, bitmap.width, 0);
      context.drawImage(bitmap, 0, 0);
      source = mirrorCanvas;
    }
    const t0 = performance.now();
    const timestamp = Math.max(request.timestamp, warmupTimestamp + 1);
    const handResult = hand.detectForVideo(source, timestamp);
    const t1 = performance.now();
    const hands: WireHand[] = handResult.landmarks.map((landmarks, i) => {
      const category = handResult.handedness[i]?.[0];
      return {
        landmarks: landmarks.map((point) => ({ x: point.x, y: point.y, z: point.z })),
        label: category?.categoryName ?? "",
        score: category?.score ?? 0,
      };
    });
    let poseLandmarks: DetectResult["pose"] = null;
    let poseMs = 0;
    if (request.runPose) {
      const t2 = performance.now();
      const poseResult = pose.detectForVideo(source, timestamp);
      poseMs = performance.now() - t2;
      poseLandmarks = poseResult.landmarks[0] ? plain(poseResult.landmarks[0]) : null;
    }
    return {
      id: request.id,
      ok: true,
      type: "detect",
      result: { hands, pose: poseLandmarks, handMs: t1 - t0, poseMs, width: bitmap.width, height: bitmap.height },
    };
  } catch (error) {
    return { id: request.id, ok: false, error: message(error), gpuFailure: delegate === "GPU" };
  } finally {
    bitmap.close();
  }
}

async function handle(request: WorkerRequest): Promise<WorkerResponse> {
  switch (request.type) {
    case "init-tracker":
      return initTracker(request);
    case "init-classifier": {
      await runner?.release();
      const chosen = await chooseRunner(new Uint8Array(request.model), request.layout, request.backend, request.threads);
      runner = chosen.runner;
      return { id: request.id, ok: true, type: "init-classifier", choice: chosen.choice, crossOriginIsolated: self.crossOriginIsolated };
    }
    case "detect":
      return detect(request);
    case "classify": {
      if (!runner) throw new Error("classifier not loaded");
      const t0 = performance.now();
      const { logits, embedding } = await runner.classify(request.input);
      return { id: request.id, ok: true, type: "classify", logits, embedding, ms: performance.now() - t0 };
    }
    case "close-tracker":
      closeTracker();
      return { id: request.id, ok: true, type: "close-tracker" };
  }
}

// Requests run one at a time, in arrival order: neither MediaPipe nor an onnxruntime session is
// re-entrant.
let chain: Promise<void> = Promise.resolve();
self.onmessage = (event: MessageEvent<WorkerRequest>) => {
  const request = event.data;
  chain = chain.then(async () => {
    let response: WorkerResponse;
    try {
      response = await handle(request);
    } catch (error) {
      response = { id: request.id, ok: false, error: message(error) };
    }
    const transfer =
      response.ok && response.type === "classify" ? [response.logits.buffer, ...(response.embedding ? [response.embedding.buffer] : [])] : [];
    self.postMessage(response, transfer as Transferable[]);
  });
};
