// The page's side of src/inference.worker.ts: a module worker that holds the MediaPipe landmarkers
// or the classifier session (main.ts runs one of each), with a promise per request.

import type { BackendChoice, BackendPreference, TensorLayout } from "./ortBackend";
import type { DetectResult, WorkerRequest, WorkerResponse } from "./inference.worker";
import type { Delegate } from "./settings";

export type { DetectResult } from "./inference.worker";

/** A detection error raised by the GPU delegate: the caller reloads the landmarkers on the CPU. */
export class GpuDetectError extends Error {}

type Pending = { resolve: (response: WorkerResponse) => void; reject: (error: Error) => void };

// Distributes the Omit over the union, so each request keeps its own fields.
type WithoutId<T> = T extends unknown ? Omit<T, "id"> : never;

export class InferenceClient {
  private readonly worker: Worker;
  private nextId = 1;
  private readonly pending = new Map<number, Pending>();

  constructor(name = "edge-vision-inference") {
    this.worker = new Worker(new URL("./inference.worker.ts", import.meta.url), { type: "module", name });
    this.worker.onmessage = (event: MessageEvent<WorkerResponse>) => {
      const response = event.data;
      const waiting = this.pending.get(response.id);
      if (!waiting) return;
      this.pending.delete(response.id);
      if (response.ok) waiting.resolve(response);
      else waiting.reject(response.gpuFailure ? new GpuDetectError(response.error) : new Error(response.error));
    };
    this.worker.onerror = (event) => {
      const error = new Error(`inference worker failed: ${event.message}`);
      for (const waiting of this.pending.values()) waiting.reject(error);
      this.pending.clear();
    };
  }

  private request(body: WithoutId<WorkerRequest>, transfer: Transferable[] = []): Promise<WorkerResponse> {
    const id = this.nextId++;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ ...body, id }, transfer);
    });
  }

  async initTracker(
    assets: { loader: string; binary: string; hand: string; pose: string },
    delegate: Delegate,
    knownGpuFailure: string | null,
  ): Promise<{ delegate: Delegate; fallbackReason: string | null; loadMs: number; warmupMs: number }> {
    const response = await this.request({ type: "init-tracker", assets, delegate, knownGpuFailure });
    if (!response.ok || response.type !== "init-tracker") throw new Error("unexpected init-tracker response");
    return { delegate: response.delegate, fallbackReason: response.fallbackReason, loadMs: response.loadMs, warmupMs: response.warmupMs };
  }

  async initClassifier(
    model: ArrayBuffer,
    layout: TensorLayout,
    backend: BackendPreference,
    threads: number,
  ): Promise<{ choice: BackendChoice; crossOriginIsolated: boolean }> {
    const copy = model.slice(0);
    const response = await this.request({ type: "init-classifier", model: copy, layout, backend, threads }, [copy]);
    if (!response.ok || response.type !== "init-classifier") throw new Error("unexpected init-classifier response");
    return { choice: response.choice, crossOriginIsolated: response.crossOriginIsolated };
  }

  /** Landmarks of one frame; the bitmap is transferred (and closed by the worker). */
  async detect(bitmap: ImageBitmap, timestamp: number, runPose: boolean, mirror: boolean): Promise<DetectResult> {
    const response = await this.request({ type: "detect", bitmap, timestamp, runPose, mirror }, [bitmap]);
    if (!response.ok || response.type !== "detect") throw new Error("unexpected detect response");
    return response.result;
  }

  async classify(input: Float32Array): Promise<{ logits: Float32Array; embedding: Float32Array | null; ms: number }> {
    const copy = Float32Array.from(input);
    const response = await this.request({ type: "classify", input: copy }, [copy.buffer]);
    if (!response.ok || response.type !== "classify") throw new Error("unexpected classify response");
    return { logits: response.logits, embedding: response.embedding, ms: response.ms };
  }

  async closeTracker(): Promise<void> {
    await this.request({ type: "close-tracker" });
  }
}
