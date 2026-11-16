// onnxruntime-web sessions for the classifier, on the wasm backend (SIMD, with threads when the
// page is cross-origin isolated) or on WebGPU. Used by the inference worker and the benchmark
// page; works in a window or a worker.
//
// The non-bundled builds are used (vite.config.ts resolve condition), with the glue and binary
// URLs passed as wasmPaths, so pthread workers can load the glue by URL. The plain wasm build
// serves the wasm backend; the WebGPU build (onnxruntime-web/webgpu, whose native WebGPU backend
// uses the asyncify binary, twice the size) is only fetched when WebGPU is tried.

import gpuGlue from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs?url";
import gpuBinary from "onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm?url";
import wasmGlue from "onnxruntime-web/ort-wasm-simd-threaded.mjs?url";
import wasmBinary from "onnxruntime-web/ort-wasm-simd-threaded.wasm?url";

export type Backend = "wasm" | "webgpu";
/** "auto" tries WebGPU when the browser has it and keeps it only if it is faster than wasm. */
export type BackendPreference = Backend | "auto";

export interface TensorLayout {
  inputName: string;
  outputName: string;
  frames: number;
  width: number;
}

export interface Runner {
  backend: Backend;
  threads: number;
  run(input: Float32Array): Promise<Float32Array>;
  release(): Promise<void>;
}

export interface BackendChoice {
  backend: Backend;
  threads: number;
  reason: string;
  /** Median ms per run of each backend tried, on a fixed input. */
  timings: Partial<Record<Backend, number>>;
}

function absolute(url: string): string {
  return new URL(url, self.location.href).href;
}

export function crossOriginIsolatedContext(): boolean {
  return typeof self !== "undefined" && self.crossOriginIsolated === true;
}

/**
 * wasm threads for the classifier: 1. Threads need cross-origin isolation (SharedArrayBuffer),
 * which the app has, but on the target laptop (i3-1115G4, 2 cores / 4 threads) they only slowed
 * this small model down: alone, 1 / 2 / 4 threads took 15 / 29 / 37 ms per call (fp32, p50), and
 * next to live tracking 1 / 2 / 3 threads gave a per-segment p95 of 27 / 45 / 82 ms
 * (results/perf/tracking_gpu_t1.json, _t2.json, _t3.json). Settings can raise it.
 */
export function defaultThreads(): number {
  return 1;
}

export function webgpuAvailable(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator && navigator.gpu !== undefined;
}

export async function createRunner(bytes: Uint8Array, layout: TensorLayout, backend: Backend, threads: number): Promise<Runner> {
  const ort = backend === "webgpu" ? await import("onnxruntime-web/webgpu") : await import("onnxruntime-web/wasm");
  ort.env.wasm.wasmPaths =
    backend === "webgpu" ? { mjs: absolute(gpuGlue), wasm: absolute(gpuBinary) } : { mjs: absolute(wasmGlue), wasm: absolute(wasmBinary) };
  ort.env.wasm.numThreads = threads;
  ort.env.wasm.simd = true;
  ort.env.wasm.proxy = false;
  // onnxruntime prints its warnings (e.g. shape ops placed on the CPU next to WebGPU) through
  // console.error; only real errors should reach the console.
  ort.env.logLevel = "error";
  const session = await ort.InferenceSession.create(bytes, {
    executionProviders: [backend],
    graphOptimizationLevel: "all",
    logSeverityLevel: 3,
  });
  const { inputName, outputName, frames, width } = layout;
  return {
    backend,
    threads,
    async run(input: Float32Array): Promise<Float32Array> {
      if (input.length !== frames * width) throw new Error(`classifier input has ${input.length} values, expected ${frames * width}`);
      const outputs = await session.run({ [inputName]: new ort.Tensor("float32", input, [1, frames, width]) });
      const logits = outputs[outputName];
      if (!logits) throw new Error(`model output ${outputName} missing`);
      const data = Float32Array.from(logits.data as Float32Array);
      logits.dispose();
      return data;
    },
    release: () => session.release(),
  };
}

export async function medianRunMs(runner: Runner, input: Float32Array, warmup: number, runs: number): Promise<number> {
  for (let i = 0; i < warmup; i += 1) await runner.run(input);
  const times: number[] = [];
  for (let i = 0; i < runs; i += 1) {
    const start = performance.now();
    await runner.run(input);
    times.push(performance.now() - start);
  }
  times.sort((a, b) => a - b);
  return times[times.length >> 1] as number;
}

/**
 * The runner for a preference. "auto": wasm, plus WebGPU when available; WebGPU is kept only if
 * its median time on a fixed input beats wasm's and its logits agree (top-1 and within 1e-2).
 * Any WebGPU failure falls back to wasm, with the reason recorded.
 */
export async function chooseRunner(
  bytes: Uint8Array,
  layout: TensorLayout,
  preference: BackendPreference,
  threads: number = defaultThreads(),
): Promise<{ runner: Runner; choice: BackendChoice }> {
  if (preference === "wasm" || (preference === "auto" && !webgpuAvailable())) {
    const runner = await createRunner(bytes, layout, "wasm", threads);
    const reason = preference === "wasm" ? "wasm requested" : "WebGPU is not available in this browser";
    return { runner, choice: { backend: "wasm", threads, reason, timings: {} } };
  }
  const probe = new Float32Array(layout.frames * layout.width).map((_, i) => Math.sin(i) * 0.5);
  let gpu: Runner | null = null;
  let gpuReason = "";
  let gpuMs = Number.POSITIVE_INFINITY;
  let gpuLogits: Float32Array | null = null;
  try {
    gpu = await createRunner(bytes, layout, "webgpu", threads);
    gpuLogits = await gpu.run(probe);
    gpuMs = await medianRunMs(gpu, probe, 5, 15);
  } catch (error) {
    gpuReason = `WebGPU failed: ${error instanceof Error ? error.message : String(error)}`;
    await gpu?.release().catch(() => undefined);
    gpu = null;
  }
  if (preference === "webgpu" && gpu) {
    return { runner: gpu, choice: { backend: "webgpu", threads, reason: "WebGPU requested", timings: { webgpu: gpuMs } } };
  }
  const wasm = await createRunner(bytes, layout, "wasm", threads);
  if (!gpu) return { runner: wasm, choice: { backend: "wasm", threads, reason: gpuReason, timings: {} } };
  const wasmLogits = await wasm.run(probe);
  const wasmMs = await medianRunMs(wasm, probe, 5, 15);
  const top = (values: Float32Array): number => values.indexOf(Math.max(...values));
  const agree =
    gpuLogits !== null && top(gpuLogits) === top(wasmLogits) && gpuLogits.every((value, i) => Math.abs(value - (wasmLogits[i] as number)) <= 1e-2);
  const timings = { wasm: wasmMs, webgpu: gpuMs };
  if (agree && gpuMs < wasmMs) {
    await wasm.release();
    return { runner: gpu, choice: { backend: "webgpu", threads, reason: "WebGPU was faster than wasm on this device", timings } };
  }
  await gpu.release();
  const reason = agree ? "wasm was faster than WebGPU on this device" : "WebGPU logits disagreed with wasm";
  return { runner: wasm, choice: { backend: "wasm", threads, reason, timings } };
}
