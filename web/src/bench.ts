// Classifier benchmark page (bench.html): one onnxruntime-web configuration per page load, timed
// on fixed inputs, with the logits checked against a wasm single-thread reference computed in the
// same page. web/scripts/benchmark.mjs drives it in Chrome and collects window.__bench.

import { createRunner, crossOriginIsolatedContext, webgpuAvailable, type Backend } from "./ortBackend";

interface BenchResult {
  backend: Backend;
  threads: number;
  crossOriginIsolated: boolean;
  webgpu: boolean;
  userAgent: string;
  hardwareConcurrency: number;
  model: { file: string; sha256: string; variant: string; bytes: number };
  sessionMs: number;
  runs: number;
  p50: number;
  p95: number;
  mean: number;
  maxAbsDiffVsWasm1: number | null;
  top1AgreesWithWasm1: boolean | null;
  error: string | null;
}

declare global {
  interface Window {
    __bench?: BenchResult | { error: string };
  }
}

const status = document.getElementById("bench-status") as HTMLElement;
const output = document.getElementById("bench-result") as HTMLElement;

function percentile(sorted: number[], q: number): number {
  return sorted[Math.min(sorted.length - 1, Math.floor(q * sorted.length))] as number;
}

async function main(): Promise<void> {
  const params = new URLSearchParams(location.search);
  const backend = (params.get("ep") ?? "wasm") as Backend;
  const threads = Number(params.get("threads") ?? "1");
  const runs = Number(params.get("runs") ?? "200");
  const base = `${import.meta.env.BASE_URL}${params.get("model") ?? "model"}/`;
  const meta = await (await fetch(`${base}meta.json`)).json();
  const bytes = new Uint8Array(await (await fetch(`${base}${meta.model_file}`)).arrayBuffer());
  const layout = {
    inputName: meta.input_layout.input_name,
    outputName: meta.input_layout.output_name,
    frames: meta.input_layout.num_frames,
    width: meta.input_layout.frame_dim,
  };
  const inputs = Array.from({ length: 8 }, (_, k) =>
    new Float32Array(layout.frames * layout.width).map((_, i) => Math.sin(i * 0.37 + k) * 0.6),
  );
  status.textContent = `Running ${backend}, ${threads} thread(s)…`;
  const t0 = performance.now();
  const runner = await createRunner(bytes, layout, backend, threads);
  const sessionMs = performance.now() - t0;
  for (let i = 0; i < 20; i += 1) await runner.run(inputs[i % inputs.length] as Float32Array);
  const times: number[] = [];
  for (let i = 0; i < runs; i += 1) {
    const start = performance.now();
    await runner.run(inputs[i % inputs.length] as Float32Array);
    times.push(performance.now() - start);
  }
  const logits = await runner.run(inputs[0] as Float32Array);
  await runner.release();
  let maxAbsDiff: number | null = null;
  let agrees: boolean | null = null;
  if (!(backend === "wasm" && threads === 1)) {
    // The reference: the plain wasm build on one thread, in this page.
    const reference = await createRunner(bytes, layout, "wasm", 1);
    const expected = await reference.run(inputs[0] as Float32Array);
    await reference.release();
    maxAbsDiff = expected.reduce((max, value, i) => Math.max(max, Math.abs(value - (logits[i] as number))), 0);
    agrees = expected.indexOf(Math.max(...expected)) === logits.indexOf(Math.max(...logits));
  }
  const sorted = [...times].sort((a, b) => a - b);
  const result: BenchResult = {
    backend,
    threads,
    crossOriginIsolated: crossOriginIsolatedContext(),
    webgpu: webgpuAvailable(),
    userAgent: navigator.userAgent,
    hardwareConcurrency: navigator.hardwareConcurrency,
    model: { file: meta.model_file, sha256: meta.model_sha256, variant: meta.variant, bytes: bytes.length },
    sessionMs,
    runs,
    p50: percentile(sorted, 0.5),
    p95: percentile(sorted, 0.95),
    mean: times.reduce((a, b) => a + b, 0) / times.length,
    maxAbsDiffVsWasm1: maxAbsDiff,
    top1AgreesWithWasm1: agrees,
    error: null,
  };
  window.__bench = result;
  output.textContent = JSON.stringify(result, null, 2);
  status.textContent = `Done: p50 ${result.p50.toFixed(2)} ms, p95 ${result.p95.toFixed(2)} ms.`;
}

main().catch((error: unknown) => {
  const message = error instanceof Error ? error.message : String(error);
  window.__bench = { error: message };
  status.textContent = `Failed: ${message}`;
});
