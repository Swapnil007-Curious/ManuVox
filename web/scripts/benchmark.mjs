// Classifier latency on this machine: drives bench.html in the installed Chrome (or Playwright's
// Chromium) once per configuration and writes the results as JSON.
//
//   npm run build && npm run preview          (in another shell; serves COOP/COEP headers)
//   node scripts/benchmark.mjs --out ../results/perf/classifier_bench.json [--headed] [--chromium]
//                              [--model eval/<name>] [--only wasm:1]
//
// Configurations: wasm with 1, 2 and 4 threads, and WebGPU. Each runs 20 warm-up and 200 timed
// calls on fixed inputs and records p50 / p95 and the logits' agreement with wasm on 1 thread.

import { mkdirSync, writeFileSync } from "node:fs";
import { dirname } from "node:path";
import { chromium } from "playwright";

const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const option = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
};
const origin = option("--origin", "http://localhost:4173");
const out = option("--out", "../results/perf/classifier_bench.json");
const model = option("--model", "model");
const only = option("--only", null);
const configs = [
  { ep: "wasm", threads: 1 },
  { ep: "wasm", threads: 2 },
  { ep: "wasm", threads: 4 },
  { ep: "webgpu", threads: 1 },
].filter((config) => only === null || only === `${config.ep}:${config.threads}`);

const browser = await chromium.launch({
  channel: flag("--chromium") ? undefined : "chrome",
  headless: !flag("--headed"),
  args: ["--enable-unsafe-webgpu", "--ignore-gpu-blocklist", "--enable-features=Vulkan"],
});
const results = [];
for (const config of configs) {
  const page = await browser.newPage();
  const errors = [];
  page.on("pageerror", (error) => errors.push(error.message));
  page.on("console", (message) => {
    if (message.type() === "error") errors.push(message.text());
  });
  await page.goto(`${origin}/bench.html?ep=${config.ep}&threads=${config.threads}&model=${model}`);
  await page.waitForFunction(() => window.__bench !== undefined, null, { timeout: 300_000 });
  const result = await page.evaluate(() => window.__bench);
  results.push({ ...config, ...result, consoleErrors: errors });
  console.log(
    `${config.ep} x${config.threads}: ` +
      (result.error ? `failed: ${result.error}` : `p50 ${result.p50.toFixed(2)} ms, p95 ${result.p95.toFixed(2)} ms, session ${result.sessionMs.toFixed(0)} ms, isolated ${result.crossOriginIsolated}`),
  );
  await page.close();
}
const version = browser.version();
await browser.close();
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify({ browser: version, measured_at: new Date().toISOString(), results }, null, 2) + "\n");
console.log(`wrote ${out}`);
