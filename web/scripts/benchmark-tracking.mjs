// End-to-end tracking benchmark on this machine: the built demo in Chrome (or Playwright's
// Chromium) with a recorded clip as its fake camera, timed by the app's own 60-second benchmark
// (src/benchmark.ts). The downloaded report is saved as JSON and summarised.
//
//   npm run build && npm run preview          (in another shell; serves COOP/COEP headers)
//   node scripts/benchmark-tracking.mjs [--delegate GPU|CPU] [--threads N] [--chromium] [--clip <y4m>] [--out <json>]
//
// The clip (default data/live_clips/water.y4m, scripts/prepare_live_clips.py) loops for the whole
// run. Every other setting is the app default, so the numbers are what the demo does as shipped:
// MediaPipe in the tracker worker, the classifier in its own worker, wasm SIMD with threads when
// the page is cross-origin isolated. --threads sets the classifier's wasm threads (0 = the app's
// automatic choice).

import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const args = process.argv.slice(2);
const flag = (name) => args.includes(name);
const option = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
};
const origin = option("--origin", "http://localhost:4173");
const delegate = option("--delegate", "GPU");
const threads = Number(option("--threads", "0"));
const channel = flag("--chromium") ? undefined : "chrome";
const clip = resolve(option("--clip", join(repo, "data", "live_clips", "water.y4m")));
const suffix = `${delegate.toLowerCase()}${threads > 0 ? `_t${threads}` : ""}${channel ? "" : "_chromium"}`;
const out = resolve(option("--out", join(repo, "results", "perf", `tracking_${suffix}.json`)));

const browser = await chromium.launch({
  channel,
  args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${clip}`],
});
const context = await browser.newContext({ permissions: ["camera"], acceptDownloads: true });
const page = await context.newPage();
const errors = [];
page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
page.on("console", (message) => {
  if (message.type() === "error" && !message.text().startsWith("INFO:")) errors.push(`console: ${message.text()}`);
});
await page.goto(`${origin}/?debug=1`);
const stored = JSON.parse((await page.evaluate(() => localStorage.getItem("edge-vision.settings.v3"))) ?? "{}");
await page.evaluate((value) => localStorage.setItem("edge-vision.settings.v3", JSON.stringify(value)), { ...stored, delegate, classifierThreads: threads });
await page.reload();
await page.waitForFunction(() => window.__edgevision !== undefined);
await page.evaluate(() => window.__edgevision.ready);
await page.waitForSelector("#start:not([disabled])", { timeout: 120_000 });
await page.click("#start");
await page.waitForFunction(() => window.__edgevision.status().running, null, { timeout: 120_000 });
// The GPU delegate compiles its shaders while warming up; start timing once frames flow.
await page.waitForFunction(() => window.__edgevision.status().frames >= 30, null, { timeout: 120_000 });
await page.waitForSelector("#benchmark:not([disabled])", { timeout: 60_000 });
const download = page.waitForEvent("download", { timeout: 180_000 });
await page.click("#benchmark");
const file = await (await download).path();
const report = JSON.parse(readFileSync(file, "utf8"));
const version = browser.version();
await browser.close();

const record = { browser: version, channel: channel ?? "chromium", clip: clip.replace(repo, "").replace(/\\/g, "/"), console_errors: errors, report };
mkdirSync(dirname(out), { recursive: true });
writeFileSync(out, JSON.stringify(record, null, 2) + "\n");

const s = report.summary;
const fmt = (summary) => (summary && summary.n > 0 ? `p50 ${summary.p50.toFixed(1)} / p95 ${summary.p95.toFixed(1)} ms (n ${summary.n})` : "–");
const ortInfo = report.context.ort_wasm;
console.log(`${version} ${channel ?? "chromium"}, delegate ${report.context.mediapipe.delegate} (requested ${delegate}), isolated ${report.environment.cross_origin_isolated}, classifier ${ortInfo.execution_provider} x${ortInfo.num_threads}`);
console.log(`tracking: ${report.frames.count} frames in ${report.duration_s.toFixed(1)} s = ${report.frames.end_to_end_fps.toFixed(1)} fps`);
console.log(`hand landmarker ${fmt(s.hand_landmarker_ms)}; pose landmarker ${fmt(s.pose_landmarker_ms)}; capture to processed ${fmt(s.frame_total_ms)}`);
console.log(`per segment: preprocess ${fmt(s.preprocess_ms)}; classifier ${fmt(s.classifier_ms)}; end of sign to result ${fmt(s.segment_end_to_end_ms)}`);
console.log(`classifier on a fixed input ${fmt(s.classifier_fixed_input_ms)}; ${errors.length} console errors`);
console.log(`wrote ${out}`);
