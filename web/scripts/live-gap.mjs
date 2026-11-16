// Live-gap evaluation, browser side: the built demo in Chromium with each clip as its fake
// camera, once per config of experiments/live_gap.json. Writes results/live_gap/web/<config>.json
// in the structure scripts/live_gap.py scores (python scripts/live_gap.py report).
//
//   npm run build && npm run preview         (in another shell)
//   node scripts/live-gap.mjs [--only c6_final x_hands] [--repeat 2] [--repeat-offset 1] [--channel chrome] [--delegate CPU]
//
// A config with "build": "head" runs the shipped build (commit 3fbafa3 with
// experiments/head_build_hook.patch applied, which only records each segment's softmax in
// window.__segments) from --head-origin (default http://localhost:4190), with its default
// settings: git archive 3fbafa3 web | tar -x -C <dir>, apply the patch, npm run fetch-assets,
// npx vite build, npx vite preview --port 4190.
//
// Each clip (data/live_clips/<clip>.y4m, scripts/prepare_live_clips.py) plays from the moment the
// camera starts; the run stops the camera when the clip's duration has passed, which also ends a
// sign in progress, and reads the segments from the page's debug hook. Byte-identical clips are
// run once and their result copied, as in the Python pipeline.

import { copyFileSync, existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
};
const list = (name) => {
  const index = args.indexOf(name);
  if (index < 0) return null;
  const values = [];
  for (let i = index + 1; i < args.length && !args[i].startsWith("--"); i += 1) values.push(args[i]);
  return values;
};
const origin = option("--origin", "http://localhost:4173");
const headOrigin = option("--head-origin", "http://localhost:4190");
const repeat = Number(option("--repeat", "1"));
// --repeat-offset N numbers the runs from N + 1, so further repeats add <config>.r<N+1>.json files
// next to an earlier run instead of replacing it.
const repeatOffset = Number(option("--repeat-offset", "0"));
const channel = option("--channel", undefined);
const delegate = option("--delegate", "CPU");
const only = list("--only");
// One browser profile for every launch: the service worker's cache and the GPU shader cache
// carry over, so only the first launch pays for them. Each clip still gets a fresh browser,
// since the fake camera file is a launch argument.
const profile = resolve(option("--profile", join(repo, "web", "test-results", "live-gap-profile")));
const clipsOnly = list("--clips");

const configs = JSON.parse(readFileSync(join(repo, "experiments", "live_gap.json"), "utf8"));
const clipsDir = join(repo, configs.clips_dir);
const manifest = JSON.parse(readFileSync(join(clipsDir, "clips.json"), "utf8"));
const outDir = join(repo, "results", "live_gap", "web");
mkdirSync(outDir, { recursive: true });

// The preview server serves web/dist; models other than the shipped one are copied under eval/.
const dist = join(repo, "web", "dist");
const modelDirs = {};
for (const [key, path] of Object.entries(configs.models)) {
  if (path === "web/public/model") {
    modelDirs[key] = "model";
    continue;
  }
  const target = join(dist, "eval", key);
  mkdirSync(target, { recursive: true });
  for (const file of ["meta.json", "model.onnx"]) copyFileSync(join(repo, path, file), join(target, file));
  modelDirs[key] = `eval/${key}`;
}

const ASPECT = { calibrated: "auto", frame: "frame", off: "off" };

function settingsFor(config) {
  return {
    mode: "auto",
    autoSegmenter: config.segmenter,
    handRule: config.hands,
    detectMirrored: config.mirrored,
    aspectMode: ASPECT[config.aspect],
    resampling: config.resampling,
    decision: config.decision,
    delegate,
    classifierBackend: "wasm",
  };
}

/** The shipped build: default settings, its own Start/Stop buttons and the patch's segment log. */
async function runHeadClip(page, entry, durationMs, errors) {
  await page.goto(`${headOrigin}/`);
  await page.waitForSelector("#start:not([disabled])", { timeout: 120_000 });
  await page.click("#start");
  await page.waitForFunction(() => document.querySelector("#stop") && !document.querySelector("#stop").disabled, null, { timeout: 120_000 });
  await page.waitForTimeout(durationMs);
  const fps = Number((await page.textContent("#fps")) ?? "NaN");
  await page.click("#stop");
  await page.waitForTimeout(1500);
  const segments = await page.evaluate(() => window.__segments ?? []);
  const statusText = await page.textContent("#status");
  return { durationMs, segments, status: { running: true, fps, handMsP50: Number.NaN, build: "head" }, statusText, errors };
}

async function runClip(entry, config) {
  const clip = join(clipsDir, entry.y4m);
  const durationMs = (entry.source_frames / entry.source_fps) * 1000;
  mkdirSync(profile, { recursive: true });
  const browser = await chromium.launchPersistentContext(profile, {
    channel,
    permissions: ["camera"],
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${clip}`],
  });
  const page = browser.pages()[0] ?? (await browser.newPage());
  const errors = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !message.text().startsWith("INFO:")) errors.push(`console: ${message.text()}`);
  });
  if (config.build === "head") {
    const result = await runHeadClip(page, entry, durationMs, errors);
    await browser.close();
    return result;
  }
  const url = `${origin}/?debug=1&model=${modelDirs[config.model]}`;
  await page.goto(url);
  await page.evaluate((value) => localStorage.setItem("edge-vision.settings.v3", JSON.stringify(value)), settingsFor(config));
  await page.evaluate(() => localStorage.setItem("edge-vision.ui.v1", JSON.stringify({ onboarded: true })));
  await page.reload();
  await page.waitForFunction(() => window.__edgevision !== undefined);
  await page.evaluate(() => window.__edgevision.ready);
  await page.waitForSelector("#start:not([disabled])", { timeout: 120_000 });
  // The app opens the camera before it waits for the landmarkers; with them loaded and warmed up
  // first, processing starts at the clip's first frame instead of some seconds into it.
  await page.waitForFunction(() => window.__edgevision.status().tracker !== null, null, { timeout: 120_000 });
  await page.click("#start");
  await page.waitForFunction(() => window.__edgevision.status().running, null, { timeout: 120_000 });
  const started = Date.now();
  await page.waitForTimeout(Math.max(0, durationMs - (Date.now() - started)));
  const status = await page.evaluate(() => window.__edgevision.status());
  await page.evaluate(() => window.__edgevision.stop());
  await page.waitForTimeout(1500); // the last segment's classification
  const segments = await page.evaluate(() => window.__edgevision.segments());
  const statusText = await page.textContent("#status");
  await browser.close();
  return { durationMs, segments, status, statusText, errors };
}

for (const config of configs.configs) {
  if (only && !only.includes(config.name)) continue;
  const meta = JSON.parse(readFileSync(join(repo, configs.models[config.model], "meta.json"), "utf8"));
  const temperature = config.decision === "calibrated" ? meta.calibration.temperature : 1;
  const threshold = config.decision === "calibrated" ? meta.calibration.threshold : meta.threshold.threshold;
  for (let r = 0; r < repeat; r += 1) {
    const clips = [];
    const done = new Map();
    for (const entry of manifest.clips) {
      if (clipsOnly && !clipsOnly.includes(entry.label)) continue;
      const twin = entry.duplicate_of.find((name) => done.has(name));
      const result = twin ? done.get(twin) : await runClip(entry, config);
      done.set(entry.label, result);
      const classified = result.segments.filter((s) => s.probabilities).length;
      console.log(
        `${config.name}${repeat > 1 || repeatOffset > 0 ? ` r${r + repeatOffset + 1}` : ""} ${entry.label}: ${result.segments.length} segments (${classified} classified), ` +
          `fps ${result.status.fps.toFixed(1)}, hand p50 ${result.status.handMsP50.toFixed(1)} ms${twin ? ` (copied from ${twin})` : ""}` +
          (result.errors.length ? `, ${result.errors.length} errors: ${result.errors[0]}` : ""),
      );
      clips.push({
        clip: entry.label,
        label: entry.label,
        label_index: meta.labels.indexOf(entry.label),
        duration_ms: result.durationMs,
        duplicate_of: entry.duplicate_of,
        copied_from: twin ?? null,
        segments: result.segments,
        status: result.status,
        status_text: result.statusText,
        errors: result.errors,
      });
    }
    const run = { pipeline: "web_tasks", config, model_sha256: meta.model_sha256, temperature, threshold, delegate, channel: channel ?? "chromium", clips };
    const index = r + repeatOffset;
    const name = index > 0 ? `${config.name}.r${index + 1}.json` : `${config.name}.json`;
    writeFileSync(join(outDir, name), JSON.stringify(run) + "\n");
  }
}
if (!existsSync(join(dist, "index.html"))) console.warn("web/dist is missing: run npm run build first");
