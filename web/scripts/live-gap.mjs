// Live-gap evaluation, browser side: the built demo in Chromium with each clip as its fake
// camera, once per config of experiments/live_gap.json. Writes results/live_gap/web/<config>.json
// in the structure scripts/live_gap.py scores (python scripts/live_gap.py report).
//
//   npm run build && npm run preview         (in another shell)
//   node scripts/live-gap.mjs [--only c6_final x_hands] [--repeat 2] [--repeat-offset 1] [--channel chrome] [--delegate CPU]
//   node scripts/live-gap.mjs --configs experiments/live_gap_v2.json --per-clip --repeat 3 --channel chrome --delegate GPU
//
// --configs picks the config file (default experiments/live_gap.json); runs go to
// <results_dir>/web/ of that file (default results/live_gap/web/). By default each config runs
// over every clip with a fresh browser per clip. --per-clip instead launches one browser per clip
// and runs every selected config in it, in a random order seeded by the clip and repeat, so slow
// drift of the laptop (heat, background load) is spread over the configs instead of falling on
// whichever ran last; the files are written after every clip and a rerun skips the clips a file
// already holds, so an interrupted run resumes. Clips are identified by their .y4m name (the
// label for test_clips, <label>_<take> for test_clips_v2).
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
const perClip = args.includes("--per-clip");

const configsPath = resolve(repo, option("--configs", join("experiments", "live_gap.json")));
const configs = JSON.parse(readFileSync(configsPath, "utf8"));
const clipsDir = join(repo, configs.clips_dir);
const manifest = JSON.parse(readFileSync(join(clipsDir, "clips.json"), "utf8"));
const outDir = join(repo, configs.results_dir ?? join("results", "live_gap"), "web");
mkdirSync(outDir, { recursive: true });
/** A clip's id: its .y4m name without the extension. */
const clipId = (entry) => entry.y4m.replace(/\.y4m$/, "");

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

/** A browser whose fake camera plays this clip; `sink.errors` collects page and console errors. */
async function launchFor(entry) {
  const clip = join(clipsDir, entry.y4m);
  mkdirSync(profile, { recursive: true });
  const browser = await chromium.launchPersistentContext(profile, {
    channel,
    permissions: ["camera"],
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${clip}`],
  });
  const page = browser.pages()[0] ?? (await browser.newPage());
  const sink = { errors: [] };
  page.on("pageerror", (error) => sink.errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    if (message.type() === "error" && !message.text().startsWith("INFO:")) sink.errors.push(`console: ${message.text()}`);
  });
  return { browser, page, sink };
}

async function runClip(entry, config) {
  const session = await launchFor(entry);
  try {
    return await runOnPage(session, entry, config);
  } finally {
    await session.browser.close();
  }
}

/** One config over one clip in an open browser (the camera restarts the clip from its first frame). */
async function runOnPage({ page, sink }, entry, config) {
  const durationMs = (entry.source_frames / entry.source_fps) * 1000;
  sink.errors = [];
  const errors = sink.errors;
  if (config.build === "head") return runHeadClip(page, entry, durationMs, errors);
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
  return { durationMs, segments, status, statusText, errors };
}

/** The decision parameters of a config's model, as the report scores them. */
function decisionOf(config) {
  const meta = JSON.parse(readFileSync(join(repo, configs.models[config.model], "meta.json"), "utf8"));
  const temperature = config.decision === "calibrated" ? meta.calibration.temperature : 1;
  const threshold = config.decision === "calibrated" ? meta.calibration.threshold : meta.threshold.threshold;
  return { meta, temperature, threshold };
}

function clipRecord(entry, meta, result, twin) {
  return {
    clip: clipId(entry),
    label: entry.label,
    label_index: meta.labels.indexOf(entry.label),
    duration_ms: result.durationMs,
    duplicate_of: entry.duplicate_of,
    copied_from: twin ?? null,
    segments: result.segments,
    status: result.status,
    status_text: result.statusText,
    errors: result.errors,
  };
}

function logResult(name, id, result, twin) {
  const classified = result.segments.filter((s) => s.probabilities).length;
  console.log(
    `${name} ${id}: ${result.segments.length} segments (${classified} classified), ` +
      `fps ${Number(result.status.fps).toFixed(1)}, hand p50 ${Number(result.status.handMsP50).toFixed(1)} ms${twin ? ` (copied from ${twin})` : ""}` +
      (result.errors.length ? `, ${result.errors.length} errors: ${result.errors[0]}` : ""),
  );
}

/** Mulberry32: a small seeded generator for the per-clip config order. */
function seeded(seed) {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
const hashString = (text) => [...text].reduce((h, c) => (Math.imul(h, 31) + c.charCodeAt(0)) >>> 0, 7);

if (perClip) {
  const selected = configs.configs.filter((config) => !only || only.includes(config.name));
  const entries = manifest.clips.filter((entry) => !clipsOnly || clipsOnly.includes(clipId(entry)));
  for (let r = 0; r < repeat; r += 1) {
    const index = r + repeatOffset;
    const fileOf = (config) => join(outDir, index > 0 ? `${config.name}.r${index + 1}.json` : `${config.name}.json`);
    // Resume: start from what an earlier, interrupted run of this repeat already wrote.
    const runs = new Map();
    for (const config of selected) {
      const { meta, temperature, threshold } = decisionOf(config);
      const existing = existsSync(fileOf(config)) ? JSON.parse(readFileSync(fileOf(config), "utf8")) : null;
      runs.set(config.name, {
        meta,
        run: existing ?? { pipeline: "web_tasks", config, model_sha256: meta.model_sha256, temperature, threshold, delegate, channel: channel ?? "chromium", order: "per_clip_shuffled", clips: [] },
      });
    }
    for (const entry of entries) {
      const id = clipId(entry);
      const pending = selected.filter((config) => !runs.get(config.name).run.clips.some((clip) => clip.clip === id));
      if (pending.length === 0) continue;
      // Byte-identical clips: copy the twin's result, as the Python pipeline does.
      const twinOf = (config) => entry.duplicate_of.find((name) => runs.get(config.name).run.clips.some((clip) => clip.clip === name));
      const order = pending.filter((config) => !twinOf(config));
      for (const config of pending.filter((c) => twinOf(c))) {
        const { run, meta } = runs.get(config.name);
        const twin = run.clips.find((clip) => clip.clip === twinOf(config));
        run.clips.push({ ...twin, clip: id, label: entry.label, label_index: meta.labels.indexOf(entry.label), copied_from: twin.clip });
      }
      const random = seeded(hashString(id) + 7919 * (index + 1));
      for (let i = order.length - 1; i > 0; i -= 1) {
        const j = Math.floor(random() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
      let session = order.length ? await launchFor(entry) : null;
      for (const config of order) {
        const { run, meta } = runs.get(config.name);
        try {
          const result = await runOnPage(session, entry, config);
          run.clips.push(clipRecord(entry, meta, result, null));
          logResult(`r${index + 1} ${config.name}`, id, result, null);
        } catch (error) {
          // Left out of the file, so a rerun of the same command retries it.
          console.log(`r${index + 1} ${config.name} ${id}: FAILED (${error.message.split("\n")[0]}); rerun to retry`);
          await session.browser.close().catch(() => {});
          session = await launchFor(entry);
        }
      }
      if (session) await session.browser.close();
      for (const config of selected) {
        const { run } = runs.get(config.name);
        // Keep the manifest's clip order, whatever order the configs ran in.
        run.clips.sort((a, b) => entries.findIndex((e) => clipId(e) === a.clip) - entries.findIndex((e) => clipId(e) === b.clip));
        run.complete = entries.every((e) => run.clips.some((clip) => clip.clip === clipId(e)));
        writeFileSync(fileOf(config), JSON.stringify(run) + "\n");
      }
    }
  }
}

for (const config of perClip ? [] : configs.configs) {
  if (only && !only.includes(config.name)) continue;
  const { meta, temperature, threshold } = decisionOf(config);
  for (let r = 0; r < repeat; r += 1) {
    const clips = [];
    const done = new Map();
    for (const entry of manifest.clips) {
      const id = clipId(entry);
      if (clipsOnly && !clipsOnly.includes(id)) continue;
      const twin = entry.duplicate_of.find((name) => done.has(name));
      const result = twin ? done.get(twin) : await runClip(entry, config);
      done.set(id, result);
      logResult(`${config.name}${repeat > 1 || repeatOffset > 0 ? ` r${r + repeatOffset + 1}` : ""}`, id, result, twin);
      clips.push(clipRecord(entry, meta, result, twin));
    }
    const run = { pipeline: "web_tasks", config, model_sha256: meta.model_sha256, temperature, threshold, delegate, channel: channel ?? "chromium", clips };
    const index = r + repeatOffset;
    const name = index > 0 ? `${config.name}.r${index + 1}.json` : `${config.name}.json`;
    writeFileSync(join(outDir, name), JSON.stringify(run) + "\n");
  }
}
if (!existsSync(join(dist, "index.html"))) console.warn("web/dist is missing: run npm run build first");
