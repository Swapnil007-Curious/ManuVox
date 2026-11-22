import { chromium, expect, test, type BrowserContext, type Page } from "@playwright/test";
import { existsSync, mkdirSync, readFileSync, rmSync, writeFileSync } from "node:fs";
import { basename } from "node:path";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import type { DebugSegment } from "../../src/debug";
import type { WebMeta } from "../../src/meta";
import { normalize, personalize, type TaughtExample } from "../../src/personalization";

// Teach Dex, record-then-test, on test_clips_v2 (25 signs x 2 takes, one signer):
//
// 1. Teach: for each sign, take 1 plays as the camera and the real Teach panel records the
//    model's examples-per-sign (3) through its countdown and the segmenter. Opening the camera
//    restarts the fake camera's clip, so the camera is stopped before each recording and the
//    take-1 clip gets LEAD_IN_S of its first frame in front (a signer waiting for the countdown);
//    the examples are thus repeated passes of the one take-1 recording. The listening window spans
//    the clip (the panel's 8 s suits a live signer who signs after "Go").
// 2. Test: each take-2 clip plays through the final live pipeline with the examples on. Every
//    segment logs the model's own distribution and the one the app showed.
// 3. A control recomputes take 2 with only 13 of the 25 signs taught (a fixed seeded choice) with
//    the same personalize() the app runs, after checking it reproduces what the app showed.
//
// The IndexedDB store persists across the per-clip browser launches through one browser profile
// per run. Run with TEACH_EVAL=1 (and TEACH_RUNS, default 3); scripts/teach_report.py summarises.
// Settings are the final pipeline of experiments/live_gap_v2.json (c6_final). Nothing is tuned here.

const REPO = join(dirname(fileURLToPath(import.meta.url)), "..", "..", "..");
const CLIPS = join(REPO, "data", "live_clips_v2");
// TEACH_LIMIT=n: a smoke test on the first n signs, written to results/teach_dex/smoke/.
const LIMIT = Number(process.env.TEACH_LIMIT ?? 0);
// TEACH_MODEL_DIR / TEACH_OUT (repo-relative) evaluate another model variant, e.g. the v3 build
// (VITE_MODEL_VARIANT=v3-citizen) with web/models/v3-citizen and results/teach_dex_v3/web.
const MODEL_DIR = join(REPO, process.env.TEACH_MODEL_DIR ?? join("web", "public", "model"));
const OUT = process.env.TEACH_OUT ? join(REPO, process.env.TEACH_OUT) : join(REPO, "results", "teach_dex", LIMIT > 0 ? "smoke" : "web");
const RUNS = Number(process.env.TEACH_RUNS ?? 3);
const ENABLED = process.env.TEACH_EVAL === "1" && existsSync(join(CLIPS, "clips.json"));
const SETTINGS = {
  mode: "auto",
  autoSegmenter: "motion",
  handRule: "nearest_pose_wrist",
  detectMirrored: false,
  aspectMode: "frame",
  resampling: "time",
  decision: "calibrated",
  delegate: "GPU",
  classifierBackend: "wasm",
  personalization: true,
};
const MAX_ATTEMPTS = 6;
// Longer than the panel's countdown plus the camera start, so the recorded sign falls in the
// listening window. The lead-in clips go to data/live_clips_v2_teach/ (gitignored, personal video).
const LEAD_IN_S = 3.5;
const LEAD_IN_DIR = join(REPO, "data", "live_clips_v2_teach");
const HALF = 13;
const K = 5;

interface Clip {
  y4m: string;
  label: string;
  take: number;
  source_fps: number;
  source_frames: number;
}

type UiHook = Record<string, (...args: unknown[]) => unknown>;

function mulberry32(seed: number): () => number {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** The control's taught signs: a seeded shuffle of the clip signs, first HALF. */
function halfTaught(signs: string[]): Set<string> {
  const random = mulberry32(2026);
  const order = [...signs].sort();
  for (let i = order.length - 1; i > 0; i -= 1) {
    const j = Math.floor(random() * (i + 1));
    [order[i], order[j]] = [order[j] as string, order[i] as string];
  }
  return new Set(order.slice(0, HALF));
}

/** A copy of a .y4m clip with its first frame repeated for ``seconds`` in front. */
function withLeadIn(source: string, seconds: number): string {
  mkdirSync(LEAD_IN_DIR, { recursive: true });
  const target = join(LEAD_IN_DIR, basename(source));
  if (existsSync(target)) return target;
  const bytes = readFileSync(source);
  const headerEnd = bytes.indexOf(0x0a) + 1;
  const fields = Object.fromEntries(
    bytes
      .subarray(0, headerEnd - 1)
      .toString("ascii")
      .split(" ")
      .slice(1)
      .map((token) => [token[0], token.slice(1)]),
  );
  const [num, den] = (fields.F ?? "30:1").split(":").map(Number);
  const frameSize = (Number(fields.W) * Number(fields.H) * 3) / 2;
  const frameHeaderEnd = bytes.indexOf(0x0a, headerEnd) + 1;
  const first = bytes.subarray(headerEnd, frameHeaderEnd + frameSize); // The FRAME line and its data.
  const copies = Math.ceil(seconds * ((num as number) / (den as number)));
  writeFileSync(target, Buffer.concat([bytes.subarray(0, headerEnd), ...Array.from({ length: copies }, () => first), bytes.subarray(headerEnd)]));
  return target;
}

async function launch(clip: string, profile: string): Promise<{ context: BrowserContext; page: Page }> {
  // The installed Chrome, as the live-gap runs (--channel chrome): the target browser, whose GPU
  // delegate is hardware-accelerated where the bundled headless Chromium renders in software.
  const context = await chromium.launchPersistentContext(profile, {
    channel: "chrome",
    permissions: ["camera"],
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${clip}`],
  });
  const page = context.pages()[0] ?? (await context.newPage());
  return { context, page };
}

async function openApp(page: Page, baseURL: string): Promise<void> {
  await page.goto(`${baseURL}?debug=1`);
  await page.evaluate((settings) => {
    localStorage.setItem("edge-vision.settings.v3", JSON.stringify(settings));
    localStorage.setItem("edge-vision.ui.v1", JSON.stringify({ onboarded: true, autoTranslate: false, autoSpeak: false }));
  }, SETTINGS);
  await page.reload();
  await page.waitForFunction(() => window.__edgevision !== undefined);
  await page.evaluate(() => window.__edgevision?.ready);
  await page.waitForFunction(() => window.__edgevision?.status().tracker !== null, null, { timeout: 120_000 });
}

function ui(page: Page, method: string, ...args: unknown[]): Promise<unknown> {
  return page.evaluate(([name, values]) => ((window as unknown as { __edgevisionUi: UiHook }).__edgevisionUi[name as string] as (...a: unknown[]) => unknown)(...(values as unknown[])), [method, args] as const);
}

function mean(rows: number[][]): number[] {
  return rows[0]!.map((_, i) => rows.reduce((sum, row) => sum + (row[i] as number), 0) / rows.length);
}

function topOf(values: number[], k: number): number[] {
  return values.map((v, i) => [v, i] as const).sort((a, b) => b[0] - a[0] || a[1] - b[1]).slice(0, k).map(([, i]) => i);
}

test.describe("Teach Dex record-then-test on test_clips_v2", () => {
  test.skip(!ENABLED, "set TEACH_EVAL=1 with data/live_clips_v2 (scripts/prepare_live_clips.py --camera)");
  test.setTimeout(6 * 60 * 60 * 1000);

  test("take 1 teaches, take 2 tests", async ({ baseURL }) => {
    const base = baseURL as string;
    const every = (JSON.parse(readFileSync(join(CLIPS, "clips.json"), "utf8")) as { clips: Clip[] }).clips;
    const limited = LIMIT > 0 ? new Set([...new Set(every.map((clip) => clip.label))].sort().slice(0, LIMIT)) : null;
    const clips = limited ? every.filter((clip) => limited.has(clip.label)) : every;
    const meta = JSON.parse(readFileSync(join(MODEL_DIR, "meta.json"), "utf8")) as WebMeta;
    expect(meta.personalization, "meta.json needs the personalization block (scripts/calibrate_teach.py)").toBeTruthy();
    const params = { weight: meta.personalization!.weight, offset: meta.personalization!.offset, influenceThreshold: meta.personalization!.influence_threshold };
    const signs = [...new Set(clips.map((clip) => clip.label))].sort();
    const half = halfTaught(signs);
    mkdirSync(OUT, { recursive: true });

    for (let run = 1; run <= RUNS; run += 1) {
      const outcomesPath = join(OUT, `run${run}.outcomes.json`);
      if (existsSync(outcomesPath)) continue; // Resumable: finished runs are kept.
      const profile = join(REPO, "web", "test-results", "teach-eval", `run${run}`);
      rmSync(profile, { recursive: true, force: true });
      mkdirSync(profile, { recursive: true });

      // A warm-up launch: the profile's service-worker and shader caches fill once, so the first
      // clip of the run is not slower than the rest (web/scripts/live-gap.mjs does the same).
      {
        const first = clips.find((c) => c.take === 1) as Clip;
        const { context, page } = await launch(join(CLIPS, first.y4m), profile);
        try {
          await openApp(page, base);
          await page.click("#start");
          await page.waitForFunction(() => window.__edgevision?.status().running, null, { timeout: 120_000 });
          await page.waitForTimeout(4000);
          await page.evaluate(() => window.__edgevision?.stop());
        } finally {
          await context.close();
        }
      }

      // 1. Teach with take 1.
      const teaching: Record<string, { examples: number; attempts: number }> = {};
      for (const clip of clips.filter((c) => c.take === 1)) {
        const { context, page } = await launch(withLeadIn(join(CLIPS, clip.y4m), LEAD_IN_S), profile);
        try {
          await openApp(page, base);
          await ui(page, "setMode", "teach");
          await ui(page, "teachSelect", clip.label);
          // The panel waits 8 s for a live signer; a replayed clip's sign may come later than a
          // signer prompted by the countdown would sign, so the wait spans the whole clip. The
          // example is still the first segment after the countdown.
          await ui(page, "teachListenMs", Math.round((LEAD_IN_S + clip.source_frames / clip.source_fps + 2) * 1000));
          let attempts = 0;
          let have = 0;
          while (have < meta.personalization!.examples_per_sign && attempts < MAX_ATTEMPTS) {
            attempts += 1;
            // Reopening the camera restarts the clip: lead-in, then the sign.
            await page.evaluate(() => window.__edgevision?.stop());
            await page.waitForFunction(() => !window.__edgevision?.status().running, null, { timeout: 30_000 });
            void ui(page, "teachRecord").catch(() => undefined);
            await page.waitForFunction(() => ((window as unknown as { __edgevisionUi: UiHook }).__edgevisionUi.teachPhase as () => string)() !== "idle", null, { timeout: 30_000 }).catch(() => undefined);
            await page.waitForFunction(() => ((window as unknown as { __edgevisionUi: UiHook }).__edgevisionUi.teachPhase as () => string)() === "idle", null, { timeout: 30_000 });
            const examples = (await ui(page, "teachExamples")) as { label: string }[];
            have = examples.filter((example) => example.label === clip.label).length;
          }
          teaching[clip.label] = { examples: have, attempts };
        } finally {
          await context.close();
        }
      }

      // 2. Test with take 2; the examples come from the profile's IndexedDB.
      const raw: { clip: string; label: string; segments: DebugSegment[]; storeMode: unknown }[] = [];
      let taught: TaughtExample[] = [];
      for (const clip of clips.filter((c) => c.take === 2)) {
        const { context, page } = await launch(join(CLIPS, clip.y4m), profile);
        try {
          await openApp(page, base);
          const exported = (await ui(page, "teachExamples")) as { label: string; embedding: number[]; modelSha256: string }[];
          taught = exported.map((e, i) => ({ id: `${i}`, label: e.label, embedding: normalize(e.embedding), createdAt: 0, modelSha256: e.modelSha256 }));
          // The build must serve the model whose meta.json scores the run.
          expect(taught.every((e) => e.modelSha256 === meta.model_sha256), "the app's examples come from the model in MODEL_DIR").toBe(true);
          const storeMode = await ui(page, "teachStoreMode");
          await page.waitForSelector("#start:not([disabled])", { timeout: 120_000 });
          await page.click("#start");
          await page.waitForFunction(() => window.__edgevision?.status().running, null, { timeout: 120_000 });
          await page.waitForTimeout((clip.source_frames / clip.source_fps) * 1000);
          await page.evaluate(() => window.__edgevision?.stop());
          await page.waitForTimeout(1500);
          const segments = (await page.evaluate(() => window.__edgevision?.segments() ?? [])) as DebugSegment[];
          raw.push({ clip: clip.y4m.replace(/\.y4m$/, ""), label: clip.label, segments, storeMode });
        } finally {
          await context.close();
        }
      }

      // 3. Clip outcomes for the three conditions, with the app-vs-recomputation check.
      const T = meta.calibration.temperature;
      const threshold = meta.calibration.threshold;
      const halfExamples = taught.filter((example) => half.has(example.label));
      let worstParity = 0;
      const outcomes = raw.map(({ clip, label, segments }) => {
        const target = meta.labels.indexOf(label);
        const classified = segments.filter((s) => s.probabilities && s.logits && s.base_probabilities);
        const score = (rows: number[][]) => {
          if (rows.length === 0) return { top1: 0, top3: 0 };
          const top = topOf(mean(rows), 3);
          return { top1: Number(top[0] === target), top3: Number(top.includes(target)) };
        };
        const halfRows = classified.map((s) => {
          const embedding = s.embedding ? Float32Array.from(s.embedding) : null;
          const all = personalize(Float32Array.from(s.logits!), embedding, meta.labels, taught, params, threshold, K, T);
          for (let i = 0; i < all.probabilities.length; i += 1) {
            worstParity = Math.max(worstParity, Math.abs((all.probabilities[i] as number) - (s.probabilities![i] as number)));
          }
          return Array.from(personalize(Float32Array.from(s.logits!), embedding, meta.labels, halfExamples, params, threshold, K, T).probabilities);
        });
        return {
          clip,
          label,
          segments: classified.length,
          badged_segments: classified.filter((s) => s.personalized).length,
          base: score(classified.map((s) => s.base_probabilities!)),
          personalized: score(classified.map((s) => s.probabilities!)),
          half: score(halfRows),
          half_taught: half.has(label),
        };
      });
      // The recomputation must reproduce what the app showed (float32 inputs, float64 maths).
      expect(worstParity, "personalize() in Node reproduces the app's decisions").toBeLessThan(1e-6);
      writeFileSync(
        outcomesPath,
        JSON.stringify(
          {
            run,
            model_sha256: meta.model_sha256,
            lead_in_s: LEAD_IN_S,
            params,
            settings: SETTINGS,
            store_modes: [...new Set(raw.map((r) => r.storeMode))],
            teaching,
            half_taught_signs: [...half].sort(),
            parity_max_abs_diff: worstParity,
            clips: outcomes,
          },
          null,
          1,
        ) + "\n",
      );
      writeFileSync(join(OUT, `run${run}.raw.json`), JSON.stringify(raw)); // Gitignored: logits and embeddings.
    }
  });
});
