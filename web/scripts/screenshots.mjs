// Screenshots of the app for docs/screenshots/: the built demo in Chrome with a recorded clip as
// its fake camera, at 1366 x 768, 1440 x 900 and iPhone 14 (390 x 844, captured at 2x), in the
// dark and light themes.
//
//   VITE_WORKER_URL=https://translate.invalid npx vite build --outDir dist-shots
//   npx vite preview --outDir dist-shots --port 4174          (another shell)
//   node scripts/screenshots.mjs [--origin http://localhost:4174] [--clip <y4m>] [--only 1366x768-dark]
//
// What is real and what is staged: onboarding, the framing guide, signing and the word landing come
// from the clip running through the real pipeline. States a recording cannot produce on cue are
// injected through the debug hook (window.__edgevision.simulate, the event the engine emits after
// classifying a segment): the low-confidence "Not sure" picker, the extra words of the example
// sentence and the practice outcomes (the clip's own results are muted meanwhile, through
// window.__edgevision.muteResults). The translation Worker is not deployed, so its reply is a
// fixed sentence returned by an intercepted /translate request. The camera-denied state is a
// browser without camera permission. The camera video is blurred in every capture (see shot()).

import { mkdirSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { chromium, devices } from "playwright";

const here = dirname(fileURLToPath(import.meta.url));
const repo = resolve(here, "..", "..");
const args = process.argv.slice(2);
const option = (name, fallback) => {
  const index = args.indexOf(name);
  return index >= 0 ? args[index + 1] : fallback;
};
const origin = option("--origin", "http://localhost:4174");
const clip = resolve(option("--clip", join(repo, "data", "live_clips", "hello.y4m")));
const only = option("--only", null);
const out = join(repo, "docs", "screenshots");
mkdirSync(out, { recursive: true });

const iphone = devices["iPhone 14"];
const VIEWPORTS = [
  { name: "1366x768", options: { viewport: { width: 1366, height: 768 }, deviceScaleFactor: 1 } },
  { name: "1440x900", options: { viewport: { width: 1440, height: 900 }, deviceScaleFactor: 1 } },
  {
    name: "iphone14",
    mobile: true,
    options: { viewport: iphone.viewport, deviceScaleFactor: 2, isMobile: true, hasTouch: true, userAgent: iphone.userAgent },
  },
];
const THEMES = ["dark", "light"];
const SENTENCE = "Hello Mom, I'd like to drink some water.";

function result(top, accepted, dominant = "right") {
  return {
    kind: "classified",
    start: 0,
    end: 1100,
    frames: 28,
    handFrames: 27,
    reason: "still",
    dominant,
    decision: { accepted, label: accepted ? top[0][0] : null, probability: top[0][1], threshold: 0.49, top: top.map(([label, probability], index) => ({ index, label, probability })) },
    excluded: null,
    classifierMs: 18,
  };
}

// Privacy: the recorded clips show the author's face, so every screenshot blurs the camera video
// (only the <video>: the hand skeleton on the HUD canvas and the interface stay sharp). The blur is
// set through the CSSOM (the app's CSP allows no injected <style>) for the capture alone, so its GPU
// cost never slows the tracking that produces the live states.
const VIDEO_BLUR = "blur(26px)";
const setVideoFilter = (page, value) =>
  page.evaluate((filter) => {
    const video = document.getElementById("video");
    if (video) video.style.filter = filter;
  }, value);

async function shot(page, name, combo) {
  const file = join(out, `${name}-${combo}.png`);
  await setVideoFilter(page, VIDEO_BLUR);
  await page.screenshot({ path: file });
  await setVideoFilter(page, "");
  console.log(`  ${name}`);
}

const simulate = (page, value) => page.evaluate((r) => window.__edgevision.simulate(r), value);
const ui = (page, method, ...values) => page.evaluate(([m, v]) => window.__edgevisionUi[m](...v), [method, values]);

async function waitIdle(page) {
  await page.waitForFunction(() => !["signing", "reading"].includes(document.getElementById("status-pill").dataset.state), null, { timeout: 30_000 });
}

async function run(viewport, theme) {
  const combo = `${viewport.name}-${theme}`;
  if (only && only !== combo) return;
  console.log(combo);
  const browser = await chromium.launch({
    channel: "chrome",
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${clip}`],
  });
  const context = await browser.newContext({ ...viewport.options, colorScheme: theme, permissions: ["camera"], reducedMotion: "no-preference" });
  await context.route("**/translate", (route) =>
    route.fulfill({ status: 200, contentType: "application/json", headers: { "access-control-allow-origin": "*" }, body: JSON.stringify({ sentence: SENTENCE, model: "mock" }) }),
  );
  const page = await context.newPage();
  page.on("pageerror", (error) => console.log(`  pageerror: ${error.message}`));
  await page.addInitScript(() => {
    // Quiet speech (headless voices) and keep the example sentence on screen.
    const stored = JSON.parse(localStorage.getItem("edge-vision.ui.v1") ?? "{}");
    localStorage.setItem("edge-vision.ui.v1", JSON.stringify({ autoSpeak: false, ...stored }));
  });
  await page.goto(`${origin}/?debug=1`);
  await page.waitForFunction(() => window.__edgevision !== undefined);
  await page.evaluate(() => window.__edgevision.ready);
  await page.waitForFunction(() => window.__edgevision.status().tracker !== null, null, { timeout: 120_000 });

  // Onboarding, step 1.
  await page.waitForSelector("#onboarding:not([hidden])");
  await page.waitForTimeout(400);
  await shot(page, "01-onboarding", combo);

  // Step 2: the camera runs and the framing guide shows.
  await page.click("#onboarding-primary");
  await page.waitForFunction(() => document.getElementById("stage").dataset.running === "true", null, { timeout: 60_000 });
  await page.waitForTimeout(900);
  await shot(page, "02-framing-guide", combo);
  await page.click("#onboarding-skip");

  // Signing: the motion segmenter has opened a segment.
  await page.waitForFunction(() => document.getElementById("status-pill").dataset.state === "signing", null, { timeout: 60_000, polling: 50 });
  await page.waitForTimeout(250);
  await shot(page, "03-signing", combo);

  // The word landing in its confidence ring (a real recognition of the clip).
  await page.waitForFunction(() => Number(getComputedStyle(document.getElementById("landing")).opacity) > 0.96, null, { timeout: 60_000, polling: "raf" });
  await shot(page, "04-word-landing", combo);

  // From here on the clip's own results are muted, so staged states are not overwritten.
  await waitIdle(page);
  await page.waitForTimeout(1200);
  await page.evaluate(() => window.__edgevision.muteResults(true));
  // The low-confidence picker (injected).
  await simulate(page, result([["drink", 0.34], ["water", 0.27], ["milk", 0.12]], false, "right"));
  await page.waitForTimeout(500);
  await shot(page, "05-top3-picker", combo);

  // The translated sentence: pick DRINK, add MOM before it and WATER after (injected), translate.
  await page.evaluate(() => window.__edgevision.stop());
  await page.waitForTimeout(400);
  await page.keyboard.press("Backspace"); // dismiss the picker
  await page.evaluate(() => {
    const words = [...document.querySelectorAll("#transcript li")].map((li) => li.textContent);
    return words;
  });
  for (const value of [
    result([["mom", 0.86], ["dad", 0.06], ["aunt", 0.03]], true, "right"),
    result([["drink", 0.71], ["water", 0.12], ["milk", 0.05]], true, "right"),
    result([["water", 0.93], ["wet", 0.03], ["drink", 0.01]], true, "right"),
  ]) {
    await simulate(page, value);
    await page.waitForTimeout(1500);
  }
  if (viewport.mobile) await ui(page, "expandSheet", true);
  await page.locator("#translate").click();
  await page.waitForFunction((s) => document.getElementById("sentence").textContent === s, SENTENCE, { timeout: 20_000 });
  await page.waitForTimeout(700);
  await shot(page, "06-sentence", combo);
  if (viewport.mobile) await ui(page, "expandSheet", false);

  // Practice: correct, then close (outcomes injected after the real countdown).
  await page.locator("#camera-toggle").click().catch(() => undefined);
  if (viewport.mobile) await page.evaluate(() => document.getElementById("start").click());
  await page.waitForFunction(() => document.getElementById("stage").dataset.running === "true", null, { timeout: 60_000 });
  await ui(page, "setMode", "practice");
  await ui(page, "practiceTarget", "hello");
  for (const [name, top, outcome] of [
    ["07-practice-correct", [["hello", 0.91], ["uncle", 0.03], ["sun", 0.01]], "correct"],
    ["08-practice-close", [["uncle", 0.41], ["hello", 0.33], ["hat", 0.08]], "close"],
  ]) {
    // The clip keeps signing, so a real segment can be scored first: retry until ours lands.
    for (let attempt = 0; attempt < 4; attempt += 1) {
      await page.waitForFunction(() => document.querySelector("#practice-go span").textContent === "Go", null, { timeout: 30_000 });
      await page.evaluate((value) => {
        const label = document.querySelector("#practice-go span");
        const observer = new MutationObserver(() => {
          if (label.textContent !== "Listening…") return;
          observer.disconnect();
          window.__edgevision.simulate(value);
        });
        observer.observe(label, { childList: true, characterData: true, subtree: true });
      }, result(top, top[0][1] > 0.49, "right"));
      await page.locator("#practice-go").click();
      await page.waitForFunction(() => document.getElementById("practice-result").dataset.outcome !== "none", null, { timeout: 30_000 });
      if ((await page.locator("#practice-result").getAttribute("data-outcome")) === outcome) break;
    }
    if (viewport.mobile) await ui(page, "expandSheet", true);
    await page.waitForTimeout(name.endsWith("correct") ? 380 : 500);
    await shot(page, name, combo);
    if (viewport.mobile) await ui(page, "expandSheet", false);
  }

  // The Signs library.
  await page.evaluate(() => window.__edgevision.stop());
  // The top bar's chip on desktop, its copy on the bottom sheet on phones.
  await page.locator("#vocab-chip:visible, #sheet-vocab-chip:visible").first().click().catch(async () => ui(page, "setMode", "signs"));
  await ui(page, "setMode", "signs");
  if (viewport.mobile) await ui(page, "expandSheet", true);
  await page.locator("#sign-search").fill("drink");
  await page.locator("#sign-search").fill("");
  await page.locator("#sign-search").blur();
  await page.waitForTimeout(500);
  await shot(page, "09-signs-library", combo);
  await browser.close();

  // Camera denied: a browser without camera permission.
  const denied = await chromium.launch({ channel: "chrome", args: ["--use-fake-device-for-media-stream"] });
  const deniedContext = await denied.newContext({ ...viewport.options, colorScheme: theme });
  const deniedPage = await deniedContext.newPage();
  await deniedPage.addInitScript(() => localStorage.setItem("edge-vision.ui.v1", JSON.stringify({ onboarded: true })));
  await deniedPage.goto(`${origin}/?debug=1`);
  await deniedPage.waitForFunction(() => window.__edgevision !== undefined);
  await deniedPage.evaluate(() => window.__edgevision.ready);
  await deniedPage.locator("#start").click();
  await deniedPage.waitForFunction(() => document.getElementById("state-eyebrow").textContent === "Camera blocked", null, { timeout: 30_000 });
  await deniedPage.waitForTimeout(400);
  await shot(deniedPage, "10-camera-denied", combo);
  await denied.close();
}

for (const viewport of VIEWPORTS) {
  for (const theme of THEMES) await run(viewport, theme);
}
console.log(`screenshots in ${out}`);
