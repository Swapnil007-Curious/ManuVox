import { expect, test, type Page } from "@playwright/test";
import type { SegmentResult } from "../../src/engine";
import { openDebugPage, skipOnboarding } from "./helpers";

// Teach Dex in the browser: the panel, recording examples (each example is a segment result
// injected through the debug hook while the panel listens, so no signer is needed), IndexedDB
// persistence across reloads, the on/off switch, the badge, Reset, and the memory fallback.

test.use({
  permissions: ["camera"],
  launchOptions: { args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"] },
});

type Ui = {
  setMode(mode: string): void;
  teachSelect(label: string): void;
  teachRecord(): Promise<void>;
  teachPhase(): string;
  teachExamples(): { label: string; embedding: number[]; modelSha256: string }[];
  teachStoreMode(): string | null;
};

/** Call one method of the app's debug UI hook (main.ts, ?debug=1). */
function call<K extends keyof Ui>(page: Page, name: K, ...args: Parameters<Ui[K]>): Promise<Awaited<ReturnType<Ui[K]>>> {
  return page.evaluate(
    ([method, values]) => {
      const hook = (window as unknown as { __edgevisionUi: Record<string, (...a: unknown[]) => unknown> }).__edgevisionUi;
      return hook[method as string]!(...(values as unknown[]));
    },
    [name, args] as const,
  ) as Promise<Awaited<ReturnType<Ui[K]>>>;
}

/** A classified segment carrying an embedding of the loaded model's width. */
async function injectExample(page: Page, seed: number): Promise<void> {
  await page.evaluate((s) => {
    const hook = window.__edgevision!;
    const dim = hook.meta()?.embedding?.dim ?? 256;
    const embedding = Float32Array.from({ length: dim }, (_, i) => Math.sin(i * 0.37 + s));
    const result: SegmentResult = {
      kind: "classified",
      start: 0,
      end: 1000,
      frames: 30,
      handFrames: 28,
      reason: "still",
      dominant: "right",
      decision: { accepted: true, label: "dad", probability: 0.7, threshold: 0.49, top: [{ index: 0, label: "dad", probability: 0.7 }] },
      excluded: null,
      classifierMs: 10,
      embedding,
    };
    hook.simulate(result);
  }, seed);
}

async function teachOne(page: Page, n: number): Promise<void> {
  void call(page, "teachRecord");
  await page.waitForFunction(() => (window as unknown as { __edgevisionUi: Ui }).__edgevisionUi.teachPhase() === "listening", null, { timeout: 60_000 });
  await injectExample(page, n);
  await expect(page.locator("#teach-dots li[data-done='true']")).toHaveCount(n);
}

test.beforeEach(async ({ page }) => {
  await skipOnboarding(page, { autoTranslate: false, autoSpeak: false });
  await openDebugPage(page);
});

test("teaching a sign three times stores it on this device and survives a reload", async ({ page }) => {
  const meta = await page.evaluate(() => window.__edgevision!.meta());
  expect(meta?.embedding?.output_name).toBe("embedding");
  expect(meta?.personalization?.computed_on).toBe("val");
  await page.locator("#tab-teach").click();
  await expect(page.locator("#view-teach")).toBeVisible();
  await expect(page.locator(".teach-privacy")).toHaveText("Your examples stay on this device.");
  expect(await call(page, "teachStoreMode")).toBe("indexeddb");
  await expect(page.locator("#teach-storage")).toBeHidden();

  await page.locator("#teach-search").fill("mom");
  await page.locator("#teach-results button", { hasText: "MOM" }).first().click();
  await expect(page.locator("#teach-word")).toHaveText("MOM");
  await expect(page.locator("#teach-dots li")).toHaveCount(3);
  for (const n of [1, 2, 3]) await teachOne(page, n);
  await expect(page.locator("#teach-status")).toContainText("Taught!");
  await expect(page.locator("#teach-list li")).toHaveCount(1);
  await expect(page.locator("#teach-count")).toHaveText("1 sign taught");
  const stored = await call(page, "teachExamples");
  expect(stored).toHaveLength(3);
  expect(stored.every((e) => e.label === "mom" && e.modelSha256 === meta?.model_sha256)).toBe(true);
  const norm = Math.hypot(...(stored[0]?.embedding ?? []));
  expect(norm).toBeCloseTo(1, 5);

  await page.reload();
  await page.evaluate(() => window.__edgevision?.ready);
  await page.locator("#tab-teach").click();
  await expect(page.locator("#teach-list li")).toHaveCount(1);
  await expect(page.locator("#teach-list li")).toContainText("3 examples");

  // The switch is a stored setting.
  await page.locator("#teach-enabled").uncheck();
  expect(await page.evaluate(() => JSON.parse(localStorage.getItem("edge-vision.settings.v3") ?? "{}").personalization)).toBe(false);
  await page.locator("#teach-enabled").check();

  page.once("dialog", (dialog) => void dialog.accept());
  await page.locator("#teach-reset").click();
  await expect(page.locator("#teach-list li")).toHaveCount(0);
  await expect(page.locator("#teach-empty")).toBeVisible();
  await page.reload();
  await page.evaluate(() => window.__edgevision?.ready);
  await page.locator("#tab-teach").click();
  await expect(page.locator("#teach-list li")).toHaveCount(0);
});

test("a personalised prediction carries the badge in the transcript and the top 3", async ({ page }) => {
  await page.evaluate(() => {
    const top = [
      { index: 1, label: "mom", probability: 0.82 },
      { index: 2, label: "dad", probability: 0.1 },
    ];
    window.__edgevision!.simulate({
      kind: "classified",
      start: 0,
      end: 1200,
      frames: 30,
      handFrames: 28,
      reason: "still",
      dominant: "right",
      decision: { accepted: true, label: "mom", probability: 0.82, threshold: 0.49, top, personalized: true },
      excluded: null,
      classifierMs: 12,
    });
  });
  const word = page.locator("#transcript li").first();
  await expect(word).toHaveAttribute("data-personalized", "true");
  await expect(word.locator(".for-you")).toHaveAttribute("title", "Personalized for you");
  await expect(page.locator("#personal-badge")).toBeVisible();
  await expect(page.locator("#personal-badge")).toHaveText("Personalized for you");
  // Replacing the word with another candidate drops the badge.
  await page.keyboard.press("2");
  await expect(page.locator("#transcript li").first()).toHaveAttribute("data-personalized", "false");
  await expect(page.locator("#personal-badge")).toBeHidden();
});

test.describe("with IndexedDB blocked", () => {
  test.beforeEach(async ({ page }) => {
    await page.addInitScript(() => {
      Object.defineProperty(window, "indexedDB", {
        configurable: true,
        get() {
          throw new DOMException("blocked by the test", "SecurityError");
        },
      });
    });
    await page.reload();
    await page.evaluate(() => window.__edgevision?.ready);
  });

  test("examples are kept for the session and the panel says so", async ({ page }) => {
    await page.locator("#tab-teach").click();
    expect(await call(page, "teachStoreMode")).toBe("memory");
    await expect(page.locator("#teach-storage")).toBeVisible();
    await expect(page.locator("#teach-storage")).toContainText("until you close this tab");
    await call(page, "teachSelect", "water");
    await teachOne(page, 1);
    await expect(page.locator("#teach-list li")).toHaveCount(1);
  });
});
