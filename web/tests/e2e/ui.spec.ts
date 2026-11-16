import { expect, test, type Page } from "@playwright/test";
import type { SegmentResult } from "../../src/engine";
import { openDebugPage, skipOnboarding } from "./helpers";

// The interface around recognition, driven by results injected through the debug hook (the same
// event the engine emits after classifying a segment), so it runs without a camera or a signer.

function result(top: [string, number][], accepted: boolean, dominant: "left" | "right" = "right"): SegmentResult {
  return {
    kind: "classified",
    start: 1000,
    end: 2200,
    frames: 30,
    handFrames: 28,
    reason: "still",
    dominant,
    decision: { accepted, label: accepted ? top[0]![0] : null, probability: top[0]![1], threshold: 0.49, top: top.map(([label, probability], index) => ({ index, label, probability })) },
    excluded: null,
    classifierMs: 20,
  };
}

async function simulate(page: Page, value: SegmentResult): Promise<void> {
  await page.evaluate((r) => window.__edgevision?.simulate(r), value);
}

test.beforeEach(async ({ page }) => {
  await skipOnboarding(page, { autoTranslate: false, autoSpeak: false });
  await openDebugPage(page);
});

test("words land in the transcript; 1/2/3 replace or pick; Backspace undoes; Enter translates (basic)", async ({ page }) => {
  await simulate(page, result([["hello", 0.92], ["uncle", 0.03], ["sun", 0.01]], true));
  await expect(page.locator("#transcript li")).toHaveText(["HELLO"]);
  await expect(page.locator("#status-pill")).toHaveAttribute("data-state", "word");
  await expect(page.locator("#lane-right .seg")).toHaveCount(1);

  await simulate(page, result([["mom", 0.81], ["dad", 0.1], ["aunt", 0.03]], true));
  await expect(page.locator("#transcript li")).toHaveText(["HELLO", "MOM"]);
  await page.keyboard.press("2");
  await expect(page.locator("#transcript li")).toHaveText(["HELLO", "DAD"]);

  // Low confidence: nothing is added until a candidate is picked.
  await simulate(page, result([["drink", 0.31], ["water", 0.22], ["milk", 0.1]], false, "left"));
  await expect(page.locator("#unsure")).toBeVisible();
  await expect(page.locator("#unsure-choices .choice-label")).toHaveText(["DRINK", "WATER", "MILK"]);
  await expect(page.locator("#lane-left .seg")).toHaveAttribute("data-kind", "unsure");
  await page.locator("#unsure-choices .choice").nth(0).click();
  await expect(page.locator("#unsure")).toBeHidden();
  await expect(page.locator("#transcript li")).toHaveText(["HELLO", "DAD", "DRINK"]);

  await page.keyboard.press("Backspace");
  await expect(page.locator("#transcript li")).toHaveText(["HELLO", "DAD"]);

  await page.locator("body").click({ position: { x: 5, y: 5 } });
  await page.keyboard.press("Enter");
  await expect(page.locator("#sentence")).toHaveText("Hello, dad.");
  await expect(page.locator("#sentence-source")).toHaveText("(basic)");
  await expect(page.locator("#history li")).toHaveCount(1);
});

test("practice shows the target sign with its description and dictionary link", async ({ page }) => {
  await page.evaluate(() => (window as unknown as { __edgevisionUi: { setMode(m: string): void; practiceTarget(l: string): void } }).__edgevisionUi.setMode("practice"));
  await page.evaluate(() => (window as unknown as { __edgevisionUi: { practiceTarget(l: string): void } }).__edgevisionUi.practiceTarget("water"));
  await expect(page.locator("#target-word")).toHaveText("WATER");
  // Without a camera, Go explains that it needs one; the scoring itself is covered with the camera
  // in the screenshot run and by the unit tests.
  await expect(page.locator("#target-desc")).toContainText("taps the chin");
  await expect(page.locator("#target-link")).toHaveAttribute("href", "https://www.signasl.org/sign/water");
});

test("the Signs library lists and searches all 250 signs", async ({ page }) => {
  await page.locator("#vocab-chip").click();
  await expect(page.locator("#view-signs")).toBeVisible();
  await expect(page.locator("#sign-list .sign-item")).toHaveCount(250);
  await expect(page.locator("#sign-list .sign-name").first()).toHaveText("AFTER"); // alphabetical
  await page.locator("#sign-search").fill("thank");
  await expect(page.locator("#sign-list .sign-name").first()).toHaveText("THANK-YOU");
  await page.locator("#sign-list .sign-item").first().getByRole("button", { name: /Practise/ }).click();
  await expect(page.locator("#view-practice")).toBeVisible();
  await expect(page.locator("#target-word")).toHaveText("THANK-YOU");
});

test("on a phone the sheet keeps the vocabulary chip and About in reach", async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 664 });
  await expect(page.locator("#vocab-chip")).toBeHidden();
  await expect(page.locator("#open-about")).toBeHidden();
  await page.locator("#sheet-about").click();
  await expect(page.locator("#about-dialog")).toBeVisible();
  await expect(page.locator("#about-dialog")).toContainText("Not a full ASL translator.");
  await page.locator("#about-close").click();
  await page.locator("#sheet-vocab-chip").click();
  await expect(page.locator("#view-signs")).toBeVisible();
  await expect(page.locator("#sheet-handle")).toHaveAttribute("aria-expanded", "true");
});

test("mode tabs follow the keyboard and settings persist", async ({ page }) => {
  await page.locator("#tab-translate").focus();
  await page.keyboard.press("ArrowRight");
  await expect(page.locator("#tab-practice")).toHaveAttribute("aria-selected", "true");
  await page.keyboard.press("End");
  await expect(page.locator("#tab-signs")).toHaveAttribute("aria-selected", "true");
  await page.locator("#open-settings").click();
  await page.locator('input[name="pref-theme"][value="light"]').check({ force: true });
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
  await page.keyboard.press("Escape");
  await page.reload();
  await expect(page.locator("html")).toHaveAttribute("data-theme", "light");
});
