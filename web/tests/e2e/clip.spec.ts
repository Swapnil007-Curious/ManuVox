import { existsSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { expect, test } from "@playwright/test";
import { skipOnboarding } from "./helpers";

// End to end on a recorded sign: a clip prepared by scripts/prepare_live_clips.py plays as the fake
// camera, and the app must classify at least one segment with no page or console error. The clips
// are personal recordings kept out of git, so the test is skipped where data/live_clips/ is absent.
const clip = fileURLToPath(new URL("../../../data/live_clips/hello.y4m", import.meta.url));

test.skip(!existsSync(clip), "data/live_clips/hello.y4m not present (scripts/prepare_live_clips.py)");
test.use({
  launchOptions: {
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream", `--use-file-for-fake-video-capture=${clip}`],
  },
  permissions: ["camera"],
});

test("a recorded clip runs end to end: segments are classified, no errors", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(`pageerror: ${error.message}`));
  page.on("console", (message) => {
    // MediaPipe prints its INFO lines through console.error.
    if (message.type() === "error" && !message.text().startsWith("INFO:")) errors.push(`console: ${message.text()}`);
  });
  await skipOnboarding(page);
  await page.goto("./?debug=1");
  await page.waitForFunction(() => window.__edgevision !== undefined);
  await page.evaluate(() => window.__edgevision?.ready);
  await page.waitForFunction(() => window.__edgevision?.status().tracker !== null, undefined, { timeout: 120_000 });
  await page.locator("#start").click();
  await expect(page.locator("#status")).toHaveText(/Running/, { timeout: 120_000 });
  await expect
    .poll(async () => page.evaluate(() => window.__edgevision?.segments().filter((s) => s.probabilities !== null).length ?? 0), {
      timeout: 60_000,
      intervals: [1000],
    })
    .toBeGreaterThan(0);
  await page.locator("#camera-toggle").click();
  await expect(page.locator("#status")).not.toHaveText(/failed|Could not/);
  expect(errors).toEqual([]);
});
