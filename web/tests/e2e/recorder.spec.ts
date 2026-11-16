import { readFileSync } from "node:fs";
import { expect, test } from "@playwright/test";
import { validateRecordingsFile, type RecordingsFile } from "../../src/recorder";

test.use({
  launchOptions: {
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
  },
  permissions: ["camera"],
});

// The fake camera shows a test pattern without a person, so the calibrated aspect mode (chosen
// here; the default takes k from the frame size) finds no shoulders and a held segment has no hand
// frames: it is discarded by the segmenter and recorded as such.
test("record test signs: one segment from the fake camera, downloaded and schema-valid", async ({ page }) => {
  await page.goto("./");
  await expect(page.locator("#start")).toBeEnabled();

  await page.locator("#recorder-details > summary").click();
  await page.locator("#recorder-on").check();
  await page.locator("#label-search").fill("bir");
  await page.locator("#label-search").press("ArrowDown");
  await expect(page.locator("#label-select")).toBeFocused();
  await page.locator("#label-select").selectOption({ label: "bird" });
  await expect(page.locator("#recorder-status")).toContainText("bird");

  await page.locator('input[name="mode"][value="manual"]').check();
  await page.locator("#settings-details > summary").click();
  await page.locator('input[name="aspectMode"][value="auto"]').check();
  await page.locator("#start").click();
  await expect(page.locator("#status")).toHaveText(/Running/, { timeout: 120_000 });
  await expect(page.locator("#calibration-status")).toContainText("Move back so your shoulders are visible", { timeout: 60_000 });
  await expect(page.locator("#diag-aspect")).toContainText("k 1.000");

  const record = page.locator("#record");
  await record.focus();
  await page.keyboard.down("Enter");
  await page.waitForTimeout(1500);
  await page.keyboard.up("Enter");
  await expect(page.locator("#recorder-status")).toContainText("1 recording", { timeout: 30_000 });
  await expect(page.locator("#segment-debug-text")).toContainText("discarded");

  const downloadPromise = page.waitForEvent("download");
  await page.locator("#recorder-download").click();
  const download = await downloadPromise;
  expect(download.suggestedFilename()).toMatch(/^edge-vision-recordings-\d{8}-\d{6}\.json$/);
  const path = await download.path();
  await download.saveAs(test.info().outputPath("recordings.json"));
  const file = JSON.parse(readFileSync(path, "utf8")) as RecordingsFile;
  expect(validateRecordingsFile(file)).toEqual([]);
  expect(file.recordings).toHaveLength(1);
  const recording = file.recordings[0] as RecordingsFile["recordings"][number];
  expect(recording.intended_label).toBe("bird");
  expect(recording.app_result.status).toBe("discarded");
  expect(recording.calibration).toMatchObject({ mode: "auto", k: 1, status: "no_shoulders" });
  expect(recording.frames.t_ms.length).toBeGreaterThan(0);
  expect(recording.frames.pose_ran.some(Boolean)).toBe(true);
  expect(recording.segment.capture_mode).toBe("manual");
  console.log(
    `recorder: ${recording.frames.t_ms.length} frames, pose ran on ${recording.frames.pose_ran.filter(Boolean).length}, ` +
      `camera ${recording.camera.width}x${recording.camera.height}, landmarks: ${await page.locator("#diag-mediapipe").textContent()}`,
  );
  await page.locator("#stop").click();
});
