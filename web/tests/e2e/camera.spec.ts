import { expect, test } from "@playwright/test";
import { skipOnboarding } from "./helpers";

test.use({
  launchOptions: {
    args: ["--use-fake-ui-for-media-stream", "--use-fake-device-for-media-stream"],
  },
  permissions: ["camera"],
});

test("camera smoke: the pipeline starts, the stage shows the video and the fps chip updates", async ({ page, baseURL }) => {
  const origin = new URL(baseURL as string).origin;
  const foreign: string[] = [];
  const cspViolations: string[] = [];
  page.on("request", (request) => {
    const url = request.url();
    if (!url.startsWith(origin) && !url.startsWith("data:") && !url.startsWith("blob:")) foreign.push(url);
  });
  page.on("console", (message) => {
    if (message.text().includes("Content Security Policy")) cspViolations.push(message.text());
  });
  await skipOnboarding(page);
  await page.goto("./");
  const start = page.locator("#start");
  await expect(start).toBeEnabled({ timeout: 120_000 });
  await start.click();
  await expect(page.locator("#status")).toHaveText(/Running/, { timeout: 120_000 });
  await expect(page.locator("#stage")).toHaveAttribute("data-running", "true");
  await expect(page.locator("#video")).toBeVisible();
  await expect(page.locator("#status-pill")).toHaveAttribute("data-state", /idle|signing|reading/);
  await expect
    .poll(async () => Number.parseFloat((await page.locator("#diag-fps").textContent()) ?? ""), { timeout: 60_000 })
    .toBeGreaterThan(0);
  await page.locator("#diag-toggle").click();
  await expect(page.locator("#diagnostics")).toBeVisible();
  await expect(page.locator("#fps")).toHaveText(/fps/);
  console.log(`camera smoke: ${await page.locator("#fps").textContent()}; ${await page.locator("#diag-backend").textContent()}`);
  await page.locator("#diag-close").click();
  await page.locator("#camera-toggle").click();
  await expect(page.locator("#stage")).toHaveAttribute("data-running", "false");
  await expect(page.locator("#status")).not.toHaveText(/failed|Could not/);
  expect(foreign, "requests to other origins").toEqual([]);
  expect(cspViolations, "Content-Security-Policy violations").toEqual([]);
});
