// Render the social preview card (Open Graph / Twitter, 1200x630) into public/og-image.png:
// the Duet mark, the ManuVox wordmark in the app's display face and the tagline, on the app's
// graphite background. Self-hosted fonts from @fontsource are inlined, so nothing is fetched.
// Run with `npm run og-image` (needs `npx playwright install chromium` once).

import { chromium } from "@playwright/test";
import { readFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const web = join(dirname(fileURLToPath(import.meta.url)), "..");
const font = (pkg, file) =>
  `data:font/woff2;base64,${readFileSync(join(web, "node_modules", "@fontsource-variable", pkg, "files", file)).toString("base64")}`;
const display = font("bricolage-grotesque", "bricolage-grotesque-latin-wght-normal.woff2");
const ui = font("manrope", "manrope-latin-wght-normal.woff2");

const html = `<!doctype html><html><head><meta charset="utf-8"><style>
@font-face { font-family: "Display"; src: url(${display}) format("woff2"); font-weight: 200 800; }
@font-face { font-family: "UI"; src: url(${ui}) format("woff2"); font-weight: 200 800; }
html, body { margin: 0; width: 1200px; height: 630px; background: #0F1115; }
body { display: grid; place-items: center; font-family: "UI", sans-serif; color: #F2F4F8; }
.card { display: grid; justify-items: center; gap: 26px; text-align: center; }
svg { width: 220px; height: 151px; }
h1 { margin: 0; font: 780 128px/1 "Display", sans-serif; letter-spacing: -0.035em; }
.tagline { margin: 0; font: 650 46px/1.2 "Display", sans-serif; color: #F5D76E; letter-spacing: -0.01em; }
.sub { margin: 6px 0 0; font: 600 26px/1.35 "UI", sans-serif; color: #A9B1C2; }
</style></head><body><div class="card">
<svg viewBox="0 0 64 44" aria-hidden="true">
  <defs><clipPath id="l"><circle cx="21" cy="22" r="20" /></clipPath></defs>
  <circle cx="21" cy="22" r="20" fill="#7AA2FF" />
  <circle cx="43" cy="22" r="20" fill="#FF9E7A" />
  <circle cx="43" cy="22" r="20" fill="#F5D76E" clip-path="url(#l)" />
</svg>
<h1>ManuVox</h1>
<p class="tagline">Your hands, spoken aloud.</p>
<p class="sub">250 ASL signs, recognised in the browser, on your device</p>
</div></body></html>`;

const browser = await chromium.launch();
try {
  const page = await browser.newPage({ viewport: { width: 1200, height: 630 }, deviceScaleFactor: 1 });
  await page.setContent(html);
  await page.evaluate(() => document.fonts.ready);
  const out = join(web, "public", "og-image.png");
  await page.screenshot({ path: out, type: "png" });
  console.log(`wrote ${out}`);
} finally {
  await browser.close();
}
