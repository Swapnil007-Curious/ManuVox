import { readFileSync } from "node:fs";
import type { Page } from "@playwright/test";
import type { DebugPreprocessResult } from "../../src/debug";
import type { LongRows } from "../../src/preprocess";

export function readJson<T>(relativeToWeb: string): T {
  return JSON.parse(readFileSync(new URL(`../../${relativeToWeb}`, import.meta.url), "utf8")) as T;
}

export interface ModelCase {
  sequence_id: number;
  label: number;
  input: number[][];
  logits: number[];
}

export interface ModelGolden {
  source: string;
  variant: string;
  model_sha256: string;
  cases: ModelCase[];
}

export async function openDebugPage(page: Page): Promise<void> {
  await page.goto("./?debug=1");
  await page.waitForFunction(() => window.__edgevision !== undefined);
  await page.evaluate(() => window.__edgevision?.ready);
}

export function classify(page: Page, input: number[]): Promise<number[]> {
  return page.evaluate((values) => (window.__edgevision as NonNullable<typeof window.__edgevision>).classify(values), input);
}

export function preprocess(page: Page, rows: LongRows): Promise<DebugPreprocessResult> {
  return page.evaluate((value) => (window.__edgevision as NonNullable<typeof window.__edgevision>).preprocess(value), rows);
}

export function argmax(values: readonly number[]): number {
  return values.indexOf(Math.max(...values));
}

export function maxAbsDiff(a: readonly number[], b: readonly number[]): number {
  return a.reduce((max, value, i) => Math.max(max, Math.abs(value - (b[i] as number))), 0);
}

/** Mark the first-run overlay as done, so tests reach the app directly. */
export async function skipOnboarding(page: Page, extra: Record<string, unknown> = {}): Promise<void> {
  await page.addInitScript((prefs) => {
    try {
      // Merged with what is stored, so a reload keeps preferences the test changed.
      const stored = JSON.parse(localStorage.getItem("edge-vision.ui.v1") ?? "{}") as Record<string, unknown>;
      localStorage.setItem("edge-vision.ui.v1", JSON.stringify({ ...prefs, ...stored, onboarded: true }));
    } catch {
      // Storage blocked: the overlay can still be skipped by the test.
    }
  }, extra);
}
