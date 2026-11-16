// Preferences and sentence history with working, blocked and corrupt storage; the WCAG contrast
// of the palette; the Worker's vocabulary staying in sync with the app's signs and model labels.

import { readFileSync } from "node:fs";
import { describe, expect, it } from "vitest";
import { auditStylesheet, parseColor, ratio } from "../../src/ui/contrast";
import { HISTORY_KEY, HISTORY_LIMIT, loadHistory, pushHistory, saveHistory, type HistoryItem } from "../../src/ui/history";
import { DEFAULT_PREFS, PREFS_KEY, loadPrefs, savePrefs, sanitizePrefs } from "../../src/ui/prefs";
import { SIGNS } from "../../src/ui/signs";

class MemoryStore {
  data = new Map<string, string>();
  getItem(key: string): string | null {
    return this.data.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.data.set(key, value);
  }
}

const blocked = {
  getItem(): string | null {
    throw new DOMException("blocked", "SecurityError");
  },
  setItem(): void {
    throw new DOMException("quota", "QuotaExceededError");
  },
};

const read = (path: string): string => readFileSync(new URL(`../../${path}`, import.meta.url), "utf8");

describe("prefs", () => {
  it("defaults to auto-speak and auto-translate on, HUD on, not onboarded", () => {
    expect(DEFAULT_PREFS).toMatchObject({ autoSpeak: true, autoTranslate: true, hud: true, onboarded: false, theme: "system" });
  });

  it("round-trips and clamps", () => {
    const store = new MemoryStore();
    expect(savePrefs(store, { ...DEFAULT_PREFS, theme: "light", rate: 1.2, onboarded: true })).toBe(true);
    expect(loadPrefs(store)).toMatchObject({ theme: "light", rate: 1.2, onboarded: true });
    expect(sanitizePrefs({ rate: 9, theme: "neon", voice: 3, hud: "yes" })).toEqual({ ...DEFAULT_PREFS, rate: 1.6 });
    store.setItem(PREFS_KEY, "nope");
    expect(loadPrefs(store)).toEqual(DEFAULT_PREFS);
  });

  it("works with blocked or missing storage", () => {
    expect(loadPrefs(blocked)).toEqual(DEFAULT_PREFS);
    expect(savePrefs(blocked, DEFAULT_PREFS)).toBe(false);
    expect(loadPrefs(null)).toEqual(DEFAULT_PREFS);
  });
});

describe("history", () => {
  const item = (text: string): HistoryItem => ({ text, source: "basic", glosses: ["HELLO"], at: "2026-11-16T10:00:00.000Z" });

  it("keeps the newest first, skips an immediate repeat and caps the length", () => {
    let items: HistoryItem[] = [];
    items = pushHistory(items, item("Hello."));
    items = pushHistory(items, item("Hello."));
    items = pushHistory(items, item("Hello, mom."));
    expect(items.map((i) => i.text)).toEqual(["Hello, mom.", "Hello."]);
    for (let i = 0; i < 40; i += 1) items = pushHistory(items, item(`s${i}`));
    expect(items).toHaveLength(HISTORY_LIMIT);
    expect(items[0]?.text).toBe("s39");
  });

  it("round-trips, drops malformed entries and survives blocked storage", () => {
    const store = new MemoryStore();
    expect(saveHistory(store, [item("A."), item("B.")])).toBe(true);
    expect(loadHistory(store).map((i) => i.text)).toEqual(["A.", "B."]);
    store.setItem(HISTORY_KEY, JSON.stringify([item("C."), { text: 3 }, null, { ...item("D."), source: "magic" }]));
    expect(loadHistory(store).map((i) => i.text)).toEqual(["C."]);
    expect(loadHistory(blocked)).toEqual([]);
    expect(saveHistory(blocked, [item("E.")])).toBe(false);
  });
});

describe("contrast", () => {
  it("computes WCAG ratios", () => {
    expect(ratio(parseColor("#000"), parseColor("#ffffff"))).toBeCloseTo(21, 5);
    expect(ratio(parseColor("#777777"), parseColor("#777777"))).toBeCloseTo(1, 5);
    expect(parseColor("rgba(22, 25, 31, 0.72)")).toEqual({ r: 22, g: 25, b: 31, a: 0.72 });
  });

  it("every text pair of both themes passes AA, glass over the video included", () => {
    const { results, problems } = auditStylesheet(read("src/styles.css"));
    expect(problems).toEqual([]);
    expect(results.length).toBeGreaterThan(40);
  });
});

describe("vocabulary", () => {
  it("lists the model's labels in order", () => {
    const meta = JSON.parse(read("public/model/meta.json")) as { labels: string[] };
    expect(SIGNS.map((sign) => sign.label)).toEqual(meta.labels);
  });

  it("matches the glosses the translation Worker accepts (node web/scripts/export-vocabulary.mjs)", () => {
    const worker = JSON.parse(read("../deploy/worker/src/vocabulary.json")) as { glosses: string[] };
    expect(worker.glosses).toEqual(SIGNS.map((sign) => sign.gloss));
  });
});
