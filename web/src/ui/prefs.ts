// Interface preferences (speech, translation, appearance, first-run state), separate from the
// pipeline settings in src/settings.ts. Stored in localStorage when the browser allows it.

import { readJson, writeJson, type KeyValueStore } from "./store";

export type Theme = "system" | "dark" | "light";

export interface Prefs {
  theme: Theme;
  /** speechSynthesis voiceURI, or null for the first on-device English voice. */
  voice: string | null;
  rate: number;
  autoSpeak: boolean;
  autoTranslate: boolean;
  /** Draw the hand skeleton and trails over the video. */
  hud: boolean;
  onboarded: boolean;
}

export const PREFS_KEY = "edge-vision.ui.v1";
export const AUTO_TRANSLATE_MS = 2000;
export const RATE_MIN = 0.6;
export const RATE_MAX = 1.6;

export const DEFAULT_PREFS: Prefs = {
  theme: "system",
  voice: null,
  rate: 1,
  autoSpeak: true,
  autoTranslate: true,
  hud: true,
  onboarded: false,
};

export function sanitizePrefs(raw: unknown): Prefs {
  const source = (raw !== null && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const bool = (key: keyof Prefs): boolean => (typeof source[key] === "boolean" ? (source[key] as boolean) : (DEFAULT_PREFS[key] as boolean));
  const rate = typeof source.rate === "number" && Number.isFinite(source.rate) ? Math.min(RATE_MAX, Math.max(RATE_MIN, source.rate)) : DEFAULT_PREFS.rate;
  return {
    theme: source.theme === "dark" || source.theme === "light" || source.theme === "system" ? source.theme : DEFAULT_PREFS.theme,
    voice: typeof source.voice === "string" && source.voice ? source.voice : null,
    rate,
    autoSpeak: bool("autoSpeak"),
    autoTranslate: bool("autoTranslate"),
    hud: bool("hud"),
    onboarded: bool("onboarded"),
  };
}

export function loadPrefs(store: KeyValueStore | null): Prefs {
  return sanitizePrefs(readJson<unknown>(store, PREFS_KEY, null));
}

export function savePrefs(store: KeyValueStore | null, prefs: Prefs): boolean {
  return writeJson(store, PREFS_KEY, prefs);
}
