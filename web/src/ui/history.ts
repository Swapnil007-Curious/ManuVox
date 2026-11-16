// The sentences translated in this browser, newest first (localStorage when allowed).

import { readJson, writeJson, type KeyValueStore } from "./store";

export interface HistoryItem {
  text: string;
  source: "worker" | "basic";
  glosses: string[];
  at: string;
}

export const HISTORY_KEY = "edge-vision.history.v1";
export const HISTORY_LIMIT = 30;

function valid(item: unknown): item is HistoryItem {
  if (item === null || typeof item !== "object") return false;
  const value = item as Record<string, unknown>;
  return (
    typeof value.text === "string" &&
    (value.source === "worker" || value.source === "basic") &&
    Array.isArray(value.glosses) &&
    value.glosses.every((gloss) => typeof gloss === "string") &&
    typeof value.at === "string"
  );
}

export function loadHistory(store: KeyValueStore | null): HistoryItem[] {
  const raw = readJson<unknown>(store, HISTORY_KEY, []);
  return Array.isArray(raw) ? raw.filter(valid).slice(0, HISTORY_LIMIT) : [];
}

/** The list with `item` first; an identical text at the top is not repeated. */
export function pushHistory(items: readonly HistoryItem[], item: HistoryItem): HistoryItem[] {
  if (items[0]?.text === item.text) return [...items];
  return [item, ...items].slice(0, HISTORY_LIMIT);
}

export function saveHistory(store: KeyValueStore | null, items: readonly HistoryItem[]): boolean {
  return writeJson(store, HISTORY_KEY, items);
}
