// JSON values in localStorage, never throwing: blocked or full storage leaves the app working
// with in-memory state (reads return the fallback, writes report false).

import type { KeyValueStore } from "../settings";

export type { KeyValueStore };

export function readJson<T>(store: KeyValueStore | null, key: string, fallback: T): T {
  try {
    const text = store?.getItem(key);
    return text ? (JSON.parse(text) as T) : fallback;
  } catch {
    return fallback;
  }
}

export function writeJson(store: KeyValueStore | null, key: string, value: unknown): boolean {
  try {
    if (!store) return false;
    store.setItem(key, JSON.stringify(value));
    return true;
  } catch {
    return false;
  }
}

export function removeKey(store: (KeyValueStore & { removeItem?: (key: string) => void }) | null, key: string): boolean {
  try {
    if (!store) return false;
    if (store.removeItem) store.removeItem(key);
    else store.setItem(key, "");
    return true;
  } catch {
    return false;
  }
}
