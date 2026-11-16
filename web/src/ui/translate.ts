// Glosses -> one natural English sentence through the translation Worker (deploy/worker), or the
// offline rule-based joiner when the Worker is not configured, unreachable, slow or refuses.
// Only the recognised glosses are sent; no video or landmarks ever leave the device.

import { basicSentence } from "./sentence";
import { glossOf } from "./signs";

export interface Translation {
  text: string;
  source: "worker" | "basic";
  /** Why the basic joiner was used (shown in the diagnostics), or null. */
  reason: string | null;
}

export interface TranslateOptions {
  workerUrl: string;
  fetchFn?: typeof fetch;
  timeoutMs?: number;
  online?: boolean;
  signal?: AbortSignal;
}

export const TRANSLATE_TIMEOUT_MS = 6000;
export const MAX_GLOSSES = 24;

function basic(labels: readonly string[], reason: string): Translation {
  return { text: basicSentence(labels), source: "basic", reason };
}

export async function translate(labels: readonly string[], options: TranslateOptions): Promise<Translation> {
  const words = labels.slice(-MAX_GLOSSES);
  if (words.length === 0) return { text: "", source: "basic", reason: null };
  if (!options.workerUrl) return basic(words, "no translation service configured");
  if (options.online === false) return basic(words, "offline");
  const fetchFn = options.fetchFn ?? fetch;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), options.timeoutMs ?? TRANSLATE_TIMEOUT_MS);
  const abort = (): void => controller.abort();
  options.signal?.addEventListener("abort", abort);
  try {
    const response = await fetchFn(`${options.workerUrl.replace(/\/+$/, "")}/translate`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ glosses: words.map(glossOf) }),
      signal: controller.signal,
    });
    const body = (await response.json().catch(() => ({}))) as { sentence?: unknown; message?: unknown };
    if (!response.ok) {
      const why = typeof body.message === "string" && body.message ? body.message : `HTTP ${response.status}`;
      return basic(words, why);
    }
    const text = typeof body.sentence === "string" ? body.sentence.trim() : "";
    return text ? { text, source: "worker", reason: null } : basic(words, "empty reply");
  } catch (error) {
    if (options.signal?.aborted) throw error;
    return basic(words, controller.signal.aborted ? "the translation service did not answer in time" : "the translation service is unreachable");
  } finally {
    clearTimeout(timer);
    options.signal?.removeEventListener("abort", abort);
  }
}
