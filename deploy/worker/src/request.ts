// Request validation and the prompt. The browser sends only recognised glosses, and every gloss
// must be one of the app's 250 (vocabulary.json, generated from web/src/data/signs.json), so no
// free text from a client can reach the model. The system prompt is fixed here.

import vocabulary from "./vocabulary.json" with { type: "json" };

export const MAX_GLOSSES = 24;
export const MAX_SENTENCE_CHARS = 300;
export const SYSTEM_PROMPT =
  "Turn this ASL gloss sequence into one short, natural English sentence; do not add facts that the glosses don't support; return only the sentence.";

const ALLOWED: ReadonlySet<string> = new Set(vocabulary.glosses);

export class ValidationError extends Error {}

export interface TranslateRequest {
  glosses: string[];
}

export function validate(body: unknown): TranslateRequest {
  if (!body || typeof body !== "object" || Array.isArray(body)) throw new ValidationError("the body must be a JSON object");
  const glosses = (body as Record<string, unknown>).glosses;
  if (!Array.isArray(glosses)) throw new ValidationError("glosses must be a list");
  if (glosses.length === 0) throw new ValidationError("glosses is empty");
  if (glosses.length > MAX_GLOSSES) throw new ValidationError(`at most ${MAX_GLOSSES} glosses per request`);
  for (const gloss of glosses) {
    if (typeof gloss !== "string" || !ALLOWED.has(gloss)) {
      throw new ValidationError(`unknown gloss: ${typeof gloss === "string" ? gloss.slice(0, 40) : typeof gloss}`);
    }
  }
  return { glosses: glosses as string[] };
}

export function buildMessages(request: TranslateRequest): { role: "system" | "user"; content: string }[] {
  return [
    { role: "system", content: SYSTEM_PROMPT },
    { role: "user", content: request.glosses.join(" ") },
  ];
}

/** The first non-empty line of the reply, without wrapping quotes, whitespace collapsed, capped. */
export function cleanSentence(text: string): string {
  const line = text
    .split(/\r?\n/)
    .map((part) => part.trim())
    .find(Boolean);
  if (!line) return "";
  const unquoted = line.replace(/^["'“”‘’]+|["'“”‘’]+$/g, "").replace(/\s+/g, " ").trim();
  return unquoted.length > MAX_SENTENCE_CHARS ? `${unquoted.slice(0, MAX_SENTENCE_CHARS - 1).trimEnd()}…` : unquoted;
}
