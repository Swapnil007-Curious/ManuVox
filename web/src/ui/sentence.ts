// The offline sentence: a rule-based joiner used when the translation Worker is unavailable.
// It does not reorder or add words; it lower-cases, splits compound glosses (THANKYOU -> "thank
// you"), capitalises the first word and adds basic punctuation. The app labels it "(basic)". Pure.

import { wordsOf } from "./signs";

const GREETINGS = new Set(["hello", "bye", "yes", "no"]);
const QUESTION_WORDS = new Set(["who", "where", "why"]);
// Words that keep their capitals.
const KEEP_CASE = new Set(["TV"]);

export function basicSentence(labels: readonly string[]): string {
  const words = labels.map((label) => wordsOf(label).trim()).filter(Boolean);
  if (words.length === 0) return "";
  const parts = words.map((word) => (KEEP_CASE.has(word) ? word : word.toLowerCase()));
  const first = parts[0] as string;
  const lead = GREETINGS.has(first);
  if (lead && parts.length > 1) parts[0] = `${first},`;
  const text = parts.join(" ");
  const capitalised = KEEP_CASE.has(words[0] as string) ? text : text.charAt(0).toUpperCase() + text.slice(1);
  const question = parts.some((word) => QUESTION_WORDS.has(word.replace(/,$/, "")));
  const end = question ? "?" : lead && parts.length === 1 && (first === "hello" || first === "bye") ? "!" : ".";
  return `${capitalised}${end}`;
}
