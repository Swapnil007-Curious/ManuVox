// The 250-sign vocabulary (src/data/signs.json): glosses for the transcript, plain words for the
// offline sentence joiner, the external dictionary link and a short description. Pure.

import data from "../data/signs.json";

export interface SignInfo {
  label: string;
  gloss: string;
  words: string;
  slug: string;
  description: string;
}

export const SIGNS: readonly SignInfo[] = data.signs;
export const DICTIONARY_BASE: string = data.dictionary;
const BY_LABEL = new Map(SIGNS.map((sign) => [sign.label, sign]));

export function signInfo(label: string): SignInfo {
  return BY_LABEL.get(label) ?? { label, gloss: label.toUpperCase(), words: label.toLowerCase(), slug: label.toLowerCase(), description: "" };
}

/** The gloss shown in the transcript: HELLO, THANK-YOU, HE/SHE/IT. */
export function glossOf(label: string): string {
  return signInfo(label).gloss;
}

export function wordsOf(label: string): string {
  return signInfo(label).words;
}

/** The external dictionary page with videos of the sign (no video is copied into the app). */
export function dictionaryUrl(label: string): string {
  return `${DICTIONARY_BASE}${encodeURIComponent(signInfo(label).slug)}`;
}

/** Alphabetical by gloss. The data is in the model's label order, where "TV" precedes "after". */
const byGloss = (a: SignInfo, b: SignInfo): number => a.gloss.localeCompare(b.gloss, "en");

/**
 * Signs whose gloss, words or description contain every word of the query (case-insensitive);
 * an empty query lists every sign alphabetically.
 */
export function searchSigns(query: string, signs: readonly SignInfo[] = SIGNS): SignInfo[] {
  const terms = query.toLowerCase().split(/\s+/).filter(Boolean);
  if (terms.length === 0) return [...signs].sort(byGloss);
  const scored = signs
    .map((sign) => {
      const name = `${sign.gloss} ${sign.words} ${sign.label}`.toLowerCase();
      const text = `${name} ${sign.description}`.toLowerCase();
      if (!terms.every((term) => text.includes(term))) return null;
      // Name matches first, then prefix matches, then description-only matches.
      const rank = terms.every((term) => name.includes(term)) ? (name.startsWith(terms[0] as string) ? 0 : 1) : 2;
      return { sign, rank };
    })
    .filter((item): item is { sign: SignInfo; rank: number } => item !== null);
  scored.sort((a, b) => a.rank - b.rank || byGloss(a.sign, b.sign));
  return scored.map((item) => item.sign);
}
