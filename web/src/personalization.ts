// Teach Dex: combine the model's logits with a cosine nearest-neighbour score over the user's own
// examples. Pure; the examples live in src/teachStore.ts and never leave the device.
//
// For each taught sign c, s_c is the highest cosine similarity between the segment's embedding
// (the model's penultimate layer, meta.json "embedding") and the user's examples of c. The
// decision then uses
//
//   z'_c = z_c / T + weight * (s_c - offset)   for taught signs,   z'_c = z_c / T   otherwise,
//
// where T is the calibrated temperature and weight and offset were fitted on the validation split
// by simulating three examples per sign (scripts/calibrate_teach.py, meta.json "personalization").
// Signs nobody taught are scored exactly as before, and the "not sure" threshold is unchanged.

import { decide, softmax, type Decision } from "./scoring";

export interface TaughtExample {
  id: string;
  label: string;
  /** L2-normalised embedding. */
  embedding: Float32Array;
  createdAt: number;
  /** The model the embedding came from; examples of another model are not comparable. */
  modelSha256: string;
}

export interface PersonalizationParams {
  weight: number;
  offset: number;
  /** A taught top sign counts as influenced when its probability moved at least this much. */
  influenceThreshold: number;
}

export interface Personalized {
  /** The decision shown: personalised when examples applied, else the model's own. */
  decision: Decision;
  /** The model's decision without the examples. */
  base: Decision;
  /** True when the user's examples changed what is shown ("Personalized for you"). */
  influenced: boolean;
  /** s_c per taught sign (highest cosine similarity), for diagnostics. */
  similarity: Map<string, number>;
  /** The full distribution behind ``decision`` (personalised when examples applied). */
  probabilities: Float64Array;
}

/** The vector scaled to unit length; a zero vector stays zero. */
export function normalize(values: ArrayLike<number>): Float32Array {
  let sum = 0;
  for (let i = 0; i < values.length; i += 1) sum += (values[i] as number) ** 2;
  const scale = sum > 0 ? 1 / Math.sqrt(sum) : 0;
  const out = new Float32Array(values.length);
  for (let i = 0; i < values.length; i += 1) out[i] = (values[i] as number) * scale;
  return out;
}

export function dot(a: ArrayLike<number>, b: ArrayLike<number>): number {
  if (a.length !== b.length) throw new Error(`embeddings differ in length: ${a.length} and ${b.length}`);
  let sum = 0;
  for (let i = 0; i < a.length; i += 1) sum += (a[i] as number) * (b[i] as number);
  return sum;
}

/** The highest cosine similarity to each taught sign's examples. */
export function similarities(query: ArrayLike<number>, examples: readonly TaughtExample[]): Map<string, number> {
  const unit = normalize(query);
  const best = new Map<string, number>();
  for (const example of examples) {
    const value = dot(unit, example.embedding);
    if (value > (best.get(example.label) ?? Number.NEGATIVE_INFINITY)) best.set(example.label, value);
  }
  return best;
}

/** Signs with at least one example, sorted. */
export function taughtSigns(examples: readonly TaughtExample[]): string[] {
  return [...new Set(examples.map((example) => example.label))].sort();
}

export function personalize(
  logits: ArrayLike<number>,
  embedding: ArrayLike<number> | null,
  labels: readonly string[],
  examples: readonly TaughtExample[],
  params: PersonalizationParams,
  threshold: number,
  k: number,
  temperature: number,
): Personalized {
  const base = decide(logits, labels, threshold, k, temperature);
  const similarity = embedding && examples.length > 0 ? similarities(embedding, examples) : new Map<string, number>();
  if (similarity.size === 0) return { decision: base, base, influenced: false, similarity, probabilities: softmax(logits, temperature) };
  const adjusted = new Float64Array(logits.length);
  for (let i = 0; i < logits.length; i += 1) {
    const s = similarity.get(labels[i] as string);
    adjusted[i] = (logits[i] as number) / temperature + (s === undefined ? 0 : params.weight * (s - params.offset));
  }
  const decision = decide(adjusted, labels, threshold, k, 1);
  const top = decision.top[0];
  const baseTop = base.top[0];
  let influenced = false;
  if (top && similarity.has(top.label)) {
    const before = baseProbability(logits, labels, top.label, temperature);
    influenced =
      top.label !== baseTop?.label ||
      decision.accepted !== base.accepted ||
      Math.abs(top.probability - before) >= params.influenceThreshold;
  }
  return {
    decision: influenced ? { ...decision, personalized: true } : decision,
    base,
    influenced,
    similarity,
    probabilities: softmax(adjusted, 1),
  };
}

function baseProbability(logits: ArrayLike<number>, labels: readonly string[], label: string, temperature: number): number {
  const index = labels.indexOf(label);
  return index < 0 ? 0 : (softmax(logits, temperature)[index] as number);
}
