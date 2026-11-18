// Teach Dex's blend (src/personalization.ts): cosine kNN over the user's examples, added to the
// temperature-scaled logits of taught signs only.

import { describe, expect, it } from "vitest";
import { dot, normalize, personalize, similarities, taughtSigns, type TaughtExample } from "../../src/personalization";
import { decide, softmax } from "../../src/scoring";

const LABELS = ["hello", "mom", "dad", "water"];
const PARAMS = { weight: 6, offset: 0.5, influenceThreshold: 0.05 };

function example(label: string, vector: number[], id = label): TaughtExample {
  return { id, label, embedding: normalize(vector), createdAt: 0, modelSha256: "a".repeat(64) };
}

describe("vectors", () => {
  it("normalises to unit length and leaves a zero vector at zero", () => {
    const unit = normalize([3, 4]);
    expect(unit[0]).toBeCloseTo(0.6);
    expect(unit[1]).toBeCloseTo(0.8);
    expect(Array.from(normalize([0, 0]))).toEqual([0, 0]);
    expect(() => dot([1], [1, 2])).toThrow(/length/);
  });

  it("keeps the best similarity per taught sign", () => {
    const examples = [example("mom", [1, 0], "m1"), example("mom", [0.6, 0.8], "m2"), example("dad", [0, 1])];
    const best = similarities([1, 0], examples);
    expect(best.get("mom")).toBeCloseTo(1);
    expect(best.get("dad")).toBeCloseTo(0);
    expect(taughtSigns(examples)).toEqual(["dad", "mom"]);
  });
});

describe("personalize", () => {
  const logits = new Float32Array([2.0, 1.5, 1.4, -1]);

  it("returns the model's own decision without examples or without an embedding", () => {
    const base = decide(logits, LABELS, 0.5, 3, 0.9);
    for (const result of [
      personalize(logits, [1, 0], LABELS, [], PARAMS, 0.5, 3, 0.9),
      personalize(logits, null, LABELS, [example("mom", [1, 0])], PARAMS, 0.5, 3, 0.9),
    ]) {
      expect(result.decision).toEqual(base);
      expect(result.influenced).toBe(false);
      expect(Array.from(result.probabilities)).toEqual(Array.from(softmax(logits, 0.9)));
    }
  });

  it("adds weight * (similarity - offset) to taught signs only", () => {
    const T = 0.9;
    const result = personalize(logits, [1, 0], LABELS, [example("mom", [1, 0]), example("dad", [0, 1])], PARAMS, 0.4, 4, T);
    const expected = Array.from(logits, (z, i) => z / T + (i === 1 ? 6 * (1 - 0.5) : i === 2 ? 6 * (0 - 0.5) : 0));
    const probabilities = softmax(expected);
    expect(Array.from(result.probabilities)).toEqual(Array.from(probabilities));
    expect(result.decision.top[0]?.label).toBe("mom");
    expect(result.base.top[0]?.label).toBe("hello");
    expect(result.influenced).toBe(true);
    expect(result.decision.personalized).toBe(true);
    expect(result.decision.threshold).toBe(0.4); // The threshold is not refitted.
  });

  it("only badges a taught top sign whose probability moved enough", () => {
    // A taught sign far from the query is pushed down; the untaught top stays: no badge.
    const away = personalize(logits, [1, 0], LABELS, [example("water", [0, 1])], PARAMS, 0.4, 3, 1);
    expect(away.decision.top[0]?.label).toBe("hello");
    expect(away.influenced).toBe(false);
    expect(away.decision.personalized).toBeUndefined();
    // A taught top sign that barely moves (similarity at the offset) is not badged either.
    const flat = personalize(logits, [1, 1], LABELS, [example("hello", [1, 0])], { ...PARAMS, offset: Math.SQRT1_2 }, 0.4, 3, 1);
    expect(flat.decision.top[0]?.label).toBe("hello");
    expect(flat.influenced).toBe(false);
  });

  it("can turn a 'not sure' into an accepted taught sign", () => {
    const unsure = new Float32Array([0.2, 0.1, 0.0, 0.05]);
    const result = personalize(unsure, [1, 0], LABELS, [example("dad", [1, 0])], PARAMS, 0.5, 3, 1);
    expect(result.base.accepted).toBe(false);
    expect(result.decision.accepted).toBe(true);
    expect(result.decision.label).toBe("dad");
    expect(result.influenced).toBe(true);
  });
});
