// The transcript: adding confident words, "not sure" candidates, replacing with 1/2/3, undo.

import { describe, expect, it } from "vitest";
import { Transcript, type Choice } from "../../src/ui/transcript";

const choices = (...labels: string[]): Choice[] => labels.map((label, i) => ({ label, probability: 0.6 - i * 0.2 }));

describe("Transcript", () => {
  it("adds the top choice of a confident result and keeps its top 3", () => {
    const t = new Transcript();
    expect(t.add(choices("hello", "uncle", "sun", "extra"), true)).toBe("added");
    expect(t.labels).toEqual(["hello"]);
    expect(t.last?.choices.map((c) => c.label)).toEqual(["hello", "uncle", "sun"]);
    expect(t.choices.map((c) => c.label)).toEqual(["hello", "uncle", "sun"]);
  });

  it("replaces the last word with candidate 2 or 3 and marks it picked", () => {
    const t = new Transcript();
    t.add(choices("mom", "dad", "aunt"), true);
    t.add(choices("drink", "water", "milk"), true);
    expect(t.pick(1)).toBe("replaced");
    expect(t.labels).toEqual(["mom", "water"]);
    expect(t.last?.picked).toBe(true);
    expect(t.pick(1)).toBe("none"); // already that word
    expect(t.pick(7)).toBe("none");
    expect(t.pick(0)).toBe("replaced");
    expect(t.labels).toEqual(["mom", "drink"]);
  });

  it("holds a not-sure result as pending until a candidate is picked", () => {
    const t = new Transcript();
    t.add(choices("hello", "uncle", "sun"), true);
    expect(t.add(choices("grass", "goose", "find"), false)).toBe("pending");
    expect(t.labels).toEqual(["hello"]);
    expect(t.choices.map((c) => c.label)).toEqual(["grass", "goose", "find"]);
    expect(t.pick(1)).toBe("added");
    expect(t.labels).toEqual(["hello", "goose"]);
    expect(t.pending).toBeNull();
    expect(t.last?.picked).toBe(true);
  });

  it("undo drops pending candidates first, then words; versions change on every change", () => {
    const t = new Transcript();
    const v0 = t.version;
    t.add(choices("yes"), true);
    t.add(choices("no", "now"), false);
    expect(t.version).toBe(v0 + 2);
    expect(t.undo()).toBe("cleared");
    expect(t.labels).toEqual(["yes"]);
    expect(t.undo()).toBe("removed");
    expect(t.labels).toEqual([]);
    expect(t.undo()).toBe("none");
    const v1 = t.version;
    expect(t.pick(0)).toBe("none");
    expect(t.version).toBe(v1);
  });

  it("a confident result replaces pending candidates; clear empties everything", () => {
    const t = new Transcript();
    t.add(choices("grass", "goose"), false);
    t.add(choices("water", "wet"), true);
    expect(t.pending).toBeNull();
    expect(t.labels).toEqual(["water"]);
    expect(t.clear()).toBe("cleared");
    expect(t.labels).toEqual([]);
    expect(t.clear()).toBe("none");
    expect(t.add([], true)).toBe("none");
  });
});

describe("Teach Dex flags", () => {
  it("marks words and pending choices from personalised decisions until the user changes them", () => {
    const t = new Transcript();
    t.add([{ label: "mom", probability: 0.8 }, { label: "dad", probability: 0.1 }], true, true);
    expect(t.last?.personalized).toBe(true);
    expect(t.choicesPersonalized).toBe(true);
    t.pick(1);
    expect(t.last?.personalized).toBe(false);
    t.add([{ label: "water", probability: 0.3 }], false, true);
    expect(t.choicesPersonalized).toBe(true);
    t.pick(0);
    expect(t.last?.personalized).toBe(false);
    t.add([{ label: "milk", probability: 0.9 }], true);
    expect(t.choicesPersonalized).toBe(false);
  });
});
