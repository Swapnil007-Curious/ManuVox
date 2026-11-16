// Practice scoring, XP and streaks, hardest signs, and storage that is blocked or corrupt.

import { describe, expect, it } from "vitest";
import {
  EMPTY_PRACTICE,
  PRACTICE_KEY,
  applyAttempt,
  hardestSigns,
  loadPractice,
  overallAccuracy,
  randomTarget,
  savePractice,
  scoreAttempt,
  xpFor,
} from "../../src/ui/practice";

const top = (...labels: string[]) => labels.map((label) => ({ label }));

class MemoryStore {
  data = new Map<string, string>();
  getItem(key: string): string | null {
    return this.data.get(key) ?? null;
  }
  setItem(key: string, value: string): void {
    this.data.set(key, value);
  }
}

const blocked = {
  getItem(): string | null {
    throw new DOMException("blocked", "SecurityError");
  },
  setItem(): void {
    throw new DOMException("blocked", "SecurityError");
  },
};

describe("scoreAttempt", () => {
  it("is correct for the top guess, close for the top 3, a miss otherwise", () => {
    expect(scoreAttempt("hello", top("hello", "uncle", "sun"))).toBe("correct");
    expect(scoreAttempt("hello", top("uncle", "sun", "hello"))).toBe("close");
    expect(scoreAttempt("hello", top("uncle", "sun", "hat", "hello"))).toBe("miss");
    expect(scoreAttempt("hello", [])).toBe("miss");
  });
});

describe("applyAttempt", () => {
  it("adds XP with a streak bonus, counts per sign and resets the streak on a miss or close", () => {
    expect(xpFor("correct", 0)).toBe(10);
    expect(xpFor("correct", 3)).toBe(16);
    expect(xpFor("correct", 50)).toBe(20);
    expect(xpFor("close", 4)).toBe(3);
    expect(xpFor("miss", 4)).toBe(0);
    let state = EMPTY_PRACTICE;
    state = applyAttempt(state, "hello", "correct").state;
    state = applyAttempt(state, "hello", "correct").state;
    const third = applyAttempt(state, "water", "correct");
    expect(third.xp).toBe(14);
    state = third.state;
    expect(state.streak).toBe(3);
    state = applyAttempt(state, "water", "close").state;
    expect(state.streak).toBe(0);
    expect(state.bestStreak).toBe(3);
    expect(state.stats.hello).toEqual({ attempts: 2, correct: 2, close: 0 });
    expect(state.stats.water).toEqual({ attempts: 2, correct: 1, close: 1 });
    expect(state.xp).toBe(10 + 12 + 14 + 3);
    expect(overallAccuracy(state)).toBe(0.75);
    expect(overallAccuracy(EMPTY_PRACTICE)).toBeNull();
  });
});

describe("hardestSigns", () => {
  it("lists signs tried at least twice, lowest accuracy first, never a perfect one", () => {
    let state = EMPTY_PRACTICE;
    for (const [label, outcome] of [
      ["dad", "miss"],
      ["dad", "miss"],
      ["mom", "correct"],
      ["mom", "miss"],
      ["yes", "correct"],
      ["yes", "correct"],
      ["no", "miss"],
    ] as const) {
      state = applyAttempt(state, label, outcome).state;
    }
    expect(hardestSigns(state).map((item) => item.label)).toEqual(["dad", "mom"]);
  });
});

describe("storage", () => {
  it("round-trips through storage and sanitises what it reads", () => {
    const store = new MemoryStore();
    const state = applyAttempt(EMPTY_PRACTICE, "hello", "correct").state;
    expect(savePractice(store, state)).toBe(true);
    expect(loadPractice(store)).toEqual(state);
    store.setItem(PRACTICE_KEY, JSON.stringify({ xp: -5, streak: "x", stats: { hello: { attempts: 2, correct: 9 }, junk: 4 } }));
    expect(loadPractice(store)).toEqual({ xp: 0, streak: 0, bestStreak: 0, stats: { hello: { attempts: 2, correct: 2, close: 0 } } });
    store.setItem(PRACTICE_KEY, "{not json");
    expect(loadPractice(store)).toEqual(EMPTY_PRACTICE);
  });

  it("keeps working when storage is blocked or missing", () => {
    expect(loadPractice(blocked)).toEqual(EMPTY_PRACTICE);
    expect(savePractice(blocked, EMPTY_PRACTICE)).toBe(false);
    expect(loadPractice(null)).toEqual(EMPTY_PRACTICE);
    expect(savePractice(null, EMPTY_PRACTICE)).toBe(false);
  });
});

describe("randomTarget", () => {
  it("never repeats the previous target when there is a choice", () => {
    const labels = ["a", "b", "c"];
    for (let i = 0; i < 30; i += 1) expect(randomTarget(labels, "b", () => i / 30)).not.toBe("b");
    expect(randomTarget(["only"], "only")).toBe("only");
  });
});
