// The sliding-window arithmetic on its own (no storage), with the Worker's limits.

import { describe, expect, it } from "vitest";
import { check } from "../src/limits.ts";

const W = [
  { max: 30, seconds: 600 },
  { max: 300, seconds: 86_400 },
];

describe("check", () => {
  it("allows 30 in 10 minutes, then asks to wait until the oldest leaves the window", () => {
    const now = 1_000_000;
    const times = Array.from({ length: 29 }, (_, i) => now - 500 + i);
    expect(check(times, now, W).wait).toBe(0);
    const full = [...times, now - 100];
    const r = check(full, now, W);
    expect(r.wait).toBeCloseTo(100, 5); // the oldest of the last 30 is 500 s old
    expect(r.kept).toHaveLength(30);
  });

  it("drops times older than a day and applies the daily limit of 300", () => {
    const now = 1_000_000;
    const old = Array.from({ length: 50 }, (_, i) => now - 90_000 - i);
    expect(check(old, now, W).kept).toHaveLength(0);
    const day = Array.from({ length: 300 }, (_, i) => now - 80_000 + i * 10);
    const r = check(day, now, W);
    expect(r.wait).toBeGreaterThan(6_000);
    expect(r.wait).toBeLessThanOrEqual(86_400);
    expect(check(day.slice(1), now, W).wait).toBe(0);
  });
});
