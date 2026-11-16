// Practice mode: scoring one attempt at a target sign, XP and streaks, accuracy per sign and the
// hardest signs, kept in localStorage. Pure apart from the storage calls.

import { readJson, writeJson, type KeyValueStore } from "./store";

export type Outcome = "correct" | "close" | "miss";

export const PRACTICE_KEY = "edge-vision.practice.v1";
export const XP_CORRECT = 10;
export const XP_STREAK_BONUS = 2;
export const MAX_STREAK_BONUS = 5;
export const XP_CLOSE = 3;

export interface SignStats {
  attempts: number;
  correct: number;
  close: number;
}

export interface PracticeState {
  xp: number;
  streak: number;
  bestStreak: number;
  stats: Record<string, SignStats>;
}

export const EMPTY_PRACTICE: PracticeState = { xp: 0, streak: 0, bestStreak: 0, stats: {} };

/** correct: the target is the top guess; close: it is in the top 3; miss: otherwise. */
export function scoreAttempt(target: string, top: readonly { label: string }[]): Outcome {
  if (top[0]?.label === target) return "correct";
  if (top.slice(0, 3).some((candidate) => candidate.label === target)) return "close";
  return "miss";
}

/** XP for an outcome given the streak before the attempt. */
export function xpFor(outcome: Outcome, streakBefore: number): number {
  if (outcome === "correct") return XP_CORRECT + XP_STREAK_BONUS * Math.min(streakBefore, MAX_STREAK_BONUS);
  return outcome === "close" ? XP_CLOSE : 0;
}

export function applyAttempt(state: PracticeState, target: string, outcome: Outcome): { state: PracticeState; xp: number } {
  const gained = xpFor(outcome, state.streak);
  const previous = state.stats[target] ?? { attempts: 0, correct: 0, close: 0 };
  const stats: SignStats = {
    attempts: previous.attempts + 1,
    correct: previous.correct + (outcome === "correct" ? 1 : 0),
    close: previous.close + (outcome === "close" ? 1 : 0),
  };
  const streak = outcome === "correct" ? state.streak + 1 : 0;
  return {
    xp: gained,
    state: {
      xp: state.xp + gained,
      streak,
      bestStreak: Math.max(state.bestStreak, streak),
      stats: { ...state.stats, [target]: stats },
    },
  };
}

export function accuracy(stats: SignStats): number {
  return stats.attempts ? stats.correct / stats.attempts : 0;
}

export function overallAccuracy(state: PracticeState): number | null {
  let attempts = 0;
  let correct = 0;
  for (const item of Object.values(state.stats)) {
    attempts += item.attempts;
    correct += item.correct;
  }
  return attempts ? correct / attempts : null;
}

/** Signs tried at least `minAttempts` times, lowest accuracy first (more attempts first on ties). */
export function hardestSigns(state: PracticeState, n = 5, minAttempts = 2): { label: string; stats: SignStats; accuracy: number }[] {
  return Object.entries(state.stats)
    .filter(([, stats]) => stats.attempts >= minAttempts)
    .map(([label, stats]) => ({ label, stats, accuracy: accuracy(stats) }))
    .filter((item) => item.accuracy < 1)
    .sort((a, b) => a.accuracy - b.accuracy || b.stats.attempts - a.stats.attempts || a.label.localeCompare(b.label))
    .slice(0, n);
}

/** A random target, never the same as `previous` when there is a choice. */
export function randomTarget(labels: readonly string[], previous: string | null, random: () => number = Math.random): string {
  const pool = labels.length > 1 && previous !== null ? labels.filter((label) => label !== previous) : labels;
  return pool[Math.floor(random() * pool.length) % pool.length] as string;
}

function sanitize(raw: unknown): PracticeState {
  const source = (raw !== null && typeof raw === "object" ? raw : {}) as Record<string, unknown>;
  const count = (value: unknown): number => (typeof value === "number" && Number.isFinite(value) && value >= 0 ? Math.floor(value) : 0);
  const stats: Record<string, SignStats> = {};
  const rawStats = (source.stats !== null && typeof source.stats === "object" ? source.stats : {}) as Record<string, unknown>;
  for (const [label, value] of Object.entries(rawStats)) {
    const item = (value !== null && typeof value === "object" ? value : {}) as Record<string, unknown>;
    const attempts = count(item.attempts);
    if (attempts === 0) continue;
    stats[label] = { attempts, correct: Math.min(count(item.correct), attempts), close: Math.min(count(item.close), attempts) };
  }
  return { xp: count(source.xp), streak: count(source.streak), bestStreak: count(source.bestStreak), stats };
}

export function loadPractice(store: KeyValueStore | null): PracticeState {
  return sanitize(readJson<unknown>(store, PRACTICE_KEY, null));
}

export function savePractice(store: KeyValueStore | null, state: PracticeState): boolean {
  return writeJson(store, PRACTICE_KEY, state);
}
