// The sentence being built: recognised signs (glosses) with their top-3 candidates. Pure.
//
// - A confident result is added as a word; its candidates stay attached, so picking candidate
//   1, 2 or 3 replaces the last word.
// - A "not sure" result adds nothing: its candidates wait as `pending` until the user picks one,
//   which adds that word.
// - Undo (Backspace) drops the pending candidates if there are any, else the last word.

export interface Choice {
  label: string;
  probability: number;
}

export interface Word {
  id: number;
  label: string;
  /** The top-3 of the segment the word came from. */
  choices: Choice[];
  /** True when the user picked or replaced it. */
  picked: boolean;
  /** True when the user's own examples (Teach Dex) shaped the prediction. */
  personalized: boolean;
}

export type Change = "added" | "pending" | "replaced" | "removed" | "cleared" | "none";

export class Transcript {
  words: Word[] = [];
  pending: Choice[] | null = null;
  /** Whether the pending choices came from a personalised decision. */
  pendingPersonalized = false;
  /** Increments on every change; the app re-translates when it differs from the last translation. */
  version = 0;
  private nextId = 1;

  get labels(): string[] {
    return this.words.map((word) => word.label);
  }

  get last(): Word | null {
    return this.words[this.words.length - 1] ?? null;
  }

  /** A classified segment: `accepted` adds its top word; otherwise its choices become pending. */
  add(choices: readonly Choice[], accepted: boolean, personalized = false): Change {
    const top = choices.slice(0, 3).map((choice) => ({ ...choice }));
    if (top.length === 0) return "none";
    this.version += 1;
    if (!accepted) {
      this.pending = top;
      this.pendingPersonalized = personalized;
      return "pending";
    }
    this.pending = null;
    this.words.push({ id: this.nextId++, label: (top[0] as Choice).label, choices: top, picked: false, personalized });
    return "added";
  }

  /** Whether the candidates on show came from a personalised decision. */
  get choicesPersonalized(): boolean {
    return this.pending ? this.pendingPersonalized : (this.last?.personalized ?? false);
  }

  /** Candidate `index` (0-2): adds a pending choice, else replaces the last word. */
  pick(index: number): Change {
    if (this.pending) {
      const choice = this.pending[index];
      if (!choice) return "none";
      this.words.push({ id: this.nextId++, label: choice.label, choices: this.pending, picked: true, personalized: false });
      this.pending = null;
      this.version += 1;
      return "added";
    }
    const last = this.last;
    const choice = last?.choices[index];
    if (!last || !choice || choice.label === last.label) return "none";
    last.label = choice.label;
    last.picked = true;
    last.personalized = false;
    this.version += 1;
    return "replaced";
  }

  undo(): Change {
    if (this.pending) {
      this.pending = null;
      this.version += 1;
      return "cleared";
    }
    if (this.words.length === 0) return "none";
    this.words.pop();
    this.version += 1;
    return "removed";
  }

  clear(): Change {
    if (this.words.length === 0 && !this.pending) return "none";
    this.words = [];
    this.pending = null;
    this.version += 1;
    return "cleared";
  }

  /** The choices the candidate picker shows: the pending ones, else the last word's. */
  get choices(): Choice[] {
    return this.pending ?? this.last?.choices ?? [];
  }
}
