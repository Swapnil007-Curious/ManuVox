// Practice mode: a target sign, Go, a 3-2-1 countdown, then the next recognised segment is scored:
// correct (top guess), close (in the top 3) or try again (with what the model saw).

import type { SegmentResult } from "../engine";
import { byId, h, percent } from "./dom";
import {
  applyAttempt,
  hardestSigns,
  loadPractice,
  overallAccuracy,
  randomTarget,
  savePractice,
  scoreAttempt,
  type PracticeState,
} from "./practice";
import { dictionaryUrl, glossOf, signInfo, SIGNS } from "./signs";
import type { KeyValueStore } from "./store";

const LISTEN_MS = 8000;

export interface PracticeHooks {
  ensureCamera(): Promise<boolean>;
  countdown(): Promise<void>;
  burst(): void;
  announce(text: string): void;
}

export class PracticeView {
  private readonly word = byId<HTMLElement>("target-word");
  private readonly desc = byId<HTMLElement>("target-desc");
  private readonly link = byId<HTMLAnchorElement>("target-link");
  readonly goButton = byId<HTMLButtonElement>("practice-go");
  readonly nextButton = byId<HTMLButtonElement>("practice-next");
  private readonly result = byId<HTMLElement>("practice-result");
  private readonly xp = byId<HTMLElement>("practice-xp");
  private readonly streak = byId<HTMLElement>("practice-streak");
  private readonly best = byId<HTMLElement>("practice-best");
  private readonly accuracyEl = byId<HTMLElement>("practice-accuracy");
  private readonly hardest = byId<HTMLOListElement>("hardest");
  private readonly hardestEmpty = byId<HTMLElement>("hardest-empty");
  target: string;
  state: PracticeState;
  phase: "idle" | "countdown" | "listening" = "idle";
  private listenTimer = 0;
  private stored = true;

  constructor(
    private readonly store: KeyValueStore | null,
    private readonly hooks: PracticeHooks,
  ) {
    this.state = loadPractice(store);
    this.target = "hello";
    this.goButton.addEventListener("click", () => void this.go());
    this.nextButton.addEventListener("click", () => this.setTarget(randomTarget(SIGNS.map((sign) => sign.label), this.target)));
    this.setTarget(this.target);
    this.renderStats();
    this.showIdle();
  }

  setTarget(label: string): void {
    this.target = label;
    const info = signInfo(label);
    this.word.textContent = info.gloss;
    this.desc.textContent = info.description;
    this.link.href = dictionaryUrl(label);
    this.link.setAttribute("aria-label", `Watch ${info.gloss} in the SignASL dictionary (opens in a new tab)`);
    this.cancel();
    this.showIdle();
  }

  private showIdle(): void {
    this.result.dataset.outcome = "none";
    this.result.textContent = "Press Go, wait for the countdown, then sign it once.";
  }

  cancel(): void {
    clearTimeout(this.listenTimer);
    this.phase = "idle";
    this.goButton.disabled = false;
    this.goButton.querySelector("span")!.textContent = "Go";
  }

  async go(): Promise<void> {
    if (this.phase !== "idle") return;
    this.goButton.disabled = true;
    if (!(await this.hooks.ensureCamera())) {
      this.goButton.disabled = false;
      this.result.dataset.outcome = "miss";
      this.result.replaceChildren(h("strong", { text: "Camera needed" }), "Start the camera to practise.");
      return;
    }
    this.phase = "countdown";
    this.result.dataset.outcome = "none";
    this.result.textContent = "Get ready…";
    await this.hooks.countdown();
    if (this.phase !== "countdown") return;
    this.phase = "listening";
    this.goButton.querySelector("span")!.textContent = "Listening…";
    this.result.textContent = `Sign ${glossOf(this.target)} now.`;
    this.listenTimer = window.setTimeout(() => {
      if (this.phase !== "listening") return;
      this.cancel();
      this.result.dataset.outcome = "miss";
      this.result.replaceChildren(h("strong", { text: "No sign caught" }), "Keep your hands in view and sign once after the countdown.");
    }, LISTEN_MS);
  }

  /** A finished segment while listening: score it. Returns true when practice consumed it. */
  handleResult(result: SegmentResult): boolean {
    if (this.phase !== "listening") return false;
    if (result.kind === "discarded") return true; // Too short to count; keep listening.
    clearTimeout(this.listenTimer);
    this.cancel();
    const top = result.decision?.top ?? [];
    const outcome = result.kind === "classified" ? scoreAttempt(this.target, top) : "miss";
    const applied = applyAttempt(this.state, this.target, outcome);
    this.state = applied.state;
    this.stored = savePractice(this.store, this.state);
    const gloss = glossOf(this.target);
    const seen = top.slice(0, 3).map((candidate) => `${glossOf(candidate.label)} ${percent(candidate.probability)}`).join(", ");
    this.result.dataset.outcome = outcome;
    if (outcome === "correct") {
      this.result.replaceChildren(h("strong", { text: `Correct! +${applied.xp} XP` }), `That was ${gloss}${this.state.streak > 1 ? `, ${this.state.streak} in a row` : ""}.`);
      this.hooks.burst();
      this.hooks.announce(`Correct, ${gloss}. Plus ${applied.xp} XP.`);
    } else if (outcome === "close") {
      this.result.replaceChildren(h("strong", { text: `Close! +${applied.xp} XP` }), `${gloss} was in the top 3. The model saw: `, h("span", { class: "seen", text: seen }), ".");
      this.hooks.announce(`Close. ${gloss} was in the top three.`);
    } else {
      const parts: (Node | string)[] = [h("strong", { text: "Try again" })];
      if (result.kind === "classified") parts.push("The model saw: ", h("span", { class: "seen", text: seen }));
      else parts.push("The sign could not be read: keep both shoulders and your hands in view.");
      this.result.replaceChildren(...parts);
      this.hooks.announce(`Try again. The model saw ${top[0] ? glossOf(top[0].label) : "nothing"}.`);
    }
    this.renderStats();
    return true;
  }

  renderStats(): void {
    this.xp.textContent = `${this.state.xp} XP`;
    this.streak.textContent = String(this.state.streak);
    this.best.textContent = String(this.state.bestStreak);
    const accuracy = overallAccuracy(this.state);
    this.accuracyEl.textContent = accuracy === null ? "–" : percent(accuracy);
    const hardest = hardestSigns(this.state);
    this.hardest.replaceChildren(
      ...hardest.map((item) => {
        const button = h(
          "button",
          { type: "button", "aria-label": `Practise ${glossOf(item.label)}: ${item.stats.correct} of ${item.stats.attempts} correct` },
          glossOf(item.label),
          h("span", { class: "mono", text: `${item.stats.correct}/${item.stats.attempts}` }),
        );
        button.addEventListener("click", () => this.setTarget(item.label));
        return h("li", {}, button);
      }),
    );
    this.hardestEmpty.hidden = hardest.length > 0;
    if (!this.stored) this.hardestEmpty.textContent = "Progress is kept for this visit only: this browser blocks storage.";
  }
}
