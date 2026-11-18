// Teach Dex: the user records a few examples of a sign (a 3-2-1 countdown, then the next segment
// the segmenter finds), and the app stores their embeddings on this device (src/teachStore.ts) and
// blends them into later decisions (src/personalization.ts). Nothing is uploaded.

import type { SegmentResult } from "../engine";
import { normalize, taughtSigns, type TaughtExample } from "../personalization";
import { forModel, type TeachStore } from "../teachStore";
import { byId, h } from "./dom";
import { glossOf, searchSigns } from "./signs";

const LISTEN_MS = 8000;
const RESULTS_SHOWN = 8;
export const DEFAULT_EXAMPLES_PER_SIGN = 3;

export interface TeachHooks {
  ensureCamera(): Promise<boolean>;
  countdown(): Promise<void>;
  burst(): void;
  announce(text: string): void;
  /** The loaded model's sha256 and whether it can be taught (it has an embedding output). */
  model(): { sha256: string; canTeach: boolean; examplesPerSign: number } | null;
  /** The examples now in use for the loaded model. */
  apply(examples: TaughtExample[]): void;
  personalizationOn(): boolean;
  setPersonalization(on: boolean): void;
  confirm(message: string): boolean;
}

function exampleId(): string {
  const random = typeof crypto !== "undefined" && "randomUUID" in crypto ? crypto.randomUUID() : `${Math.random()}`.slice(2);
  return `${Date.now()}-${random}`;
}

export class TeachView {
  private readonly count = byId<HTMLElement>("teach-count");
  private readonly unavailable = byId<HTMLElement>("teach-unavailable");
  private readonly storageNote = byId<HTMLElement>("teach-storage");
  private readonly search = byId<HTMLInputElement>("teach-search");
  private readonly results = byId<HTMLUListElement>("teach-results");
  private readonly card = byId<HTMLElement>("teach-card");
  private readonly word = byId<HTMLElement>("teach-word");
  private readonly dots = byId<HTMLOListElement>("teach-dots");
  readonly recordButton = byId<HTMLButtonElement>("teach-record");
  private readonly status = byId<HTMLElement>("teach-status");
  private readonly list = byId<HTMLUListElement>("teach-list");
  private readonly empty = byId<HTMLElement>("teach-empty");
  private readonly enabled = byId<HTMLInputElement>("teach-enabled");
  private readonly resetButton = byId<HTMLButtonElement>("teach-reset");

  target: string | null = null;
  phase: "idle" | "countdown" | "listening" = "idle";
  /** Every stored example (any model); the loaded model uses forModel(...) of them. */
  private stored: TaughtExample[] = [];
  private store: TeachStore | null = null;
  private listenTimer = 0;
  /** How long a recording waits for the sign after the countdown (the evaluation lengthens it). */
  listenMs = LISTEN_MS;

  constructor(private readonly hooks: TeachHooks) {
    this.search.addEventListener("input", () => this.renderResults());
    this.recordButton.addEventListener("click", () => void this.record());
    this.enabled.addEventListener("change", () => {
      this.hooks.setPersonalization(this.enabled.checked);
      this.hooks.announce(this.enabled.checked ? "Dex uses your examples." : "Dex ignores your examples.");
    });
    this.resetButton.addEventListener("click", () => void this.reset());
    this.renderResults();
    this.render();
  }

  /** The store opened (IndexedDB, or memory with a reason); loads the saved examples. */
  async attach(store: TeachStore): Promise<void> {
    this.store = store;
    this.storageNote.hidden = store.mode === "indexeddb";
    this.storageNote.textContent =
      store.mode === "memory" ? "This browser won’t let Dex save examples, so they last until you close this tab." : "";
    try {
      this.stored = await store.all();
    } catch {
      this.stored = [];
    }
    this.sync();
  }

  /** The model loaded or changed: apply its examples. */
  modelReady(): void {
    this.sync();
  }

  get examples(): TaughtExample[] {
    const model = this.hooks.model();
    return model ? forModel(this.stored, model.sha256) : [];
  }

  private sync(): void {
    this.hooks.apply(this.examples);
    this.render();
  }

  private perSign(): number {
    return this.hooks.model()?.examplesPerSign ?? DEFAULT_EXAMPLES_PER_SIGN;
  }

  select(label: string): void {
    this.cancel();
    this.target = label;
    this.status.textContent = "";
    delete this.status.dataset.tone;
    this.renderResults();
    this.render();
  }

  cancel(): void {
    clearTimeout(this.listenTimer);
    this.phase = "idle";
    this.render();
  }

  private countFor(label: string): number {
    return this.examples.filter((example) => example.label === label).length;
  }

  async record(): Promise<void> {
    const model = this.hooks.model();
    if (this.phase !== "idle" || !this.target || !model?.canTeach) return;
    if (this.countFor(this.target) >= this.perSign()) {
      // Teaching a sign again replaces its examples.
      await this.removeSign(this.target, false);
    }
    this.recordButton.disabled = true;
    if (!(await this.hooks.ensureCamera())) {
      this.recordButton.disabled = false;
      this.setStatus("Start the camera to teach Dex.", "miss");
      return;
    }
    this.phase = "countdown";
    this.setStatus("Get ready…");
    this.render();
    await this.hooks.countdown();
    if (this.phase !== "countdown") return;
    this.phase = "listening";
    this.setStatus(`Sign ${glossOf(this.target)} now.`);
    this.render();
    this.listenTimer = window.setTimeout(() => {
      if (this.phase !== "listening") return;
      this.cancel();
      this.setStatus("No sign caught. Keep your hands in view and sign once after the countdown.", "miss");
    }, this.listenMs);
  }

  /** A finished segment: while listening it becomes an example. Returns true when consumed. */
  handleResult(result: SegmentResult): boolean {
    if (this.phase !== "listening") return false;
    if (result.kind === "discarded") return true; // Too short; keep listening.
    clearTimeout(this.listenTimer);
    const target = this.target;
    const model = this.hooks.model();
    this.phase = "idle";
    if (!target || !model || result.kind !== "classified" || !result.embedding) {
      this.setStatus("Dex couldn’t read that one. Try again with both hands in view.", "miss");
      this.render();
      return true;
    }
    const example: TaughtExample = {
      id: exampleId(),
      label: target,
      embedding: normalize(result.embedding),
      createdAt: Date.now(),
      modelSha256: model.sha256,
    };
    this.stored.push(example);
    void this.store?.add(example).catch(() => this.setStatus("Saved for this session only: the browser refused to store it.", "miss"));
    this.sync();
    const have = this.countFor(target);
    const need = this.perSign();
    const gloss = glossOf(target);
    if (have >= need) {
      this.setStatus(`Taught! Dex will lean toward your ${gloss}.`, "done");
      this.hooks.burst();
      this.hooks.announce(`Taught ${gloss}. Dex will lean toward your way of signing it.`);
    } else {
      this.setStatus(`Got it: ${have} of ${need}. Record the next one.`, "done");
      this.hooks.announce(`Example ${have} of ${need} recorded.`);
    }
    return true;
  }

  async removeSign(label: string, announce = true): Promise<void> {
    this.stored = this.stored.filter((example) => example.label !== label);
    try {
      await this.store?.removeLabel(label);
    } catch {
      // The in-memory list is already updated; the next load may bring them back.
    }
    this.sync();
    if (announce) this.hooks.announce(`Forgot your ${glossOf(label)}.`);
  }

  async reset(): Promise<void> {
    if (this.stored.length === 0) return;
    if (!this.hooks.confirm("Delete every sign you taught Dex on this device?")) return;
    this.cancel();
    this.stored = [];
    try {
      await this.store?.clear();
    } catch {
      // Cleared in memory regardless.
    }
    this.sync();
    this.setStatus("");
    this.hooks.announce("Your signs were deleted from this device.");
  }

  private setStatus(text: string, tone?: "done" | "miss"): void {
    this.status.textContent = text;
    if (tone) this.status.dataset.tone = tone;
    else delete this.status.dataset.tone;
  }

  private renderResults(): void {
    const matches = searchSigns(this.search.value).slice(0, RESULTS_SHOWN);
    this.results.replaceChildren(
      ...matches.map((sign) => {
        const button = h("button", { type: "button", class: "button small", "aria-pressed": String(sign.label === this.target) }, sign.gloss);
        button.addEventListener("click", () => this.select(sign.label));
        return h("li", {}, button);
      }),
    );
  }

  render(): void {
    const model = this.hooks.model();
    const canTeach = model?.canTeach ?? false;
    this.unavailable.hidden = !model || canTeach;
    this.search.disabled = !canTeach;
    this.enabled.checked = this.hooks.personalizationOn();
    const examples = this.examples;
    const signs = taughtSigns(examples);
    this.count.textContent = `${signs.length} sign${signs.length === 1 ? "" : "s"} taught`;
    this.resetButton.disabled = this.stored.length === 0;
    this.empty.hidden = signs.length > 0;
    this.list.replaceChildren(
      ...signs.map((label) => {
        const n = examples.filter((example) => example.label === label).length;
        const forget = h("button", { type: "button", class: "button ghost small", "aria-label": `Forget your ${glossOf(label)}` }, "Forget");
        forget.addEventListener("click", () => void this.removeSign(label));
        return h(
          "li",
          { "data-label": label },
          h("span", {}, h("span", { class: "teach-sign", text: glossOf(label) }), h("span", { class: "teach-meta", text: `${n} example${n === 1 ? "" : "s"}` })),
          forget,
        );
      }),
    );
    this.card.hidden = !this.target || !canTeach;
    if (!this.target) return;
    const have = this.countFor(this.target);
    const need = this.perSign();
    this.word.textContent = glossOf(this.target);
    this.dots.replaceChildren(...Array.from({ length: need }, (_, i) => h("li", { "data-done": String(i < have), "aria-label": i < have ? "recorded" : "not yet" })));
    const label = this.phase === "countdown" ? "Get ready…" : this.phase === "listening" ? "Listening…" : have >= need ? "Teach again" : "Record example";
    this.recordButton.querySelector("span")!.textContent = label;
    this.recordButton.disabled = this.phase !== "idle" || !canTeach;
  }
}
