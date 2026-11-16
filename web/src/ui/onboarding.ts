// First run: three steps. 1, why the camera is needed (before the browser's own prompt); 2, frame
// yourself (the card moves aside so the framing guide shows, and the step completes when the body
// and a hand are in view); 3, try "hello" (completes on the first recognised sign).

import { byId } from "./dom";

export type Step = 1 | 2 | 3;

const COPY: Record<Step, { title: string; body: string; primary: string }> = {
  1: {
    title: "Allow the camera",
    body: "Edge-Vision reads your hands from the camera to recognise signs. Video stays on this device and nothing is recorded. Your browser asks for permission next.",
    primary: "Allow camera",
  },
  2: {
    title: "Frame yourself",
    body: "Sit back until your head, shoulders and both hands fit in the frame. The outline turns green when you're set.",
    primary: "Next",
  },
  3: {
    title: "Try “hello”",
    body: "Flat hand at your forehead, then move it out, like a salute. The word lands in the transcript when it's recognised.",
    primary: "Finish",
  },
};

export interface OnboardingHooks {
  allowCamera(): Promise<boolean>;
  finish(): void;
}

export class Onboarding {
  private readonly root = byId<HTMLElement>("onboarding");
  private readonly title = byId<HTMLElement>("onboarding-title");
  private readonly body = byId<HTMLElement>("onboarding-body");
  private readonly primary = byId<HTMLButtonElement>("onboarding-primary");
  private readonly skip = byId<HTMLButtonElement>("onboarding-skip");
  step: Step = 1;
  private previousFocus: HTMLElement | null = null;

  constructor(private readonly hooks: OnboardingHooks) {
    this.primary.addEventListener("click", () => void this.onPrimary());
    this.skip.addEventListener("click", () => this.close(true));
    this.root.addEventListener("keydown", (event) => {
      if (event.key === "Escape") this.close(true);
    });
  }

  get open(): boolean {
    return !this.root.hidden;
  }

  show(step: Step = 1): void {
    if (!this.open) this.previousFocus = document.activeElement as HTMLElement | null;
    this.step = step;
    const copy = COPY[step];
    this.root.hidden = false;
    this.root.dataset.step = String(step);
    // Step 1 blocks the page; steps 2 and 3 leave the stage usable.
    this.root.setAttribute("aria-modal", String(step === 1));
    this.title.textContent = copy.title;
    this.body.textContent = copy.body;
    this.primary.textContent = copy.primary;
    this.primary.disabled = false;
    this.skip.textContent = step === 3 ? "Done" : "Skip";
    for (const item of this.root.querySelectorAll<HTMLElement>(".steps li")) {
      const n = Number(item.dataset.step);
      if (n === step) item.setAttribute("aria-current", "step");
      else item.removeAttribute("aria-current");
      item.dataset.done = String(n < step);
    }
    this.primary.focus();
  }

  private async onPrimary(): Promise<void> {
    if (this.step === 1) {
      this.primary.disabled = true;
      this.primary.textContent = "Waiting for permission…";
      const ok = await this.hooks.allowCamera();
      if (ok) this.show(2);
      else this.close(false);
      return;
    }
    if (this.step === 2) {
      this.show(3);
      return;
    }
    this.close(true);
  }

  /** The framing check passed (step 2). */
  framed(): void {
    if (this.open && this.step === 2) window.setTimeout(() => this.open && this.step === 2 && this.show(3), 700);
  }

  /** A sign was recognised (step 3). */
  recognised(gloss: string): void {
    if (!this.open || this.step !== 3) return;
    this.title.textContent = gloss === "HELLO" ? "That's “hello”" : `Recognised “${gloss.toLowerCase()}”`;
    this.body.textContent = "Keep signing to build a sentence. It's translated and spoken after a short pause.";
    this.primary.textContent = "Start signing";
  }

  close(done: boolean): void {
    if (!this.open) return;
    this.root.hidden = true;
    if (done) this.hooks.finish();
    this.previousFocus?.focus?.();
  }
}
