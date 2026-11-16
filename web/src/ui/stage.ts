// Everything drawn over the video: the status pill, hand chips, hints, the framing guide, the
// recognised word landing in its confidence ring and flying into the transcript, the "Not sure"
// picker, the practice countdown and burst, the two-lane timeline and the camera-state cards.

import type { EngineState, Hand } from "../engine";
import type { GuidanceMessage } from "../guidance";
import { byId, h, percent } from "./dom";
import { prefersReducedMotion, spring, SPRINGS, place } from "./motion";
import { glossOf } from "./signs";
import type { Choice } from "./transcript";

export type StateKind = "loading" | "ready" | "denied" | "no_camera" | "busy" | "insecure" | "unsupported" | "error";

const STATE_COPY: Record<StateKind, { eyebrow: string; title: string; body: string; action: string | null }> = {
  loading: {
    eyebrow: "Loading",
    title: "Getting the sign model ready",
    body: "The sign model and hand tracking download once, then work offline.",
    action: "Start camera",
  },
  ready: {
    eyebrow: "Ready",
    title: "Sign, and hear it spoken",
    body: "Start the camera, keep your upper body in view, and sign one word at a time.",
    action: "Start camera",
  },
  denied: {
    eyebrow: "Camera blocked",
    title: "The camera is blocked for this site",
    body: "<ol><li>Click the camera or lock icon in the address bar and choose <b>Allow</b>.</li><li>On iPhone: Settings, Safari, Camera, Allow.</li><li>Then press Try again.</li></ol>",
    action: "Try again",
  },
  no_camera: {
    eyebrow: "No camera",
    title: "No camera was found",
    body: "Connect a camera, or check that it is enabled, then try again.",
    action: "Try again",
  },
  busy: {
    eyebrow: "Camera busy",
    title: "Another app is using the camera",
    body: "Close the other app or tab that uses the camera, then try again.",
    action: "Try again",
  },
  insecure: {
    eyebrow: "Camera unavailable",
    title: "The camera needs a secure page",
    body: "Open Edge-Vision over https (or on localhost) to use the camera.",
    action: null,
  },
  unsupported: {
    eyebrow: "Browser not supported",
    title: "This browser cannot run Edge-Vision",
    body: "It needs WebAssembly, Web Workers and camera access. Use a recent Chrome, Edge, Safari or Firefox.",
    action: null,
  },
  error: {
    eyebrow: "Something went wrong",
    title: "Edge-Vision could not start",
    body: "",
    action: "Reload",
  },
};

const HINT_TEXT: Partial<Record<GuidanceMessage["code"], string>> = {
  low_light: "Too dark: add light in front of you",
  hand_out_of_frame: "Hands partly out of frame",
  too_close: "Move back a little",
  too_far: "Move a little closer",
  shoulders_hidden: "Fit your shoulders in the frame",
  calibration_no_shoulders: "Move back so your shoulders are visible",
};
const HINT_DELAY_MS = 900;
const NO_HANDS_HINT_S = 10;
const LANE_LIMIT = 14;
const STATUS_TEXT: Record<EngineState, string> = { off: "Camera off", idle: "Ready", signing: "Signing", reading: "Reading" };

export class StageView {
  readonly stage = byId<HTMLElement>("stage");
  readonly start = byId<HTMLButtonElement>("start");
  private readonly pill = byId<HTMLElement>("status-pill");
  private readonly pillLabel = byId<HTMLElement>("status-label");
  private readonly hintList = byId<HTMLUListElement>("hints");
  private readonly guide = byId<HTMLElement>("framing-guide");
  private readonly guideText = byId<HTMLElement>("framing-text");
  private readonly landing = byId<HTMLElement>("landing");
  private readonly landingWord = byId<HTMLElement>("landing-word");
  private readonly landingArc = document.getElementById("landing-arc") as unknown as SVGCircleElement;
  private readonly landingConfidence = byId<HTMLElement>("landing-confidence");
  private readonly unsureBox = byId<HTMLElement>("unsure");
  private readonly unsureChoices = byId<HTMLOListElement>("unsure-choices");
  private readonly countdownEl = byId<HTMLElement>("countdown");
  private readonly burstEl = byId<HTMLElement>("burst");
  private readonly timeline = byId<HTMLElement>("timeline");
  private readonly lanes: Record<Hand, HTMLOListElement> = { left: byId("lane-left"), right: byId("lane-right") };
  private readonly stateBox = byId<HTMLElement>("stage-state");
  private readonly stateEyebrow = byId<HTMLElement>("state-eyebrow");
  private readonly stateTitle = byId<HTMLElement>("state-title");
  private readonly stateBody = byId<HTMLElement>("state-body");
  private readonly progress = byId<HTMLElement>("progress");
  private readonly progressBar = byId<HTMLElement>("progress-bar");
  private readonly progressDetail = byId<HTMLElement>("progress-detail");
  readonly secondary = byId<HTMLButtonElement>("state-secondary");
  private readonly handChips = new Map<Hand, HTMLElement>(
    (["left", "right"] as const).map((hand) => [hand, this.stage.querySelector<HTMLElement>(`.hand-chip[data-hand="${hand}"]`) as HTMLElement]),
  );
  private engineState: EngineState = "off";
  private flash: { label: string; kind: "word" | "unsure"; until: number } | null = null;
  private flashTimer = 0;
  private hintSince = new Map<string, number>();
  private shownHints = "";
  private framing: "off" | "watching" | "ok" = "off";
  private framingTimer = 0;
  private landingToken = 0;
  stateKind: StateKind = "loading";

  // -------------------------------------------------------------------------------------------
  // Camera state cards

  showState(kind: StateKind, detail = ""): void {
    this.stateKind = kind;
    const copy = STATE_COPY[kind];
    this.stateBox.dataset.kind = kind === "loading" ? "loading" : kind === "ready" ? "ready" : "error";
    this.stateEyebrow.textContent = copy.eyebrow;
    this.stateTitle.textContent = copy.title;
    if (kind === "error") this.stateBody.textContent = detail || "Reload the page to try again.";
    else this.stateBody.innerHTML = copy.body;
    this.start.hidden = copy.action === null || kind === "error";
    const label = this.start.querySelector("span");
    if (label && copy.action && kind !== "error") label.textContent = copy.action;
    this.secondary.hidden = kind !== "error";
    if (kind === "error") this.secondary.textContent = "Reload";
  }

  setProgress(fraction: number, detail: string): void {
    this.progressBar.style.transform = `scaleX(${fraction.toFixed(3)})`;
    this.progress.setAttribute("aria-valuenow", String(Math.round(fraction * 100)));
    this.progressDetail.textContent = detail ? `${detail} ${Math.round(fraction * 100)}%` : "";
  }

  setRunning(running: boolean): void {
    this.stage.dataset.running = String(running);
    if (!running) {
      this.setHands(false, false);
      this.renderHints([]);
      this.hideUnsure();
      this.stopFraming();
    }
  }

  // -------------------------------------------------------------------------------------------
  // Status pill: READY -> SIGNING (pulsing ring) -> READING (spinner) -> the word

  setEngineState(state: EngineState): void {
    this.engineState = state;
    this.stage.dataset.state = state;
    this.renderPill();
  }

  /** Show a recognised word (or "Not sure") in the pill for a moment. */
  flashPill(label: string, kind: "word" | "unsure", ms = 1600): void {
    this.flash = { label, kind, until: performance.now() + ms };
    clearTimeout(this.flashTimer);
    this.flashTimer = window.setTimeout(() => {
      this.flash = null;
      this.renderPill();
    }, ms);
    this.renderPill();
  }

  private renderPill(): void {
    const busy = this.engineState === "signing" || this.engineState === "reading";
    if (this.flash && !busy) {
      this.pill.dataset.state = this.flash.kind;
      this.pillLabel.textContent = this.flash.label;
      return;
    }
    this.pill.dataset.state = this.engineState;
    this.pillLabel.textContent = STATUS_TEXT[this.engineState];
  }

  setHands(left: boolean, right: boolean): void {
    for (const [hand, on] of [["left", left], ["right", right]] as const) {
      const chip = this.handChips.get(hand) as HTMLElement;
      // Written only on change: an attribute write every frame would restyle the stage each frame.
      if (chip.dataset.on !== String(on)) chip.dataset.on = String(on);
    }
  }

  // -------------------------------------------------------------------------------------------
  // Hints (debounced so they do not flicker)

  updateHints(messages: readonly GuidanceMessage[], secondsWithoutHands: number, now: number): void {
    const wanted: { key: string; text: string; kind: string }[] = [];
    for (const message of messages) {
      const text = HINT_TEXT[message.code];
      if (text) wanted.push({ key: message.code, text, kind: message.code === "low_light" || message.code === "hand_out_of_frame" ? "warn" : "info" });
    }
    if (secondsWithoutHands >= NO_HANDS_HINT_S) wanted.push({ key: "no_hands_10s", text: "Raise your hands into view to sign", kind: "info" });
    const keys = new Set(wanted.map((item) => item.key));
    for (const key of [...this.hintSince.keys()]) if (!keys.has(key)) this.hintSince.delete(key);
    for (const item of wanted) if (!this.hintSince.has(item.key)) this.hintSince.set(item.key, now);
    const visible = wanted.filter((item) => item.key === "no_hands_10s" || now - (this.hintSince.get(item.key) ?? now) >= HINT_DELAY_MS).slice(0, 2);
    this.renderHints(visible);
  }

  private renderHints(items: { key: string; text: string; kind: string }[]): void {
    const signature = items.map((item) => item.key).join(",");
    if (signature === this.shownHints) return;
    this.shownHints = signature;
    this.hintList.replaceChildren(...items.map((item) => h("li", { "data-kind": item.kind, text: item.text })));
  }

  // -------------------------------------------------------------------------------------------
  // Framing guide: shown at camera start, green once body and hands are in view, then fades.

  startFraming(): void {
    clearTimeout(this.framingTimer);
    this.framing = "watching";
    this.guide.hidden = false;
    this.guide.classList.remove("fading");
    this.guide.dataset.ok = "false";
    this.guideText.textContent = "Fit your upper body in the frame";
  }

  updateFraming(bodyInView: boolean, handInView: boolean): void {
    if (this.framing !== "watching") return;
    if (bodyInView && !handInView) this.guideText.textContent = "Now raise a hand";
    if (!bodyInView) this.guideText.textContent = "Fit your upper body in the frame";
    if (bodyInView && handInView) {
      this.framing = "ok";
      this.guide.dataset.ok = "true";
      this.guideText.textContent = "You're in frame";
      this.framingTimer = window.setTimeout(() => {
        this.guide.classList.add("fading");
        this.framingTimer = window.setTimeout(() => this.stopFraming(), 650);
      }, 900);
    }
  }

  get framingDone(): boolean {
    return this.framing === "ok" || this.framing === "off";
  }

  stopFraming(): void {
    clearTimeout(this.framingTimer);
    this.framing = "off";
    this.guide.hidden = true;
    this.guide.classList.remove("fading");
  }

  // -------------------------------------------------------------------------------------------
  // The recognition moment

  /**
   * The word rises into the centre of the stage inside its confidence ring (the arc fills to the
   * calibrated confidence), with the one bouncy spring of the interface. Resolves once it has
   * landed; `fly` then sends it into the transcript.
   */
  async land(word: string, confidence: number): Promise<number> {
    const token = ++this.landingToken;
    this.landingWord.textContent = word;
    // Long glosses (REFRIGERATOR, FRENCH-FRIES) shrink so they stay inside a phone's width.
    this.landingWord.style.setProperty("--fit", Math.min(1, 8 / Math.max(1, word.length)).toFixed(3));
    this.landingConfidence.textContent = percent(confidence);
    // Empty arc, then fill (stroke-dashoffset on a 220 px ring: the one non-transform paint).
    this.landingArc.style.transition = "none";
    this.landingArc.style.strokeDashoffset = "100";
    void this.landingArc.getBoundingClientRect();
    this.landingArc.style.transition = prefersReducedMotion() ? "none" : "stroke-dashoffset 620ms cubic-bezier(0.2, 0.8, 0.2, 1) 80ms";
    this.landingArc.style.strokeDashoffset = String(100 - Math.max(0, Math.min(1, confidence)) * 100);
    await spring(this.landing, { y: 0, scale: 1, opacity: 1 }, SPRINGS.land, { from: { y: 44, scale: 0.82, opacity: 0 } });
    return token;
  }

  /** Shared-element flight from the stage centre to `target` (a transcript word). */
  async fly(token: number, target: HTMLElement | null): Promise<void> {
    if (token !== this.landingToken) return;
    await new Promise((resolve) => setTimeout(resolve, 520));
    if (token !== this.landingToken) return;
    const source = this.landingWord.getBoundingClientRect();
    const visible = target && target.getClientRects().length > 0 && target.getBoundingClientRect().height > 0;
    if (!target || !visible || prefersReducedMotion()) {
      void spring(this.landing, { opacity: 0, scale: 0.96 }, SPRINGS.quick);
      if (target) target.dataset.arriving = "false";
      return;
    }
    const goal = target.getBoundingClientRect();
    const flying = h("p", { class: "flying-word", "aria-hidden": "true", text: this.landingWord.textContent ?? "" });
    // Start at the landed word's actual size (fitted, and smaller on phones), so the flight has no jump.
    flying.style.fontSize = getComputedStyle(this.landingWord).fontSize;
    document.body.append(flying);
    place(flying, { x: source.left, y: source.top, scale: 1, opacity: 1 });
    place(this.landing, { opacity: 0 });
    const scale = goal.height / source.height;
    await spring(flying, { x: goal.left, y: goal.top, scale }, SPRINGS.move);
    target.dataset.arriving = "false";
    await spring(flying, { opacity: 0 }, SPRINGS.quick);
    flying.remove();
  }

  /** Low confidence: a "Not sure" picker with the top-3; nothing is added until one is tapped. */
  showUnsure(choices: readonly Choice[], onPick: (index: number) => void): void {
    this.unsureChoices.replaceChildren(...choices.map((choice, index) => choiceButton(choice, index, false, () => onPick(index), glossOf(choice.label))));
    this.unsureBox.hidden = false;
    void spring(this.unsureBox, { y: 0, opacity: 1 }, SPRINGS.move, { from: { y: 16, opacity: 0 } });
  }

  hideUnsure(): void {
    if (this.unsureBox.hidden) return;
    this.unsureBox.hidden = true;
  }

  dismissLanding(): void {
    this.landingToken += 1;
    void spring(this.landing, { opacity: 0 }, SPRINGS.quick);
  }

  // -------------------------------------------------------------------------------------------
  // Practice: 3-2-1 and a small burst

  async countdown(from = 3, stepMs = 750): Promise<void> {
    this.countdownEl.hidden = false;
    for (let n = from; n >= 1; n -= 1) {
      const digit = h("span", { text: String(n) });
      this.countdownEl.replaceChildren(digit);
      void spring(digit, { scale: 1, opacity: 1 }, SPRINGS.move, { from: { scale: 1.5, opacity: 0 } });
      await new Promise((resolve) => setTimeout(resolve, stepMs));
    }
    const go = h("span", { class: "cue", text: "Sign now" });
    this.countdownEl.replaceChildren(go);
    await spring(go, { scale: 1, opacity: 1 }, SPRINGS.quick, { from: { scale: 0.85, opacity: 0 } });
    window.setTimeout(() => this.hideCountdown(), 450);
  }

  /** Clears the countdown (also when a result arrives before its cue has faded). */
  hideCountdown(): void {
    const cue = this.countdownEl.firstElementChild as HTMLElement | null;
    if (!cue || this.countdownEl.hidden) return;
    void spring(cue, { opacity: 0 }, SPRINGS.quick).then(() => {
      this.countdownEl.hidden = true;
      this.countdownEl.replaceChildren();
    });
  }

  burst(): void {
    if (prefersReducedMotion()) return;
    const colors = ["#7AA2FF", "#FF9E7A", "#F5D76E", "#5ED39B"];
    const pieces: HTMLElement[] = [];
    for (let i = 0; i < 22; i += 1) {
      const piece = h("i");
      piece.style.background = colors[i % colors.length] as string;
      this.burstEl.append(piece);
      pieces.push(piece);
      const angle = (i / 22) * Math.PI * 2 + Math.random() * 0.3;
      const distance = 110 + Math.random() * 120;
      const dx = Math.cos(angle) * distance;
      const dy = Math.sin(angle) * distance - 40;
      piece.animate(
        [
          { transform: "translate(0, 0) rotate(0deg) scale(1)", opacity: 1 },
          { transform: `translate(${dx}px, ${dy}px) rotate(${Math.round(Math.random() * 360)}deg) scale(0.9)`, opacity: 1, offset: 0.7 },
          { transform: `translate(${dx * 1.05}px, ${dy + 60}px) rotate(${Math.round(Math.random() * 540)}deg) scale(0.6)`, opacity: 0 },
        ],
        { duration: 1100 + Math.random() * 300, easing: "cubic-bezier(0.15, 0.7, 0.3, 1)", fill: "forwards" },
      );
    }
    window.setTimeout(() => pieces.forEach((piece) => piece.remove()), 1600);
  }

  // -------------------------------------------------------------------------------------------
  // Timeline: one chip per sign in the lane of the hand that signed it, width from its duration

  addSegment(hand: Hand, label: string, durationS: number, kind: "word" | "unsure" | "skipped"): void {
    const lane = this.lanes[hand];
    const chip = h("li", { class: "seg", "data-kind": kind, title: `${label}, ${durationS.toFixed(1)} s, ${hand} hand` }, label, h("span", { class: "mono", text: `${durationS.toFixed(1)}s` }));
    chip.style.minWidth = `${Math.round(44 + Math.min(3, durationS) * 26)}px`;
    lane.append(chip);
    while (lane.children.length > LANE_LIMIT) lane.firstElementChild?.remove();
    this.timeline.dataset.filled = "true";
  }

  clearTimeline(): void {
    this.lanes.left.replaceChildren();
    this.lanes.right.replaceChildren();
    this.timeline.dataset.filled = "false";
  }
}

/** A candidate chip: key hint, gloss, confidence bar. */
export function choiceButton(choice: Choice, index: number, current: boolean, onPick: () => void, gloss?: string): HTMLLIElement {
  const bar = h("i");
  bar.style.transform = `scaleX(${Math.max(0.02, Math.min(1, choice.probability)).toFixed(3)})`;
  const button = h(
    "button",
    { type: "button", class: "choice", "data-current": String(current), "aria-label": `${gloss ?? choice.label}, ${percent(choice.probability)}. Press ${index + 1}.` },
    h("kbd", { text: String(index + 1) }),
    h("span", { class: "choice-label", text: gloss ?? choice.label }),
    h("span", { class: "choice-pct mono", text: percent(choice.probability) }),
    h("span", { class: "choice-bar" }, bar),
  );
  button.addEventListener("click", onPick);
  return h("li", {}, button);
}
