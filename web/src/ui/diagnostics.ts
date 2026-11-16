// The diagnostics drawer (fps, model latency, hands, lighting, backend, aspect, benchmark, Record
// test signs) and the advanced recognition settings form. Both talk to the engine only.

import type { Engine, SegmentResult } from "../engine";
import { RecordingStore, filterLabels, recordingsFileName } from "../recorder";
import { downloadJson } from "../benchmark";
import { DEFAULT_SETTINGS, NUMBER_RULES, sanitizeSettings, saveSettings, type KeyValueStore, type Settings } from "../settings";
import { byId, h, percent } from "./dom";
import { glossOf } from "./signs";

const NUMBER_FIELDS = Object.keys(NUMBER_RULES) as (keyof typeof NUMBER_RULES)[];
const REFRESH_MS = 500;
const RECORDER_LIST_LENGTH = 10;
const CLEAR_CONFIRM_MS = 4000;

function format(value: number, digits = 1): string {
  return Number.isFinite(value) ? value.toFixed(digits) : "–";
}

export class Diagnostics {
  private readonly drawer = byId<HTMLElement>("diagnostics");
  readonly toggle = byId<HTMLButtonElement>("diag-toggle");
  private readonly toggleFps = byId<HTMLElement>("diag-fps");
  private readonly fps = byId<HTMLElement>("fps");
  private readonly latency = byId<HTMLElement>("diag-latency");
  private readonly hands = byId<HTMLElement>("diag-hands");
  private readonly light = byId<HTMLElement>("diag-light");
  private readonly backend = byId<HTMLElement>("diag-backend");
  private readonly mediapipe = byId<HTMLElement>("diag-mediapipe");
  private readonly cameraInfo = byId<HTMLElement>("diag-camera");
  private readonly aspect = byId<HTMLElement>("diag-aspect");
  private readonly wrists = byId<HTMLElement>("diag-wrists");
  private readonly calibration = byId<HTMLElement>("calibration-status");
  private readonly segmentText = byId<HTMLElement>("segment-debug-text");
  readonly recalibrate = byId<HTMLButtonElement>("recalibrate");
  readonly benchmark = byId<HTMLButtonElement>("benchmark");
  private readonly benchmarkStatus = byId<HTMLElement>("benchmark-status");
  private readonly recorderOn = byId<HTMLInputElement>("recorder-on");
  private readonly labelSearch = byId<HTMLInputElement>("label-search");
  private readonly labelSelect = byId<HTMLSelectElement>("label-select");
  private readonly recorderStatus = byId<HTMLElement>("recorder-status");
  private readonly recorderDownload = byId<HTMLButtonElement>("recorder-download");
  private readonly recorderClear = byId<HTMLButtonElement>("recorder-clear");
  private readonly recorderList = byId<HTMLOListElement>("recorder-list");
  private readonly form = byId<HTMLFormElement>("settings");
  private readonly settingsNote = byId<HTMLElement>("settings-note");
  private readonly aspectKValue = byId<HTMLOutputElement>("aspectK-value");
  private lastRefresh = 0;
  private handsDetected = 0;
  private clearArmedUntil = 0;

  constructor(
    private readonly engine: Engine,
    private readonly store: KeyValueStore | null,
  ) {
    this.toggle.addEventListener("click", () => this.setOpen(!this.open));
    byId<HTMLButtonElement>("diag-close").addEventListener("click", () => this.setOpen(false));
    this.drawer.addEventListener("keydown", (event) => {
      if (event.key === "Escape") this.setOpen(false);
    });
    this.recalibrate.addEventListener("click", () => engine.recalibrate());
    this.benchmark.addEventListener("click", () => {
      if (engine.startBenchmark()) this.benchmark.disabled = true;
    });
    engine.on("calibration", (text) => {
      if (this.calibration.textContent !== text) this.calibration.textContent = text;
      this.recalibrate.disabled = !engine.running || engine.settings.aspectMode !== "auto";
    });
    engine.on("benchmark", (text) => {
      this.benchmarkStatus.textContent = text;
      this.benchmark.disabled = !engine.running || engine.benchmark.isRunning;
    });
    engine.on("recorder", (text) => this.renderRecorder(text));
    this.bindSettings();
    this.bindRecorder();
    this.renderSettings();
    this.renderRecorder();
  }

  get open(): boolean {
    return !this.drawer.hidden;
  }

  setOpen(open: boolean): void {
    this.drawer.hidden = !open;
    this.toggle.setAttribute("aria-expanded", String(open));
    if (open) {
      this.refresh(performance.now(), true);
      (this.drawer.querySelector("button") as HTMLButtonElement | null)?.focus();
    } else {
      this.toggle.focus();
    }
  }

  setRunning(running: boolean): void {
    this.benchmark.disabled = !running || this.engine.benchmark.isRunning;
    this.recalibrate.disabled = !running || this.engine.settings.aspectMode !== "auto";
    if (running) this.benchmarkStatus.textContent = "Ready. Sign a few times during the run so whole segments are timed.";
  }

  onFrame(handsDetected: number, now: number): void {
    this.handsDetected = handsDetected;
    this.refresh(now);
  }

  onSegment(result: SegmentResult, frames: string): void {
    const outcome =
      result.kind === "classified"
        ? result.decision?.accepted
          ? `recognised ${glossOf(result.decision.top[0]?.label ?? "")}`
          : "below the confidence threshold"
        : result.kind === "excluded"
          ? `not classified (${result.excluded})`
          : "discarded (too few frames with a hand)";
    this.segmentText.textContent = `Last segment: ${frames}; ${((result.end - result.start) / 1000).toFixed(2)} s, ${result.dominant ?? "no"} hand; ${outcome}.`;
  }

  /** The fps chip on the stage updates always; the drawer only while it is open. */
  refresh(now: number, force = false): void {
    if (!force && now - this.lastRefresh < REFRESH_MS) return;
    this.lastRefresh = now;
    const perf = this.engine.perf(now);
    this.toggleFps.textContent = this.engine.running ? `${format(perf.fps, 0)} fps` : "– fps";
    if (this.drawer.hidden) return;
    this.fps.textContent = this.engine.running
      ? `${format(perf.fps)} fps · hands ${format(perf.hand.p50)} ms · pose ${format(perf.pose.p50)} ms (p50)`
      : "camera off";
    this.latency.textContent = perf.classifier.n ? `${format(perf.classifier.p50)} ms p50 · ${format(perf.classifier.p95)} ms p95 per sign` : "no sign classified yet";
    this.hands.textContent = this.engine.running ? String(this.handsDetected) : "–";
    const low = this.engine.settings.lowLightThreshold;
    this.light.textContent =
      perf.luminance === null ? "–" : perf.luminance < low ? `Too dark (${perf.luminance.toFixed(0)} / 255)` : `OK (${perf.luminance.toFixed(0)} / 255)`;
    const classifier = perf.classifier_backend;
    this.backend.textContent = classifier
      ? `${classifier.backend === "webgpu" ? "WebGPU" : `WebAssembly × ${classifier.threads}`} (${classifier.reason})`
      : "loading";
    const tracker = perf.tracker;
    const poseEvery = `pose every ${perf.poseEvery === 1 ? "frame" : `${perf.poseEvery} frames`}`;
    this.mediapipe.textContent = tracker
      ? `${tracker.delegate}${tracker.fallbackReason ? ` (${tracker.fallbackReason})` : ""}; ${poseEvery}`
      : `${perf.delegateRequested} requested; ${poseEvery}`;
    this.cameraInfo.textContent = perf.camera
      ? `${perf.camera.width} × ${perf.camera.height}${perf.camera.frameRate ? ` at ${perf.camera.frameRate.toFixed(0)} fps` : ""}${perf.framing === "portrait" ? ", portrait crop" : ""}`
      : "–";
    const rTrain = perf.aspect.rTrain;
    this.aspect.textContent = `k ${perf.aspect.k.toFixed(3)} (${perf.aspect.mode}); r_live ${perf.aspect.rLive === null ? "–" : perf.aspect.rLive.toFixed(3)}, r_train ${rTrain === null ? "–" : rTrain.toFixed(3)}`;
    this.wrists.textContent = perf.wrists.checked === 0 ? "no hand and pose seen together yet" : `${percent(perf.wrists.consistent / perf.wrists.checked)} of recent hands`;
  }

  // -------------------------------------------------------------------------------------------
  // Advanced settings form (src/settings.ts)

  private field<T extends HTMLElement>(name: string): T {
    return byId<T>(name);
  }

  renderSettings(): void {
    const settings = this.engine.settings;
    const meta = this.engine.meta;
    for (const name of NUMBER_FIELDS) {
      const input = document.getElementById(name) as HTMLInputElement | HTMLSelectElement | null;
      if (!input) continue;
      const value = settings[name];
      input.value = value === null ? "" : String(value);
      input.removeAttribute("aria-invalid");
    }
    this.field<HTMLInputElement>("minHandFrames").placeholder = meta ? String(meta.preprocessing.min_hand_frames) : "";
    this.field<HTMLInputElement>("confidenceThreshold").placeholder = meta ? this.engine.threshold().toFixed(3) : "";
    this.field<HTMLSelectElement>("framing").value = settings.framing;
    this.field<HTMLSelectElement>("delegate").value = settings.delegate;
    this.field<HTMLSelectElement>("classifierBackend").value = settings.classifierBackend;
    this.field<HTMLSelectElement>("autoSegmenter").value = settings.autoSegmenter;
    this.field<HTMLInputElement>("swapHands").checked = settings.swapHands;
    for (const radio of this.form.querySelectorAll<HTMLInputElement>('input[name="mode"]')) radio.checked = radio.value === settings.mode;
    for (const radio of this.form.querySelectorAll<HTMLInputElement>('input[name="aspectMode"]')) radio.checked = radio.value === settings.aspectMode;
    const slider = this.field<HTMLInputElement>("aspectK");
    slider.disabled = settings.aspectMode !== "manual";
    this.aspectKValue.textContent = settings.aspectK.toFixed(2);
  }

  private apply(next: Settings): void {
    this.engine.applySettings(next);
    const saved = saveSettings(this.store, next);
    if (!saved) this.settingsNote.textContent = "Settings could not be stored in this browser; they apply until the page is closed.";
    this.renderSettings();
  }

  private readNumber(name: keyof typeof NUMBER_RULES): number | null | undefined {
    const input = document.getElementById(name) as HTMLInputElement;
    const text = input.value.trim();
    const nullable = name === "minHandFrames" || name === "confidenceThreshold";
    if (text === "" && nullable) return null;
    const value = Number(text);
    const rule = NUMBER_RULES[name];
    const valid = text !== "" && Number.isFinite(value) && value >= rule.min && value <= rule.max && (!rule.integer || Number.isInteger(value));
    input.setAttribute("aria-invalid", valid ? "false" : "true");
    return valid ? value : undefined;
  }

  private bindSettings(): void {
    this.form.addEventListener("change", (event) => {
      const target = event.target as HTMLInputElement | HTMLSelectElement;
      const name = target.name as keyof Settings;
      const raw: Record<string, unknown> = { ...this.engine.settings };
      if ((NUMBER_FIELDS as string[]).includes(name)) {
        const value = this.readNumber(name as keyof typeof NUMBER_RULES);
        if (value === undefined) {
          this.settingsNote.textContent = `${target.labels?.[0]?.textContent ?? name}: value out of range; previous value kept.`;
          return;
        }
        raw[name] = value;
      } else if (target instanceof HTMLInputElement && target.type === "checkbox") {
        raw[name] = target.checked;
      } else {
        raw[name] = target.value;
      }
      const next = sanitizeSettings(raw);
      if (next.minShoulderDistance !== raw.minShoulderDistance || next.maxShoulderDistance !== raw.maxShoulderDistance) {
        this.settingsNote.textContent = '"Move closer" distance must be below "move back" distance; both were reset.';
      } else {
        this.settingsNote.textContent = "";
      }
      this.apply(next);
    });
    this.field<HTMLInputElement>("aspectK").addEventListener("input", (event) => {
      this.aspectKValue.textContent = Number((event.target as HTMLInputElement).value).toFixed(2);
    });
    this.form.addEventListener("submit", (event) => event.preventDefault());
    byId<HTMLButtonElement>("reset-settings").addEventListener("click", () => {
      this.apply({ ...DEFAULT_SETTINGS });
      this.settingsNote.textContent = "Recognition settings reset to defaults.";
    });
  }

  // -------------------------------------------------------------------------------------------
  // Record test signs

  private get recordings(): RecordingStore {
    return this.engine.recordings;
  }

  renderLabelOptions(): void {
    const meta = this.engine.meta;
    if (!meta) return;
    const matches = filterLabels(meta.labels, this.labelSearch.value);
    this.labelSelect.replaceChildren(
      ...matches.map(({ index, label }) => {
        const option = h("option", { value: String(index), text: label });
        option.selected = index === this.engine.intendedIndex;
        return option;
      }),
    );
  }

  renderRecorder(note = ""): void {
    const count = this.recordings.count;
    const summary = count === 0 ? "No recordings." : `${count} recording${count === 1 ? "" : "s"} in memory.`;
    this.recorderStatus.textContent = note ? `${note} ${summary}` : summary;
    this.recorderDownload.disabled = count === 0;
    this.recorderClear.disabled = count === 0;
    this.recorderClear.textContent = performance.now() < this.clearArmedUntil ? "Press again to clear" : "Clear recordings";
    this.recorderList.replaceChildren(
      ...this.recordings.recordings
        .slice(-RECORDER_LIST_LENGTH)
        .reverse()
        .map((recording) => {
          const outcome = recording.app_result;
          const predicted = outcome.prediction ? `predicted ${outcome.prediction} (${percent(outcome.max_probability ?? 0)})` : outcome.status;
          return h("li", { text: `#${recording.id + 1} ${recording.intended_label}: ${predicted}, ${recording.frames.t_ms.length} frames` });
        }),
    );
  }

  private bindRecorder(): void {
    this.recorderOn.addEventListener("change", () => {
      this.engine.recorderOn = this.recorderOn.checked;
      this.renderRecorder(this.recorderOn.checked ? (this.engine.intendedIndex === null ? "Pick the intended sign." : "") : "Recording paused.");
    });
    this.labelSearch.addEventListener("input", () => this.renderLabelOptions());
    this.labelSearch.addEventListener("keydown", (event) => {
      if (event.key === "ArrowDown" && this.labelSelect.options.length > 0) {
        event.preventDefault();
        if (this.labelSelect.selectedIndex < 0) this.labelSelect.selectedIndex = 0;
        this.labelSelect.dispatchEvent(new Event("change"));
        this.labelSelect.focus();
      }
    });
    this.labelSelect.addEventListener("change", () => {
      const value = this.labelSelect.value;
      this.engine.intendedIndex = value === "" ? null : Number(value);
      const meta = this.engine.meta;
      if (meta && this.engine.intendedIndex !== null) this.renderRecorder(`Intended sign: ${meta.labels[this.engine.intendedIndex]}.`);
    });
    this.recorderDownload.addEventListener("click", () => {
      const meta = this.engine.meta;
      if (!meta || this.recordings.count === 0) return;
      downloadJson(this.recordings.toFile(this.engine.appInfo(meta), navigator.userAgent), recordingsFileName());
      this.renderRecorder("Downloaded.");
    });
    this.recorderClear.addEventListener("click", () => {
      const now = performance.now();
      if (now >= this.clearArmedUntil) {
        this.clearArmedUntil = now + CLEAR_CONFIRM_MS;
        this.renderRecorder();
        window.setTimeout(() => this.renderRecorder(), CLEAR_CONFIRM_MS + 50);
        return;
      }
      this.clearArmedUntil = 0;
      this.recordings.clear();
      this.renderRecorder("Recordings cleared.");
    });
  }
}
