// ManuVox "Duet": the app shell. The recognition pipeline lives in src/engine.ts; this file
// wires its events to the interface (stage, transcript, practice, signs library, diagnostics),
// sentence translation and speech, keyboard shortcuts, onboarding and the service worker.

import "./styles.css";

import { debugEnabled, installDebugHook } from "./debug";
import { Engine, type SegmentResult } from "./engine";
import { loadSettings, browserStorage, saveSettings } from "./settings";
import { openTeachStore, type TeachStore } from "./teachStore";
import { Diagnostics } from "./ui/diagnostics";
import { byId, isControl, isTypingTarget } from "./ui/dom";
import { loadHistory, pushHistory, saveHistory, type HistoryItem } from "./ui/history";
import { Hud } from "./ui/hud";
import { ModeTabs, type Mode } from "./ui/modes";
import { prefersReducedMotion } from "./ui/motion";
import { Onboarding } from "./ui/onboarding";
import { PracticeView } from "./ui/practiceView";
import { AUTO_TRANSLATE_MS, loadPrefs, savePrefs, type Prefs, type Theme } from "./ui/prefs";
import { BottomSheet } from "./ui/sheet";
import { SIGNS, glossOf } from "./ui/signs";
import { SignsView } from "./ui/signsView";
import { Speech } from "./ui/speech";
import { StageView } from "./ui/stage";
import { TeachView } from "./ui/teachView";
import { Transcript, type Choice } from "./ui/transcript";
import { TranslateView } from "./ui/translateView";
import { translate, type Translation } from "./ui/translate";

declare const __WORKER_URL__: string;
/** The shipped model's directory under BASE (web/model-variant.ts; "model" for v2-kaggle). */
declare const __MODEL_DIR__: string;

const BASE = import.meta.env.BASE_URL;
const DEBUG = debugEnabled();
// The model directory under BASE; a debug page may load another one (the live-gap harness).
const MODEL_DIR = (DEBUG && new URLSearchParams(location.search).get("model")) || __MODEL_DIR__;

const storage = browserStorage();
let prefs: Prefs = loadPrefs(storage);
applyTheme(prefs.theme);

const app = byId("app");
const video = byId<HTMLVideoElement>("video");
const status = byId("status");
const announcer = byId("announcer");
const cameraToggle = byId<HTMLButtonElement>("camera-toggle");
const cameraToggleLabel = byId("camera-toggle-label");

const engine = new Engine({ video, settings: loadSettings(storage), storage, modelDir: MODEL_DIR, logSegments: DEBUG });
const stage = new StageView();
const hud = new Hud(byId<HTMLCanvasElement>("hud"), video);
hud.setEnabled(prefs.hud);
hud.setTrails(!prefersReducedMotion());
const transcript = new Transcript();
const translateView = new TranslateView();
const diagnostics = new Diagnostics(engine, storage);
const speech = new Speech();
const sheet = new BottomSheet(byId("panel"), byId("sheet-handle"));
let history: HistoryItem[] = loadHistory(storage);
let translatedVersion = -1;
let translateTimer = 0;
let translating: AbortController | null = null;
let sentence: Translation | null = null;
let startingCamera: Promise<boolean> | null = null;

function announce(text: string): void {
  announcer.textContent = "";
  window.setTimeout(() => {
    announcer.textContent = text;
  }, 30);
}

function applyTheme(theme: Theme): void {
  if (theme === "system") delete document.documentElement.dataset.theme;
  else document.documentElement.dataset.theme = theme;
}

function persistPrefs(next: Prefs): void {
  prefs = next;
  savePrefs(storage, prefs);
}

// ---------------------------------------------------------------------------------------------
// Modes

const practice = new PracticeView(storage, {
  ensureCamera: () => startCamera(),
  countdown: () => stage.countdown(),
  burst: () => stage.burst(),
  announce,
});
const teach = new TeachView({
  ensureCamera: () => startCamera(),
  countdown: () => stage.countdown(),
  burst: () => stage.burst(),
  announce,
  model: () =>
    engine.meta
      ? { sha256: engine.meta.model_sha256, canTeach: engine.canTeach(), examplesPerSign: engine.meta.personalization?.examples_per_sign ?? 3 }
      : null,
  apply: (examples) => engine.setTaughtExamples(examples),
  personalizationOn: () => engine.settings.personalization,
  setPersonalization: (on) => {
    const next = { ...engine.settings, personalization: on };
    engine.applySettings(next);
    saveSettings(storage, next);
  },
  confirm: (text) => window.confirm(text),
});
let teachStore: TeachStore | null = null;
void openTeachStore().then((store) => {
  teachStore = store;
  return teach.attach(store);
});
const signsView = new SignsView((label) => {
  practice.setTarget(label);
  modes.select("practice");
  sheet.setExpanded(true);
});
const modes = new ModeTabs(document.querySelector(".modes") as HTMLElement, (mode) => setMode(mode));

function setMode(mode: Mode): void {
  app.dataset.mode = mode;
  for (const name of ["translate", "practice", "teach", "signs"] as const) {
    byId(`view-${name}`).hidden = name !== mode;
  }
  if (mode !== "practice") practice.cancel();
  if (mode !== "teach") teach.cancel();
  if (mode === "signs") signsView.show();
}

function openSignList(): void {
  modes.select("signs");
  sheet.setExpanded(true);
  signsView.focusSearch();
}
// The top bar's chip, and its copy on the phone's bottom sheet.
for (const id of ["vocab-chip", "sheet-vocab-chip"]) byId<HTMLButtonElement>(id).addEventListener("click", openSignList);

// ---------------------------------------------------------------------------------------------
// Camera

function setCameraUi(running: boolean): void {
  stage.setRunning(running);
  diagnostics.setRunning(running);
  cameraToggle.disabled = !engine.meta;
  cameraToggleLabel.textContent = running ? "Stop camera" : "Start camera";
  cameraToggle.querySelector("use")?.setAttribute("href", running ? "#i-stop" : "#i-camera");
  stage.start.disabled = !engine.meta;
  renderRecordButton();
  if (running) {
    hud.resize();
    stage.startFraming();
  } else {
    hud.clear();
    if (stage.stateKind === "loading" && engine.meta) return;
    if (!["denied", "no_camera", "busy", "insecure", "unsupported", "error"].includes(stage.stateKind)) stage.showState("ready");
  }
}

async function startCamera(): Promise<boolean> {
  if (engine.running) return true;
  if (startingCamera) return startingCamera;
  if (!engine.meta) return false;
  stage.start.disabled = true;
  cameraToggle.disabled = true;
  const label = stage.start.querySelector("span");
  if (label) label.textContent = "Starting…";
  startingCamera = engine.start().finally(() => {
    startingCamera = null;
  });
  const ok = await startingCamera;
  setCameraUi(ok);
  if (!ok && label && stage.stateKind === "ready") label.textContent = "Start camera";
  return ok;
}

function stopCamera(): void {
  engine.stop();
  setCameraUi(false);
}

function toggleCamera(): void {
  if (engine.running) stopCamera();
  else if (!prefs.onboarded && !onboarding.open) onboarding.show(1);
  else void startCamera();
}

stage.start.addEventListener("click", () => {
  if (!prefs.onboarded && !onboarding.open) onboarding.show(1);
  else void startCamera();
});
stage.secondary.addEventListener("click", () => location.reload());

// Hold-to-record (Settings > Advanced > Capture): press and hold the button, or hold Enter on it.
const recordButton = byId<HTMLButtonElement>("record");
function setRecording(on: boolean): void {
  if (engine.settings.mode !== "manual") return;
  engine.setManualRecording(on);
  recordButton.setAttribute("aria-pressed", String(on));
  recordButton.textContent = on ? "Recording… release to recognise" : "Hold to record a sign";
}
function renderRecordButton(): void {
  recordButton.hidden = engine.settings.mode !== "manual" || !engine.running;
}
recordButton.addEventListener("pointerdown", (event) => {
  recordButton.setPointerCapture(event.pointerId);
  setRecording(true);
});
for (const type of ["pointerup", "pointercancel", "lostpointercapture"] as const) recordButton.addEventListener(type, () => setRecording(false));
recordButton.addEventListener("click", (event) => event.preventDefault());
recordButton.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.repeat) {
    event.preventDefault();
    setRecording(true);
  }
});
recordButton.addEventListener("keyup", (event) => {
  if (event.key === "Enter") setRecording(false);
});
byId<HTMLFormElement>("settings").addEventListener("change", () => window.setTimeout(renderRecordButton, 0));
cameraToggle.addEventListener("click", () => toggleCamera());
document.addEventListener("visibilitychange", () => {
  if (document.hidden && engine.running) stopCamera();
});

// ---------------------------------------------------------------------------------------------
// Results: the transcript, the word landing, the "Not sure" picker, the timeline

function choicesOf(result: SegmentResult): Choice[] {
  return (result.decision?.top ?? []).slice(0, 3).map((candidate) => ({ label: candidate.label, probability: candidate.probability }));
}

function renderTranscript(arriving = false): HTMLElement | null {
  return translateView.render(transcript, (index) => pick(index), arriving);
}

function pick(index: number): void {
  const change = transcript.pick(index);
  if (change === "none") return;
  stage.hideUnsure();
  const word = transcript.last ? glossOf(transcript.last.label) : "";
  renderTranscript();
  stage.flashPill(word, "word");
  announce(change === "added" ? `Added ${word}.` : `Replaced with ${word}.`);
  scheduleTranslate();
}

function undo(): void {
  const change = transcript.undo();
  if (change === "none") return;
  stage.hideUnsure();
  stage.dismissLanding();
  renderTranscript();
  announce(change === "cleared" ? "Candidates dismissed." : "Removed the last sign.");
  scheduleTranslate();
}

function clearTranscript(): void {
  if (transcript.clear() === "none") return;
  stage.hideUnsure();
  stage.clearTimeline();
  renderTranscript();
  setSentence(null);
  translatedVersion = transcript.version;
  announce("Transcript cleared.");
}

function onResult(result: SegmentResult): void {
  stage.hideCountdown();
  diagnostics.onSegment(result, `${result.frames} frames, ${result.handFrames} with a hand`);
  const hand = result.dominant ?? "right";
  const duration = Math.max(0, (result.end - result.start) / 1000);
  const decision = result.decision;
  if (result.kind === "excluded") stage.addSegment(hand, "skipped", duration, "skipped");
  if (result.kind === "classified" && decision) {
    const top = decision.top[0];
    stage.addSegment(hand, `${glossOf(top?.label ?? "")}${decision.accepted ? "" : "?"}`, duration, decision.accepted ? "word" : "unsure");
  }
  if (teach.handleResult(result)) return;
  if (practice.handleResult(result)) return;
  if (result.kind !== "classified" || !decision) return;
  const choices = choicesOf(result);
  const gloss = glossOf(choices[0]?.label ?? "");
  const inTranslate = modes.current === "translate";
  const personalized = decision.personalized === true;
  const forYou = personalized ? ", personalized for you" : "";
  if (decision.accepted) {
    stage.hideUnsure();
    stage.flashPill(gloss, "word");
    onboarding.recognised(gloss);
    let target: HTMLElement | null = null;
    if (inTranslate) {
      transcript.add(choices, true, personalized);
      target = renderTranscript(true);
      scheduleTranslate();
    }
    announce(`${gloss}, ${Math.round(decision.probability * 100)} percent${forYou}.`);
    void stage.land(gloss, decision.probability).then((token) => stage.fly(token, inTranslate ? target : null));
  } else {
    stage.flashPill("Not sure", "unsure");
    if (inTranslate) {
      transcript.add(choices, false, personalized);
      renderTranscript();
      stage.showUnsure(choices, (index) => pick(index));
    }
    announce(`Not sure${forYou}. Candidates: ${choices.map((choice) => glossOf(choice.label)).join(", ")}.`);
  }
}

// ---------------------------------------------------------------------------------------------
// Sentences: translation after a pause, speech, history

function setSentence(translation: Translation | null): void {
  sentence = translation;
  translateView.setSentence(translation);
}

function scheduleTranslate(): void {
  window.clearTimeout(translateTimer);
  if (!prefs.autoTranslate) return;
  if (engine.state === "signing" || engine.state === "reading") return;
  if (transcript.words.length === 0 || transcript.version === translatedVersion || transcript.pending) return;
  translateTimer = window.setTimeout(() => void translateNow(), AUTO_TRANSLATE_MS);
}

async function translateNow(): Promise<void> {
  window.clearTimeout(translateTimer);
  const labels = transcript.labels;
  if (labels.length === 0) {
    setSentence(null);
    return;
  }
  translating?.abort();
  const controller = new AbortController();
  translating = controller;
  const version = transcript.version;
  translateView.setWorking();
  let result: Translation;
  try {
    result = await translate(labels, { workerUrl: __WORKER_URL__, online: navigator.onLine, signal: controller.signal });
  } catch {
    return; // Superseded by a newer request.
  }
  if (controller !== translating) return;
  translating = null;
  translatedVersion = version;
  setSentence(result);
  announce(result.text);
  history = pushHistory(history, { text: result.text, source: result.source, glosses: labels.map(glossOf), at: new Date().toISOString() });
  saveHistory(storage, history);
  translateView.renderHistory(history);
  if (prefs.autoSpeak) speech.speak(result.text, prefs.voice, prefs.rate);
}

function speakSentence(): void {
  const text = sentence?.text ?? "";
  if (!text) return;
  if (!speech.speak(text, prefs.voice, prefs.rate)) announce("No on-device voice is available to speak.");
}

translateView.translateButton.addEventListener("click", () => void translateNow());
translateView.undoButton.addEventListener("click", () => undo());
translateView.clearButton.addEventListener("click", () => clearTranscript());
translateView.speakButton.addEventListener("click", () => speakSentence());
translateView.copyButton.addEventListener("click", () => {
  const text = sentence?.text;
  if (!text) return;
  navigator.clipboard?.writeText(text).then(
    () => {
      const label = translateView.copyButton.querySelector("span");
      if (label) {
        label.textContent = "Copied";
        window.setTimeout(() => (label.textContent = "Copy"), 1400);
      }
      announce("Sentence copied.");
    },
    () => announce("Copy is not allowed here; select the sentence to copy it."),
  );
});
translateView.historyClear.addEventListener("click", () => {
  history = [];
  saveHistory(storage, history);
  translateView.renderHistory(history);
});

// ---------------------------------------------------------------------------------------------
// Engine events

engine.on("status", ({ text }) => {
  status.textContent = text;
});
engine.on("progress", ({ stage: step, fraction, detail }) => {
  stage.setProgress(fraction, detail);
  if (step === "ready" && !engine.running && stage.stateKind === "loading") stage.showState("ready");
});
engine.on("ready", (meta) => {
  byId("vocab-count").textContent = String(meta.num_classes);
  stage.start.disabled = false;
  cameraToggle.disabled = false;
  document.body.dataset.modelReady = "true";
  status.textContent = "Model ready. Press Start camera.";
  byId("model-info").textContent =
    `Model: ${meta.model_name} (${meta.variant}, ${meta.num_classes} signs, ${(meta.model_bytes / 1e6).toFixed(1)} MB), confidence threshold ` +
    `${engine.threshold().toFixed(2)} at temperature ${engine.temperature().toFixed(2)}, both set on validation data.`;
  diagnostics.renderSettings();
  diagnostics.renderLabelOptions();
  teach.modelReady();
  if (!prefs.onboarded) onboarding.show(1);
});
engine.on("state", (state) => {
  stage.setEngineState(state);
  // The camera can stop without the app's button (debug hook, a GPU fallback reload): follow it.
  if (state === "off" && stage.stage.dataset.running === "true") setCameraUi(false);
  if (state === "signing" || state === "reading") window.clearTimeout(translateTimer);
  else scheduleTranslate();
});
engine.on("guess", (guess) => {
  if (guess.accepted && guess.top[0]) stage.flashPill(`Signing · ${glossOf(guess.top[0].label)}?`, "word", 700);
});
engine.on("frame", (frame) => {
  hud.draw(frame);
  stage.setHands(frame.left !== null, frame.right !== null);
  const now = performance.now();
  stage.updateHints(frame.guidance, frame.secondsWithoutHands, now);
  stage.updateFraming(frame.bodyInView, frame.handInView);
  if (frame.bodyInView && frame.handInView) onboarding.framed();
  diagnostics.onFrame(frame.handsDetected, now);
});
engine.on("result", (result) => onResult(result));
engine.on("cameraError", ({ kind, message }) => {
  stage.showState(kind === "other" ? "error" : kind, message);
  setCameraUi(false);
});
engine.benchmarkExtras = () => ({ hud: prefs.hud });

// ---------------------------------------------------------------------------------------------
// Onboarding

const onboarding = new Onboarding({
  allowCamera: () => startCamera(),
  finish: () => persistPrefs({ ...prefs, onboarded: true }),
});

// ---------------------------------------------------------------------------------------------
// Settings and About dialogs

const settingsDialog = byId<HTMLDialogElement>("settings-dialog");
const aboutDialog = byId<HTMLDialogElement>("about-dialog");
const voiceSelect = byId<HTMLSelectElement>("pref-voice");
const rateInput = byId<HTMLInputElement>("pref-rate");
const rateValue = byId<HTMLOutputElement>("pref-rate-value");

function renderVoices(): void {
  const voices = speech.voices();
  voiceSelect.replaceChildren(
    ...(voices.length
      ? voices.map((voice) => {
          const option = document.createElement("option");
          option.value = voice.uri;
          option.textContent = `${voice.name} (${voice.lang})`;
          option.selected = voice.uri === prefs.voice;
          return option;
        })
      : [Object.assign(document.createElement("option"), { value: "", textContent: "No on-device voice available" })]),
  );
  voiceSelect.disabled = voices.length === 0;
}

function renderPrefs(): void {
  renderVoices();
  rateInput.value = String(prefs.rate);
  rateValue.textContent = `${prefs.rate.toFixed(1)}×`;
  byId<HTMLInputElement>("pref-autospeak").checked = prefs.autoSpeak;
  byId<HTMLInputElement>("pref-autotranslate").checked = prefs.autoTranslate;
  byId<HTMLInputElement>("pref-hud").checked = prefs.hud;
  for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="pref-theme"]')) radio.checked = radio.value === prefs.theme;
}

speech.onVoices(() => renderVoices());
byId("open-settings").addEventListener("click", () => {
  renderPrefs();
  diagnostics.renderSettings();
  settingsDialog.showModal();
});
for (const id of ["open-about", "sheet-about"]) byId(id).addEventListener("click", () => aboutDialog.showModal());
byId("about-close").addEventListener("click", () => aboutDialog.close());
for (const dialog of [settingsDialog, aboutDialog]) {
  dialog.addEventListener("click", (event) => {
    if (event.target === dialog) dialog.close();
  });
}
voiceSelect.addEventListener("change", () => persistPrefs({ ...prefs, voice: voiceSelect.value || null }));
rateInput.addEventListener("input", () => {
  const rate = Number(rateInput.value);
  rateValue.textContent = `${rate.toFixed(1)}×`;
  persistPrefs({ ...prefs, rate });
});
byId<HTMLInputElement>("pref-autospeak").addEventListener("change", (event) => persistPrefs({ ...prefs, autoSpeak: (event.target as HTMLInputElement).checked }));
byId<HTMLInputElement>("pref-autotranslate").addEventListener("change", (event) => {
  persistPrefs({ ...prefs, autoTranslate: (event.target as HTMLInputElement).checked });
  scheduleTranslate();
});
byId<HTMLInputElement>("pref-hud").addEventListener("change", (event) => {
  persistPrefs({ ...prefs, hud: (event.target as HTMLInputElement).checked });
  hud.setEnabled(prefs.hud);
});
for (const radio of document.querySelectorAll<HTMLInputElement>('input[name="pref-theme"]')) {
  radio.addEventListener("change", () => {
    if (!radio.checked) return;
    persistPrefs({ ...prefs, theme: radio.value as Theme });
    applyTheme(prefs.theme);
  });
}

// ---------------------------------------------------------------------------------------------
// Keyboard: Space start/stop, 1/2/3 pick, Backspace undo, Enter translate, S speak

document.addEventListener("keydown", (event) => {
  if (event.defaultPrevented || event.ctrlKey || event.metaKey || event.altKey) return;
  if (settingsDialog.open || aboutDialog.open || (onboarding.open && onboarding.step === 1)) return;
  const typing = isTypingTarget(event.target);
  if (event.code === "Space" && !typing && !isControl(event.target)) {
    event.preventDefault();
    if (!event.repeat) toggleCamera();
    return;
  }
  if (typing) return;
  if (["1", "2", "3"].includes(event.key) && modes.current === "translate") {
    event.preventDefault();
    pick(Number(event.key) - 1);
  } else if (event.key === "Backspace" && modes.current === "translate") {
    event.preventDefault();
    undo();
  } else if (event.key === "Enter" && !isControl(event.target) && modes.current === "translate") {
    event.preventDefault();
    void translateNow();
  } else if (event.key === "s" || event.key === "S") {
    event.preventDefault();
    speakSentence();
  }
});

// ---------------------------------------------------------------------------------------------
// Layout

new ResizeObserver(() => hud.resize()).observe(stage.stage);
matchMedia("(prefers-reduced-motion: reduce)").addEventListener("change", () => hud.setTrails(!prefersReducedMotion()));

// ---------------------------------------------------------------------------------------------
// Service worker (offline use and cross-origin isolation on GitHub Pages)

function updateOfflineStatus(): void {
  const controlled = "serviceWorker" in navigator && navigator.serviceWorker.controller !== null;
  byId("offline-status").textContent = controlled
    ? "Offline use: available. The app, model and tracking files are stored on this device."
    : "Offline use: not yet available (stored after the first complete load).";
}

const ISOLATION_RELOAD_KEY = "edge-vision.isolation-reload";

/**
 * The service worker adds the COOP/COEP headers that GitHub Pages cannot send, but a page loaded
 * before it took control is not cross-origin isolated. Reload such a page once per tab session
 * when the worker takes over, unless the camera is running; otherwise isolation starts next load.
 */
function reloadForIsolation(): void {
  if (self.crossOriginIsolated || engine.running || navigator.serviceWorker.controller === null) return;
  try {
    if (sessionStorage.getItem(ISOLATION_RELOAD_KEY)) return;
    sessionStorage.setItem(ISOLATION_RELOAD_KEY, "1");
  } catch {
    return;
  }
  location.reload();
}

function registerServiceWorker(): void {
  if (!import.meta.env.PROD || !("serviceWorker" in navigator)) {
    updateOfflineStatus();
    return;
  }
  navigator.serviceWorker.addEventListener("controllerchange", () => {
    updateOfflineStatus();
    reloadForIsolation();
  });
  navigator.serviceWorker.register(`${BASE}sw.js`, { scope: BASE }).then(
    () => updateOfflineStatus(),
    (error: unknown) => {
      byId("offline-status").textContent = `Offline use: unavailable (${error instanceof Error ? error.message : String(error)}).`;
    },
  );
}

// ---------------------------------------------------------------------------------------------
// Boot

/** The missing capability, or null when the browser can run the app. */
function unsupported(): string | null {
  if (typeof Worker === "undefined") return "Web Workers";
  if (typeof WebAssembly === "undefined") return "WebAssembly";
  const simd = new Uint8Array([0, 97, 115, 109, 1, 0, 0, 0, 1, 5, 1, 96, 0, 1, 123, 3, 2, 1, 0, 10, 10, 1, 8, 0, 65, 0, 253, 15, 253, 98, 11]);
  if (!WebAssembly.validate(simd)) return "WebAssembly SIMD";
  if (typeof createImageBitmap === "undefined" || typeof OffscreenCanvas === "undefined") return "OffscreenCanvas";
  return null;
}

if (DEBUG) {
  installDebugHook(engine.debugHook());
  (window as unknown as { __edgevisionUi?: object }).__edgevisionUi = {
    setMode: (mode: Mode) => modes.select(mode),
    practiceTarget: (label: string) => practice.setTarget(label),
    expandSheet: (on: boolean) => sheet.setExpanded(on),
    labels: () => SIGNS.map((sign) => sign.label),
    // Teach Dex, for the e2e tests and the record-then-test evaluation (tests/e2e/teach*.spec.ts).
    teachSelect: (label: string) => teach.select(label),
    teachRecord: () => teach.record(),
    teachPhase: () => teach.phase,
    teachListenMs: (ms: number) => {
      teach.listenMs = ms;
    },
    teachExamples: () => teach.examples.map((example) => ({ label: example.label, embedding: Array.from(example.embedding), modelSha256: example.modelSha256 })),
    teachStoreMode: () => teachStore?.mode ?? null,
  };
}

renderTranscript();
translateView.renderHistory(history);
setMode("translate");
modes.select("translate");
registerServiceWorker();
const missing = unsupported();
if (missing) {
  stage.showState("unsupported");
  status.textContent = `This browser lacks ${missing}.`;
} else {
  stage.showState("loading");
  engine.load().catch((error: unknown) => stage.showState("error", `Could not load the model: ${error instanceof Error ? error.message : String(error)}`));
}
