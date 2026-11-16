// The Translate panel: the transcript of glosses, the top-3 picker, the English sentence (with
// Copy and Speak), the controls and the sentence history.

import { byId, h } from "./dom";
import type { HistoryItem } from "./history";
import { glossOf } from "./signs";
import { choiceButton } from "./stage";
import type { Transcript } from "./transcript";
import type { Translation } from "./translate";

export class TranslateView {
  private readonly list = byId<HTMLOListElement>("transcript");
  private readonly empty = byId<HTMLElement>("transcript-empty");
  private readonly count = byId<HTMLElement>("word-count");
  private readonly candidates = byId<HTMLElement>("candidates");
  private readonly candidateList = byId<HTMLOListElement>("candidate-list");
  private readonly candidatesHint = byId<HTMLElement>("candidates-hint");
  private readonly card = byId<HTMLElement>("sentence-card");
  private readonly sentence = byId<HTMLElement>("sentence");
  private readonly source = byId<HTMLElement>("sentence-source");
  readonly copyButton = byId<HTMLButtonElement>("copy-sentence");
  readonly speakButton = byId<HTMLButtonElement>("speak-sentence");
  readonly translateButton = byId<HTMLButtonElement>("translate");
  readonly undoButton = byId<HTMLButtonElement>("undo");
  readonly clearButton = byId<HTMLButtonElement>("clear-transcript");
  private readonly history = byId<HTMLOListElement>("history");
  private readonly historyCount = byId<HTMLElement>("history-count");
  readonly historyClear = byId<HTMLButtonElement>("history-clear");
  current: Translation | null = null;

  /**
   * Renders the transcript; returns the element of the newest word (marked as arriving when
   * `arriving` is set, so the flying word can land on it).
   */
  render(transcript: Transcript, onPick: (index: number) => void, arriving = false): HTMLElement | null {
    const items = transcript.words.map((word, index) => {
      const item = h("li", { "data-picked": String(word.picked) }, glossOf(word.label));
      if (arriving && index === transcript.words.length - 1) item.dataset.arriving = "true";
      return item;
    });
    if (transcript.pending) items.push(h("li", { "data-pending": "true", "aria-label": "Waiting for you to pick a sign", text: "?" }));
    this.list.replaceChildren(...items);
    this.empty.hidden = transcript.words.length > 0 || transcript.pending !== null;
    const n = transcript.words.length;
    this.count.textContent = `${n} sign${n === 1 ? "" : "s"}`;
    const choices = transcript.choices;
    this.candidates.hidden = choices.length === 0;
    this.candidatesHint.textContent = transcript.pending ? "pick one to add it" : "tap to replace the last sign";
    const last = transcript.last;
    this.candidateList.replaceChildren(
      ...choices.map((choice, index) =>
        choiceButton(choice, index, !transcript.pending && last?.label === choice.label, () => onPick(index), glossOf(choice.label)),
      ),
    );
    this.translateButton.disabled = n === 0;
    this.undoButton.disabled = n === 0 && !transcript.pending;
    this.clearButton.disabled = n === 0 && !transcript.pending;
    return items.length && !transcript.pending ? (items[items.length - 1] as HTMLElement) : null;
  }

  setWorking(): void {
    this.card.dataset.state = "working";
    this.sentence.textContent = "Translating…";
    this.source.textContent = "";
  }

  setSentence(translation: Translation | null): void {
    this.current = translation && translation.text ? translation : null;
    this.card.classList.remove("shimmer");
    if (!this.current) {
      this.card.dataset.state = "empty";
      this.sentence.textContent = "Your sentence appears here after a short pause in signing.";
      this.source.textContent = "";
      this.copyButton.disabled = true;
      this.speakButton.disabled = true;
      return;
    }
    this.card.dataset.state = "done";
    this.sentence.textContent = this.current.text;
    this.source.textContent = this.current.source === "basic" ? "(basic)" : "";
    this.source.title = this.current.source === "basic" && this.current.reason ? `Joined on this device: ${this.current.reason}` : "";
    void this.card.offsetWidth;
    this.card.classList.add("shimmer");
    this.copyButton.disabled = false;
    this.speakButton.disabled = false;
  }

  renderHistory(items: readonly HistoryItem[]): void {
    this.historyCount.textContent = items.length ? String(items.length) : "";
    this.history.replaceChildren(
      ...items.map((item) =>
        h(
          "li",
          {},
          item.text,
          h("span", { class: "meta", text: `${item.glosses.join(" ")}${item.source === "basic" ? " · basic" : ""} · ${new Date(item.at).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })}` }),
        ),
      ),
    );
    this.historyClear.disabled = items.length === 0;
  }
}
