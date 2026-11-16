// The Signs library: all 250 supported signs, searchable, each with a short reminder of the
// handshape and motion, a link to a dictionary video (no video is copied) and "Practise".

import { byId, h, icon } from "./dom";
import { dictionaryUrl, searchSigns, SIGNS, type SignInfo } from "./signs";

export class SignsView {
  private readonly input = byId<HTMLInputElement>("sign-search");
  private readonly list = byId<HTMLUListElement>("sign-list");
  private readonly count = byId<HTMLElement>("signs-count");
  private rendered = false;

  constructor(private readonly onPractice: (label: string) => void) {
    this.input.addEventListener("input", () => this.render());
  }

  /** Renders on first show (250 rows are cheap, but not needed before). */
  show(): void {
    if (!this.rendered) this.render();
  }

  focusSearch(): void {
    this.input.focus();
  }

  private item(sign: SignInfo): HTMLLIElement {
    const practise = h("button", { type: "button", class: "button small", "aria-label": `Practise ${sign.gloss}` }, "Practise");
    practise.addEventListener("click", () => this.onPractice(sign.label));
    const watch = h(
      "a",
      { href: dictionaryUrl(sign.label), target: "_blank", rel: "noopener noreferrer", "aria-label": `Watch ${sign.gloss} on SignASL.org (opens in a new tab)`, title: "Watch on SignASL.org" },
      icon("external"),
    );
    return h(
      "li",
      { class: "sign-item" },
      h("p", { class: "sign-name", text: sign.gloss }),
      h("div", { class: "sign-actions" }, practise, watch),
      h("p", { class: "sign-desc", text: sign.description }),
    );
  }

  render(): void {
    this.rendered = true;
    const matches = searchSigns(this.input.value);
    this.count.textContent = matches.length === SIGNS.length ? `${SIGNS.length} signs` : `${matches.length} of ${SIGNS.length}`;
    if (matches.length === 0) {
      this.list.replaceChildren(h("li", { class: "muted", text: `No sign matches “${this.input.value}”. The app knows 250 signs; try another word.` }));
      return;
    }
    this.list.replaceChildren(...matches.map((sign) => this.item(sign)));
  }
}
