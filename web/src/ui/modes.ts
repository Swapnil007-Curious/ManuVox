// The mode switch in the top bar: a segmented control (ARIA tabs) whose indicator slides between
// the tabs on a critically damped spring. Arrow keys, Home and End move between tabs.

import { place, spring, SPRINGS } from "./motion";

export type Mode = "translate" | "practice" | "signs";

export class ModeTabs {
  private readonly tabs: HTMLButtonElement[];
  private readonly indicator: HTMLElement;
  current: Mode = "translate";
  private placed = false;

  constructor(
    root: HTMLElement,
    private readonly onChange: (mode: Mode) => void,
  ) {
    this.tabs = [...root.querySelectorAll<HTMLButtonElement>('[role="tab"]')];
    this.indicator = root.querySelector(".modes-indicator") as HTMLElement;
    for (const tab of this.tabs) {
      tab.addEventListener("click", () => this.select(tab.dataset.mode as Mode, true));
      tab.addEventListener("keydown", (event) => this.onKey(event, tab));
    }
    new ResizeObserver(() => this.moveIndicator(false)).observe(root);
  }

  select(mode: Mode, focus = false): void {
    const changed = mode !== this.current;
    this.current = mode;
    for (const tab of this.tabs) {
      const selected = tab.dataset.mode === mode;
      tab.setAttribute("aria-selected", String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && focus) tab.focus();
    }
    this.moveIndicator(true);
    if (changed) this.onChange(mode);
  }

  private moveIndicator(animate: boolean): void {
    const tab = this.tabs.find((candidate) => candidate.dataset.mode === this.current);
    if (!tab) return;
    this.indicator.style.width = `${tab.offsetWidth}px`;
    const x = tab.offsetLeft;
    if (!animate || !this.placed) {
      place(this.indicator, { x, y: 0, scale: 1, opacity: 1 });
      this.placed = true;
      return;
    }
    void spring(this.indicator, { x }, SPRINGS.move);
  }

  private onKey(event: KeyboardEvent, tab: HTMLButtonElement): void {
    const index = this.tabs.indexOf(tab);
    let next = -1;
    if (event.key === "ArrowRight") next = (index + 1) % this.tabs.length;
    else if (event.key === "ArrowLeft") next = (index - 1 + this.tabs.length) % this.tabs.length;
    else if (event.key === "Home") next = 0;
    else if (event.key === "End") next = this.tabs.length - 1;
    if (next < 0) return;
    event.preventDefault();
    this.select(this.tabs[next]?.dataset.mode as Mode, true);
  }
}
