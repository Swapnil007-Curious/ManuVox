// The mobile bottom sheet: two detents (peek and expanded), dragged 1:1 from the handle with the
// grab offset respected, thrown with momentum (the release velocity projects the resting point,
// then hands over to a critically damped spring), rubber-banded past its ends, and toggled by a
// tap or Enter/Space on the handle. Inactive on wide screens, where the panel is a column.

import { project, rubberband, springValue, SPRINGS } from "./motion";

const MOBILE = "(max-width: 900px)";
const HISTORY_MS = 100;

export class BottomSheet {
  private readonly media = matchMedia(MOBILE);
  private offset = 0;
  private expandedState = false;
  private drag: { pointerId: number; startY: number; startOffset: number; moved: boolean; samples: { t: number; y: number }[] } | null = null;
  private readonly key = {};
  private listeners: ((expanded: boolean) => void)[] = [];

  constructor(
    private readonly panel: HTMLElement,
    private readonly handle: HTMLElement,
  ) {
    this.media.addEventListener("change", () => this.reset());
    window.addEventListener("resize", () => this.snapTo(this.expandedState, false));
    handle.addEventListener("pointerdown", (event) => this.onDown(event));
    handle.addEventListener("pointermove", (event) => this.onMove(event));
    handle.addEventListener("pointerup", (event) => this.onUp(event));
    handle.addEventListener("pointercancel", (event) => this.onUp(event));
    handle.addEventListener("keydown", (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        this.setExpanded(!this.expandedState);
      }
    });
    this.reset();
  }

  get active(): boolean {
    return this.media.matches;
  }

  get expanded(): boolean {
    return this.expandedState;
  }

  onChange(listener: (expanded: boolean) => void): void {
    this.listeners.push(listener);
  }

  private bounds(): { expanded: number; collapsed: number } {
    const height = this.panel.getBoundingClientRect().height || window.innerHeight * 0.88;
    const peek = parseFloat(getComputedStyle(document.documentElement).getPropertyValue("--sheet-peek")) || 292;
    return { expanded: 0, collapsed: Math.max(0, height - peek) };
  }

  private render(offset: number): void {
    this.offset = offset;
    this.panel.style.transform = `translate3d(0, ${offset.toFixed(1)}px, 0)`;
  }

  private reset(): void {
    if (!this.active) {
      this.panel.style.transform = "";
      this.panel.dataset.expanded = "true";
      return;
    }
    this.snapTo(this.expandedState, false);
  }

  setExpanded(expanded: boolean, animate = true): void {
    if (!this.active) return;
    this.snapTo(expanded, animate);
  }

  private snapTo(expanded: boolean, animate: boolean, velocity = 0): void {
    if (!this.active) return;
    const { expanded: top, collapsed } = this.bounds();
    const target = expanded ? top : collapsed;
    const changed = this.expandedState !== expanded;
    this.expandedState = expanded;
    this.panel.dataset.expanded = String(expanded);
    this.handle.setAttribute("aria-expanded", String(expanded));
    this.handle.setAttribute("aria-label", expanded ? "Collapse the panel" : "Expand the panel");
    if (animate) void springValue(this.key, this.offset, target, SPRINGS.sheet, (value) => this.render(value), velocity);
    else this.render(target);
    if (changed) for (const listener of this.listeners) listener(expanded);
  }

  private onDown(event: PointerEvent): void {
    if (!this.active) return;
    this.handle.setPointerCapture(event.pointerId);
    this.drag = { pointerId: event.pointerId, startY: event.clientY, startOffset: this.offset, moved: false, samples: [{ t: event.timeStamp, y: event.clientY }] };
    // Grabbing a moving sheet stops it where it is (interruptible).
    void springValue(this.key, this.offset, this.offset, SPRINGS.sheet, (value) => this.render(value));
  }

  private onMove(event: PointerEvent): void {
    const drag = this.drag;
    if (!drag || event.pointerId !== drag.pointerId) return;
    const dy = event.clientY - drag.startY;
    if (!drag.moved && Math.abs(dy) < 6) return;
    drag.moved = true;
    const { expanded: top, collapsed } = this.bounds();
    let next = drag.startOffset + dy;
    if (next < top) next = top - rubberband(top - next, collapsed || 1);
    if (next > collapsed) next = collapsed + rubberband(next - collapsed, collapsed || 1);
    this.render(next);
    drag.samples.push({ t: event.timeStamp, y: event.clientY });
    while (drag.samples.length > 2 && event.timeStamp - (drag.samples[0] as { t: number }).t > HISTORY_MS) drag.samples.shift();
  }

  private onUp(event: PointerEvent): void {
    const drag = this.drag;
    if (!drag || event.pointerId !== drag.pointerId) return;
    this.drag = null;
    if (!drag.moved) {
      this.setExpanded(!this.expandedState);
      return;
    }
    const first = drag.samples[0] as { t: number; y: number };
    const last = drag.samples[drag.samples.length - 1] as { t: number; y: number };
    const velocity = last.t > first.t ? ((last.y - first.y) / (last.t - first.t)) * 1000 : 0;
    const { expanded: top, collapsed } = this.bounds();
    const projected = this.offset + project(velocity);
    const expanded = Math.abs(projected - top) < Math.abs(projected - collapsed);
    this.snapTo(expanded, true, velocity);
  }
}
