// Small DOM helpers.

export function byId<T extends HTMLElement = HTMLElement>(id: string): T {
  const found = document.getElementById(id);
  if (!found) throw new Error(`missing #${id}`);
  return found as T;
}

type Child = Node | string | null | undefined | false;

/** Create an element with attributes (strings; `true` sets an empty attribute) and children. */
export function h<K extends keyof HTMLElementTagNameMap>(
  tag: K,
  attrs: Record<string, string | number | boolean | undefined> = {},
  ...children: Child[]
): HTMLElementTagNameMap[K] {
  const el = document.createElement(tag);
  for (const [name, value] of Object.entries(attrs)) {
    if (value === undefined || value === false) continue;
    if (name === "class") el.className = String(value);
    else if (name === "text") el.textContent = String(value);
    else el.setAttribute(name, value === true ? "" : String(value));
  }
  for (const child of children) {
    if (child === null || child === undefined || child === false) continue;
    el.append(typeof child === "string" ? document.createTextNode(child) : child);
  }
  return el;
}

/** An inline SVG icon from the sprite in index.html. */
export function icon(name: string): SVGSVGElement {
  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("aria-hidden", "true");
  const use = document.createElementNS("http://www.w3.org/2000/svg", "use");
  use.setAttribute("href", `#i-${name}`);
  svg.append(use);
  return svg;
}

export function percent(value: number): string {
  return `${Math.round(value * 100)}%`;
}

/** True when keyboard shortcuts must leave the event alone (typing, or a focused control). */
export function isTypingTarget(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  return target.closest("input, select, textarea, [contenteditable], dialog") !== null;
}

export function isControl(target: EventTarget | null): boolean {
  if (!(target instanceof Element)) return false;
  return target.closest("button, a, summary, [role='button'], [role='tab'], input, select, textarea") !== null;
}
