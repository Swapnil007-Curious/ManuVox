// WCAG 2.x contrast of the Duet palette, read from src/styles.css so the stylesheet stays the
// single source of the colours. Glass surfaces are translucent: a pair over the camera video is
// checked against the surface composited over both a black and a white frame (the extremes the
// video can show), and must pass in both. Pure: tests/unit/contrast.test.ts and
// scripts/check-contrast.mjs (npm run contrast) call it.

export interface Rgba {
  r: number;
  g: number;
  b: number;
  a: number;
}

export interface Pair {
  /** Token holding the text colour. */
  fg: string;
  /** Token holding the background; may be translucent. */
  bg: string;
  /** What lies under a translucent background: the theme's base, or the camera video. */
  over: "base" | "video";
  /** 4.5 for body text, 3 for large text and non-text indicators. */
  min: number;
  use: string;
}

/**
 * Every text / background pair the interface uses, per theme. The hand colours themselves are
 * pinned by the design (periwinkle #7AA2FF reaches 2.8:1 on glass over a pure white frame), so on
 * glass they never carry meaning alone: every hand swatch and timeline lane has a text label in
 * --hud-text next to it, and text is never set in a hand colour on glass.
 */
export const PAIRS: Pair[] = [
  { fg: "--text", bg: "--bg", over: "base", min: 4.5, use: "body text" },
  { fg: "--muted", bg: "--bg", over: "base", min: 4.5, use: "secondary text" },
  { fg: "--text", bg: "--panel", over: "base", min: 4.5, use: "text in the side panel" },
  { fg: "--muted", bg: "--panel", over: "base", min: 4.5, use: "secondary text in the side panel" },
  { fg: "--text", bg: "--raised", over: "base", min: 4.5, use: "chip and button labels" },
  { fg: "--muted", bg: "--raised", over: "base", min: 4.5, use: "secondary text on chips" },
  { fg: "--accent-ink", bg: "--panel", over: "base", min: 4.5, use: "confirmed words in the transcript" },
  { fg: "--on-accent", bg: "--accent", over: "base", min: 4.5, use: "primary button label" },
  { fg: "--success-ink", bg: "--panel", over: "base", min: 4.5, use: "correct / success text" },
  { fg: "--warn-ink", bg: "--panel", over: "base", min: 4.5, use: "warnings and 'not sure'" },
  { fg: "--danger-ink", bg: "--panel", over: "base", min: 4.5, use: "errors" },
  { fg: "--left-ink", bg: "--panel", over: "base", min: 4.5, use: "left-hand labels" },
  { fg: "--right-ink", bg: "--panel", over: "base", min: 4.5, use: "right-hand labels" },
  { fg: "--focus", bg: "--bg", over: "base", min: 3, use: "focus ring" },
  { fg: "--hud-text", bg: "--hud-glass", over: "video", min: 4.5, use: "text on glass over the video" },
  { fg: "--hud-muted", bg: "--hud-glass", over: "video", min: 4.5, use: "secondary text on glass over the video" },
  { fg: "--hud-muted", bg: "--hud-glass-live", over: "video", min: 4.5, use: "secondary text on unblurred glass while the camera runs" },
  { fg: "--accent", bg: "--hud-glass", over: "video", min: 4.5, use: "accent text on glass over the video" },
  { fg: "--on-warn", bg: "--warn", over: "base", min: 4.5, use: "the solid 'Not sure' badge" },
  { fg: "--warn", bg: "--stage-bg", over: "base", min: 4.5, use: "error eyebrow on the stage's state card" },
  { fg: "--warn", bg: "--hud-glass", over: "video", min: 3, use: "'not sure' dots, bars and borders on glass (non-text)" },
  { fg: "--success", bg: "--hud-glass", over: "video", min: 3, use: "ready / framing-ok dots on glass (non-text)" },
  { fg: "--seg-ink", bg: "--left", over: "base", min: 4.5, use: "labels on left-hand timeline chips" },
  { fg: "--seg-ink", bg: "--right", over: "base", min: 4.5, use: "labels on right-hand timeline chips" },
];

export function parseColor(text: string): Rgba {
  const value = text.trim().toLowerCase();
  const hex = /^#([0-9a-f]{3}|[0-9a-f]{6})$/.exec(value);
  if (hex) {
    const digits = (hex[1] as string).length === 3 ? [...(hex[1] as string)].map((d) => d + d).join("") : (hex[1] as string);
    return { r: parseInt(digits.slice(0, 2), 16), g: parseInt(digits.slice(2, 4), 16), b: parseInt(digits.slice(4, 6), 16), a: 1 };
  }
  const rgba = /^rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)(?:[\s,/]+([\d.]+%?))?\s*\)$/.exec(value);
  if (rgba) {
    const alpha = rgba[4] === undefined ? 1 : (rgba[4] as string).endsWith("%") ? parseFloat(rgba[4] as string) / 100 : parseFloat(rgba[4] as string);
    return { r: Number(rgba[1]), g: Number(rgba[2]), b: Number(rgba[3]), a: alpha };
  }
  throw new Error(`unsupported colour: ${text}`);
}

/** `top` composited over an opaque `bottom`. */
export function composite(top: Rgba, bottom: Rgba): Rgba {
  const a = top.a;
  return { r: top.r * a + bottom.r * (1 - a), g: top.g * a + bottom.g * (1 - a), b: top.b * a + bottom.b * (1 - a), a: 1 };
}

function channel(value: number): number {
  const c = value / 255;
  return c <= 0.03928 ? c / 12.92 : ((c + 0.055) / 1.055) ** 2.4;
}

export function luminance(color: Rgba): number {
  return 0.2126 * channel(color.r) + 0.7152 * channel(color.g) + 0.0722 * channel(color.b);
}

export function ratio(a: Rgba, b: Rgba): number {
  const la = luminance(a);
  const lb = luminance(b);
  return (Math.max(la, lb) + 0.05) / (Math.min(la, lb) + 0.05);
}

/** The custom properties declared in the first block of `css` whose selector matches exactly. */
export function tokensOf(css: string, selector: string): Record<string, string> {
  const start = css.indexOf(`${selector} {`);
  if (start < 0) throw new Error(`no "${selector} {" block in the stylesheet`);
  const open = css.indexOf("{", start);
  const close = css.indexOf("}", open);
  const tokens: Record<string, string> = {};
  for (const match of css.slice(open + 1, close).matchAll(/(--[\w-]+)\s*:\s*([^;]+);/g)) {
    tokens[match[1] as string] = (match[2] as string).trim();
  }
  return tokens;
}

export interface Result {
  theme: string;
  pair: Pair;
  ratio: number;
  pass: boolean;
  detail: string;
}

const BLACK: Rgba = { r: 0, g: 0, b: 0, a: 1 };
const WHITE: Rgba = { r: 255, g: 255, b: 255, a: 1 };

export function audit(theme: string, tokens: Record<string, string>, pairs: Pair[] = PAIRS): Result[] {
  const color = (name: string): Rgba => {
    const value = tokens[name];
    if (value === undefined) throw new Error(`${theme}: token ${name} is not defined`);
    return parseColor(value);
  };
  const base = color("--bg");
  return pairs.map((pair) => {
    const fg = color(pair.fg);
    const bg = color(pair.bg);
    const unders = pair.over === "video" ? [BLACK, WHITE] : [base];
    const ratios = unders.map((under) => {
      const background = composite(bg, under);
      return ratio(composite(fg, background), background);
    });
    const worst = Math.min(...ratios);
    return {
      theme,
      pair,
      ratio: worst,
      pass: worst >= pair.min,
      detail: `${theme}: ${pair.fg} on ${pair.bg}${pair.over === "video" ? " over black/white video" : ""} = ${worst.toFixed(2)} (min ${pair.min}, ${pair.use})`,
    };
  });
}

/** Both themes of a stylesheet; also checks that the system-light block repeats the light one. */
export function auditStylesheet(css: string): { results: Result[]; problems: string[] } {
  const dark = tokensOf(css, ":root");
  const light = { ...dark, ...tokensOf(css, ':root[data-theme="light"]') };
  const system = { ...dark, ...tokensOf(css, ':root:not([data-theme="dark"])') };
  const problems: string[] = [];
  for (const [name, value] of Object.entries(light)) {
    if (system[name] !== value) problems.push(`the prefers-color-scheme: light block differs from [data-theme="light"] for ${name}`);
  }
  const results = [...audit("dark", dark), ...audit("light", light)];
  for (const result of results) if (!result.pass) problems.push(`below AA: ${result.detail}`);
  return { results, problems };
}
