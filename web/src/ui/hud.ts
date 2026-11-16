// The live HUD: each hand's skeleton over the video in its own colour, glowing joints and a short
// fading trail of the fingertips. Drawn on a canvas laid over the <video>; both are mirrored by
// the same CSS transform, so points are drawn in unmirrored frame coordinates. Kept cheap for the
// i3 budget: no shadowBlur (the glow is a wide translucent stroke under the bone), one path per
// colour, the canvas at 1 device pixel per CSS pixel, and nothing drawn when the HUD is off.

import type { CropRect } from "../camera";
import { HAND_CONNECTIONS } from "../landmarker";
import type { Point2 } from "../preprocess";

export const HAND_COLORS = { left: "#7AA2FF", right: "#FF9E7A" } as const;
const CORE = "#F2F4F8";
const POSE_COLOR = "rgba(242, 244, 248, 0.28)";
const FINGERTIPS = [4, 8, 12, 16, 20] as const;
export const TRAIL_FRAMES = 8;
const MAX_DPR = 1;
const POSE_LINES: readonly (readonly [number, number])[] = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
];

export interface HudFrame {
  left: readonly Point2[] | null;
  right: readonly Point2[] | null;
  pose: readonly Point2[] | null;
  crop: CropRect;
}

/** object-fit: cover of a (width x height) frame in a (boxWidth x boxHeight) box. */
export function coverMapping(width: number, height: number, boxWidth: number, boxHeight: number): { scale: number; dx: number; dy: number } {
  const scale = Math.max(boxWidth / width, boxHeight / height);
  return { scale, dx: (boxWidth - width * scale) / 2, dy: (boxHeight - height * scale) / 2 };
}

export class Hud {
  private readonly context: CanvasRenderingContext2D;
  private enabled = true;
  private trails = true;
  private width = 0;
  private height = 0;
  private dpr = 1;
  private readonly history: Record<"left" | "right", Point2[][]> = { left: [], right: [] };

  constructor(
    private readonly canvas: HTMLCanvasElement,
    private readonly video: HTMLVideoElement,
  ) {
    this.context = canvas.getContext("2d", { alpha: true }) as CanvasRenderingContext2D;
  }

  setEnabled(on: boolean): void {
    this.enabled = on;
    if (!on) this.clear();
  }

  setTrails(on: boolean): void {
    this.trails = on;
    if (!on) this.history.left = this.history.right = [];
  }

  /** Match the canvas to its box (call on resize). */
  resize(): void {
    const box = this.canvas.getBoundingClientRect();
    this.dpr = Math.min(MAX_DPR, window.devicePixelRatio || 1);
    this.width = box.width;
    this.height = box.height;
    this.canvas.width = Math.max(1, Math.round(box.width * this.dpr));
    this.canvas.height = Math.max(1, Math.round(box.height * this.dpr));
  }

  clear(): void {
    this.context.setTransform(1, 0, 0, 1, 0, 0);
    this.context.clearRect(0, 0, this.canvas.width, this.canvas.height);
    this.history.left = [];
    this.history.right = [];
  }

  draw(frame: HudFrame): void {
    if (!this.enabled) return;
    const { videoWidth, videoHeight } = this.video;
    if (!videoWidth || !videoHeight || !this.width) return;
    const ctx = this.context;
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.clearRect(0, 0, this.width, this.height);
    const map = coverMapping(videoWidth, videoHeight, this.width, this.height);
    const { crop } = frame;
    const sw = crop.sw || videoWidth;
    const sh = crop.sh || videoHeight;
    const toPx = (point: Point2): [number, number] => [
      map.dx + (crop.sx + point.x * sw) * map.scale,
      map.dy + (crop.sy + point.y * sh) * map.scale,
    ];
    const unit = Math.max(1, Math.min(this.width, this.height) / 420);

    if (frame.pose) this.drawPose(frame.pose, toPx, unit);
    for (const side of ["left", "right"] as const) {
      const hand = frame[side];
      if (this.trails) this.pushTrail(side, hand);
      if (this.trails) this.drawTrail(side, toPx, unit);
      if (hand) this.drawHand(hand, HAND_COLORS[side], toPx, unit);
    }
  }

  private pushTrail(side: "left" | "right", hand: readonly Point2[] | null): void {
    const list = this.history[side];
    if (!hand) {
      // A lost hand fades its trail out frame by frame instead of cutting it.
      list.shift();
      return;
    }
    list.push(FINGERTIPS.map((index) => hand[index] as Point2));
    if (list.length > TRAIL_FRAMES) list.shift();
  }

  private drawTrail(side: "left" | "right", toPx: (p: Point2) => [number, number], unit: number): void {
    const list = this.history[side];
    if (list.length < 2) return;
    const ctx = this.context;
    ctx.strokeStyle = HAND_COLORS[side];
    ctx.lineCap = "round";
    for (let i = 1; i < list.length; i += 1) {
      const age = i / list.length;
      ctx.globalAlpha = 0.45 * age * age;
      ctx.lineWidth = (1 + 2.2 * age) * unit;
      ctx.beginPath();
      const previous = list[i - 1] as Point2[];
      const next = list[i] as Point2[];
      for (let tip = 0; tip < FINGERTIPS.length; tip += 1) {
        const [ax, ay] = toPx(previous[tip] as Point2);
        const [bx, by] = toPx(next[tip] as Point2);
        ctx.moveTo(ax, ay);
        ctx.lineTo(bx, by);
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  private drawPose(pose: readonly Point2[], toPx: (p: Point2) => [number, number], unit: number): void {
    const ctx = this.context;
    ctx.strokeStyle = POSE_COLOR;
    ctx.lineWidth = 1.5 * unit;
    ctx.lineCap = "round";
    ctx.beginPath();
    for (const [a, b] of POSE_LINES) {
      const p = pose[a];
      const q = pose[b];
      if (!p || !q) continue;
      const [ax, ay] = toPx(p);
      const [bx, by] = toPx(q);
      ctx.moveTo(ax, ay);
      ctx.lineTo(bx, by);
    }
    ctx.stroke();
  }

  private drawHand(hand: readonly Point2[], color: string, toPx: (p: Point2) => [number, number], unit: number): void {
    const ctx = this.context;
    const points = hand.map(toPx);
    const bones = (): void => {
      ctx.beginPath();
      for (const { start, end } of HAND_CONNECTIONS) {
        const a = points[start];
        const b = points[end];
        if (!a || !b) continue;
        ctx.moveTo(a[0], a[1]);
        ctx.lineTo(b[0], b[1]);
      }
      ctx.stroke();
    };
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
    ctx.strokeStyle = color;
    // Glow: a wide translucent pass, then the bone itself.
    ctx.globalAlpha = 0.22;
    ctx.lineWidth = 7 * unit;
    bones();
    ctx.globalAlpha = 1;
    ctx.lineWidth = 2.2 * unit;
    bones();
    // Joints: a soft halo in the hand colour around a light core; fingertips slightly larger.
    ctx.fillStyle = color;
    ctx.globalAlpha = 0.3;
    ctx.beginPath();
    for (const [x, y] of points) {
      ctx.moveTo(x + 5 * unit, y);
      ctx.arc(x, y, 5 * unit, 0, Math.PI * 2);
    }
    ctx.fill();
    ctx.globalAlpha = 1;
    ctx.fillStyle = CORE;
    ctx.beginPath();
    points.forEach(([x, y], index) => {
      const r = (FINGERTIPS as readonly number[]).includes(index) ? 2.6 * unit : 1.9 * unit;
      ctx.moveTo(x + r, y);
      ctx.arc(x, y, r, 0, Math.PI * 2);
    });
    ctx.fill();
  }
}
