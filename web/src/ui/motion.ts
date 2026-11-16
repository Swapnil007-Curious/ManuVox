// Springs for interface motion: velocity-carrying, interruptible, transform and opacity only.
// Parameters follow Apple's damping ratio + response: damping 1.0 (critically damped, no
// overshoot) for movement; a lower damping only for the recognised word's landing. With
// prefers-reduced-motion, positions jump and only opacity fades.

export interface SpringConfig {
  /** Damping ratio: 1 = no overshoot, < 1 = bounce. */
  damping: number;
  /** Seconds to (roughly) reach the target; not a fixed duration. */
  response: number;
}

export const SPRINGS = {
  move: { damping: 1, response: 0.38 },
  quick: { damping: 1, response: 0.24 },
  sheet: { damping: 1, response: 0.34 },
  land: { damping: 0.58, response: 0.46 },
} satisfies Record<string, SpringConfig>;

export interface Channels {
  x: number;
  y: number;
  scale: number;
  opacity: number;
}

const KEYS = ["x", "y", "scale", "opacity"] as const;
const REST: Channels = { x: 0, y: 0, scale: 1, opacity: 1 };

export function prefersReducedMotion(): boolean {
  return typeof matchMedia === "function" && matchMedia("(prefers-reduced-motion: reduce)").matches;
}

/** One step of a damped spring (semi-implicit Euler, unit mass). */
export function stepSpring(value: number, velocity: number, target: number, config: SpringConfig, dt: number): [number, number] {
  const stiffness = (2 * Math.PI / config.response) ** 2;
  const damping = (4 * Math.PI * config.damping) / config.response;
  const acceleration = -stiffness * (value - target) - damping * velocity;
  const v = velocity + acceleration * dt;
  return [value + v * dt, v];
}

interface Motion {
  value: Channels;
  velocity: Channels;
  target: Channels;
  config: SpringConfig;
  resolve: (() => void)[];
  apply: (value: Channels) => void;
}

const motions = new Map<object, Motion>();
const current = new WeakMap<object, Channels>();
let frame = 0;
let last = 0;

function render(el: HTMLElement | SVGElement, value: Channels): void {
  el.style.transform = `translate3d(${value.x.toFixed(2)}px, ${value.y.toFixed(2)}px, 0) scale(${value.scale.toFixed(4)})`;
  el.style.opacity = value.opacity.toFixed(3);
}

function settled(motion: Motion): boolean {
  return KEYS.every((key) => {
    const scale = key === "x" || key === "y" ? 0.5 : 0.002;
    return Math.abs(motion.value[key] - motion.target[key]) < scale && Math.abs(motion.velocity[key]) < scale * 10;
  });
}

function tick(now: number): void {
  const dt = Math.min(0.032, Math.max(0.001, (now - last) / 1000));
  last = now;
  for (const [key, motion] of motions) {
    for (const channel of KEYS) {
      const [value, velocity] = stepSpring(motion.value[channel], motion.velocity[channel], motion.target[channel], motion.config, dt);
      motion.value[channel] = value;
      motion.velocity[channel] = velocity;
    }
    if (settled(motion)) {
      motion.value = { ...motion.target };
      motion.velocity = { x: 0, y: 0, scale: 0, opacity: 0 };
      motion.apply(motion.value);
      current.set(key, { ...motion.value });
      motions.delete(key);
      for (const resolve of motion.resolve) resolve();
      continue;
    }
    motion.apply(motion.value);
    current.set(key, { ...motion.value });
  }
  frame = motions.size > 0 ? requestAnimationFrame(tick) : 0;
}

/** Where an element sits now (its live, presentation value), for interruption. */
export function valueOf(el: object): Channels {
  return { ...(current.get(el) ?? REST) };
}

/** Set an element's channels without animating (cancels a running spring). */
export function place(el: HTMLElement | SVGElement, value: Partial<Channels>): void {
  const motion = motions.get(el);
  if (motion) {
    motions.delete(el);
    for (const resolve of motion.resolve) resolve();
  }
  const next = { ...valueOf(el), ...value };
  current.set(el, next);
  render(el, next);
}

/**
 * Spring `el` to `to`, starting from where it is now (and how fast it is moving), so a new call
 * mid-flight redirects the motion instead of restarting it. `velocity` hands a gesture's speed
 * over (px/s). Resolves when it settles.
 */
export function spring(
  el: HTMLElement | SVGElement,
  to: Partial<Channels>,
  config: SpringConfig = SPRINGS.move,
  options: { from?: Partial<Channels>; velocity?: Partial<Channels>; apply?: (value: Channels) => void } = {},
): Promise<void> {
  const apply = options.apply ?? ((value: Channels) => render(el, value));
  const running = motions.get(el);
  let start = { ...(running ? running.value : valueOf(el)), ...options.from };
  const target = { ...(running ? running.target : start), ...to };
  const reduced = prefersReducedMotion();
  if (reduced) {
    // Positions and scale jump; only opacity fades, so changes stay noticeable without motion.
    start = { ...target, opacity: start.opacity };
    current.set(el, start);
    apply(start);
  }
  return new Promise<void>((resolve) => {
    const velocity = running && !reduced ? running.velocity : { x: 0, y: 0, scale: 0, opacity: 0 };
    const motion: Motion = {
      value: start,
      velocity: reduced ? velocity : { ...velocity, ...options.velocity },
      target,
      config: reduced ? SPRINGS.quick : config,
      resolve: [...(running?.resolve ?? []), resolve],
      apply,
    };
    motions.set(el, motion);
    if (!frame) {
      last = performance.now();
      frame = requestAnimationFrame(tick);
    }
  });
}

/** A scalar spring (e.g. a ring's fill, a sheet's offset) driven through `apply`. */
export function springValue(key: object, from: number, to: number, config: SpringConfig, apply: (value: number) => void, velocity = 0): Promise<void> {
  const channels = (value: number): Partial<Channels> => ({ x: value });
  const running = motions.get(key);
  const startValue = running ? running.value.x : from;
  if (prefersReducedMotion()) {
    apply(to);
    current.set(key, { ...REST, x: to });
    if (running) motions.delete(key);
    return Promise.resolve();
  }
  return new Promise<void>((resolve) => {
    motions.set(key, {
      value: { ...REST, x: startValue },
      velocity: { x: running ? running.velocity.x : velocity, y: 0, scale: 0, opacity: 0 },
      target: { ...REST, ...channels(to) },
      config,
      resolve: [...(running?.resolve ?? []), resolve],
      apply: (value) => apply(value.x),
    });
    if (!frame) {
      last = performance.now();
      frame = requestAnimationFrame(tick);
    }
  });
}

/** Apple's momentum projection: where a flick at `velocity` px/s comes to rest. */
export function project(velocity: number, decelerationRate = 0.998): number {
  return ((velocity / 1000) * decelerationRate) / (1 - decelerationRate);
}

/** Progressive resistance past a boundary. */
export function rubberband(overshoot: number, dimension: number, constant = 0.55): number {
  return (overshoot * dimension * constant) / (dimension + constant * Math.abs(overshoot));
}
