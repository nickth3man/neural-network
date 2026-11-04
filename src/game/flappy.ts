// Headless Flappy Bird simulation — no rendering, no DOM.
// All birds in a generation share an identical pipe schedule so fitness
// comparisons are fair and exactly one pipe set needs rendering.

import type { Network } from "../nn/network";

export const GAME_W = 480;
export const GAME_H = 640;
export const GRAVITY = 1400;
export const FLAP_V = -430;
export const MAX_FALL = 620;
export const PIPE_SPEED = 140;
export const PIPE_WIDTH = 72;
export const GAP_HEIGHT = 170;
export const PIPE_SPAWN_EVERY = 1.55;
export const BIRD_X = 96;
export const BIRD_R = 12;
export const MAX_TICKS = 3600;
export const TICK_DT = 1 / 60;

export interface Pipe {
 x: number;
 gapY: number;
 passed: boolean;
}

export interface PipeScheduleEntry {
 delay: number;
 gapY: number;
}

/** Precomputed pipe spawn sequence — identical for every bird in a generation. */
export function makeSchedule(rng: () => number, count: number): PipeScheduleEntry[] {
 const schedule: PipeScheduleEntry[] = [];
 for (let i = 0; i < count; i++) {
  const lo = GAP_HEIGHT / 2 + 48;
  const hi = GAME_H - GAP_HEIGHT / 2 - 48;
  schedule.push({
   delay: PIPE_SPAWN_EVERY * (0.85 + rng() * 0.3),
   gapY: lo + rng() * (hi - lo),
  });
 }
 return schedule;
}

export class FlappyEnv {
 y = GAME_H / 2;
 vy = 0;
 pipes: Pipe[] = [];
 score = 0;
 ticks = 0;
 dead = false;

 private readonly schedule: PipeScheduleEntry[];
 private scheduleIdx = 0;
 private spawnTimer = PIPE_SPAWN_EVERY * 1.2;

 constructor(schedule: PipeScheduleEntry[]) {
  this.schedule = schedule;
 }

 reset(): void {
  this.y = GAME_H / 2;
  this.vy = 0;
  this.pipes = [];
  this.score = 0;
  this.ticks = 0;
  this.dead = false;
  this.scheduleIdx = 0;
  this.spawnTimer = PIPE_SPAWN_EVERY * 1.2;
 }

 /** Advance one tick (TICK_DT seconds). `flap` applies upward impulse. */
 step(flap: boolean): void {
  if (this.dead) return;
  this.ticks++;
  this.vy = Math.min(this.vy + GRAVITY * TICK_DT, MAX_FALL);
  if (flap) this.vy = FLAP_V;
  this.y += this.vy * TICK_DT;

  // Spawn and scroll pipes.
  this.spawnTimer -= TICK_DT;
  if (this.spawnTimer <= 0 && this.scheduleIdx < this.schedule.length) {
   const entry = this.schedule[this.scheduleIdx++];
   this.pipes.push({ x: GAME_W, gapY: entry.gapY, passed: false });
   this.spawnTimer += entry.delay;
  }
  for (const p of this.pipes) p.x -= PIPE_SPEED * TICK_DT;
  this.pipes = this.pipes.filter((p) => p.x + PIPE_WIDTH > -10);

  for (const p of this.pipes) {
   if (!p.passed && p.x + PIPE_WIDTH < BIRD_X) {
    p.passed = true;
    this.score++;
   }
  }

  // Ceiling and floor are lethal.
  if (this.y < -BIRD_R || this.y > GAME_H - BIRD_R) {
   this.dead = true;
   return;
  }
  for (const p of this.pipes) {
   if (hitPipe(BIRD_X, this.y, BIRD_R, p)) {
    this.dead = true;
    return;
   }
  }
 }

 /** The next pipe the bird still has to pass, or null when none is ahead. */
 nextPipe(): Pipe | null {
  for (const p of this.pipes) {
   if (p.x + PIPE_WIDTH >= BIRD_X) return p;
  }
  return null;
 }

 /** Normalized sensor vector: [y, vy, dx-to-pipe, gap center, gap offset]. */
 sensorsInto(out: Float32Array): void {
  const p = this.nextPipe();
  out[0] = this.y / GAME_H;
  out[1] = clamp(this.vy / MAX_FALL, -1, 1);
  out[2] = p ? clamp((p.x + PIPE_WIDTH - BIRD_X) / GAME_W, 0, 1) : 1;
  out[3] = p ? p.gapY / GAME_H : 0.5;
  out[4] = p ? clamp(((p.gapY - this.y) / GAME_H) * 2, -1, 1) : 0;
 }
}

export interface EpisodeResult {
 score: number;
 ticks: number;
 fitness: number;
}

/**
 * Run one bird to death (or MAX_TICKS) and return its fitness.
 * `net === null` plays a naive periodic-flap baseline.
 */
export function runEpisode(env: FlappyEnv, net: Network | null, maxTicks = MAX_TICKS): EpisodeResult {
 env.reset();
 const result = net ? net.createResult() : null;
 const sensors = new Float32Array(5);
 while (!env.dead && env.ticks < maxTicks) {
  let flap = false;
  if (net) {
   env.sensorsInto(sensors);
   net.forward(sensors, result!);
   flap = result!.outputs[0] > 0.5;
  } else {
   flap = env.ticks % 8 === 0;
  }
  env.step(flap);
 }
 return { score: env.score, ticks: env.ticks, fitness: env.score * 100 + env.ticks };
}

export function hitPipe(bx: number, by: number, r: number, p: Pipe): boolean {
 const topH = p.gapY - GAP_HEIGHT / 2;
 const bottomY = p.gapY + GAP_HEIGHT / 2;
 const hitsTop = circleRect(bx, by, r, p.x, 0, p.x + PIPE_WIDTH, topH);
 const hitsBottom = circleRect(bx, by, r, p.x, bottomY, p.x + PIPE_WIDTH, GAME_H);
 return hitsTop || hitsBottom;
}

function circleRect(cx: number, cy: number, r: number, rx: number, ry: number, rw: number, rh: number): boolean {
 const nx = clamp(cx, rx, rw);
 const ny = clamp(cy, ry, rh);
 const dx = cx - nx;
 const dy = cy - ny;
 return dx * dx + dy * dy <= r * r;
}

function clamp(v: number, lo: number, hi: number): number {
 let out = v;
 if (out < lo) out = lo;
 if (out > hi) out = hi;
 return out;
}
