import { describe, expect, it } from "vitest";
import {
  BIRD_R,
  BIRD_X,
  FlappyEnv,
  GAME_H,
  GAME_W,
  GRAVITY,
  makeSchedule,
  runEpisode,
  TICK_DT,
  type PipeScheduleEntry,
} from "../src/game/flappy";
import { mulberry32 } from "../src/nn/ga";

function fixedSchedule(): PipeScheduleEntry[] {
  return [{ delay: 1.2, gapY: 320 }];
}

describe("FlappyEnv physics", () => {
  it("falls under gravity when not flapping", () => {
    const env = new FlappyEnv(fixedSchedule());
    env.reset();
    const y0 = env.y;
    env.step(false);
    expect(env.vy).toBeCloseTo(GRAVITY * TICK_DT, 10);
    expect(env.y).toBeGreaterThan(y0);
  });

  it("applies the flap impulse", () => {
    const env = new FlappyEnv(fixedSchedule());
    env.reset();
    env.step(true);
    expect(env.vy).toBeLessThan(0);
    const yAfter = env.y + env.vy * TICK_DT;
    expect(yAfter).toBeLessThan(env.y);
  });

  it("dies on the floor", () => {
    const env = new FlappyEnv(fixedSchedule());
    env.reset();
    env.y = GAME_H - BIRD_R + 1;
    env.step(false);
    expect(env.dead).toBe(true);
  });

  it("steps are no-ops after death", () => {
    const env = new FlappyEnv(fixedSchedule());
    env.reset();
    env.dead = true;
    const ticks = env.ticks;
    env.step(false);
    expect(env.ticks).toBe(ticks);
  });
});

describe("FlappyEnv pipes", () => {
  it("spawns and scrolls pipes", () => {
    const env = new FlappyEnv(fixedSchedule());
    env.reset();
    // Keep the bird near mid-screen so it survives the grace period (1.2*1.55s)
    // plus the spawn delay, then assert a pipe exists.
    const ticks = Math.ceil((1.2 * 1.55 + 1.2) / TICK_DT) + 1;
    for (let i = 0; i < ticks && !env.dead; i++) {
      env.step(env.y > GAME_H / 2);
    }
    expect(env.dead).toBe(false);
    expect(env.pipes.length).toBe(1);
    const xBefore = env.pipes[0].x;
    for (let i = 0; i < 10 && !env.dead; i++) {
      env.step(env.y > GAME_H / 2);
    }
    expect(env.pipes[0].x).toBeLessThan(xBefore);
  });

  it("scores a point when a pipe passes the bird", () => {
    const env = new FlappyEnv(fixedSchedule());
    env.reset();
    // Right edge (x + width) already behind the bird: x + 72 < BIRD_X.
    env.pipes.push({ x: 20, gapY: GAME_H / 2, passed: false });
    env.step(false);
    expect(env.score).toBe(1);
    expect(env.dead).toBe(false);
  });

  it("kills the bird inside a pipe rect", () => {
    const env = new FlappyEnv(fixedSchedule());
    env.reset();
    env.pipes.push({ x: BIRD_X, gapY: 320, passed: false });
    env.y = 100; // inside the top rect (top rect spans y 0..235)
    env.step(false);
    expect(env.dead).toBe(true);
  });
});

describe("FlappyEnv sensors", () => {
  it("reports normalized idle state with no pipes ahead", () => {
    const env = new FlappyEnv(fixedSchedule());
    env.reset();
    const s = new Float32Array(5);
    env.sensorsInto(s);
    expect(Array.from(s)).toEqual([0.5, 0, 1, 0.5, 0]);
  });

  it("reports distance and gap offset with a pipe ahead", () => {
    const env = new FlappyEnv(fixedSchedule());
    env.reset();
    env.pipes.push({ x: 200, gapY: 320, passed: false });
    env.y = 100;
    const s = new Float32Array(5);
    env.sensorsInto(s);
    expect(s[0]).toBeCloseTo(100 / GAME_H, 6);
    expect(s[2]).toBeCloseTo((200 + 72 - BIRD_X) / GAME_W, 6);
    expect(s[3]).toBeCloseTo(320 / GAME_H, 6);
    expect(s[4]).toBeCloseTo(((320 - 100) / GAME_H) * 2, 6);
    for (const v of s) {
      expect(Number.isFinite(v)).toBe(true);
    }
  });

  it("is identical across birds sharing a schedule", () => {
    const schedule = makeSchedule(mulberry32(5), 20);
    const a = new FlappyEnv(schedule);
    const b = new FlappyEnv(schedule);
    const sa = new Float32Array(5);
    const sb = new Float32Array(5);
    for (let i = 0; i < 300 && !a.dead && !b.dead; i++) {
      const flap = i % 13 === 0;
      a.step(flap);
      b.step(flap);
    }
    a.sensorsInto(sa);
    b.sensorsInto(sb);
    expect(Array.from(sa)).toEqual(Array.from(sb));
  });
});

describe("runEpisode", () => {
  it("returns fitness = score*100 + ticks for a survivor", () => {
    const env = new FlappyEnv(fixedSchedule());
    const result = runEpisode(env, null, 30);
    expect(result.score).toBe(0);
    expect(result.ticks).toBe(30);
    expect(result.fitness).toBe(30);
  });

  it("never exceeds maxTicks", () => {
    const env = new FlappyEnv(fixedSchedule());
    const result = runEpisode(env, null, 1000);
    expect(result.ticks).toBeLessThanOrEqual(1000);
    expect(result.fitness).toBe(result.score * 100 + result.ticks);
  });
});
