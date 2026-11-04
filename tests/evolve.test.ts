import { describe, expect, it } from "vitest";
import { Evolver } from "../src/evolve";

describe("Evolver", () => {
  it("advances generations and records fitness history", () => {
    const evolver = new Evolver(20260804);
    let gens = 0;
    for (let steps = 0; steps < 200_000 && gens < 3; steps++) {
      if (evolver.step()) gens++;
    }
    expect(gens).toBe(3);
    expect(evolver.pop.generation).toBe(3);
    expect(evolver.pop.historyBest.length).toBe(3);
    expect(evolver.pop.champion).not.toBeNull();
    expect(evolver.stats.championFitness).toBeGreaterThan(0);
  });

  it("improves the best fitness over several generations", () => {
    const evolver = new Evolver(7);
    let gens = 0;
    for (let steps = 0; steps < 300_000 && gens < 8; steps++) {
      if (evolver.step()) gens++;
    }
    const best = evolver.pop.historyBest;
    expect(best.length).toBe(8);
    // Later-generation champions must beat the initial random best.
    expect(best[best.length - 1]).toBeGreaterThan(best[0]);
  });

  it("serves tracked data for the overlay during a generation", () => {
    const evolver = new Evolver(123);
    evolver.step();
    evolver.step();
    const data = evolver.trackedData();
    expect(data).not.toBeNull();
    expect(data!.inputs.length).toBe(5);
    expect(data!.layers.length).toBe(2);
    expect(data!.weights.length).toBeGreaterThan(0);
  });

  it("reports the champion's pipes passed instead of a fitness-derived score", () => {
    const evolver = new Evolver(20260804);
    for (let steps = 0; steps < 200_000 && !evolver.envs.every((env) => env.dead); steps++) {
      evolver.step();
    }
    expect(evolver.envs.every((env) => env.dead)).toBe(true);
    let bestIdx = 0;
    for (let i = 1; i < evolver.fitnessAcc.length; i++) {
      if (evolver.fitnessAcc[i] > evolver.fitnessAcc[bestIdx]) bestIdx = i;
    }
    const pipes = evolver.envs[bestIdx].score;
    const ticks = evolver.envs[bestIdx].ticks;
    evolver.step(); // closes the generation and starts the champion replay
    expect(evolver.stats.inReplay).toBe(true);
    expect(evolver.stats.alive).toBe(0);
    expect(evolver.stats.championScore).toBe(pipes);
    // This seed's champion survives past 100 ticks, so fitness/100 is not the score.
    expect(ticks).toBeGreaterThanOrEqual(100);
    expect(Math.floor(evolver.stats.championFitness / 100)).not.toBe(pipes);
  });

  it("switches tracking to a living bird when the leader dies", () => {
    const evolver = new Evolver(123);
    let switched = false;
    for (let steps = 0; steps < 50_000 && !switched; steps++) {
      const leader = evolver.trackedIndex();
      if (leader < 0) {
        evolver.step();
        continue;
      }
      const wasAlive = !evolver.envs[leader].dead;
      evolver.step();
      if (evolver.replayTicks > 0) break;
      if (wasAlive && evolver.envs[leader].dead && evolver.envs.some((env) => !env.dead)) {
        const next = evolver.trackedIndex();
        expect(next).toBeGreaterThanOrEqual(0);
        expect(evolver.envs[next].dead).toBe(false);
        expect(evolver.trackedData()).not.toBeNull();
        switched = true;
      }
    }
    expect(switched).toBe(true);
  });

  it("renders pipes from a living environment when envs[0] dies", () => {
    const evolver = new Evolver(20260804);
    let found = false;
    for (let steps = 0; steps < 50_000; steps++) {
      if (
        evolver.replayTicks === 0 &&
        evolver.envs[0].dead &&
        evolver.envs.some((env, i) => i > 0 && !env.dead && env.pipes.length > 0)
      ) {
        found = true;
        break;
      }
      evolver.step();
    }
    expect(found).toBe(true);
    const env = evolver.sceneEnv();
    expect(env.dead).toBe(false);
    expect(env).not.toBe(evolver.envs[0]);
    const before = env.pipes.map((p) => p.x);
    evolver.step();
    if (!env.dead && evolver.replayTicks === 0) {
      expect(env.pipes.map((p) => p.x)).not.toEqual(before);
    }
  });

  it("installs a loaded champion in a live slot and replays it", () => {
    const source = new Evolver(20260804);
    for (let steps = 0; steps < 200_000 && source.replayTicks === 0; steps++) {
      source.step();
    }
    const champion = source.pop.champion!.clone();

    const target = new Evolver(123);
    for (let steps = 0; steps < 1_000 && !target.envs[0].dead && target.replayTicks === 0; steps++) {
      target.step();
    }
    expect(target.envs[0].dead).toBe(true);

    target.loadChampion(champion);
    expect(target.envs[0].dead).toBe(false);
    expect(target.envs[0].ticks).toBe(0);
    expect(target.fitnessAcc[0]).toBe(0);
    expect(target.replayTicks).toBeGreaterThan(0);
    expect(Array.from(target.pop.genomes[0].weights)).toEqual(Array.from(champion.weights));
    expect(Array.from(target.pop.champion!.weights)).toEqual(Array.from(champion.weights));

    target.step();
    expect(target.replayEnv.ticks).toBe(1);
    while (target.replayTicks > 0) target.step();
    target.step();
    expect(target.envs[0].ticks).toBe(1);
  });
});
