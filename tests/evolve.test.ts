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
});
