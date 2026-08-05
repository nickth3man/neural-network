import { describe, expect, it } from "vitest";
import { DEFAULT_GA, gaussian, mulberry32, Population } from "../src/nn/ga";

describe("mulberry32", () => {
  it("is deterministic for the same seed", () => {
    const a = mulberry32(42);
    const b = mulberry32(42);
    for (let i = 0; i < 100; i++) {
      expect(a()).toBe(b());
    }
  });

  it("differs for different seeds", () => {
    const a = mulberry32(1);
    const b = mulberry32(2);
    let same = 0;
    for (let i = 0; i < 100; i++) {
      if (a() === b()) same++;
    }
    expect(same).toBeLessThan(10);
  });
});

describe("gaussian", () => {
  it("has near-zero mean and unit variance", () => {
    const rng = mulberry32(7);
    const n = 20000;
    let sum = 0;
    let sq = 0;
    for (let i = 0; i < n; i++) {
      const v = gaussian(rng);
      sum += v;
      sq += v * v;
    }
    const mean = sum / n;
    const variance = sq / n - mean * mean;
    expect(Math.abs(mean)).toBeLessThan(0.03);
    expect(Math.abs(variance - 1)).toBeLessThan(0.05);
  });
});

function singleWeightPop(seed: number, popSize: number, sigma: number): Population {
  return new Population({
    ...DEFAULT_GA,
    sizes: { input: 1, hidden: 1, output: 1 },
    popSize,
    mutationSigma: sigma,
    rng: mulberry32(seed),
  });
}

describe("Population.evaluate", () => {
  it("selects the fittest genome as champion", () => {
    const pop = singleWeightPop(11, 30, 0.5);
    pop.evaluate((net) => net.weights[0]);
    const maxW = Math.max(...pop.fitness);
    expect(pop.championFitness).toBe(maxW);
    expect(pop.champion!.weights[0]).toBe(maxW);
    expect(pop.historyBest[pop.historyBest.length - 1]).toBe(maxW);
    expect(pop.historyAvg.length).toBe(1);
  });

  it("preserves population size and elites across evolve", () => {
    const pop = singleWeightPop(22, 40, 0.5);
    pop.evaluate((net) => net.weights[0]);
    const elite = pop.champion!.clone();
    pop.evolve();
    expect(pop.genomes.length).toBe(40);
    expect(pop.generation).toBe(1);
    const hasElite = pop.genomes.some(
      (g) => g.weights[0] === elite.weights[0] && g.weights.length === elite.weights.length,
    );
    expect(hasElite).toBe(true);
  });
});

describe("Population evolution", () => {
  it("hill-climbs a single weight toward the maximum", () => {
    const pop = singleWeightPop(20260804, 40, 0.6);
    for (let gen = 0; gen < 60; gen++) {
      pop.evaluate((net) => net.weights[0]);
      pop.evolve();
    }
    expect(pop.championFitness).toBeGreaterThan(0.85);
    expect(pop.historyBest.length).toBe(60);
  });

  it("is deterministic end to end for a fixed seed", () => {
    const run = (): number[] => {
      const pop = singleWeightPop(99, 20, 0.4);
      const trace: number[] = [];
      for (let gen = 0; gen < 10; gen++) {
        pop.evaluate((net) => net.weights[0]);
        trace.push(pop.championFitness);
        pop.evolve();
      }
      return trace;
    };
    expect(run()).toEqual(run());
  });
});
