// Genetic algorithm over fixed-topology feed-forward networks.
// Deterministic when given a seeded RNG — unit-testable end to end.
// Selection: elitism + rank-based tournament parents.
// Variation: uniform crossover + gaussian mutation (+ occasional weight reset).

import { Network, type NetworkSizes } from "./network";

export type Rng = () => number;

/** Deterministic PRNG (mulberry32). */
export function mulberry32(seed: number): Rng {
  let a = seed >>> 0;
  return () => {
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Standard normal via Box-Muller. */
export function gaussian(rng: Rng): number {
  let u = 0;
  let v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export interface GaOptions {
  sizes: NetworkSizes;
  popSize: number;
  elitism: number;
  tournamentSize: number;
  mutationRate: number; // per-weight probability of gaussian perturbation
  mutationSigma: number;
  resetRate: number; // per-weight probability of full re-randomization
  rng: Rng;
}

export const DEFAULT_GA: Omit<GaOptions, "sizes" | "rng"> = {
  popSize: 60,
  elitism: 2,
  tournamentSize: 3,
  mutationRate: 0.15,
  mutationSigma: 0.45,
  resetRate: 0.05,
};

export class Population {
  readonly options: GaOptions;
  genomes: Network[];
  /** fitness[i] corresponds to genomes[i]; valid after evaluate() */
  fitness: Float32Array;
  /** champion = fittest genome of the most recent completed generation */
  champion: Network | null = null;
  championFitness = 0;
  /** Index of the champion within genomes at the last evaluate(); -1 before any evaluation. */
  championIndex = -1;
  generation = 0;
  historyBest: number[] = [];
  historyAvg: number[] = [];

  constructor(options: GaOptions) {
    this.options = options;
    this.genomes = [];
    this.fitness = new Float32Array(options.popSize);
    for (let i = 0; i < options.popSize; i++) {
      this.genomes.push(this.randomNetwork());
    }
  }

  private randomNetwork(): Network {
    const net = new Network(this.options.sizes);
    const { weights } = net;
    for (let i = 0; i < weights.length; i++) {
      weights[i] = gaussian(this.options.rng) * 0.7;
    }
    return net;
  }

  /** Score every genome; records champion and history. */
  evaluate(fitnessFn: (net: Network, index: number) => number): void {
    const n = this.genomes.length;
    for (let i = 0; i < n; i++) {
      this.fitness[i] = fitnessFn(this.genomes[i], i);
    }
    let bestIdx = 0;
    let sum = 0;
    for (let i = 0; i < n; i++) {
      sum += this.fitness[i];
      if (this.fitness[i] > this.fitness[bestIdx]) bestIdx = i;
    }
    this.champion = this.genomes[bestIdx].clone();
    this.championIndex = bestIdx;
    this.championFitness = this.fitness[bestIdx];
    this.historyBest.push(this.fitness[bestIdx]);
    this.historyAvg.push(sum / n);
  }

  /** Produce the next generation. Requires evaluate() to have run. */
  evolve(): void {
    const { popSize, elitism, tournamentSize, mutationRate, mutationSigma, resetRate, rng } =
      this.options;
    const order = Array.from({ length: this.genomes.length }, (_, i) => i).sort(
      (a, b) => this.fitness[b] - this.fitness[a],
    );
    const next: Network[] = [];

    // Elitism: copy the fittest genomes unchanged.
    for (let e = 0; e < elitism; e++) {
      next.push(this.genomes[order[e]].clone());
    }

    const pickParent = (): Network => {
      let best = order[Math.floor(rng() * order.length)];
      for (let t = 1; t < tournamentSize; t++) {
        const cand = order[Math.floor(rng() * order.length)];
        if (this.fitness[cand] > this.fitness[best]) best = cand;
      }
      return this.genomes[best];
    };

    while (next.length < popSize) {
      const a = pickParent();
      const b = pickParent();
      const child = crossover(a, b, rng);
      mutate(child, mutationRate, mutationSigma, resetRate, rng);
      next.push(child);
    }

    this.genomes = next;
    this.generation++;
  }
}

/** Uniform crossover: each weight comes from parent a or b. */
export function crossover(a: Network, b: Network, rng: Rng): Network {
  const child = new Network(a.sizes);
  for (let i = 0; i < child.weights.length; i++) {
    child.weights[i] = rng() < 0.5 ? a.weights[i] : b.weights[i];
  }
  return child;
}

/** Gaussian perturbation per weight, plus occasional full reset. */
export function mutate(
  net: Network,
  mutationRate: number,
  mutationSigma: number,
  resetRate: number,
  rng: Rng,
): void {
  for (let i = 0; i < net.weights.length; i++) {
    if (rng() < resetRate) {
      net.weights[i] = gaussian(rng) * 0.7;
    } else if (rng() < mutationRate) {
      net.weights[i] += gaussian(rng) * mutationSigma;
    }
  }
}
