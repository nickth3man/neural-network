// Generation loop: steps the whole population one tick at a time, scores
// birds as they die, and evolves when the generation completes.
// After each generation the champion gets a short solo replay so the
// brain overlay has a champion to display while the next gen resets.

import {
  FlappyEnv,
  makeSchedule,
  MAX_TICKS,
  type PipeScheduleEntry,
} from "./game/flappy";
import { DEFAULT_GA, mulberry32, Population, type GaOptions, type Rng } from "./nn/ga";
import type { ForwardResult } from "./nn/network";

export const SENSOR_COUNT = 5;
const REPLAY_TICKS = 420;

export interface EvolverStats {
  generation: number;
  alive: number;
  bestScoreThisGen: number;
  championFitness: number;
  championScore: number;
  inReplay: boolean;
  historyBest: number[];
  historyAvg: number[];
}

export class Evolver {
  readonly pop: Population;
  readonly envs: FlappyEnv[];
  readonly sensors: Float32Array[];
  readonly results: ForwardResult[];
  readonly fitnessAcc: Float32Array;

  readonly replayEnv: FlappyEnv;
  readonly replaySensors = new Float32Array(SENSOR_COUNT);
  readonly replayResult: ForwardResult;
  replayTicks = 0;

  private readonly schedule: PipeScheduleEntry[];
  private rng: Rng;
  private bestAliveIdx = 0;
  private bestAliveScore = 0;

  constructor(seed = 20260804, opts?: Partial<GaOptions>) {
    this.rng = mulberry32(seed);
    this.pop = new Population({
      ...DEFAULT_GA,
      sizes: { input: SENSOR_COUNT, hidden: 4, output: 1 },
      rng: this.rng,
      ...opts,
    });
    this.schedule = makeSchedule(this.rng, 80);
    this.envs = this.pop.genomes.map(() => new FlappyEnv(this.schedule));
    this.sensors = this.envs.map(() => new Float32Array(SENSOR_COUNT));
    this.results = this.pop.genomes.map((g) => g.createResult());
    this.fitnessAcc = new Float32Array(this.pop.genomes.length);
    this.replayEnv = new FlappyEnv(this.schedule);
    this.replayResult = this.pop.genomes[0].createResult();
  }

  /** Advance one tick. Returns true when a new generation just started. */
  step(): boolean {
    if (this.replayTicks > 0) {
      this.replayTicks--;
      this.stepChampionReplay();
      return false;
    }
    let allDead = true;
    for (let i = 0; i < this.envs.length; i++) {
      const env = this.envs[i];
      if (env.dead) continue;
      allDead = false;
      env.sensorsInto(this.sensors[i]);
      this.pop.genomes[i].forward(this.sensors[i], this.results[i]);
      env.step(this.results[i].outputs[0] > 0.5);
      if (env.dead) {
        this.fitnessAcc[i] = env.score * 100 + env.ticks;
      } else if (env.ticks >= MAX_TICKS) {
        env.dead = true;
        this.fitnessAcc[i] = env.score * 100 + env.ticks;
      } else if (env.score > this.bestAliveScore) {
        this.bestAliveScore = env.score;
        this.bestAliveIdx = i;
      }
    }
    if (allDead) {
      this.endGeneration();
      return true;
    }
    return false;
  }

  /** Index of the bird the brain overlay should track this tick. */
  trackedIndex(): number {
    return this.replayTicks > 0 ? -1 : this.bestAliveIdx;
  }

  /** Champion replay state, or null when a live population bird is tracked. */
  replayData(): { inputs: Float32Array; layers: Float32Array[]; weights: Float32Array } | null {
    if (this.replayTicks <= 0 || !this.pop.champion) return null;
    return {
      inputs: this.replaySensors,
      layers: this.replayResult.layers,
      weights: this.pop.champion.weights,
    };
  }

  /** Live state of the tracked population bird (its forward pass just ran). */
  trackedData(): { inputs: Float32Array; layers: Float32Array[]; weights: Float32Array } | null {
    const i = this.trackedIndex();
    if (i < 0) return null;
    return {
      inputs: this.sensors[i],
      layers: this.results[i].layers,
      weights: this.pop.genomes[i].weights,
    };
  }

  get stats(): EvolverStats {
    let alive = 0;
    for (const env of this.envs) {
      if (!env.dead) alive++;
    }
    return {
      generation: this.pop.generation,
      alive,
      bestScoreThisGen: this.bestAliveScore,
      championFitness: Math.round(this.pop.championFitness),
      championScore: Math.floor(this.pop.championFitness / 100),
      inReplay: this.replayTicks > 0,
      historyBest: this.pop.historyBest,
      historyAvg: this.pop.historyAvg,
    };
  }

  private stepChampionReplay(): void {
    const champion = this.pop.champion;
    if (!champion) return;
    this.replayEnv.sensorsInto(this.replaySensors);
    champion.forward(this.replaySensors, this.replayResult);
    this.replayEnv.step(this.replayResult.outputs[0] > 0.5);
    if (this.replayEnv.dead) this.replayTicks = 0;
  }

  private endGeneration(): void {
    this.pop.evaluate((_net, i) => this.fitnessAcc[i]);
    this.pop.evolve();
    // Rebuild per-bird simulation state for the new population.
    for (let i = 0; i < this.envs.length; i++) {
      this.envs[i] = new FlappyEnv(this.schedule);
      this.sensors[i] = new Float32Array(SENSOR_COUNT);
      this.results[i] = this.pop.genomes[i].createResult();
      this.fitnessAcc[i] = 0;
    }
    this.replayEnv.reset();
    this.replayTicks = REPLAY_TICKS;
    this.bestAliveScore = 0;
    this.bestAliveIdx = 0;
  }
}
