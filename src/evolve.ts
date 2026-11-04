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
import type { ForwardResult, Network } from "./nn/network";

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
  /** Index of the highest-scoring living bird, or -1 when none are alive. */
  private bestAliveIdx = 0;
  /** Best score (pipes passed) achieved by any bird in this generation. */
  private bestScoreThisGen = 0;
  /** Pipes passed by the champion selected at the last generation boundary. */
  private championPipes = 0;

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

  /**
   * Install a persisted champion as slot 0, reset its episode/fitness, and
   * restart the replay so the loaded network is immediately visible and scored.
   */
  loadChampion(net: Network): void {
    this.pop.genomes[0] = net.clone();
    this.pop.fitness[0] = 0;
    this.pop.champion = net.clone();
    this.pop.championFitness = 0;
    this.pop.championIndex = 0;
    this.envs[0].reset();
    this.results[0] = net.createResult();
    this.fitnessAcc[0] = 0;
    this.replayEnv.reset();
    this.replayTicks = REPLAY_TICKS;
    this.championPipes = 0;
    this.refreshTracking();
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
      }
    }
    if (allDead) {
      this.endGeneration();
      return true;
    }
    this.refreshTracking();
    return false;
  }

  /** Index of the bird the brain overlay should track this tick. */
  trackedIndex(): number {
    return this.replayTicks > 0 ? -1 : this.bestAliveIdx;
  }

  /**
   * Environment whose pipes should be rendered: the replay env during a
   * champion replay, otherwise the environment of a bird that is still
   * stepping (the tracked bird when possible).
   */
  sceneEnv(): FlappyEnv {
    if (this.replayTicks > 0) return this.replayEnv;
    const idx = this.trackedIndex();
    if (idx >= 0 && !this.envs[idx].dead) return this.envs[idx];
    for (const env of this.envs) {
      if (!env.dead) return env;
    }
    return this.envs[0] ?? this.replayEnv;
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
    if (i < 0 || this.envs[i].dead) return null;
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
      alive: this.replayTicks > 0 ? 0 : alive,
      bestScoreThisGen: this.bestScoreThisGen,
      championFitness: Math.round(this.pop.championFitness),
      championScore: this.championPipes,
      inReplay: this.replayTicks > 0,
      historyBest: this.pop.historyBest,
      historyAvg: this.pop.historyAvg,
    };
  }

  private refreshTracking(): void {
    let idx = -1;
    let aliveScore = -1;
    let bestScore = 0;
    for (let i = 0; i < this.envs.length; i++) {
      const env = this.envs[i];
      if (env.score > bestScore) bestScore = env.score;
      if (!env.dead && env.score > aliveScore) {
        aliveScore = env.score;
        idx = i;
      }
    }
    this.bestAliveIdx = idx;
    this.bestScoreThisGen = bestScore;
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
    const championIdx = this.pop.championIndex;
    this.championPipes = championIdx >= 0 ? this.envs[championIdx].score : 0;
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
    this.bestScoreThisGen = 0;
    this.bestAliveIdx = 0;
  }
}
