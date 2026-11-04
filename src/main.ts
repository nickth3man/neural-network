// Browser entry point: wires the game canvas, evolution loop, brain
// overlay, fitness chart, and HUD controls.

import { BrainView, type NodeSelection } from "./brainView";
import { drawChart } from "./chart";
import { Evolver, SENSOR_COUNT } from "./evolve";
import { FlappyEnv, GAME_H, makeSchedule } from "./game/flappy";
import { mulberry32 } from "./nn/ga";
import type { NetworkSizes } from "./nn/network";
import { deserializeChampion, serializeChampion } from "./persistence";
import { drawFrame, poseBirds } from "./render";

const SIZES: NetworkSizes = { input: SENSOR_COUNT, hidden: 4, output: 1 };
const CHAMPION_KEY = "flappy-brain-champion";

const canvas = document.getElementById("game") as HTMLCanvasElement;
const ctx = canvas.getContext("2d")!;
const statsEl = document.getElementById("stats") as HTMLElement;
const btnStart = document.getElementById("btn-start") as HTMLButtonElement;
const btnReset = document.getElementById("btn-reset") as HTMLButtonElement;
const btnHuman = document.getElementById("btn-human") as HTMLButtonElement;
const btnSave = document.getElementById("btn-save") as HTMLButtonElement;
const btnLoad = document.getElementById("btn-load") as HTMLButtonElement;
const speedSelect = document.getElementById("speed") as HTMLSelectElement;
const detailEl = document.getElementById("node-detail") as HTMLElement;
const announceEl = document.getElementById("node-announce") as HTMLElement;
const actionEl = document.getElementById("action-status") as HTMLElement;
const chartCanvas = document.getElementById("chart") as HTMLCanvasElement;

const brainView = new BrainView(
  document.getElementById("brain-svg-wrap")!,
  detailEl,
  SIZES,
  (sel: NodeSelection) => {
    announceEl.textContent = brainView.renderDetail(sel, lastInputs, lastLayers, lastWeights);
  },
);

// --- State -------------------------------------------------------------

let evolver = new Evolver();
let running = false;
let humanMode = false;
let humanEnv = new FlappyEnv(makeSchedule(mulberry32(999), 80));
let humanDead = false;
let flapHeld = false;

// Reusable buffers for the tracked bird (drives the overlay).
const lastInputs = new Float32Array(SENSOR_COUNT);
let lastLayers: Float32Array[] | null = null;
let lastWeights: Float32Array | null = null;

function speeds(): number {
  return Number(speedSelect.value);
}

function stepHuman(): void {
  if (humanDead) return;
  humanEnv.step(flapHeld);
  if (humanEnv.dead) humanDead = true;
}

function updateTrackedBuffers(): void {
  const replay = evolver.replayData();
  if (replay) {
    lastInputs.set(replay.inputs);
    lastLayers = replay.layers;
    lastWeights = replay.weights;
    return;
  }
  const live = evolver.trackedData();
  if (live) {
    lastInputs.set(live.inputs);
    lastLayers = live.layers;
    lastWeights = live.weights;
  } else {
    lastLayers = null;
    lastWeights = null;
  }
}

function render(): void {
  const replaying = evolver.replayTicks > 0;
  const birds = humanMode ? [] : poseBirds(evolver.envs, replaying ? evolver.replayEnv : null);
  const trackedIdx = humanMode ? -1 : replaying ? 0 : evolver.trackedIndex();
  const env = humanMode ? humanEnv : evolver.sceneEnv();
  drawFrame(ctx, env, birds, trackedIdx, humanMode ? humanEnv.y : null, humanMode && humanDead);

  updateTrackedBuffers();
  if (lastLayers && lastWeights) {
    brainView.update(lastInputs, lastLayers, lastWeights);
  }
  drawChart(chartCanvas, evolver.stats.historyBest, evolver.stats.historyAvg);
}

function updateHUD(): void {
  const s = evolver.stats;
  const parts = [
    `gen ${s.generation}`,
    `alive ${s.alive}`,
    `best this gen ${s.bestScoreThisGen}`,
    `champion score ${s.championScore} (fitness ${s.championFitness})`,
  ];
  if (s.inReplay) parts.push("champion replay…");
  if (humanMode) {
    parts.push(`you: score ${humanEnv.score}${humanDead ? " — dead, press Space to restart" : ""}`);
  }
  statsEl.textContent = parts.join("  ·  ");
}

// --- Loop --------------------------------------------------------------

function frame(): void {
  const steps = running ? (humanMode ? 1 : speeds()) : 0;
  for (let i = 0; i < steps; i++) {
    if (humanMode) {
      stepHuman();
    } else {
      evolver.step();
    }
  }
  render();
  updateHUD();
  requestAnimationFrame(frame);
}

// --- Controls ----------------------------------------------------------

btnStart.addEventListener("click", () => {
  running = !running;
  btnStart.textContent = running ? "Pause" : "Resume";
  if (running && humanMode && humanDead) startHumanRun();
});

btnReset.addEventListener("click", () => {
  evolver = new Evolver(Date.now() % 0xffffffff);
  humanEnv = new FlappyEnv(makeSchedule(mulberry32(999), 80));
  humanDead = false;
  running = false;
  humanMode = false;
  btnStart.textContent = "Start evolution";
  btnHuman.textContent = "Play as human";
});

function startHumanRun(): void {
  humanEnv = new FlappyEnv(makeSchedule(mulberry32(999), 80));
  humanDead = false;
}

btnHuman.addEventListener("click", () => {
  humanMode = !humanMode;
  btnHuman.textContent = humanMode ? "Back to evolution" : "Play as human";
  if (humanMode) {
    running = true;
    btnStart.textContent = "Pause";
    startHumanRun();
  } else {
    running = false;
    btnStart.textContent = "Start evolution";
  }
});

btnSave.addEventListener("click", () => {
  const champion = evolver.pop.champion;
  if (!champion) {
    actionEl.textContent = "save failed: no champion yet";
    return;
  }
  localStorage.setItem(CHAMPION_KEY, serializeChampion(champion, evolver.pop.generation));
  actionEl.textContent = `saved champion (gen ${evolver.pop.generation})`;
});

btnLoad.addEventListener("click", () => {
  const raw = localStorage.getItem(CHAMPION_KEY);
  if (!raw) {
    actionEl.textContent = "load failed: no saved champion";
    return;
  }
  const loaded = deserializeChampion(raw, SIZES);
  if (!loaded.ok) {
    actionEl.textContent = `load failed: ${loaded.error}`;
    return;
  }
  evolver.loadChampion(loaded.network);
  actionEl.textContent = `loaded champion from gen ${loaded.gen}`;
});

speedSelect.addEventListener("change", () => {
  if (!humanMode) {
    running = true;
    btnStart.textContent = "Pause";
  }
});

window.addEventListener("keydown", (e) => {
  if (e.code === "Space") {
    e.preventDefault();
    if (humanMode) {
      flapHeld = true;
      if (humanDead) startHumanRun();
    }
  }
  if (e.code === "KeyP") {
    running = !running;
    btnStart.textContent = running ? "Pause" : "Resume";
  }
});

window.addEventListener("keyup", (e) => {
  if (e.code === "Space") flapHeld = false;
});

// --- Debug hook for headless verification ------------------------------

declare global {
  interface Window {
    __nn?: {
      getStats: () => {
        generation: number;
        alive: number;
        bestScoreThisGen: number;
        championFitness: number;
        championScore: number;
        inReplay: boolean;
      };
    };
  }
}

window.__nn = {
  getStats: () => evolver.stats,
};

// --- Boot --------------------------------------------------------------

canvas.setAttribute("width", "480");
canvas.setAttribute("height", String(GAME_H));
chartCanvas.width = chartCanvas.clientWidth * 2 || 520;
chartCanvas.height = 110;
requestAnimationFrame(frame);
