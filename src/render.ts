// Canvas rendering for the game viewport. Pure presentation: reads env
// states, draws background, pipes, and birds.

import {
  BIRD_R,
  BIRD_X,
  GAME_H,
  GAME_W,
  GAP_HEIGHT,
  PIPE_WIDTH,
  type FlappyEnv,
} from "./game/flappy";

export interface BirdPose {
  y: number;
  dead: boolean;
}

export function drawFrame(
  ctx: CanvasRenderingContext2D,
  env: FlappyEnv,
  birds: BirdPose[],
  trackedIdx: number,
  humanY: number | null,
  humanDead: boolean,
): void {
  // Background
  const grad = ctx.createLinearGradient(0, 0, 0, GAME_H);
  grad.addColorStop(0, "#0b1e33");
  grad.addColorStop(1, "#0f172a");
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, GAME_W, GAME_H);

  // Pipes (identical schedule for every bird, so one set renders all)
  for (const p of env.pipes) {
    const topH = p.gapY - GAP_HEIGHT / 2;
    ctx.fillStyle = "#2d6a4f";
    ctx.fillRect(p.x, 0, PIPE_WIDTH, topH);
    ctx.fillRect(p.x, p.gapY + GAP_HEIGHT / 2, PIPE_WIDTH, GAME_H);
    ctx.fillStyle = "#40916c";
    ctx.fillRect(p.x - 3, topH - 16, PIPE_WIDTH + 6, 16);
    ctx.fillRect(p.x - 3, p.gapY + GAP_HEIGHT / 2, PIPE_WIDTH + 6, 16);
  }

  // Population birds
  for (let i = 0; i < birds.length; i++) {
    if (birds[i].dead) continue;
    const tracked = i === trackedIdx;
    ctx.beginPath();
    ctx.arc(BIRD_X, birds[i].y, BIRD_R, 0, Math.PI * 2);
    ctx.fillStyle = tracked ? "#facc15" : "rgba(148, 163, 184, 0.5)";
    ctx.fill();
    if (tracked) {
      ctx.strokeStyle = "#fde047";
      ctx.lineWidth = 2.5;
      ctx.stroke();
    }
  }

  // Human bird
  if (humanY !== null) {
    ctx.beginPath();
    ctx.arc(BIRD_X, humanY, BIRD_R, 0, Math.PI * 2);
    ctx.fillStyle = humanDead ? "#7f1d1d" : "#fb923c";
    ctx.fill();
    ctx.strokeStyle = "#fdba74";
    ctx.lineWidth = 2.5;
    ctx.stroke();
  }
}
