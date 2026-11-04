import { describe, expect, it } from "vitest";
import { FlappyEnv } from "../src/game/flappy";
import { drawFrame, poseBirds } from "../src/render";

interface Arc {
  x: number;
  y: number;
  r: number;
  fill: string;
}

function fakeCtx(arcs: Arc[]): CanvasRenderingContext2D {
  let pending: Arc | null = null;
  const ctx = {
    fillStyle: "",
    strokeStyle: "",
    lineWidth: 0,
    createLinearGradient: () => ({ addColorStop: () => {} }),
    fillRect: () => {},
    beginPath: () => {},
    arc: (x: number, y: number, r: number) => {
      pending = { x, y, r, fill: String(ctx.fillStyle) };
      arcs.push(pending);
    },
    fill: () => {
      if (pending) pending.fill = String(ctx.fillStyle);
    },
    stroke: () => {},
  };
  return ctx as unknown as CanvasRenderingContext2D;
}

function envAt(y: number): FlappyEnv {
  const env = new FlappyEnv([{ delay: 1.2, gapY: 320 }]);
  env.reset();
  env.y = y;
  return env;
}

describe("poseBirds", () => {
  it("returns every population pose during a live generation", () => {
    expect(poseBirds([{ y: 1, dead: false }, { y: 2, dead: true }], null)).toEqual([
      { y: 1, dead: false },
      { y: 2, dead: true },
    ]);
  });

  it("returns only the replay bird during a champion replay", () => {
    expect(poseBirds([{ y: 1, dead: false }, { y: 2, dead: false }], { y: 3, dead: false })).toEqual([
      { y: 3, dead: false },
    ]);
  });
});

describe("drawFrame", () => {
  it("draws one tracked champion at the replay position, not the reset population", () => {
    const replay = envAt(300);
    const arcs: Arc[] = [];
    drawFrame(
      fakeCtx(arcs),
      replay,
      poseBirds([{ y: 320, dead: false }, { y: 320, dead: false }], replay),
      0,
      null,
      false,
    );
    expect(arcs).toHaveLength(1);
    expect(arcs[0].y).toBe(300);
    expect(arcs[0].fill).toBe("#facc15");
  });

  it("skips dead population birds in a live frame", () => {
    const arcs: Arc[] = [];
    drawFrame(
      fakeCtx(arcs),
      envAt(320),
      poseBirds([{ y: 100, dead: false }, { y: 200, dead: true }], null),
      0,
      null,
      false,
    );
    expect(arcs).toHaveLength(1);
    expect(arcs[0].y).toBe(100);
  });
});
