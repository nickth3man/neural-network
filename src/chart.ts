// Tiny fitness-history chart: best and average per generation.

const BEST_COLOR = "#3ddc84";
const AVG_COLOR = "#58a6ff";

export function drawChart(
  canvas: HTMLCanvasElement,
  best: number[],
  avg: number[],
): void {
  const ctx = canvas.getContext("2d");
  if (!ctx) return;
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  if (best.length < 2) {
    ctx.fillStyle = "#8b949e";
    ctx.font = "12px system-ui";
    ctx.fillText("Waiting for generations…", 12, h / 2);
    return;
  }

  let max = 0;
  for (const v of best) {
    if (v > max) max = v;
  }
  for (const v of avg) {
    if (v > max) max = v;
  }
  max = Math.max(max, 1);

  // Grid
  ctx.strokeStyle = "#21262d";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  ctx.fillStyle = "#8b949e";
  ctx.font = "10px system-ui";
  ctx.fillText(`max ${Math.round(max)}`, 4, 10);

  const x = (i: number) => (i / (best.length - 1)) * (w - 8) + 4;
  const y = (v: number) => h - 4 - (v / max) * (h - 14);

  const polyline = (values: number[], color: string): void => {
    ctx.strokeStyle = color;
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i < values.length; i++) {
      const px = x(i);
      const py = y(values[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
  };
  polyline(best, BEST_COLOR);
  polyline(avg, AVG_COLOR);
}
