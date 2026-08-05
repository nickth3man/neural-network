// Live brain overlay (CNN-Explainer / Netron style): an SVG graph of the
// tracked network where edge thickness = |weight|, edge color = sign, and
// node brightness = current activation. Clicking a node shows a detail
// panel with its incoming weights and live value.

import type { NetworkSizes } from "./nn/network";

const SVG_NS = "http://www.w3.org/2000/svg";
const VIEW_W = 520;
const VIEW_H = 300;
const NODE_R = 13;
const COL_X = [90, 260, 430];
const POS_COLOR = "#3ddc84";
const NEG_COLOR = "#ff5d5d";

export const SENSOR_NAMES = [
  "bird y",
  "velocity",
  "pipe distance",
  "gap center",
  "gap offset",
];

interface EdgeSpec {
  line: SVGLineElement;
  /** index into the genome's weight array */
  weightIdx: number;
}

interface NodeSpec {
  circle: SVGCircleElement;
  label: SVGTextElement | null;
  valueText: SVGTextElement | null;
  x: number;
  y: number;
}

export interface NodeSelection {
  layer: "input" | "hidden" | "output";
  idx: number;
}

export class BrainView {
  private readonly svg: SVGSVGElement;
  private readonly detailEl: HTMLElement;
  private readonly sizes: NetworkSizes;
  private readonly inputNodes: NodeSpec[] = [];
  private readonly hiddenNodes: NodeSpec[] = [];
  private readonly outputNodes: NodeSpec[] = [];
  private readonly edges: EdgeSpec[] = [];
  private readonly onSelect: (sel: NodeSelection) => void;
  private selected: NodeSelection | null = null;

  constructor(container: HTMLElement, detailEl: HTMLElement, sizes: NetworkSizes, onSelect: (sel: NodeSelection) => void) {
    this.detailEl = detailEl;
    this.sizes = sizes;
    this.onSelect = onSelect;
    this.svg = document.createElementNS(SVG_NS, "svg");
    this.svg.setAttribute("viewBox", `0 0 ${VIEW_W} ${VIEW_H}`);
    container.appendChild(this.svg);
    this.buildLayout();
    this.svg.addEventListener("click", (ev) => {
      const target = ev.target as Element;
      const layer = target.getAttribute("data-layer");
      const idx = Number(target.getAttribute("data-idx"));
      if (layer && !Number.isNaN(idx)) this.selectNode(layer as NodeSelection["layer"], idx);
    });
  }

  private nodeY(count: number, i: number): number {
    return (VIEW_H / (count + 1)) * (i + 1);
  }

  private buildLayout(): void {
    const { input, hidden, output } = this.sizes;
    const mkNode = (layer: string, idx: number, x: number, y: number, label: string | null): NodeSpec => {
      const circle = document.createElementNS(SVG_NS, "circle");
      circle.setAttribute("cx", String(x));
      circle.setAttribute("cy", String(y));
      circle.setAttribute("r", String(NODE_R));
      circle.setAttribute("fill", "#1e2a3a");
      circle.setAttribute("stroke", "#30363d");
      circle.setAttribute("stroke-width", "1.5");
      circle.setAttribute("data-layer", layer);
      circle.setAttribute("data-idx", String(idx));
      circle.style.cursor = "pointer";
      const labelEl = label
        ? this.makeText(x - 74, y + 4, label, "end", "#8b949e")
        : null;
      const valueEl = this.makeText(x + NODE_R + 6, y + 4, "", "start", "#e6edf3");
      this.svg.appendChild(circle);
      if (labelEl) this.svg.appendChild(labelEl);
      this.svg.appendChild(valueEl);
      return { circle, label: labelEl, valueText: valueEl, x, y };
    };
    const mkEdge = (x1: number, y1: number, x2: number, y2: number): SVGLineElement => {
      const line = document.createElementNS(SVG_NS, "line");
      line.setAttribute("x1", String(x1));
      line.setAttribute("y1", String(y1));
      line.setAttribute("x2", String(x2));
      line.setAttribute("y2", String(y2));
      line.setAttribute("stroke", POS_COLOR);
      line.setAttribute("stroke-width", "1");
      line.setAttribute("opacity", "0.75");
      this.svg.appendChild(line);
      return line;
    };

    for (let i = 0; i < input; i++) {
      this.inputNodes.push(mkNode("input", i, COL_X[0], this.nodeY(input, i), SENSOR_NAMES[i] ?? `in${i}`));
    }
    for (let i = 0; i < hidden; i++) {
      this.hiddenNodes.push(mkNode("hidden", i, COL_X[1], this.nodeY(hidden, i), null));
    }
    for (let i = 0; i < output; i++) {
      this.outputNodes.push(mkNode("output", i, COL_X[2], this.nodeY(output, i), null));
    }

    // Edge weight layout mirrors Network's weight array:
    // w1 rows (hidden x input+1), then w2 rows (output x hidden+1).
    let weightIdx = 0;
    for (let i = 0; i < hidden; i++) {
      for (let j = 0; j < input; j++) {
        const a = this.inputNodes[j];
        const b = this.hiddenNodes[i];
        this.edges.push({ line: mkEdge(a.x, a.y, b.x, b.y), weightIdx });
        weightIdx++;
      }
      weightIdx++; // bias slot
    }
    for (let k = 0; k < output; k++) {
      for (let i = 0; i < hidden; i++) {
        const a = this.hiddenNodes[i];
        const b = this.outputNodes[k];
        this.edges.push({ line: mkEdge(a.x, a.y, b.x, b.y), weightIdx });
        weightIdx++;
      }
      weightIdx++; // bias slot
    }
  }

  private makeText(x: number, y: number, text: string, anchor: string, fill: string): SVGTextElement {
    const el = document.createElementNS(SVG_NS, "text");
    el.setAttribute("x", String(x));
    el.setAttribute("y", String(y));
    el.setAttribute("text-anchor", anchor);
    el.setAttribute("fill", fill);
    el.setAttribute("font-size", "11");
    el.textContent = text;
    return el;
  }

  /** Refresh activations and weights for the current tick. */
  update(inputs: Float32Array, layers: Float32Array[], weights: Float32Array): void {
    this.applyActivations(this.inputNodes, inputs);
    this.applyActivations(this.hiddenNodes, layers[0]);
    this.applyActivations(this.outputNodes, layers[1]);
    for (let e = 0; e < this.edges.length; e++) {
      const spec = this.edges[e];
      const w = weights[spec.weightIdx];
      const width = Math.min(5, 0.5 + Math.abs(w) * 2.2);
      spec.line.setAttribute("stroke-width", width.toFixed(2));
      spec.line.setAttribute("stroke", w >= 0 ? POS_COLOR : NEG_COLOR);
    }
    if (this.selected) this.renderDetail(this.selected, inputs, layers, weights);
  }

  private applyActivations(nodes: NodeSpec[], values: Float32Array): void {
    for (let i = 0; i < nodes.length; i++) {
      const a = i < values.length ? values[i] : 0;
      const mag = Math.abs(a);
      const color = a >= 0 ? POS_COLOR : NEG_COLOR;
      nodes[i].circle.setAttribute("fill", color);
      nodes[i].circle.setAttribute("fill-opacity", String(0.15 + 0.85 * Math.min(1, mag)));
      const v = nodes[i].valueText;
      if (v) v.textContent = a.toFixed(2);
    }
  }

  private selectNode(layer: NodeSelection["layer"], idx: number): void {
    this.selected = { layer, idx };
    for (const n of [...this.inputNodes, ...this.hiddenNodes, ...this.outputNodes]) {
      n.circle.setAttribute("stroke", "#30363d");
    }
    const map: Record<string, NodeSpec[]> = { input: this.inputNodes, hidden: this.hiddenNodes, output: this.outputNodes };
    map[layer][idx]?.circle.setAttribute("stroke", "#fde047");
    this.onSelect(this.selected);
  }

  /** Render the detail panel for the selected node (or clear with null). */
  renderDetail(sel: NodeSelection | null, inputs: Float32Array | null, layers: Float32Array[] | null, weights: Float32Array | null): void {
    if (!sel) {
      this.detailEl.textContent = "Click a neuron to inspect its weights.";
      return;
    }
    const { input, hidden, output } = this.sizes;
    const parts: string[] = [];

    if (sel.layer === "input") {
      const value = inputs ? inputs[sel.idx] : 0;
      parts.push(`<b>${SENSOR_NAMES[sel.idx] ?? `input ${sel.idx}`}</b> — value ${value.toFixed(3)}`);
      parts.push("Outgoing: hidden neurons weighted by:");
      const rows: [string, number][] = [];
      for (let i = 0; i < hidden; i++) {
        const w = weights ? weights[i * (input + 1) + sel.idx] : 0;
        rows.push([`h${i}`, w]);
      }
      rows.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      parts.push(rows.map(([name, w]) => `${name}: ${w.toFixed(2)}`).join(" &nbsp; "));
    } else if (sel.layer === "hidden") {
      const value = layers ? layers[0][sel.idx] : 0;
      parts.push(`<b>Hidden neuron ${sel.idx}</b> — activation ${value.toFixed(3)}`);
      const ins: [string, number][] = [];
      for (let j = 0; j < input; j++) {
        const w = weights ? weights[sel.idx * (input + 1) + j] : 0;
        ins.push([SENSOR_NAMES[j] ?? `in${j}`, w]);
      }
      ins.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      parts.push("Top sensors by weight: " + ins.slice(0, 3).map(([name, w]) => `${name} ${w.toFixed(2)}`).join(", "));
      const outs: [string, number][] = [];
      for (let k = 0; k < output; k++) {
        const w = weights ? weights[hidden * (input + 1) + k * (hidden + 1) + sel.idx] : 0;
        outs.push([`out${k}`, w]);
      }
      parts.push("Output: " + outs.map(([name, w]) => `${name} ${w.toFixed(2)}`).join(", "));
    } else {
      const value = layers ? layers[1][sel.idx] : 0;
      parts.push(`<b>Output ${sel.idx}</b> — flap probability ${value.toFixed(3)} (flap when &gt; 0.5)`);
      const ins: [string, number][] = [];
      for (let i = 0; i < hidden; i++) {
        const w = weights ? weights[hidden * (input + 1) + sel.idx * (hidden + 1) + i] : 0;
        ins.push([`h${i}`, w]);
      }
      ins.sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
      parts.push("Hidden inputs: " + ins.map(([name, w]) => `${name} ${w.toFixed(2)}`).join(", "));
    }
    this.detailEl.innerHTML = parts.join("<br/>");
  }
}
