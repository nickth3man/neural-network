// Feed-forward neural network with one hidden layer.
// Pure math, no RNG and no DOM: runs headless and is fully unit-testable.
//
// Weight layout (single Float32Array):
//   w1: hidden * (input + 1)     — per hidden neuron: [input weights..., bias]
//   w2: output * (hidden + 1)    — per output neuron: [hidden weights..., bias]

export interface NetworkSizes {
  input: number;
  hidden: number;
  output: number;
}

export interface ForwardResult {
  /** final outputs (sigmoid, in (0,1)) */
  outputs: Float32Array;
  /** activations per layer after the input: [hidden (tanh), output (sigmoid)] */
  layers: Float32Array[];
}

const TANH_LIMIT = 20; // guard against exp overflow

export function tanh(x: number): number {
  if (x > TANH_LIMIT) return 1;
  if (x < -TANH_LIMIT) return -1;
  const e = Math.exp(2 * x);
  return (e - 1) / (e + 1);
}

export function sigmoid(x: number): number {
  if (x >= 0) {
    const z = Math.exp(-x);
    return 1 / (1 + z);
  }
  const z = Math.exp(x);
  return z / (1 + z);
}

export class Network {
  readonly sizes: NetworkSizes;
  readonly weights: Float32Array;
  readonly offW1: number;
  readonly lenW1: number;
  readonly offW2: number;
  readonly lenW2: number;

  constructor(sizes: NetworkSizes, weights?: Float32Array) {
    const { input, hidden, output } = sizes;
    if (input < 1 || hidden < 1 || output < 1) {
      throw new Error(`invalid sizes ${JSON.stringify(sizes)}`);
    }
    this.sizes = sizes;
    this.offW1 = 0;
    this.lenW1 = hidden * (input + 1);
    this.offW2 = this.lenW1;
    this.lenW2 = output * (hidden + 1);
    const total = this.lenW1 + this.lenW2;
    if (weights && weights.length !== total) {
      throw new Error(`weights length ${weights.length} != expected ${total}`);
    }
    this.weights = weights ? new Float32Array(weights) : new Float32Array(total);
  }

  /** Reusable forward result buffer — avoids per-call allocation in the evolution loop. */
  createResult(): ForwardResult {
    return {
      outputs: new Float32Array(this.sizes.output),
      layers: [new Float32Array(this.sizes.hidden), new Float32Array(this.sizes.output)],
    };
  }

  forward(inputs: Float32Array, out: ForwardResult): ForwardResult {
    if (inputs.length !== this.sizes.input) {
      throw new Error(`expected ${this.sizes.input} inputs, got ${inputs.length}`);
    }
    const { hidden, output } = this.sizes;
    const w = this.weights;
    const hid = out.layers[0];
    const outL = out.layers[1];

    // hidden layer (tanh)
    for (let i = 0; i < hidden; i++) {
      const row = i * (this.sizes.input + 1);
      let sum = w[this.offW1 + row + this.sizes.input]; // bias
      for (let j = 0; j < this.sizes.input; j++) {
        sum += w[this.offW1 + row + j] * inputs[j];
      }
      hid[i] = tanh(sum);
    }

    // output layer (sigmoid)
    for (let k = 0; k < output; k++) {
      const row = k * (hidden + 1);
      let sum = w[this.offW2 + row + hidden]; // bias
      for (let i = 0; i < hidden; i++) {
        sum += w[this.offW2 + row + i] * hid[i];
      }
      outL[k] = sigmoid(sum);
    }
    out.outputs.set(outL);
    return out;
  }

  clone(): Network {
    return new Network(this.sizes, this.weights);
  }

  /** JSON-safe serialization for save/load. */
  toJSON(): { sizes: NetworkSizes; weights: number[] } {
    return { sizes: { ...this.sizes }, weights: Array.from(this.weights) };
  }

  static fromJSON(json: { sizes: NetworkSizes; weights: number[] }): Network {
    return new Network(json.sizes, Float32Array.from(json.weights));
  }
}
