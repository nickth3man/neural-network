import { describe, expect, it } from "vitest";
import { Network, sigmoid, tanh } from "../src/nn/network";

describe("Network.forward", () => {
  it("matches a hand-computed forward pass", () => {
    // input=2, hidden=2, output=1
    // h0 = tanh(1.0*x0 + 0*x1 + 0) ; h1 = tanh(0*x0 + 1*x1 + 0)
    // out = sigmoid(1*h0 + 1*h1 + 0)
    const net = new Network({ input: 2, hidden: 2, output: 1 });
    net.weights.set([1, 0, 0, 0, 1, 0, 1, 1, 0]);
    const inputs = new Float32Array([0.5, -0.25]);
    const out = net.forward(inputs, net.createResult());
    const expected = sigmoid(tanh(0.5) + tanh(-0.25));
    // Network math runs in float32; allow float32 rounding (~1e-8).
    expect(out.outputs[0]).toBeCloseTo(expected, 7);
    expect(out.layers[0][0]).toBeCloseTo(tanh(0.5), 7);
    expect(out.layers[0][1]).toBeCloseTo(tanh(-0.25), 7);
    expect(out.layers[1][0]).toBeCloseTo(expected, 7);
  });

  it("is deterministic for identical inputs", () => {
    const net = new Network({ input: 3, hidden: 4, output: 2 });
    net.weights.set(Array.from({ length: net.weights.length }, (_, i) => Math.sin(i) * 0.5));
    const inputs = new Float32Array([0.1, -0.7, 0.9]);
    const a = net.forward(inputs, net.createResult());
    const b = net.forward(inputs, net.createResult());
    expect(Array.from(a.outputs)).toEqual(Array.from(b.outputs));
  });

  it("bounds outputs to (0,1) and hidden activations to (-1,1)", () => {
    const net = new Network({ input: 2, hidden: 3, output: 1 });
    net.weights.set(Array.from({ length: net.weights.length }, () => 2)); // near-saturating but not rounded to 1.0 in f32
    const result = net.forward(new Float32Array([1, -1]), net.createResult());
    expect(result.outputs[0]).toBeGreaterThan(0);
    expect(result.outputs[0]).toBeLessThan(1);
    for (const h of result.layers[0]) {
      expect(Math.abs(h)).toBeLessThanOrEqual(1);
    }
  });

  it("rejects wrong input length", () => {
    const net = new Network({ input: 2, hidden: 2, output: 1 });
    expect(() => net.forward(new Float32Array([1]), net.createResult())).toThrow();
  });

  it("rejects malformed weight buffers", () => {
    expect(() => new Network({ input: 2, hidden: 2, output: 1 }, new Float32Array(3))).toThrow();
  });
});

describe("Network serialization", () => {
  it("round-trips weights and behavior through JSON", () => {
    const net = new Network({ input: 2, hidden: 3, output: 1 });
    net.weights.set(Array.from({ length: net.weights.length }, (_, i) => (i % 3) * 0.37 - 0.5));
    const restored = Network.fromJSON(net.toJSON());
    const inputs = new Float32Array([0.3, 0.8]);
    const a = net.forward(inputs, net.createResult());
    const b = restored.forward(inputs, restored.createResult());
    expect(Array.from(a.outputs)).toEqual(Array.from(b.outputs));
    expect(Array.from(a.layers[0])).toEqual(Array.from(b.layers[0]));
  });
});

describe("Network.clone", () => {
  it("produces an independent copy", () => {
    const net = new Network({ input: 1, hidden: 1, output: 1 });
    net.weights.set([1, 0, 1, 0]);
    const copy = net.clone();
    copy.weights[0] = 99;
    expect(net.weights[0]).toBe(1);
    expect(copy.weights[0]).toBe(99);
  });
});
