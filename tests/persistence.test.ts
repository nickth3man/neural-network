import { describe, expect, it } from "vitest";
import { Network } from "../src/nn/network";
import { deserializeChampion, serializeChampion } from "../src/persistence";

const SIZES = { input: 5, hidden: 4, output: 1 };

function champion(): Network {
  const net = new Network(SIZES);
  net.weights.set(Array.from({ length: net.weights.length }, (_, i) => Math.sin(i) * 0.5));
  return net;
}

describe("champion persistence", () => {
  it("round-trips a champion and preserves its behavior", () => {
    const net = champion();
    const loaded = deserializeChampion(serializeChampion(net, 7), SIZES);
    expect(loaded.ok).toBe(true);
    if (!loaded.ok) return;
    expect(loaded.gen).toBe(7);
    const inputs = new Float32Array([0.1, -0.2, 0.3, -0.4, 0.5]);
    const a = net.forward(inputs, net.createResult());
    const b = loaded.network.forward(inputs, loaded.network.createResult());
    expect(Array.from(b.outputs)).toEqual(Array.from(a.outputs));
  });

  it("rejects a saved champion with a different topology", () => {
    const raw = serializeChampion(champion(), 3);
    const loaded = deserializeChampion(raw, { input: 3, hidden: 8, output: 1 });
    expect(loaded.ok).toBe(false);
    if (loaded.ok) return;
    expect(loaded.error).toContain("topology");
  });

  it("rejects malformed JSON and missing weight arrays", () => {
    const invalidJson = deserializeChampion("{not json", SIZES);
    expect(invalidJson.ok).toBe(false);
    const missingWeights = deserializeChampion(JSON.stringify({ gen: 1, sizes: SIZES }), SIZES);
    expect(missingWeights.ok).toBe(false);
  });

  it("rejects non-finite and wrongly sized weight arrays", () => {
    const nonFinite = deserializeChampion(
      JSON.stringify({ gen: 1, sizes: SIZES, weights: [1, 2, NaN] }),
      SIZES,
    );
    expect(nonFinite.ok).toBe(false);
    const short = deserializeChampion(
      JSON.stringify({ gen: 1, sizes: SIZES, weights: [0, 0] }),
      SIZES,
    );
    expect(short.ok).toBe(false);
  });
});
