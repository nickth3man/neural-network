// Champion persistence: the payload written to localStorage is untrusted
// input when it is read back (corrupt storage, an older/newer build, or a
// hand-edited value), so decode it before it can reach per-bird buffers.

import { Network, type NetworkSizes } from "./nn/network";

export interface ChampionRecord {
  gen: number;
  sizes: NetworkSizes;
  weights: number[];
}

export type ChampionLoad =
  | { ok: true; gen: number; network: Network }
  | { ok: false; error: string };

/** Serialize a champion in the shape read by deserializeChampion(). */
export function serializeChampion(net: Network, gen: number): string {
  return JSON.stringify({ gen, ...net.toJSON() } satisfies ChampionRecord);
}

/** Decode and validate a saved champion against the running topology. */
export function deserializeChampion(raw: string, expected: NetworkSizes): ChampionLoad {
  let parsed: unknown;
  try {
    parsed = JSON.parse(raw);
  } catch {
    return { ok: false, error: "saved champion is not valid JSON" };
  }
  if (typeof parsed !== "object" || parsed === null) {
    return { ok: false, error: "saved champion is not an object" };
  }
  const record = parsed as Partial<ChampionRecord>;
  if (
    !record.sizes ||
    record.sizes.input !== expected.input ||
    record.sizes.hidden !== expected.hidden ||
    record.sizes.output !== expected.output
  ) {
    return {
      ok: false,
      error: `saved topology does not match ${expected.input}/${expected.hidden}/${expected.output}`,
    };
  }
  if (!Array.isArray(record.weights)) {
    return { ok: false, error: "saved champion has no weight array" };
  }
  if (record.weights.some((w) => typeof w !== "number" || !Number.isFinite(w))) {
    return { ok: false, error: "saved champion has non-finite weights" };
  }
  try {
    const network = new Network(record.sizes, Float32Array.from(record.weights));
    const gen =
      typeof record.gen === "number" && Number.isFinite(record.gen) ? record.gen : 0;
    return { ok: true, gen, network };
  } catch (err) {
    return {
      ok: false,
      error: err instanceof Error ? err.message : "invalid saved champion",
    };
  }
}
