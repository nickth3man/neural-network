# neural-network

Two demonstrations of learning agents:

* **[`racer/`](racer/)** — the main project: a Rust racing game with procedurally
  generated tracks and a neural-network driver evolved to complete two clean
  laps on unseen layouts. Live viewer, headless training, evaluation and
  screenshots. See [`racer/README.md`](racer/README.md).
* The original TypeScript Flappy Bird neuroevolution demo remains at the
  repository root (`index.html`, `src/`, `tests/`).

```bash
cd racer
cargo run --release -- watch --model models/champion.json
```
