//! Acceptance test for the shipped champion model: two clean laps on unseen
//! random tracks. This is the deliverable's behavioral contract.

use racer::eval::{eval_seeds, evaluate_on_seeds};
use racer::model::load;

#[test]
fn shipped_champion_completes_two_laps_on_unseen_tracks() {
    let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let path = dir.join("models/champion.json");
    let model = load(&path).expect("champion model parses");
    assert_eq!(
        model.net().input_size(),
        racer::sim::INPUTS,
        "champion must match the current observation size"
    );
    let seeds = eval_seeds(1_600_000, 20);
    let runs = evaluate_on_seeds(&model.net(), &seeds, 1.0, 2.0, 150.0);
    for r in &runs {
        assert!(
            r.finished && !r.crashed,
            "failed track {}: laps {:.2}, crashed {}, progress {:.1}m",
            r.seed,
            r.laps,
            r.crashed,
            r.progress
        );
    }
}
