//! Evaluation helpers: aggregate statistics over runs and batch evaluation.

use serde::{Deserialize, Serialize};

use crate::agent::{RunStats, rollout_net};
use crate::nn::Net;
use crate::rng::mix;
use crate::track::Track;

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Aggregate {
    pub runs: usize,
    pub successes: usize,
    pub success_rate: f32,
    pub crashed: usize,
    pub mean_laps: f32,
    pub min_laps: f32,
    pub mean_progress_frac: f32,
    pub mean_time: f32,
    pub mean_max_speed: f32,
    pub mean_abs_lateral: f32,
}

pub fn aggregate(runs: &[RunStats]) -> Aggregate {
    let mut a = Aggregate {
        runs: runs.len(),
        ..Default::default()
    };
    if runs.is_empty() {
        return a;
    }
    let mut time_sum = 0.0;
    let mut finished = 0;
    for r in runs {
        if r.finished && !r.crashed {
            a.successes += 1;
        }
        if r.crashed {
            a.crashed += 1;
        }
        a.mean_laps += r.laps;
        a.mean_progress_frac += r.progress_frac;
        a.mean_max_speed += r.max_speed;
        a.mean_abs_lateral += r.mean_abs_lateral;
        if r.finished {
            time_sum += r.time;
            finished += 1;
        }
        a.min_laps = if a.min_laps == 0.0 {
            r.laps
        } else {
            a.min_laps.min(r.laps)
        };
    }
    let n = runs.len() as f32;
    a.success_rate = a.successes as f32 / n;
    a.mean_laps /= n;
    a.mean_progress_frac /= n;
    a.mean_max_speed /= n;
    a.mean_abs_lateral /= n;
    a.mean_time = if finished > 0 {
        time_sum / finished as f32
    } else {
        0.0
    };
    a
}

/// Lower is better ordering for "how good is this aggregate", used to keep the
/// best champion seen during training.
pub fn score(a: &Aggregate) -> f32 {
    a.success_rate * 1000.0 + a.mean_progress_frac * 100.0 - a.mean_time * 0.2
}

pub fn eval_seeds(base: u64, n: usize) -> Vec<u64> {
    (0..n).map(|i| mix(base, 0x7E57_0001, i as u64)).collect()
}

pub fn evaluate_on_seeds(
    net: &Net,
    seeds: &[u64],
    difficulty: f32,
    laps: f32,
    time_limit: f32,
) -> Vec<RunStats> {
    use rayon::prelude::*;
    assert_eq!(
        net.input_size(),
        crate::sim::INPUTS,
        "model expects {} inputs but this build produces {}; retrain the model",
        net.input_size(),
        crate::sim::INPUTS
    );
    seeds
        .par_iter()
        .map(|&seed| {
            let track = Track::generate(seed, difficulty);
            rollout_net(net, track, laps, time_limit)
        })
        .collect()
}
