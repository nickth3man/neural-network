//! Running a policy (learned network or baseline) through a full episode.

use serde::Serialize;

use crate::car::CarSpec;
use crate::nn::{ForwardPass, Net};
use crate::sim::{Episode, POLICY_EVERY};
use crate::track::Track;

pub const HIDDEN: [usize; 2] = [24, 16];

pub fn default_sizes() -> Vec<usize> {
    vec![crate::sim::INPUTS, HIDDEN[0], HIDDEN[1], 2]
}

#[derive(Clone, Debug, Serialize)]
pub struct RunStats {
    pub seed: u64,
    pub difficulty: f32,
    pub track_length: f32,
    pub finished: bool,
    pub crashed: bool,
    pub laps: f32,
    pub progress: f32,
    pub progress_frac: f32,
    pub time: f32,
    pub max_speed: f32,
    pub mean_abs_lateral: f32,
    pub fitness: f32,
}

pub fn stats(seed: u64, difficulty: f32, ep: &Episode) -> RunStats {
    let steps = ep.steps.max(1) as f32;
    RunStats {
        seed,
        difficulty,
        track_length: ep.track.length,
        finished: ep.finished,
        crashed: ep.crashed,
        laps: ep.laps(),
        progress: ep.progress,
        progress_frac: ep.progress / ep.track.length,
        time: ep.time,
        max_speed: ep.max_speed,
        mean_abs_lateral: ep.lateral_accum / steps,
        fitness: ep.fitness(),
    }
}

/// Drive an episode to completion (crash, finish or timeout) with a policy
/// called at the control rate.
pub fn rollout_with<F>(
    track: Track,
    laps: f32,
    time_limit: f32,
    mut policy: F,
) -> (Episode, RunStats)
where
    F: FnMut(&Episode) -> [f32; 2],
{
    let spec = CarSpec::default();
    let seed = track.seed;
    let difficulty = track.difficulty;
    let mut ep = Episode::new(track, &spec);
    ep.target_laps = laps;
    ep.time_limit = time_limit;
    while !ep.done {
        if ep.steps % POLICY_EVERY == 0 {
            let c = policy(&ep);
            ep.apply_controls(c[0], c[1]);
        }
        ep.advance(&spec);
        if ep.steps % POLICY_EVERY == 0 {
            ep.record_diagnostics();
        }
    }
    let st = stats(seed, difficulty, &ep);
    (ep, st)
}

pub fn rollout_net(net: &Net, track: Track, laps: f32, time_limit: f32) -> RunStats {
    let mut pass = ForwardPass::new();
    let (_, st) = rollout_with(track, laps, time_limit, |ep| {
        let obs = ep.observe();
        let out = net.forward(&obs, &mut pass);
        [out[0], out[1]]
    });
    st
}

pub fn rollout_baseline(track: Track, laps: f32, time_limit: f32) -> RunStats {
    let spec = CarSpec::default();
    let (_, st) = rollout_with(track, laps, time_limit, |ep| {
        crate::sim::Baseline::controls(ep, &spec)
    });
    st
}
