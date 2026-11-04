//! Episode simulation: sensors, collisions, progress and lap accounting.

use crate::car::{Car, CarSpec};
use crate::math::{Vec2, angle_delta, clampf};
use crate::track::Track;

pub const DT: f32 = 1.0 / 60.0;
/// The policy runs at 30 Hz while physics runs at 60 Hz.
pub const POLICY_EVERY: u32 = 2;
pub const N_RAYS: usize = 9;
pub const RAY_SPAN: f32 = 80.0_f32.to_radians();
pub const RAY_RANGE: f32 = 34.0;
pub const INPUTS: usize = N_RAYS + 9;
pub const TARGET_LAPS: f32 = 2.0;
pub const TIME_LIMIT: f32 = 150.0;

#[derive(Clone, Debug)]
pub struct Episode {
    pub track: Track,
    pub car: Car,
    /// Unwrapped distance travelled along the centerline (m).
    pub progress: f32,
    pub prev_s: f32,
    pub center_idx: usize,
    pub lateral: f32,
    /// Largest |lateral| / half-width seen this episode; used to reward margin.
    pub max_lateral_ratio: f32,
    /// Integrated "braking too late for the previewed curvature" measure.
    pub safety_violation: f32,
    pub heading_error: f32,
    pub controls: [f32; 2],
    pub sensors: [f32; N_RAYS],
    pub sensor_hits: [Option<Vec2>; N_RAYS],
    pub time: f32,
    pub steps: u32,
    pub crashed: bool,
    pub finished: bool,
    pub done: bool,
    pub target_laps: f32,
    pub time_limit: f32,
    pub max_speed: f32,
    pub lateral_accum: f32,
}

impl Episode {
    pub fn new(track: Track, spec: &CarSpec) -> Self {
        // Spawn slightly behind the line so the first crossing arms the lap.
        let start = track.center[0];
        let car = Car::new(start, track.start_heading);
        let (idx, lateral, s) = track.project(start, Some(0));
        let mut ep = Self {
            track,
            car,
            progress: 0.0,
            prev_s: s,
            center_idx: idx,
            lateral,
            max_lateral_ratio: 0.0,
            safety_violation: 0.0,
            heading_error: 0.0,
            controls: [0.0, 0.0],
            sensors: [0.0; N_RAYS],
            sensor_hits: [None; N_RAYS],
            time: 0.0,
            steps: 0,
            crashed: false,
            finished: false,
            done: false,
            target_laps: TARGET_LAPS,
            time_limit: TIME_LIMIT,
            max_speed: 0.0,
            lateral_accum: 0.0,
        };
        ep.refresh_observation();
        let _ = spec;
        ep
    }

    /// Accumulate per-control-tick diagnostics for evaluation reports.
    pub fn record_diagnostics(&mut self) {
        self.max_speed = self.max_speed.max(self.car.speed);
        self.lateral_accum += self.lateral.abs();
    }

    #[inline]
    pub fn laps(&self) -> f32 {
        self.progress / self.track.length
    }

    #[inline]
    pub fn body_center(&self, spec: &CarSpec) -> Vec2 {
        self.car.body_center(spec)
    }

    /// Recompute sensors and the derived features used by the network.
    pub fn refresh_observation(&mut self) {
        let spec = CarSpec::default();
        let origin = self.car.body_center(&spec);
        for (k, ray) in self.sensors.iter_mut().enumerate() {
            let angle =
                self.car.heading - RAY_SPAN * 0.5 + RAY_SPAN * k as f32 / (N_RAYS - 1) as f32;
            let dir = Vec2::from_angle(angle);
            match self.track.raycast(origin, dir, RAY_RANGE, self.center_idx) {
                Some((d, hit)) => {
                    *ray = clampf(1.0 - d / RAY_RANGE, 0.0, 1.0);
                    self.sensor_hits[k] = Some(hit);
                }
                None => {
                    *ray = 0.0;
                    self.sensor_hits[k] = None;
                }
            }
        }
        let hw = self.track.half_width(self.center_idx).max(1.0);
        self.max_lateral_ratio = self
            .max_lateral_ratio
            .max((self.lateral.abs() / hw).clamp(0.0, 1.0));
    }

    /// Network inputs, all in roughly [-1, 1]. Besides the lidar fan, the
    /// driver sees signed curvature samples and the worst curvature in two
    /// lookahead windows, which is what makes braking in time learnable.
    pub fn observe(&self) -> [f32; INPUTS] {
        let mut o = [0.0f32; INPUTS];
        o[..N_RAYS].copy_from_slice(&self.sensors);
        let hw = self.track.half_width(self.center_idx).max(1.0);
        o[N_RAYS] = clampf(self.car.speed / CarSpec::default().max_speed, -1.0, 1.0);
        o[N_RAYS + 1] = clampf(self.lateral / hw, -1.0, 1.0);
        o[N_RAYS + 2] = clampf(self.heading_error / std::f32::consts::PI, -1.0, 1.0);
        let n = self.track.n();
        let advance = |meters: f32| {
            let s = self.progress + meters;
            let i = ((s / self.track.ds).rem_euclid(n as f32)) as usize % n;
            self.track.curvature[i]
        };
        o[N_RAYS + 3] = clampf(advance(22.0) * 24.0, -1.0, 1.0);
        o[N_RAYS + 4] = clampf(advance(60.0) * 50.0, -1.0, 1.0);
        // Worst curvature and where it occurs, per lookahead window.
        let worst = |from: f32, to: f32| {
            let mut m = 0.0f32;
            let mut at = from;
            let mut d = from;
            while d < to {
                let k = advance(d).abs();
                if k > m {
                    m = k;
                    at = d;
                }
                d += 2.5;
            }
            (m, at)
        };
        let (near_k, _) = worst(2.0, 45.0);
        let (far_k, _) = worst(45.0, 90.0);
        o[N_RAYS + 5] = clampf(near_k * 26.0, 0.0, 1.0);
        o[N_RAYS + 6] = clampf(far_k * 55.0, 0.0, 1.0);
        o[N_RAYS + 7] = clampf(self.car.yaw_rate(&CarSpec::default()) / 2.5, -1.0, 1.0);
        let need = self.brake_need(2.0, 90.0);
        o[N_RAYS + 8] = clampf(need, 0.0, 2.0) * 0.5;
        o
    }

    /// Deceleration needed right now to reach the tightest sample in
    /// `[from, to]` at the grip-limited speed. >1 means even full braking is
    /// late, so this is both a network input and a training penalty.
    pub fn brake_need(&self, from: f32, to: f32) -> f32 {
        let n = self.track.n();
        let advance = |m: f32| {
            let s = self.progress + m;
            let i = ((s / self.track.ds).rem_euclid(n as f32)) as usize % n;
            self.track.curvature[i].abs()
        };
        let mut k = 0.0f32;
        let mut at = from;
        let mut d = from;
        while d < to {
            let v = advance(d);
            if v > k {
                k = v;
                at = d;
            }
            d += 2.5;
        }
        let spec = CarSpec::default();
        let v_safe = (spec.max_lat_accel / k.max(1e-4)).sqrt();
        (self.car.speed * self.car.speed - v_safe * v_safe).max(0.0)
            / (2.0 * spec.brake_accel * at.max(2.0))
    }

    #[inline]
    fn accumulate_safety(&mut self) {
        let need = self.brake_need(2.0, 90.0);
        let throttle = self.controls[0].max(0.0);
        if need > 0.5 {
            self.safety_violation += (need - 0.5) * throttle * DT;
        }
    }

    pub fn apply_controls(&mut self, throttle: f32, steer: f32) {
        self.controls = [clampf(throttle, -1.0, 1.0), clampf(steer, -1.0, 1.0)];
        self.refresh_observation();
    }

    pub fn advance(&mut self, spec: &CarSpec) {
        if self.done {
            return;
        }
        self.time += DT;
        self.steps += 1;
        self.car.step(self.controls[0], self.controls[1], DT, spec);

        let (idx, lateral, s) = self.track.project(self.car.pos, Some(self.center_idx));
        let mut delta = s - self.prev_s;
        if delta > self.track.length * 0.5 {
            delta -= self.track.length;
        } else if delta < -self.track.length * 0.5 {
            delta += self.track.length;
        }
        self.progress += delta;
        self.prev_s = s;
        self.center_idx = idx;
        self.lateral = lateral;
        self.heading_error = angle_delta(self.car.heading, self.track.tangent[idx].angle());
        self.accumulate_safety();

        // Off-road check on the body corners, so the car cannot hang its nose
        // over the grass without registering a crash.
        let mut off_road = false;
        for c in self.car.corners(spec) {
            let (ci, clat, _) = self.track.project(c, Some(self.center_idx));
            if clat.abs() > self.track.half_width(ci) {
                off_road = true;
                break;
            }
        }
        if off_road || self.progress < -self.track.length * 0.25 {
            self.crashed = true;
            self.done = true;
        }
        if self.progress >= self.target_laps * self.track.length {
            self.finished = true;
            self.done = true;
        }
        if self.time >= self.time_limit {
            self.done = true;
        }
    }

    /// Selection objective. Every metre is positive, so the gradient never
    /// disappears; finishing adds a large flat bonus plus a speed bonus, so a
    /// clean two laps always beats a fast near-miss at the same distance.
    pub fn fitness(&self) -> f32 {
        let cap = self.target_laps * self.track.length;
        let mut f = self.progress.clamp(0.0, cap);
        if self.finished {
            f += 500.0 + 150.0 * (1.0 - self.time / self.time_limit).max(0.0);
        } else if self.crashed {
            // Still monotone in distance, but a clean slow run is worth much
            // more than a fast crash: the deliverable is crash-free laps.
            f *= 0.5;
        }
        // Keep a margin from the walls: touching the edge leaves no room for
        // the next bend.
        f *= 1.0 - 0.6 * (self.max_lateral_ratio - 0.70).max(0.0);
        // Penalise carrying throttle when the preview says braking is already
        // needed. Multiplicative so the score stays positive and monotone.
        f /= 1.0 + 2.0 * self.safety_violation;
        f
    }
}

/// Hand-written pure-pursuit driver used to validate physics and lap counting.
/// It is deliberately not used by the learned agent.
pub struct Baseline;

impl Baseline {
    pub fn controls(ep: &Episode, spec: &CarSpec) -> [f32; 2] {
        let n = ep.track.n();
        let look = 7.0 + ep.car.speed * 0.55;
        let s = ep.progress + look;
        let ti = ((s / ep.track.ds).rem_euclid(n as f32)) as usize % n;
        let target = ep.track.center[ti];
        let origin = ep.car.body_center(spec);
        let alpha = angle_delta(ep.car.heading, (target - origin).angle());
        let delta = (2.0 * spec.wheelbase * alpha.sin() / look.max(1.0)).atan();
        let steer = clampf(delta / spec.max_steer, -1.0, 1.0);

        // Racing speed profile: the fastest speed now that still allows
        // braking down to the corner speed of every upcoming bend.
        let horizon = 12.0 + ep.car.speed * 2.4;
        let steps = (horizon / ep.track.ds).ceil() as usize;
        let mut v_target = spec.max_speed;
        for k in 0..steps {
            let d = 2.0 + k as f32 * ep.track.ds;
            let i = ((ep.progress + d) / ep.track.ds).rem_euclid(n as f32) as usize % n;
            let kappa = ep.track.curvature[i].abs();
            if kappa > 1e-5 {
                let v_curve = (6.0 / kappa).sqrt();
                let v_now = (v_curve * v_curve + 2.0 * 8.0 * d).sqrt();
                v_target = v_target.min(v_now);
            }
        }
        v_target = v_target.clamp(5.0, spec.max_speed);
        let throttle = if ep.car.speed < v_target - 0.4 {
            1.0
        } else if ep.car.speed > v_target + 0.7 {
            -1.0
        } else {
            ((v_target - ep.car.speed) * 1.5).clamp(-1.0, 1.0)
        };
        [throttle, steer]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn drive_baseline(seed: u64, difficulty: f32, laps: f32, limit: f32) -> Episode {
        let spec = CarSpec::default();
        let track = Track::generate(seed, difficulty);
        let len = track.length;
        let mut ep = Episode::new(track, &spec);
        ep.target_laps = laps;
        ep.time_limit = limit;
        let mut guard = 0;
        while !ep.done && guard < 100_000 {
            if ep.steps % POLICY_EVERY == 0 {
                let c = Baseline::controls(&ep, &spec);
                ep.apply_controls(c[0], c[1]);
            }
            ep.advance(&spec);
            guard += 1;
        }
        assert!(len > 0.0);
        ep
    }

    #[test]
    fn baseline_completes_two_laps_without_crashing() {
        let mut completed = 0;
        for seed in 0..12u64 {
            let ep = drive_baseline(seed, 0.9, 2.0, 200.0);
            if ep.finished && !ep.crashed {
                completed += 1;
            }
            assert!(
                !ep.crashed,
                "seed {seed} crashed at s={} t={}",
                ep.progress, ep.time
            );
        }
        assert_eq!(completed, 12, "baseline should finish every track");
    }

    #[test]
    fn observations_are_finite_and_normalized() {
        let spec = CarSpec::default();
        let track = Track::generate(5, 0.7);
        let mut ep = Episode::new(track, &spec);
        ep.apply_controls(1.0, 0.2);
        for _ in 0..300 {
            ep.advance(&spec);
        }
        let o = ep.observe();
        for (i, v) in o.iter().enumerate() {
            assert!(v.is_finite(), "input {i} not finite");
            assert!(*v >= -1.01 && *v <= 1.01, "input {i} = {v}");
        }
        assert!(ep.sensors.iter().all(|s| (0.0..=1.0).contains(s)));
    }

    #[test]
    fn leaving_the_road_crashes_and_stops_the_episode() {
        let spec = CarSpec::default();
        let track = Track::generate(9, 0.5);
        let mut ep = Episode::new(track, &spec);
        // Drive straight off the first corner at full throttle.
        ep.apply_controls(1.0, 0.0);
        for _ in 0..1200 {
            ep.advance(&spec);
            if ep.done {
                break;
            }
        }
        assert!(ep.crashed, "expected a crash, progress {}", ep.progress);
        assert!(ep.done);
        let t = ep.time;
        ep.advance(&spec);
        assert_eq!(t, ep.time, "episode must freeze after done");
    }

    #[test]
    fn fitness_rewards_progress_and_clean_finishes() {
        let spec = CarSpec::default();
        let track = Track::generate(5, 1.0);
        let len = track.length;
        let mut ep = Episode::new(track, &spec);
        ep.progress = 50.0;
        assert!((ep.fitness() - 50.0).abs() < 1e-3);
        ep.progress = 300.0;
        assert!(ep.fitness() > 299.0);
        ep.crashed = true;
        assert!((ep.fitness() - 150.0).abs() < 1e-3);
        // A clean finish always beats crashing just short of the line.
        ep.progress = len * 2.0 - 1.0;
        ep.crashed = true;
        let near_miss = ep.fitness();
        ep.crashed = false;
        ep.finished = true;
        ep.time = 50.0;
        assert!(
            ep.fitness() > near_miss + 400.0,
            "finish {} miss {}",
            ep.fitness(),
            near_miss
        );
    }

    #[test]
    fn same_seed_and_policy_replay_exactly() {
        let spec = CarSpec::default();
        let run = || {
            let track = Track::generate(77, 1.0);
            let mut ep = Episode::new(track, &spec);
            ep.target_laps = 2.0;
            let mut k = 0u32;
            while !ep.done {
                if ep.steps % POLICY_EVERY == 0 {
                    // Deterministic open-loop-ish policy.
                    let s = (k as f32 * 0.13).sin();
                    ep.apply_controls(0.7, s);
                    k += 1;
                }
                ep.advance(&spec);
            }
            (ep.progress.to_bits(), ep.time.to_bits(), ep.crashed)
        };
        assert_eq!(run(), run());
    }

    #[test]
    fn lap_counting_matches_progress() {
        let ep = drive_baseline(4, 0.5, 2.0, 200.0);
        assert!(ep.finished);
        assert!((ep.laps() - 2.0).abs() < 1e-3, "laps {}", ep.laps());
        assert!(ep.progress >= 2.0 * ep.track.length - 1e-2);
    }
}
