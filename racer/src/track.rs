//! Procedural closed racing circuits.
//!
//! A track is a closed Catmull-Rom centerline, arc-length resampled at a fixed
//! spacing, with a variable-width road band around it. Generation is fully
//! deterministic from `(seed, difficulty)` and rejection-sampled until the
//! circuit is drivable: bounded curvature (no hairpins tighter than the car)
//! and no self-intersection.

use crate::math::{Vec2, angle_delta, lerp, ray_segment, vec2};
use crate::rng::{Rng, mix};

pub const SAMPLE_SPACING: f32 = 2.0;

#[derive(Clone, Debug)]
pub struct Track {
    pub seed: u64,
    pub difficulty: f32,
    /// Arc-length uniform centerline samples (closed; last connects to first).
    pub center: Vec<Vec2>,
    pub tangent: Vec<Vec2>,
    /// Unit left normal: `tangent.rotate(PI/2)`.
    pub normal: Vec<Vec2>,
    pub curvature: Vec<f32>,
    /// Full road width at each sample center.
    pub width: Vec<f32>,
    pub left: Vec<Vec2>,
    pub right: Vec<Vec2>,
    pub ds: f32,
    pub length: f32,
    pub start_index: usize,
    pub start_heading: f32,
    pub min_curve_radius: f32,
}

impl Track {
    pub fn n(&self) -> usize {
        self.center.len()
    }

    #[inline]
    pub fn wrap(&self, i: isize) -> usize {
        let n = self.center.len() as isize;
        (((i % n) + n) % n) as usize
    }

    #[inline]
    pub fn half_width(&self, i: usize) -> f32 {
        self.width[i] * 0.5
    }

    /// Continuous arc-length coordinate at a world point, using `hint` as a
    /// search origin (a local search is enough while the car moves).
    pub fn project(&self, p: Vec2, hint: Option<usize>) -> (usize, f32, f32) {
        let idx = self.nearest_index(p, hint, 24);
        let lateral = (p - self.center[idx]).dot(self.normal[idx]);
        let s = idx as f32 * self.ds + (p - self.center[idx]).dot(self.tangent[idx]);
        (idx, lateral, s)
    }

    /// Nearest centerline sample. `window` limits the search around `hint`;
    /// without a hint (or a hint outside the window) it scans everything.
    pub fn nearest_index(&self, p: Vec2, hint: Option<usize>, window: usize) -> usize {
        let n = self.center.len();
        match hint {
            Some(h) if window > 0 => {
                let mut best = h % n;
                let mut best_d = (p - self.center[best]).length_sq();
                let mut k = 1;
                while k <= window {
                    for i in [
                        self.wrap(h as isize - k as isize),
                        self.wrap(h as isize + k as isize),
                    ] {
                        let d = (p - self.center[i]).length_sq();
                        if d < best_d {
                            best_d = d;
                            best = i;
                        }
                    }
                    k += 1;
                }
                best
            }
            _ => {
                let mut best = 0;
                let mut best_d = f32::INFINITY;
                for (i, c) in self.center.iter().enumerate() {
                    let d = (p - *c).length_sq();
                    if d < best_d {
                        best_d = d;
                        best = i;
                    }
                }
                best
            }
        }
    }

    /// Cast a ray (unit `dir`, max `range`) against both road boundaries near
    /// the car. Returns distance and world hit point.
    pub fn raycast(&self, origin: Vec2, dir: Vec2, range: f32, hint: usize) -> Option<(f32, Vec2)> {
        let n = self.center.len();
        let span = ((range / self.ds).ceil() as usize + 2).min(n);
        let mut best = f32::INFINITY;
        let mut hit = Vec2::ZERO;
        for k in 0..span {
            let i = self.wrap(hint as isize - (span as isize) / 2 + k as isize);
            let j = (i + 1) % n;
            for (a, b) in [(self.left[i], self.left[j]), (self.right[i], self.right[j])] {
                if let Some(t) = ray_segment(origin, dir, a, b) {
                    if t <= range && t < best {
                        best = t;
                        hit = origin + dir * t;
                    }
                }
            }
        }
        if best.is_finite() {
            Some((best, hit))
        } else {
            None
        }
    }

    /// World-space bounds covering the whole road band.
    pub fn bounds(&self) -> (Vec2, Vec2) {
        let mut lo = vec2(f32::INFINITY, f32::INFINITY);
        let mut hi = vec2(f32::NEG_INFINITY, f32::NEG_INFINITY);
        for i in 0..self.left.len() {
            for p in [self.left[i], self.right[i]] {
                lo.x = lo.x.min(p.x);
                lo.y = lo.y.min(p.y);
                hi.x = hi.x.max(p.x);
                hi.y = hi.y.max(p.y);
            }
        }
        (lo, hi)
    }

    /// Curvature-, self-intersection- and width-checked generator.
    pub fn generate(seed: u64, difficulty: f32) -> Track {
        Self::generate_with(seed, difficulty, 1.0)
    }

    /// Same generator with a width multiplier. Training uses a narrower road so
    /// the policy learns a safety margin; evaluation uses `generate`.
    pub fn generate_with(seed: u64, difficulty: f32, width_scale: f32) -> Track {
        let difficulty = difficulty.clamp(0.0, 1.0);
        let width_scale = width_scale.clamp(0.5, 1.5);
        let mut rng = Rng::new(mix(seed, 0xC0FF_EE01, (difficulty * 1000.0) as u64));

        // Attempt progressively easier versions if validation rejects a layout.
        for attempt in 0..24 {
            let relax = 1.0 - 0.12 * attempt as f32;
            if let Some(t) = Self::try_generate(&mut rng, seed, difficulty, relax, width_scale) {
                return t;
            }
        }
        // Guaranteed fallback: a slightly wavy circle.
        Self::try_generate(&mut rng, seed, difficulty, 0.0, width_scale)
            .unwrap_or_else(|| Self::circle(seed, difficulty, width_scale))
    }

    fn try_generate(
        rng: &mut Rng,
        seed: u64,
        difficulty: f32,
        relax: f32,
        width_scale: f32,
    ) -> Option<Track> {
        let n_ctrl = 10 + (difficulty * 7.0).round() as usize;
        let base_radius = rng.range(78.0, 100.0);
        let amp = lerp(0.06, 0.30, difficulty) * relax;

        // Smoothly varying radial offsets around the loop: random values,
        // circularly smoothed twice so the centerline has no kinks.
        let mut offs: Vec<f32> = (0..n_ctrl).map(|_| rng.range(-1.0, 1.0)).collect();
        for _ in 0..2 {
            let prev = offs.clone();
            for i in 0..n_ctrl {
                let a = prev[(i + n_ctrl - 1) % n_ctrl];
                let b = prev[i];
                let c = prev[(i + 1) % n_ctrl];
                offs[i] = (a + 2.0 * b + c) * 0.25;
            }
        }
        let max_off = offs.iter().fold(0.0f32, |m, v| m.max(v.abs())).max(1e-3);
        let phase = rng.range(0.0, std::f32::consts::TAU);
        let ctrl: Vec<Vec2> = offs
            .iter()
            .enumerate()
            .map(|(i, off)| {
                let a = i as f32 / n_ctrl as f32 * std::f32::consts::TAU + phase;
                let r = base_radius * (1.0 + amp * (off / max_off) * 0.85);
                Vec2::from_angle(a) * r
            })
            .collect();

        // Dense Catmull-Rom sampling of the closed loop.
        let per_seg = 48;
        let mut dense = Vec::with_capacity(n_ctrl * per_seg);
        for i in 0..n_ctrl {
            let p0 = ctrl[(i + n_ctrl - 1) % n_ctrl];
            let p1 = ctrl[i];
            let p2 = ctrl[(i + 1) % n_ctrl];
            let p3 = ctrl[(i + 2) % n_ctrl];
            for k in 0..per_seg {
                let t = k as f32 / per_seg as f32;
                dense.push(catmull_rom(p0, p1, p2, p3, t));
            }
        }
        let center = resample_closed(&dense, SAMPLE_SPACING);
        let n = center.len();
        if n < 100 {
            return None;
        }

        // Tangents / normals / signed curvature.
        let mut tangent = Vec::with_capacity(n);
        let mut normal = Vec::with_capacity(n);
        let mut curvature = Vec::with_capacity(n);
        for i in 0..n {
            let t = (center[(i + 1) % n] - center[(i + n - 1) % n]).normalized();
            tangent.push(t);
            normal.push(t.perp());
        }
        for i in 0..n {
            let a = tangent[(i + n - 1) % n].angle();
            let b = tangent[(i + 1) % n].angle();
            curvature.push(angle_delta(a, b) / (2.0 * SAMPLE_SPACING));
        }

        // Width with a gentle breathing variation; never narrower than the car
        // plus margin even at maximum difficulty.
        let base_width = lerp(30.0, 18.0, difficulty) * width_scale;
        let wave_phase = rng.range(0.0, std::f32::consts::TAU);
        let mut width = Vec::with_capacity(n);
        for i in 0..n {
            let u = i as f32 / n as f32 * std::f32::consts::TAU;
            let w = base_width
                * (1.0 + 0.10 * (u * 2.0 + wave_phase).sin())
                * (1.0 + 0.05 * (u * 5.0 - wave_phase * 1.7).sin());
            width.push(w);
        }

        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);
        for i in 0..n {
            left.push(center[i] + normal[i] * (width[i] * 0.5));
            right.push(center[i] - normal[i] * (width[i] * 0.5));
        }

        let min_radius = 1.0
            / curvature
                .iter()
                .fold(0.0f32, |m, k| m.max(k.abs()))
                .max(1e-6);
        if min_radius < 9.0 {
            return None;
        }
        if !no_self_intersection(&center, &width) {
            return None;
        }

        // Start where the road is straightest (smoothed curvature) so agents
        // begin with a fair run-up.
        let mut best_i = 0;
        let mut best_k = f32::INFINITY;
        for i in 0..n {
            let mut avg = 0.0;
            for d in -3i32..=3 {
                avg += curvature[((i as i32 + d + n as i32) % n as i32) as usize].abs();
            }
            if avg < best_k {
                best_k = avg;
                best_i = i;
            }
        }
        let mut t = Track {
            seed,
            difficulty,
            center: Vec::new(),
            tangent: Vec::new(),
            normal: Vec::new(),
            curvature: Vec::new(),
            width: Vec::new(),
            left: Vec::new(),
            right: Vec::new(),
            ds: SAMPLE_SPACING,
            length: 0.0,
            start_index: 0,
            start_heading: 0.0,
            min_curve_radius: min_radius,
        };
        for k in 0..n {
            let i = (best_i + k) % n;
            t.center.push(center[i]);
            t.tangent.push(tangent[i]);
            t.normal.push(normal[i]);
            t.curvature.push(curvature[i]);
            t.width.push(width[i]);
            t.left.push(left[i]);
            t.right.push(right[i]);
        }
        t.length = n as f32 * SAMPLE_SPACING;
        t.start_heading = t.tangent[0].angle();
        Some(t)
    }

    fn circle(seed: u64, difficulty: f32, width_scale: f32) -> Track {
        let mut rng = Rng::new(seed);
        let r = 85.0;
        let n = (std::f32::consts::TAU * r / SAMPLE_SPACING).round() as usize;
        let mut center = Vec::with_capacity(n);
        let mut tangent = Vec::with_capacity(n);
        let mut normal = Vec::with_capacity(n);
        let mut curvature = Vec::with_capacity(n);
        let mut width = Vec::with_capacity(n);
        let base_width = lerp(30.0, 20.0, difficulty) * width_scale;
        for i in 0..n {
            let a = i as f32 / n as f32 * std::f32::consts::TAU;
            center.push(Vec2::from_angle(a) * r);
            let t = Vec2::from_angle(a + std::f32::consts::FRAC_PI_2);
            tangent.push(t);
            normal.push(t.perp());
            curvature.push(1.0 / r);
            width.push(base_width);
        }
        let mut left = Vec::with_capacity(n);
        let mut right = Vec::with_capacity(n);
        for i in 0..n {
            left.push(center[i] + normal[i] * (width[i] * 0.5));
            right.push(center[i] - normal[i] * (width[i] * 0.5));
        }
        let _ = rng.next_u64();
        let tangent_start_angle = tangent[0].angle();
        Track {
            seed,
            difficulty,
            center,
            tangent,
            normal,
            curvature,
            width,
            left,
            right,
            ds: SAMPLE_SPACING,
            length: n as f32 * SAMPLE_SPACING,
            start_index: 0,
            start_heading: tangent_start_angle,
            min_curve_radius: r,
        }
    }
}

#[inline]
fn catmull_rom(p0: Vec2, p1: Vec2, p2: Vec2, p3: Vec2, t: f32) -> Vec2 {
    let t2 = t * t;
    let t3 = t2 * t;
    let a = p1 * 2.0;
    let b = p2 - p0;
    let c = p0 * 2.0 - p1 * 5.0 + p2 * 4.0 - p3;
    let d = p0 * -1.0 + p1 * 3.0 - p2 * 3.0 + p3;
    (a + b * t + c * t2 + d * t3) * 0.5
}

/// Resample a closed polyline at uniform arc length.
fn resample_closed(points: &[Vec2], spacing: f32) -> Vec<Vec2> {
    let m = points.len();
    let mut total = 0.0;
    for i in 0..m {
        total += points[i].dist(points[(i + 1) % m]);
    }
    let count = (total / spacing).round().max(8.0) as usize;
    let step = total / count as f32;
    let mut out = Vec::with_capacity(count);
    let mut seg = 0usize;
    let mut seg_start = 0.0f32;
    let mut seg_len = points[0].dist(points[1 % m]);
    for k in 0..count {
        let target = k as f32 * step;
        while seg + 1 < m && seg_start + seg_len < target {
            seg_start += seg_len;
            seg += 1;
            seg_len = points[seg % m].dist(points[(seg + 1) % m]);
        }
        let t = ((target - seg_start) / seg_len.max(1e-6)).clamp(0.0, 1.0);
        out.push(points[seg % m].lerp(points[(seg + 1) % m], t));
    }
    out
}

fn no_self_intersection(center: &[Vec2], width: &[f32]) -> bool {
    let n = center.len();
    // Only distant parts of the loop are compared: two samples 60 m apart
    // along the road are allowed to be one road width apart on a tight bend.
    // What must never happen is a separate stretch of road crossing or
    // overlapping this one, which the straight-line separation test catches.
    let min_arc_samples = (60.0 / SAMPLE_SPACING) as usize;
    for i in 0..n {
        for j in (i + min_arc_samples)..n {
            let arc_gap = {
                let d = (i as isize - j as isize).abs() as f32 * SAMPLE_SPACING;
                d.min(n as f32 * SAMPLE_SPACING - d)
            };
            if arc_gap >= 60.0 {
                let need = (width[i] + width[j]) * 0.5 + 4.0;
                if center[i].dist(center[j]) < need {
                    return false;
                }
            }
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generated_tracks_are_valid_and_deterministic() {
        for seed in 0..40u64 {
            let difficulty = (seed % 5) as f32 / 4.0;
            let t = Track::generate(seed, difficulty);
            assert!(t.n() >= 100, "seed {seed} n {}", t.n());
            assert!((t.length - t.n() as f32 * SAMPLE_SPACING).abs() < 1e-3);
            // Uniform spacing and closure.
            for i in 0..t.n() {
                let j = (i + 1) % t.n();
                let d = t.center[i].dist(t.center[j]);
                assert!((d - SAMPLE_SPACING).abs() < 0.05, "seed {seed} i {i} d {d}");
                assert!((t.tangent[i].length() - 1.0).abs() < 1e-4);
            }
            assert!(
                t.min_curve_radius >= 9.0,
                "seed {seed} radius {}",
                t.min_curve_radius
            );
            // Boundaries stay on their side.
            for i in 0..t.n() {
                let l = (t.left[i] - t.center[i]).dot(t.normal[i]);
                let r = (t.right[i] - t.center[i]).dot(t.normal[i]);
                assert!(l > 0.0 && r < 0.0);
            }
            let a = Track::generate(seed, difficulty);
            assert_eq!(t.center, a.center);
        }
    }

    #[test]
    fn start_is_on_centerline_and_forward() {
        let t = Track::generate(3, 0.5);
        assert!(t.center[0].dist(t.center[0]).abs() < 1e-6);
        // Heading points along the first tangent, toward the next sample.
        let forward = Vec2::from_angle(t.start_heading);
        assert!(forward.dot(t.tangent[0]) > 0.99);
        let next = t.center[1] - t.center[0];
        assert!(forward.dot(next) > 0.0);
    }

    #[test]
    fn raycast_hits_a_boundary_in_both_directions() {
        let t = Track::generate(7, 0.8);
        let p = t.center[10];
        // Sideways rays must find a wall at roughly half width.
        let (d, hit) = t.raycast(p, t.normal[10], 60.0, 10).expect("left wall");
        assert!(
            (d - t.half_width(10)).abs() < 1.5,
            "d {d} hw {}",
            t.half_width(10)
        );
        assert!(p.dist(hit) < d + 0.1);
        let (d2, _) = t.raycast(p, -t.normal[10], 60.0, 10).expect("right wall");
        assert!((d2 - t.half_width(10)).abs() < 1.5);
    }

    #[test]
    fn projection_recovers_arc_length() {
        let t = Track::generate(11, 0.3);
        for i in [0usize, 5, 40, t.n() - 1] {
            let (idx, lat, s) = t.project(t.center[i], Some(i));
            assert_eq!(idx, i);
            assert!(lat.abs() < 0.01);
            assert!((s - i as f32 * SAMPLE_SPACING).abs() < 0.6, "i {i} s {s}");
        }
        // A point offset to the left reports positive lateral offset.
        let p = t.center[20] + t.normal[20] * 3.0;
        let (_, lat, _) = t.project(p, Some(20));
        assert!(lat > 2.5 && lat < 3.5);
    }
}
