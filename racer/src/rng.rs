//! Dependency-free deterministic PRNG (SplitMix64).
//!
//! Training must be reproducible across runs and machines: every random draw in
//! track generation and evolution goes through this type, so replaying a seed
//! reproduces a run bit-for-bit.

#[derive(Clone, Debug)]
pub struct Rng {
    state: u64,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        // Avoid the degenerate all-zero stream and decorrelate similar seeds.
        Self {
            state: seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ 0xA076_1D64_78BD_642F,
        }
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform float in [0, 1).
    #[inline]
    pub fn unit(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) * (1.0 / 16_777_216.0)
    }

    /// Uniform float in [lo, hi).
    #[inline]
    pub fn range(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (hi - lo) * self.unit()
    }

    /// Uniform integer in [0, n), n > 0.
    #[inline]
    pub fn below(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }

    /// Standard normal via Box-Muller.
    #[inline]
    pub fn normal(&mut self) -> f32 {
        let u1 = self.unit().max(1e-7);
        let u2 = self.unit();
        (-2.0 * u1.ln()).sqrt() * (std::f32::consts::TAU * u2).cos()
    }

    pub fn chance(&mut self, p: f32) -> bool {
        self.unit() < p
    }
}

/// Stable seed derivation for `(base, a, b)` style keys.
pub fn mix(seed: u64, a: u64, b: u64) -> u64 {
    let mut z =
        seed ^ a.wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ b.wrapping_mul(0xC2B2_AE3D_27D4_EB4F);
    z ^= z >> 33;
    z = z.wrapping_mul(0xFF51_AFD7_ED55_8CCD);
    z ^= z >> 33;
    z = z.wrapping_mul(0xC4CE_B9FE_1A85_EC53);
    z ^ (z >> 33)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_same_stream() {
        let mut a = Rng::new(7);
        let mut b = Rng::new(7);
        for _ in 0..1000 {
            assert_eq!(a.next_u64(), b.next_u64());
        }
    }

    #[test]
    fn uniform_bounds_and_mean() {
        let mut r = Rng::new(1);
        let mut sum = 0.0;
        for _ in 0..100_000 {
            let v = r.unit();
            assert!((0.0..1.0).contains(&v));
            sum += v;
        }
        let mean = sum / 100_000.0;
        assert!((mean - 0.5).abs() < 0.01, "mean {mean}");
    }

    #[test]
    fn normal_is_sane() {
        let mut r = Rng::new(2);
        let n = 50_000;
        let mut sum = 0.0;
        let mut sq = 0.0;
        for _ in 0..n {
            let v = r.normal();
            sum += v;
            sq += v * v;
        }
        let mean = sum / n as f32;
        let var = sq / n as f32 - mean * mean;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "var {var}");
    }

    #[test]
    fn mix_changes_with_inputs() {
        assert_ne!(mix(1, 2, 3), mix(1, 2, 4));
        assert_ne!(mix(1, 2, 3), mix(1, 3, 2));
        assert_eq!(mix(42, 7, 7), mix(42, 7, 7));
    }
}
