//! Feed-forward neural network with tanh activations.
//!
//! The network is a flat parameter vector plus layer sizes so it can be
//! mutated, crossed over and serialized without per-layer allocation games.

use serde::{Deserialize, Serialize};

use crate::rng::Rng;

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct Net {
    pub sizes: Vec<usize>,
    /// Layer-major: for each layer, `out*in` weights then `out` biases.
    pub params: Vec<f32>,
}

impl Net {
    #[inline]
    pub fn input_size(&self) -> usize {
        self.sizes.first().copied().unwrap_or(0)
    }

    pub fn param_count(sizes: &[usize]) -> usize {
        sizes.windows(2).map(|w| w[0] * w[1] + w[1]).sum()
    }

    pub fn random(sizes: &[usize], rng: &mut Rng) -> Self {
        let mut params = Vec::with_capacity(Self::param_count(sizes));
        for w in sizes.windows(2) {
            let (in_n, out_n) = (w[0], w[1]);
            let scale = 1.5 / (in_n as f32).sqrt();
            for _ in 0..in_n * out_n {
                params.push(rng.normal() * scale);
            }
            for _ in 0..out_n {
                params.push(rng.normal() * 0.15);
            }
        }
        Self {
            sizes: sizes.to_vec(),
            params,
        }
    }

    /// All layer activations for visualization (allocates; renderer only).
    pub fn activations(&self, input: &[f32]) -> Vec<Vec<f32>> {
        let mut layers = vec![input.to_vec()];
        let mut off = 0usize;
        for li in 0..self.sizes.len() - 1 {
            let in_n = self.sizes[li];
            let out_n = self.sizes[li + 1];
            let mut out = vec![0.0f32; out_n];
            for o in 0..out_n {
                let mut sum = self.params[off + in_n * out_n + o];
                let base = off + o * in_n;
                for i in 0..in_n {
                    sum += self.params[base + i] * layers[li][i];
                }
                out[o] = sum.tanh();
            }
            off += in_n * out_n + out_n;
            layers.push(out);
        }
        layers
    }

    /// Run a forward pass. Hidden layers use tanh; the output layer also uses
    /// tanh so controls stay in [-1, 1].
    pub fn forward<'a>(&self, input: &[f32], pass: &'a mut ForwardPass) -> &'a [f32] {
        debug_assert_eq!(input.len(), self.sizes[0]);
        pass.a.clear();
        pass.a.extend_from_slice(input);
        let mut off = 0usize;
        for li in 0..self.sizes.len() - 1 {
            let in_n = self.sizes[li];
            let out_n = self.sizes[li + 1];
            pass.b.resize(out_n, 0.0);
            let w = &self.params[off..off + in_n * out_n];
            let b = &self.params[off + in_n * out_n..off + in_n * out_n + out_n];
            for o in 0..out_n {
                let row = &w[o * in_n..(o + 1) * in_n];
                let mut sum = b[o];
                for i in 0..in_n {
                    sum += row[i] * pass.a[i];
                }
                pass.b[o] = sum.tanh();
            }
            off += in_n * out_n + out_n;
            std::mem::swap(&mut pass.a, &mut pass.b);
        }
        &pass.a
    }
}

#[derive(Clone, Debug, Default)]
pub struct ForwardPass {
    a: Vec<f32>,
    b: Vec<f32>,
}

impl ForwardPass {
    pub fn new() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn param_layout_matches_counts() {
        let sizes = [5, 7, 3, 2];
        let n = Net::param_count(&sizes);
        assert_eq!(n, (5 * 7 + 7) + (7 * 3 + 3) + (3 * 2 + 2));
        let mut rng = Rng::new(1);
        let net = Net::random(&sizes, &mut rng);
        assert_eq!(net.params.len(), n);
    }

    #[test]
    fn forward_is_deterministic_bounded_and_shaped() {
        let mut rng = Rng::new(2);
        let net = Net::random(&[4, 6, 2], &mut rng);
        let input = [0.1, -0.5, 0.0, 1.0];
        let mut p1 = ForwardPass::new();
        let mut p2 = ForwardPass::new();
        let out1 = net.forward(&input, &mut p1).to_vec();
        let out2 = net.forward(&input, &mut p2);
        assert_eq!(out1, out2);
        assert_eq!(out1.len(), 2);
        assert!(
            out1.iter()
                .all(|v| v.is_finite() && (-1.0..=1.0).contains(v))
        );
    }

    #[test]
    fn single_layer_matches_manual_math() {
        // 2 inputs -> 1 output, weights [2, -1], bias 0.5
        let net = Net {
            sizes: vec![2, 1],
            params: vec![2.0, -1.0, 0.5],
        };
        let mut pass = ForwardPass::new();
        let out = net.forward(&[1.0, 0.5], &mut pass)[0];
        assert!((out - (2.0f32).tanh()).abs() < 1e-6);
    }

    #[test]
    fn serde_roundtrip_preserves_behavior() {
        let mut rng = Rng::new(3);
        let net = Net::random(&[crate::sim::INPUTS, 8, 2], &mut rng);
        let json = serde_json::to_string(&net).unwrap();
        let back: Net = serde_json::from_str(&json).unwrap();
        assert_eq!(net, back);
        let input = [0.3; crate::sim::INPUTS];
        let mut p1 = ForwardPass::new();
        let mut p2 = ForwardPass::new();
        assert_eq!(net.forward(&input, &mut p1), back.forward(&input, &mut p2));
    }
}
