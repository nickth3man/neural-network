//! Genetic algorithm over network weights: elitism, tournament selection,
//! uniform crossover and Gaussian mutation.

use crate::nn::Net;
use crate::rng::Rng;

#[derive(Clone, Debug)]
pub struct GaConfig {
    pub elite: usize,
    pub tournament: usize,
    pub mutation_rate: f32,
    pub mutation_sigma: f32,
    /// Probability that a gene is fully re-rolled instead of nudged.
    pub reset_rate: f32,
    pub crossover_rate: f32,
}

impl Default for GaConfig {
    fn default() -> Self {
        Self {
            elite: 4,
            tournament: 5,
            mutation_rate: 0.07,
            mutation_sigma: 0.18,
            reset_rate: 0.004,
            crossover_rate: 0.65,
        }
    }
}

pub fn seed_population(sizes: &[usize], n: usize, rng: &mut Rng) -> Vec<Net> {
    (0..n).map(|_| Net::random(sizes, rng)).collect()
}

fn tournament(fitness: &[f32], k: usize, rng: &mut Rng) -> usize {
    let mut best = rng.below(fitness.len());
    for _ in 1..k {
        let c = rng.below(fitness.len());
        if fitness[c] > fitness[best] {
            best = c;
        }
    }
    best
}

pub fn mutate(net: &mut Net, cfg: &GaConfig, rng: &mut Rng) {
    for p in net.params.iter_mut() {
        if rng.chance(cfg.mutation_rate) {
            if rng.chance(cfg.reset_rate) {
                *p = rng.normal() * 0.5;
            } else {
                *p += rng.normal() * cfg.mutation_sigma;
            }
            *p = p.clamp(-6.0, 6.0);
        }
    }
}

pub fn crossover(a: &Net, b: &Net, rng: &mut Rng) -> Net {
    debug_assert_eq!(a.sizes, b.sizes);
    let params = a
        .params
        .iter()
        .zip(&b.params)
        .map(|(x, y)| if rng.chance(0.5) { *x } else { *y })
        .collect();
    Net {
        sizes: a.sizes.clone(),
        params,
    }
}

/// Produce the next population, preserving the best genomes unchanged.
pub fn next_generation(pop: &[Net], fitness: &[f32], cfg: &GaConfig, rng: &mut Rng) -> Vec<Net> {
    assert_eq!(pop.len(), fitness.len());
    let mut order: Vec<usize> = (0..pop.len()).collect();
    order.sort_by(|&a, &b| fitness[b].total_cmp(&fitness[a]));

    let mut next: Vec<Net> = order
        .iter()
        .take(cfg.elite.min(pop.len()))
        .map(|&i| pop[i].clone())
        .collect();

    while next.len() < pop.len() {
        let pa = tournament(fitness, cfg.tournament, rng);
        let mut child = if rng.chance(cfg.crossover_rate) {
            let pb = tournament(fitness, cfg.tournament, rng);
            crossover(&pop[pa], &pop[pb], rng)
        } else {
            pop[pa].clone()
        };
        mutate(&mut child, cfg, rng);
        next.push(child);
    }
    next
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn elites_survive_unchanged() {
        let mut rng = Rng::new(4);
        let pop = seed_population(&[3, 4, 2], 20, &mut rng);
        let fit: Vec<f32> = (0..20).map(|i| i as f32).collect();
        let cfg = GaConfig::default();
        let next = next_generation(&pop, &fit, &cfg, &mut rng);
        assert_eq!(next.len(), 20);
        for k in 0..cfg.elite {
            assert_eq!(next[k], pop[19 - k]);
        }
    }

    #[test]
    fn mutation_changes_some_genes_but_keeps_shape() {
        let mut rng = Rng::new(5);
        let mut net = Net::random(&[4, 5, 2], &mut rng);
        let before = net.clone();
        let cfg = GaConfig {
            mutation_rate: 0.5,
            ..Default::default()
        };
        mutate(&mut net, &cfg, &mut rng);
        assert_eq!(net.sizes, before.sizes);
        assert_eq!(net.params.len(), before.params.len());
        assert!(net.params != before.params);
        assert!(net.params.iter().all(|v| v.is_finite() && v.abs() <= 6.0));
    }

    #[test]
    fn next_generation_is_deterministic_for_a_seed() {
        let mut a = Rng::new(9);
        let mut b = Rng::new(9);
        let pop = seed_population(&[2, 3, 1], 8, &mut a);
        let pop_b = seed_population(&[2, 3, 1], 8, &mut b);
        assert_eq!(pop, pop_b);
        let fit = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let cfg = GaConfig::default();
        let n1 = next_generation(&pop, &fit, &cfg, &mut a);
        let n2 = next_generation(&pop, &fit, &cfg, &mut b);
        assert_eq!(n1, n2);
    }
}
