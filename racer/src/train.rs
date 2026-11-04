//! Headless neuroevolution training with a difficulty curriculum and a
//! held-out validation set used to pick and save the champion.

use std::io::Write;
use std::path::PathBuf;
use std::time::Instant;

use rayon::prelude::*;
use serde::Serialize;

use crate::agent::{RunStats, default_sizes, rollout_net};
use crate::eval::{Aggregate, aggregate, eval_seeds, evaluate_on_seeds, score};
use crate::ga::{self, GaConfig};
use crate::model::{ModelFile, ModelMeta, save};
use crate::nn::Net;
use crate::rng::{Rng, mix};
use crate::track::Track;

#[derive(Clone, Debug)]
pub struct TrainConfig {
    pub seed: u64,
    pub generations: u32,
    pub population: usize,
    pub tracks_per_gen: usize,
    pub ramp_gens: u32,
    pub laps: f32,
    pub time_limit: f32,
    pub out: PathBuf,
    pub resume: Option<PathBuf>,
    pub validation_tracks: usize,
    pub validation_candidates: usize,
    pub validation_every: u32,
    pub target_success: f32,
    /// 0 = derive the validation seed base from `seed`.
    pub validation_seed: u64,
    /// Optional second validation base; half the tracks come from each base.
    pub validation_seed2: u64,
    /// Road width multiplier used for training tracks (1.0 = evaluation width).
    pub width_scale: f32,
    pub quiet: bool,
}

impl Default for TrainConfig {
    fn default() -> Self {
        Self {
            seed: 1,
            generations: 700,
            population: 320,
            tracks_per_gen: 8,
            ramp_gens: 100,
            laps: 2.0,
            time_limit: 150.0,
            out: PathBuf::from("models/champion.json"),
            resume: None,
            validation_tracks: 60,
            validation_candidates: 10,
            validation_every: 4,
            target_success: 1.0,
            validation_seed: 0,
            validation_seed2: 0,
            width_scale: 0.85,
            quiet: false,
        }
    }
}

#[derive(Clone, Debug, Default, Serialize)]
pub struct GenStat {
    pub generation: u32,
    pub difficulty: f32,
    pub best_fitness: f32,
    pub mean_fitness: f32,
    pub finished: usize,
    pub crashes: usize,
    pub runs: usize,
    pub best_progress_frac: f32,
    pub elapsed_s: f64,
}

#[derive(Clone, Debug)]
pub struct TrainReport {
    pub best: Net,
    pub validation: Aggregate,
    pub generations_run: u32,
    pub history: Vec<GenStat>,
    pub champion_path: PathBuf,
}

#[derive(Default)]
struct Batch {
    fitness: f32,
    finished: usize,
    crashed: usize,
    runs: usize,
    progress_frac: f32,
    best_run_laps: f32,
}

impl Batch {
    fn add(&mut self, st: &RunStats) {
        self.fitness += st.fitness;
        self.runs += 1;
        self.progress_frac += st.progress_frac;
        self.best_run_laps = self.best_run_laps.max(st.laps);
        if st.finished && !st.crashed {
            self.finished += 1;
        }
        if st.crashed {
            self.crashed += 1;
        }
    }
    fn mean(&self) -> f32 {
        self.fitness / self.runs.max(1) as f32
    }
}

fn evaluate_population(pop: &[Net], tracks: &[Track], laps: f32, limit: f32) -> Vec<Batch> {
    pop.par_iter()
        .map(|net| {
            let mut b = Batch::default();
            for t in tracks {
                let st = rollout_net(net, t.clone(), laps, limit);
                b.add(&st);
            }
            b
        })
        .collect()
}

fn validate(net: &Net, seeds: &[u64], laps: f32, limit: f32) -> Aggregate {
    aggregate(&evaluate_on_seeds(net, seeds, 1.0, laps, limit))
}

pub fn train(cfg: &TrainConfig) -> Result<TrainReport, String> {
    let sizes = default_sizes();
    let mut rng = Rng::new(cfg.seed);
    let mut pop = ga::seed_population(&sizes, cfg.population, &mut rng);
    if let Some(path) = &cfg.resume {
        let model = crate::model::load(path)?;
        if model.sizes != sizes {
            return Err(format!(
                "resume model shape {:?} does not match {:?}",
                model.sizes, sizes
            ));
        }
        pop[0] = model.net();
    }

    let mut ga_cfg = GaConfig::default();
    ga_cfg.mutation_rate = 0.08;
    ga_cfg.mutation_sigma = 0.15;
    let val_base = if cfg.validation_seed != 0 {
        cfg.validation_seed
    } else {
        mix(cfg.seed, 0x5EED_1111, 1)
    };
    let mut validation_seeds = eval_seeds(
        val_base,
        if cfg.validation_seed2 != 0 {
            cfg.validation_tracks / 2
        } else {
            cfg.validation_tracks
        },
    );
    if cfg.validation_seed2 != 0 {
        validation_seeds.extend(eval_seeds(
            cfg.validation_seed2,
            cfg.validation_tracks - cfg.validation_tracks / 2,
        ));
    }
    let validation_tracks_note = format!("{} fixed tracks", validation_seeds.len());
    if !cfg.quiet {
        println!(
            "training seed={} pop={} tracks/gen={} ramp={} gens | validation: {}",
            cfg.seed, cfg.population, cfg.tracks_per_gen, cfg.ramp_gens, validation_tracks_note
        );
    }

    let started = Instant::now();
    let mut history = Vec::new();
    let mut best_validation: Option<Aggregate> = None;
    let mut best_net = pop[0].clone();
    let mut best_score = f32::NEG_INFINITY;
    let mut halt_checks = 0u32;
    let mut generations_run: u32 = 0;

    // Fresh CSV log for this run.
    let csv_path = cfg.out.with_extension("history.csv");
    if let Some(dir) = csv_path.parent() {
        let _ = std::fs::create_dir_all(dir);
    }
    let mut csv = std::fs::File::create(&csv_path)
        .map_err(|e| format!("create {}: {e}", csv_path.display()))?;
    writeln!(
        csv,
        "generation,difficulty,best_fitness,mean_fitness,finished,crashes,runs,best_progress_frac,elapsed_s"
    )
    .ok();

    // Resuming trains at full difficulty instead of replaying the curriculum.
    let gen_offset = if cfg.resume.is_some() {
        cfg.ramp_gens
    } else {
        0
    };
    for g in gen_offset..gen_offset + cfg.generations {
        let gen_start = Instant::now();
        // Anneal mutation strength as the population matures.
        ga_cfg.mutation_sigma = 0.22 - 0.15 * (g as f32 / (gen_offset + cfg.generations) as f32);
        let difficulty = if cfg.ramp_gens == 0 {
            1.0
        } else {
            (0.30 + 0.70 * g as f32 / cfg.ramp_gens as f32).min(1.0)
        };
        let tracks: Vec<Track> = (0..cfg.tracks_per_gen)
            .map(|k| {
                Track::generate_with(
                    mix(cfg.seed, g as u64, k as u64),
                    difficulty,
                    cfg.width_scale,
                )
            })
            .collect();
        // Keep the all-time validation champion in the gene pool so it can
        // survive generations whose random tracks happen to suit it badly.
        if g > 0 && pop.len() > 1 {
            let last = pop.len() - 1;
            pop[last] = best_net.clone();
        }

        let batches = evaluate_population(&pop, &tracks, cfg.laps, cfg.time_limit);
        let fitness: Vec<f32> = batches.iter().map(|b| b.mean()).collect();

        let mut best_i = 0usize;
        let mut fit_sum = 0.0;
        let mut finished = 0usize;
        let mut crashes = 0usize;
        let mut runs = 0usize;
        for (i, b) in batches.iter().enumerate() {
            fit_sum += b.mean();
            finished += b.finished;
            crashes += b.crashed;
            runs += b.runs;
            if b.mean() > batches[best_i].mean() {
                best_i = i;
            }
        }
        let best_batch = &batches[best_i];
        let elapsed = gen_start.elapsed().as_secs_f64();
        let stat = GenStat {
            generation: g,
            difficulty,
            best_fitness: best_batch.mean(),
            mean_fitness: fit_sum / pop.len() as f32,
            finished,
            crashes,
            runs,
            best_progress_frac: best_batch.progress_frac / best_batch.runs.max(1) as f32,
            elapsed_s: elapsed,
        };
        if !cfg.quiet {
            println!(
                "gen {:4} d={:.2} best={:8.1} mean={:8.1} fin={:3}/{:3} crash={:3} prog={:.2} {:.2}s",
                g,
                stat.difficulty,
                stat.best_fitness,
                stat.mean_fitness,
                stat.finished,
                stat.runs,
                stat.crashes,
                stat.best_progress_frac,
                elapsed
            );
            let _ = std::io::stdout().flush();
        }
        writeln!(
            csv,
            "{},{:.4},{:.3},{:.3},{},{},{},{:.4},{:.3}",
            g,
            stat.difficulty,
            stat.best_fitness,
            stat.mean_fitness,
            stat.finished,
            stat.crashes,
            stat.runs,
            stat.best_progress_frac,
            elapsed
        )
        .ok();
        history.push(stat);
        generations_run = g + 1;

        // Periodic honest checkpoint: evaluate the top candidates on fixed
        // tracks at full difficulty, and keep the best champion so far.
        let due = g % cfg.validation_every == 0 || g + 1 == cfg.generations;
        if due {
            let mut order: Vec<usize> = (0..pop.len()).collect();
            order.sort_by(|&a, &b| fitness[b].total_cmp(&fitness[a]));
            let take = cfg.validation_candidates.min(order.len());
            let mut best_candidate: Option<(f32, usize, Aggregate)> = None;
            for &idx in order.iter().take(take) {
                let agg = validate(&pop[idx], &validation_seeds, cfg.laps, cfg.time_limit);
                let sc = score(&agg);
                if best_candidate.as_ref().is_none_or(|(bs, _, _)| sc > *bs) {
                    best_candidate = Some((sc, idx, agg));
                }
            }
            if let Some((sc, idx, agg)) = best_candidate {
                let improved = sc > best_score;
                if improved {
                    best_score = sc;
                    best_net = pop[idx].clone();
                    let model = ModelFile::from_net(
                        &best_net,
                        ModelMeta {
                            training_seed: cfg.seed,
                            generation: g,
                            validation: Some(agg.clone()),
                        },
                    );
                    save(&cfg.out, &model)?;
                }
                if !cfg.quiet {
                    println!(
                        "  val gen {:4}: success {}/{} laps {:.2} time {:.1}s progress {:.2} {}",
                        g,
                        agg.successes,
                        agg.runs,
                        agg.mean_laps,
                        agg.mean_time,
                        agg.mean_progress_frac,
                        if improved { "-> champion saved" } else { "" }
                    );
                    let _ = std::io::stdout().flush();
                }
                best_validation = Some(agg.clone());
                let ramp_done = cfg.ramp_gens == 0 || g + 1 >= cfg.ramp_gens;
                if ramp_done && agg.success_rate >= cfg.target_success {
                    halt_checks += 1;
                } else {
                    halt_checks = 0;
                }
                if halt_checks >= 2 {
                    if !cfg.quiet {
                        println!(
                            "early stop: validation success {:.0}% on {} full-difficulty tracks",
                            agg.success_rate * 100.0,
                            agg.runs
                        );
                    }
                    break;
                }
            }
        }

        // Stop before breeding if this was the last generation.
        if g + 1 >= cfg.generations {
            break;
        }
        ga_cfg.mutation_sigma = 0.22 - 0.15 * (g as f32 / (gen_offset + cfg.generations) as f32);
        pop = ga::next_generation(&pop, &fitness, &ga_cfg, &mut rng);
        // Fresh random immigrants keep the search from collapsing onto one
        // driving style; the elite is preserved above.
        let immigrants = (pop.len() / 25).max(1);
        for k in 0..immigrants {
            let idx = pop.len() - 1 - k;
            pop[idx] = Net::random(&sizes, &mut rng);
        }
    }

    // Guarantee an artifact exists even if validation never ran (0 generations).
    if !cfg.out.exists() {
        let model = ModelFile::from_net(
            &best_net,
            ModelMeta {
                training_seed: cfg.seed,
                generation: generations_run.saturating_sub(1),
                validation: best_validation.clone(),
            },
        );
        save(&cfg.out, &model)?;
    }

    if !cfg.quiet {
        println!("total {:.1}s", started.elapsed().as_secs_f64());
    }
    Ok(TrainReport {
        best: best_net,
        validation: best_validation.unwrap_or_default(),
        generations_run,
        history,
        champion_path: cfg.out.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn short_training_run_produces_a_model() {
        let dir = std::env::temp_dir().join("racer_train_test");
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = TrainConfig {
            seed: 3,
            generations: 2,
            population: 8,
            tracks_per_gen: 1,
            ramp_gens: 1,
            laps: 0.25,
            time_limit: 20.0,
            out: dir.join("champion.json"),
            validation_tracks: 2,
            validation_candidates: 2,
            validation_every: 2,
            target_success: 1.0,
            quiet: true,
            ..Default::default()
        };
        let report = train(&cfg).unwrap();
        assert!(report.champion_path.exists());
        assert_eq!(report.generations_run, 2);
        let model = crate::model::load(&report.champion_path).unwrap();
        assert_eq!(model.sizes, default_sizes());
        let _ = std::fs::remove_dir_all(dir);
    }
}
