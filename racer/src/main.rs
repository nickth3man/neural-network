//! Command line interface: train, evaluate, baseline and watch.

use std::path::PathBuf;

use racer::agent::{RunStats, rollout_baseline};
use racer::eval::{aggregate, eval_seeds, evaluate_on_seeds};
use racer::model::load;
use racer::track::Track;
use racer::train::{TrainConfig, train};

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let cmd = args.first().map(String::as_str).unwrap_or("help");
    let rest = if args.is_empty() {
        &args[..]
    } else {
        &args[1..]
    };
    let code = match cmd {
        "train" => run_train(rest),
        "eval" => run_eval(rest),
        "baseline" => run_baseline(rest),
        "watch" => racer::render::run_watch(rest),
        "help" | "--help" | "-h" => {
            print_help();
            0
        }
        other => {
            eprintln!("unknown command: {other}");
            print_help();
            2
        }
    };
    std::process::exit(code);
}

fn print_help() {
    println!(
        "racer - neuroevolution racing\n\n\
         USAGE:\n\
         \x20 racer train    [--seed 1] [--generations 400] [--population 160]\n\
         \x20                [--tracks-per-gen 3] [--ramp 60] [--laps 2] [--time-limit 150]\n\
         \x20                [--out models/champion.json] [--resume PATH]\n\
         \x20                [--validation-tracks 16] [--validation-candidates 8]\n\
         \x20                [--validation-every 4] [--target-success 1.0] [--validation-seed N]\n\
         \x20 racer eval     --model models/champion.json [--tracks 40] [--seed 700000]\n\
         \x20                [--difficulty 1.0] [--laps 2] [--min-success 1.0] [--json out.json]\n\
         \x20 racer baseline [--tracks 20] [--seed 500000] [--difficulty 1.0]\n\
         \x20 racer watch    --model models/champion.json [--seed 123] [--difficulty 1.0]\n\
         \x20                [--shots screenshots] [--shot-every 1.0] [--auto-track 8]\n"
    );
}

fn get<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn parse<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> Result<T, String> {
    match get(args, name) {
        Some(v) => v
            .parse()
            .map_err(|_| format!("invalid value for {name}: {v}")),
        None => Ok(default),
    }
}

fn run_train(args: &[String]) -> i32 {
    let cfg = match (|| {
        Ok::<_, String>(TrainConfig {
            seed: parse(args, "--seed", 1)?,
            generations: parse(args, "--generations", 400)?,
            population: parse(args, "--population", 160)?,
            tracks_per_gen: parse(args, "--tracks-per-gen", 3)?,
            ramp_gens: parse(args, "--ramp", 60)?,
            laps: parse(args, "--laps", 2.0)?,
            time_limit: parse(args, "--time-limit", 150.0)?,
            out: PathBuf::from(get(args, "--out").unwrap_or("models/champion.json")),
            resume: get(args, "--resume").map(PathBuf::from),
            validation_tracks: parse(args, "--validation-tracks", 16)?,
            validation_candidates: parse(args, "--validation-candidates", 8)?,
            validation_every: parse(args, "--validation-every", 4)?,
            target_success: parse(args, "--target-success", 1.0)?,
            validation_seed: parse(args, "--validation-seed", 0)?,
            validation_seed2: parse(args, "--validation-seed2", 0)?,
            width_scale: parse(args, "--width-scale", 0.85)?,
            quiet: false,
        })
    })() {
        Ok(c) => c,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    let started = std::time::Instant::now();
    match train(&cfg) {
        Ok(report) => {
            println!(
                "\nfinished {} generations in {:.1}s",
                report.generations_run,
                started.elapsed().as_secs_f64()
            );
            println!(
                "champion: {} | validation success {}/{} | mean laps {:.2} | mean time {:.1}s",
                report.champion_path.display(),
                report.validation.successes,
                report.validation.runs,
                report.validation.mean_laps,
                report.validation.mean_time
            );
            0
        }
        Err(e) => {
            eprintln!("training failed: {e}");
            2
        }
    }
}

fn print_runs(runs: &[RunStats]) {
    for r in runs {
        println!(
            "seed {:>8} {} laps={:.2} frac={:.2} t={:6.2}s vmax={:5.1} lat={:.2}{}",
            r.seed,
            if r.finished && !r.crashed {
                "OK  "
            } else if r.crashed {
                "CRASH"
            } else {
                "SLOW "
            },
            r.laps,
            r.progress_frac,
            r.time,
            r.max_speed,
            r.mean_abs_lateral,
            if r.crashed {
                format!(" at s={:.0}", r.progress)
            } else {
                String::new()
            }
        );
    }
}

fn run_eval(args: &[String]) -> i32 {
    let model_path = match get(args, "--model") {
        Some(p) => PathBuf::from(p),
        None => {
            eprintln!("--model is required");
            return 2;
        }
    };
    let tracks: usize = parse(args, "--tracks", 40).unwrap_or(40);
    let seed_base: u64 = parse(args, "--seed", 700_000).unwrap_or(700_000);
    let difficulty: f32 = parse(args, "--difficulty", 1.0).unwrap_or(1.0);
    let laps: f32 = parse(args, "--laps", 2.0).unwrap_or(2.0);
    let time_limit: f32 = parse(args, "--time-limit", 150.0).unwrap_or(150.0);
    let min_success: f32 = parse(args, "--min-success", 1.0).unwrap_or(1.0);
    let json_path = get(args, "--json").map(PathBuf::from);

    let model = match load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    if model.sizes.first().copied() != Some(racer::sim::INPUTS) {
        eprintln!(
            "model expects {} inputs, this build produces {} (retrain the model)",
            model.sizes.first().copied().unwrap_or(0),
            racer::sim::INPUTS
        );
        return 2;
    }
    let net = model.net();
    println!(
        "model {} (gen {}, training seed {})",
        model_path.display(),
        model.meta.generation,
        model.meta.training_seed
    );
    let seeds = eval_seeds(seed_base, tracks);
    let runs = evaluate_on_seeds(&net, &seeds, difficulty, laps, time_limit);
    print_runs(&runs);
    let agg = aggregate(&runs);
    println!(
        "\nRESULT: success {}/{} ({:.0}%) | laps mean {:.2} min {:.2} | time mean {:.1}s | crashes {}",
        agg.successes,
        agg.runs,
        agg.success_rate * 100.0,
        agg.mean_laps,
        agg.min_laps,
        agg.mean_time,
        agg.crashed
    );
    if let Some(path) = json_path {
        #[derive(serde::Serialize)]
        struct EvalReport<'a> {
            model: String,
            generation: u32,
            difficulty: f32,
            laps: f32,
            time_limit: f32,
            runs: &'a [RunStats],
            aggregate: racer::eval::Aggregate,
        }
        let report = EvalReport {
            model: model_path.display().to_string(),
            generation: model.meta.generation,
            difficulty,
            laps,
            time_limit,
            runs: &runs,
            aggregate: agg.clone(),
        };
        if let Some(dir) = path.parent() {
            let _ = std::fs::create_dir_all(dir);
        }
        match serde_json::to_string_pretty(&report) {
            Ok(js) => {
                if let Err(e) = std::fs::write(&path, js) {
                    eprintln!("write {}: {e}", path.display());
                    return 2;
                }
                println!("wrote {}", path.display());
            }
            Err(e) => {
                eprintln!("serialize eval report: {e}");
                return 2;
            }
        }
    }
    if agg.success_rate + 1e-6 < min_success {
        eprintln!(
            "FAIL: success rate {:.2} below required {:.2}",
            agg.success_rate, min_success
        );
        return 1;
    }
    0
}

fn run_baseline(args: &[String]) -> i32 {
    let tracks: usize = parse(args, "--tracks", 20).unwrap_or(20);
    let seed_base: u64 = parse(args, "--seed", 500_000).unwrap_or(500_000);
    let difficulty: f32 = parse(args, "--difficulty", 1.0).unwrap_or(1.0);
    let laps: f32 = parse(args, "--laps", 2.0).unwrap_or(2.0);
    let seeds = eval_seeds(seed_base, tracks);
    use rayon::prelude::*;
    let runs: Vec<RunStats> = seeds
        .par_iter()
        .map(|&seed| {
            let track = Track::generate(seed, difficulty);
            rollout_baseline(track, laps, 150.0)
        })
        .collect();
    print_runs(&runs);
    let agg = aggregate(&runs);
    println!(
        "\nBASELINE: success {}/{} | laps mean {:.2} min {:.2} | time mean {:.1}s | crashes {}",
        agg.successes, agg.runs, agg.mean_laps, agg.min_laps, agg.mean_time, agg.crashed
    );
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_flags() {
        let a = vec!["--seed".to_string(), "9".to_string(), "--laps".to_string()];
        assert_eq!(parse::<u64>(&a, "--seed", 1).unwrap(), 9);
        assert_eq!(parse::<u64>(&a, "--nope", 1).unwrap(), 1);
        assert!(parse::<f32>(&a, "--laps", 2.0).is_ok());
        let b = vec!["--laps".to_string(), "fast".to_string()];
        assert!(parse::<f32>(&b, "--laps", 2.0).is_err());
    }

    #[test]
    fn get_reads_value() {
        let a = vec!["--model".to_string(), "x.json".to_string()];
        assert_eq!(get(&a, "--model"), Some("x.json"));
        assert_eq!(get(&a, "--missing"), None);
    }
}
