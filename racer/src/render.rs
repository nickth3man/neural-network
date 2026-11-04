//! Live viewer: renders the learned driver, its sensors and its network, and
//! can save framebuffer screenshots without user interaction.

use std::path::PathBuf;

use macroquad::prelude::*;

use crate::agent::default_sizes;
use crate::car::CarSpec;
use crate::math::Vec2 as WV;
use crate::model::load;
use crate::nn::{ForwardPass, Net};
use crate::rng::mix;
use crate::sim::{DT, Episode, POLICY_EVERY};
use crate::track::Track;

const PANEL_W: f32 = 380.0;

const BG: Color = Color::new(0.051, 0.067, 0.09, 1.0);
const PANEL: Color = Color::new(0.086, 0.106, 0.137, 1.0);
const TEXT: Color = Color::new(0.90, 0.93, 0.95, 1.0);
const MUTED: Color = Color::new(0.55, 0.58, 0.63, 1.0);
const ACCENT: Color = Color::new(0.24, 0.86, 0.52, 1.0);
const RED: Color = Color::new(1.0, 0.36, 0.36, 1.0);
const ROAD: Color = Color::new(0.17, 0.19, 0.23, 1.0);
const GRASS: Color = Color::new(0.055, 0.13, 0.095, 1.0);
const EDGE: Color = Color::new(0.82, 0.85, 0.89, 1.0);

struct Watch {
    net: Net,
    spec: CarSpec,
    ep: Episode,
    base_seed: u64,
    track_idx: u32,
    difficulty: f32,
    laps: f32,
    time_limit: f32,
    paused: bool,
    speed: f32,
    accum: f32,
    history: Vec<f32>,
    lap_floor: i32,
    runs: u32,
    successes: u32,
    best_laps: f32,
    best_time: f32,
    last_result: Option<String>,
    done_timer: f32,
    auto_track: bool,
    auto_remaining: i32,
    shots_dir: Option<PathBuf>,
    shot_every: f32,
    real_time: f32,
    next_auto_shot: f32,
    shot_seq: u32,
    shot_limit: u32,
    pending_shots: Vec<String>,
    pass: ForwardPass,
    activations: Vec<Vec<f32>>,
    last_controls: [f32; 2],
    generation: u32,
    model_path: PathBuf,
    exit_after: f32,
    track_seed: Option<u64>,
}

impl Watch {
    fn new(net: Net, meta_generation: u32, model_path: PathBuf, opts: WatchOptions) -> Self {
        let spec = CarSpec::default();
        let mut w = Self {
            net,
            spec,
            ep: Episode::new(Track::generate(1, 1.0), &CarSpec::default()),
            base_seed: opts.seed,
            track_idx: 0,
            difficulty: opts.difficulty,
            laps: opts.laps,
            time_limit: 150.0,
            paused: false,
            speed: opts.speed,
            accum: 0.0,
            history: Vec::new(),
            lap_floor: 0,
            runs: 0,
            successes: 0,
            best_laps: 0.0,
            best_time: f32::INFINITY,
            last_result: None,
            done_timer: 0.0,
            auto_track: opts.auto_track > 0,
            auto_remaining: opts.auto_track,
            shots_dir: opts.shots,
            shot_every: opts.shot_every,
            real_time: 0.0,
            next_auto_shot: 0.0,
            shot_seq: 0,
            shot_limit: opts.shot_limit,
            pending_shots: Vec::new(),
            pass: ForwardPass::new(),
            activations: Vec::new(),
            last_controls: [0.0, 0.0],
            generation: meta_generation,
            model_path,
            exit_after: 0.0,
            track_seed: opts.track_seed,
        };
        w.next_track();
        w
    }

    fn next_track(&mut self) {
        self.track_idx += 1;
        let seed = self
            .track_seed
            .unwrap_or_else(|| mix(self.base_seed, self.track_idx as u64, 0xA11CE));
        let track = Track::generate(seed, self.difficulty);
        self.ep = Episode::new(track, &self.spec);
        self.ep.target_laps = self.laps;
        self.ep.time_limit = self.time_limit;
        self.history.clear();
        self.lap_floor = 0;
        self.last_result = None;
        self.done_timer = 0.0;
        self.accum = 0.0;
    }

    fn tick(&mut self) {
        if self.ep.done {
            self.done_timer += DT;
            if self.auto_track && self.done_timer > 1.2 && self.auto_remaining != 0 {
                self.auto_remaining -= 1;
                if self.auto_remaining >= 0 {
                    self.next_track();
                } else {
                    self.auto_track = false;
                }
            }
            return;
        }
        if self.ep.steps % POLICY_EVERY == 0 {
            let obs = self.ep.observe();
            let out = self.net.forward(&obs, &mut self.pass);
            let (t, s) = (out[0], out[1]);
            self.ep.apply_controls(t, s);
            self.last_controls = [t, s];
            self.activations = self.net.activations(&obs);
        }
        self.ep.advance(&self.spec);
        if self.ep.steps % POLICY_EVERY == 0 {
            self.ep.record_diagnostics();
            self.history.push(self.ep.car.speed);
            if self.history.len() > 900 {
                self.history.remove(0);
            }
        }
        let lap = self.ep.laps().floor() as i32;
        if lap > self.lap_floor && lap >= 1 {
            self.lap_floor = lap;
            self.pending_shots.push(format!("lap{lap}"));
        }
        if self.ep.done {
            self.runs += 1;
            self.best_laps = self.best_laps.max(self.ep.laps());
            if self.ep.finished && !self.ep.crashed {
                self.successes += 1;
                self.best_time = self.best_time.min(self.ep.time);
                self.last_result = Some(format!(
                    "COMPLETED {:.0} LAPS in {:.1}s",
                    self.ep.laps(),
                    self.ep.time
                ));
                self.pending_shots.push("finish".into());
            } else if self.ep.crashed {
                self.last_result = Some(format!(
                    "CRASHED at {:.0}m ({:.1}s)",
                    self.ep.progress, self.ep.time
                ));
                self.pending_shots.push("crash".into());
            } else {
                self.last_result = Some(format!(
                    "TIMEOUT at {:.0}m ({:.1}s)",
                    self.ep.progress, self.ep.time
                ));
                self.pending_shots.push("timeout".into());
            }
        }
    }

    fn speed_kmh(&self) -> f32 {
        self.ep.car.speed * 3.6
    }

    fn save_shot(&mut self, tag: &str) {
        if self.shots_dir.is_none() {
            return;
        }
        if self.shot_seq >= self.shot_limit {
            return;
        }
        let dir = self.shots_dir.as_ref().unwrap();
        if std::fs::create_dir_all(dir).is_err() {
            return;
        }
        self.shot_seq += 1;
        let name = format!("{:04}_{}_{}.png", self.shot_seq, tag, self.ep.track.seed);
        let path = dir.join(name);
        let image = get_screen_data();
        if let Some(p) = path.to_str() {
            image.export_png(p);
        }
        println!("screenshot {}", path.display());
    }
}

#[derive(Clone, Debug)]
struct WatchOptions {
    seed: u64,
    difficulty: f32,
    laps: f32,
    speed: f32,
    auto_track: i32,
    shots: Option<PathBuf>,
    shot_every: f32,
    shot_limit: u32,
    exit_after: f32,
    track_seed: Option<u64>,
}

fn get<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .map(String::as_str)
}

fn parse<T: std::str::FromStr>(args: &[String], name: &str, default: T) -> T {
    get(args, name)
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

pub fn run_watch(args: &[String]) -> i32 {
    let model_path = PathBuf::from(get(args, "--model").unwrap_or("models/champion.json"));
    let model = match load(&model_path) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    if model.sizes != default_sizes() {
        eprintln!(
            "model shape {:?} does not match this build {:?}",
            model.sizes,
            default_sizes()
        );
        return 2;
    }
    let opts = WatchOptions {
        seed: parse(args, "--seed", 123u64),
        difficulty: parse(args, "--difficulty", 1.0f32),
        laps: parse(args, "--laps", 2.0f32),
        speed: parse(args, "--speed", 1.0f32).max(0.1),
        auto_track: parse(args, "--auto-track", 1i32),
        shots: get(args, "--shots").map(PathBuf::from),
        shot_every: parse(args, "--shot-every", 1.0f32),
        shot_limit: parse(args, "--shot-limit", 120u32),
        exit_after: parse(args, "--exit-after", 0.0f32),
        track_seed: get(args, "--track-seed").and_then(|v| v.parse().ok()),
    };
    let start_seed = opts.seed;
    let _ = start_seed;
    let exit_after = opts.exit_after;
    let mut watch = Watch::new(
        model.net(),
        model.meta.generation,
        model_path.clone(),
        opts.clone(),
    );
    println!(
        "watch model {} gen={} seed={} diff={:.2} shots={:?}",
        model_path.display(),
        model.meta.generation,
        opts.seed,
        opts.difficulty,
        opts.shots
    );
    // Base seed is used as given; track index 1 is the first circuit.
    watch.base_seed = opts.seed;
    macroquad::Window::from_config(
        Conf {
            window_title: "Racer — evolved driver".to_string(),
            window_width: 1440,
            window_height: 900,
            high_dpi: true,
            sample_count: 4,
            window_resizable: true,
            ..Default::default()
        },
        async move {
            watch.exit_after = exit_after;
            watch_loop(watch).await;
        },
    );
    0
}

async fn watch_loop(mut w: Watch) {
    let mut last = get_time();
    loop {
        let now = get_time();
        let frame_dt = ((now - last) as f32).clamp(0.0, 0.1);
        last = now;
        w.real_time += frame_dt;
        if w.exit_after > 0.0 && w.real_time >= w.exit_after {
            std::process::exit(0);
        }

        if is_key_pressed(KeyCode::Escape) {
            std::process::exit(0);
        }
        if is_key_pressed(KeyCode::Space) {
            w.paused = !w.paused;
        }
        if is_key_pressed(KeyCode::R) {
            w.next_track();
        }
        if is_key_pressed(KeyCode::S) || is_key_pressed(KeyCode::P) {
            w.pending_shots.push("manual".into());
        }
        if is_key_pressed(KeyCode::Down) || is_key_pressed(KeyCode::Minus) {
            w.speed = (w.speed * 0.5).max(0.25);
        }
        if is_key_pressed(KeyCode::Up) || is_key_pressed(KeyCode::Equal) {
            w.speed = (w.speed * 2.0).min(32.0);
        }

        if !w.paused {
            w.accum += frame_dt * w.speed;
            let mut steps = 0;
            while w.accum >= DT && steps < 4000 {
                w.tick();
                w.accum -= DT;
                steps += 1;
            }
        }

        w.draw();

        if w.shots_dir.is_some() {
            if w.shot_every > 0.0 && w.real_time >= w.next_auto_shot {
                w.next_auto_shot = w.real_time + w.shot_every;
                w.pending_shots.push("auto".into());
            }
            let requests = std::mem::take(&mut w.pending_shots);
            for tag in requests {
                w.save_shot(&tag);
            }
        }

        next_frame().await;
    }
}

impl Watch {
    fn draw(&self) {
        clear_background(BG);
        let sw = screen_width();
        let sh = screen_height();
        let game_w = (sw - PANEL_W).max(200.0);

        // ---- world projection -------------------------------------------
        let (lo, hi) = self.ep.track.bounds();
        let pad = 28.0;
        let world_w = (hi.x - lo.x).max(1.0);
        let world_h = (hi.y - lo.y).max(1.0);
        let scale = ((game_w - pad * 2.0) / world_w).min((sh - pad * 2.0) / world_h);
        let cx = game_w * 0.5;
        let cy = sh * 0.5;
        let mid = (lo + hi) * 0.5;
        let project =
            |p: WV| -> Vec2 { Vec2::new(cx + (p.x - mid.x) * scale, cy + (p.y - mid.y) * scale) };

        // grass
        draw_rectangle(0.0, 0.0, game_w, sh, GRASS);

        // road band
        let n = self.ep.track.n();
        for i in 0..n {
            let j = (i + 1) % n;
            let a = project(self.ep.track.left[i]);
            let b = project(self.ep.track.left[j]);
            let c = project(self.ep.track.right[j]);
            let d = project(self.ep.track.right[i]);
            draw_triangle(a, b, c, ROAD);
            draw_triangle(a, c, d, ROAD);
        }
        // edges
        for i in 0..n {
            let j = (i + 1) % n;
            draw_line(
                project(self.ep.track.left[i]).x,
                project(self.ep.track.left[i]).y,
                project(self.ep.track.left[j]).x,
                project(self.ep.track.left[j]).y,
                1.6,
                EDGE,
            );
            draw_line(
                project(self.ep.track.right[i]).x,
                project(self.ep.track.right[i]).y,
                project(self.ep.track.right[j]).x,
                project(self.ep.track.right[j]).y,
                1.6,
                EDGE,
            );
        }
        // dashed centerline
        for i in (0..n).step_by(4) {
            let j = (i + 2) % n;
            let a = project(self.ep.track.center[i]);
            let b = project(self.ep.track.center[j]);
            draw_line(a.x, a.y, b.x, b.y, 1.2, Color::new(0.95, 0.78, 0.25, 0.5));
        }
        // start/finish checkered band
        draw_start_line(self, &project);

        // sensors
        let spec = self.spec;
        let origin = project(self.ep.car.body_center(&spec));
        for hit in self.ep.sensor_hits.iter().flatten() {
            let hp = project(*hit);
            let d = (hp - origin).length();
            let t = (d / (60.0 * scale).max(1.0)).clamp(0.0, 1.0);
            let col = Color::new(0.3 + 0.7 * t, 1.0 - 0.6 * t, 0.35, 0.65);
            draw_line(origin.x, origin.y, hp.x, hp.y, 1.0, col);
            draw_circle(hp.x, hp.y, 2.0, col);
        }

        // car
        let corners = self.ep.car.corners(&spec);
        let q: Vec<Vec2> = corners.iter().map(|p| project(*p)).collect();
        let body = Color::new(0.95, 0.25, 0.25, 1.0);
        draw_triangle(q[0], q[1], q[2], body);
        draw_triangle(q[0], q[2], q[3], body);
        for k in 0..4 {
            draw_line(
                q[k].x,
                q[k].y,
                q[(k + 1) % 4].x,
                q[(k + 1) % 4].y,
                2.0,
                Color::new(1.0, 0.85, 0.85, 1.0),
            );
        }
        // windshield towards the front
        let f = self.ep.car.forward();
        let front = project(self.ep.car.pos + f * (spec.wheelbase + 1.4));
        let left = project(self.ep.car.pos + f * (spec.wheelbase * 0.6) + f.perp() * 0.75);
        let right = project(self.ep.car.pos + f * (spec.wheelbase * 0.6) - f.perp() * 0.75);
        draw_triangle(front, left, right, Color::new(0.35, 0.55, 0.9, 0.9));

        // ---- HUD text ---------------------------------------------------
        let x = 18.0;
        let mut y = 30.0;
        draw_text("RACER - evolved neural driver", x, y, 24.0, TEXT);
        y += 28.0;
        draw_text(
            &format!(
                "LAP {:.0}/{:.0}   speed {:>5.0} km/h   time {:5.1}s",
                self.ep.laps().max(0.0).ceil().min(self.laps),
                self.laps,
                self.speed_kmh(),
                self.ep.time
            ),
            x,
            y,
            18.0,
            TEXT,
        );
        y += 24.0;
        draw_text(
            &format!(
                "track seed {} diff {:.2}  turn radius {:.0}m  len {:.0}m",
                self.ep.track.seed,
                self.ep.track.difficulty,
                self.ep.track.min_curve_radius,
                self.ep.track.length
            ),
            x,
            y,
            16.0,
            MUTED,
        );
        y += 22.0;
        let c = if self.ep.crashed {
            RED
        } else if self.ep.finished {
            ACCENT
        } else {
            TEXT
        };
        draw_text(
            self.last_result
                .as_deref()
                .unwrap_or(if self.paused { "PAUSED" } else { "running" }),
            x,
            y,
            18.0,
            c,
        );
        y += 22.0;
        draw_text(
            &format!(
                "runs {}  success {}  best {:.2} laps  best time {}",
                self.runs,
                self.successes,
                self.best_laps,
                if self.best_time.is_finite() {
                    format!("{:.1}s", self.best_time)
                } else {
                    "-".to_string()
                }
            ),
            x,
            y,
            16.0,
            MUTED,
        );
        y += 22.0;
        draw_text(
            &format!(
                "throttle {:+.2}  steer {:+.2}   x{:.1}  [S] shot [R] new track [space] pause [↑↓] speed",
                self.last_controls[0], self.last_controls[1], self.speed
            ),
            x,
            y,
            15.0,
            MUTED,
        );

        // speed sparkline (bottom left)
        draw_history(self, 18.0, sh - 100.0, 360.0, 70.0);

        // ---- right panel ------------------------------------------------
        let px = sw - PANEL_W;
        draw_rectangle(px, 0.0, PANEL_W, sh, PANEL);
        draw_line(px, 0.0, px, sh, 1.0, Color::new(0.2, 0.23, 0.28, 1.0));
        let mut py = 30.0;
        draw_text("BRAIN", px + 18.0, py, 20.0, ACCENT);
        py += 24.0;
        draw_text(
            &format!(
                "model gen {}  |  {}",
                self.generation,
                self.model_path.display()
            ),
            px + 18.0,
            py,
            13.0,
            MUTED,
        );
        py += 30.0;
        self.draw_network(px, py, sh);
    }

    fn draw_network(&self, px: f32, top: f32, sh: f32) {
        let layers = &self.activations;
        if layers.len() < 2 {
            return;
        }
        draw_text("live activations", px + 18.0, top, 14.0, MUTED);
        let area_top = top + 16.0;
        let area_h = (sh - area_top - 130.0).max(200.0);
        let xs = [px + 38.0, px + 140.0, px + 245.0, px + 338.0];
        let names = ["sensors+state", "hidden 24", "hidden 16", "controls"];

        // connections
        for li in 0..layers.len().saturating_sub(1) {
            let (from, to) = (&layers[li], &layers[li + 1]);
            let (in_n, out_n) = (from.len(), to.len());
            let mut off = 0usize;
            for k in 0..li {
                off += layers[k].len() * layers[k + 1].len() + layers[k + 1].len();
            }
            let weights = &self.net.params[off..off + in_n * out_n];
            for o in 0..out_n.min(40) {
                let y2 = if out_n == 1 {
                    area_top + area_h * 0.5
                } else {
                    area_top + o as f32 * area_h / (out_n - 1) as f32
                };
                for i in 0..in_n {
                    let w = weights[o * in_n + i];
                    if w.abs() < 0.85 {
                        continue;
                    }
                    let y1 = if in_n == 1 {
                        area_top + area_h * 0.5
                    } else {
                        area_top + i as f32 * area_h / (in_n - 1) as f32
                    };
                    let alpha = ((w.abs() - 0.85) * 0.55).clamp(0.03, 0.45);
                    let col = if w > 0.0 {
                        Color::new(0.2, 0.9, 0.5, alpha)
                    } else {
                        Color::new(1.0, 0.35, 0.35, alpha)
                    };
                    draw_line(xs[li], y1, xs[li + 1], y2, 0.7, col);
                }
            }
        }
        // nodes
        for (li, layer) in layers.iter().enumerate() {
            let n = layer.len();
            for (i, v) in layer.iter().enumerate() {
                let y = if n == 1 {
                    area_top + area_h * 0.5
                } else {
                    area_top + i as f32 * area_h / (n - 1) as f32
                };
                let a = v.clamp(-1.0, 1.0);
                let col = if *v >= 0.0 {
                    Color::new(0.15 + 0.85 * a, 0.3 + 0.6 * a, 0.35, 1.0)
                } else {
                    Color::new(0.95, 0.3 + 0.5 * a.abs(), 0.3, 1.0)
                };
                let r = if li == 0 {
                    3.2
                } else if li == 3 {
                    6.0
                } else {
                    4.0
                };
                draw_circle(xs[li], y, r, col);
                draw_circle_lines(xs[li], y, r, 0.7, Color::new(0.0, 0.0, 0.0, 0.5));
            }
            draw_text(names[li], xs[li] - 26.0, sh - 112.0, 11.0, MUTED);
        }
        // output labels
        let out = &layers[layers.len() - 1];
        if out.len() == 2 {
            let y0 = area_top + area_h * 0.5;
            draw_text("throttle", xs[3] - 24.0, y0 - 12.0, 11.0, MUTED);
            draw_text("steer", xs[3] - 24.0, y0 + 22.0, 11.0, MUTED);
        }
    }
}

fn draw_start_line<F: Fn(WV) -> Vec2>(w: &Watch, project: &F) {
    let n = w.ep.track.n();
    let i = w.ep.track.start_index;
    let j = (i + 1) % n;
    let l = w.ep.track.left[i];
    let r = w.ep.track.right[i];
    let steps = 10;
    for k in 0..steps {
        let t0 = k as f32 / steps as f32;
        let t1 = (k + 1) as f32 / steps as f32;
        let a = project(l.lerp(r, t0));
        let b = project(l.lerp(r, t1));
        let a2 = project(
            l.lerp(r, t0) + (w.ep.track.center[j] - w.ep.track.center[i]).normalized() * 2.0,
        );
        let b2 = project(
            l.lerp(r, t1) + (w.ep.track.center[j] - w.ep.track.center[i]).normalized() * 2.0,
        );
        let col = if k % 2 == 0 {
            Color::new(0.95, 0.95, 0.95, 1.0)
        } else {
            Color::new(0.1, 0.1, 0.1, 1.0)
        };
        draw_triangle(a, b, a2, col);
        draw_triangle(b, b2, a2, col);
    }
}

fn draw_history(w: &Watch, x: f32, y: f32, width: f32, height: f32) {
    draw_rectangle(x, y, width, height, Color::new(0.07, 0.09, 0.12, 0.9));
    draw_rectangle_lines(x, y, width, height, 1.0, Color::new(0.2, 0.23, 0.28, 1.0));
    draw_text("speed", x + 6.0, y + 14.0, 11.0, MUTED);
    if w.history.len() < 2 {
        return;
    }
    let max_v = 36.0f32;
    let n = w.history.len();
    for i in 1..n {
        let x0 = x + (i - 1) as f32 / (n - 1) as f32 * width;
        let x1 = x + i as f32 / (n - 1) as f32 * width;
        let y0 = y + height - (w.history[i - 1] / max_v).clamp(0.0, 1.0) * (height - 6.0) - 3.0;
        let y1 = y + height - (w.history[i] / max_v).clamp(0.0, 1.0) * (height - 6.0) - 3.0;
        draw_line(x0, y0, x1, y1, 1.3, ACCENT);
    }
}
