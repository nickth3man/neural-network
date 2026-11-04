# Racing AI — Rust 2D top-down racer + neuroevolution

Notes for a Rust crate: macroquad top-down racer, procedurally generated closed tracks, an
agent trained by neuroevolution to finish 2 laps without crashing. Versions read from
crates.io / docs.rs on **2026-11-03**; `S1`… resolve in the Sources table.

## 0. Versions

| crate | latest stable | notes |
|---|---|---|
| `macroquad` | **0.4.16** (2026-07-30) [S1] | deps `glam 0.27` (feature `scalar-math`), `image 0.24` (png+tga only), `miniquad =0.4.11` [S2] |
| `rand` | **0.10.3** (2026-09-20) [S3] | `ChaCha*Rng` behind the non-default `chacha` feature |
| `rand_chacha` | **0.10.0** (2026-02-02) [S4] | standalone ChaCha crate; `rand` 0.10 re-exports these |
| `rayon` | **1.12.0** (2026-04-14) [S5] | MSRV 1.80 |

```toml
macroquad = "0.4.16"
rand      = { version = "0.10.3", features = ["chacha"] }
rayon     = "1.12.0"
```

Two facts that matter later: macroquad enables glam's **`scalar-math`** feature, so vector
math is plain scalar `f32` with no target-dependent SIMD path (a determinism win) [S2]; and
`image` ships only **png + tga** decoders, so sprites must be PNG [S2].

## 1. macroquad

### 1.1 Window & entry point

`#[macroquad::main]` expands to `Window::new`/`Window::from_config`, which call
`miniquad::start(miniquad_conf, …)` — window and GL/Metal context are created **before** user
code runs [S6]. `window_conf() -> Conf` customises it [S7]:

```rust
fn window_conf() -> Conf {
    Conf { window_title: "racing-ai".into(), window_width: 1280, window_height: 720,
           window_resizable: false, sample_count: 4, high_dpi: false, ..Default::default() }
}
#[macroquad::main(window_conf)] async fn main() { /* loop { … next_frame().await } */ }
```

`macroquad::conf::Conf` wraps `miniquad::conf::Conf` (field `miniquad_conf`) plus
`update_on`, `default_filter_mode`, `draw_call_vertex_capacity` [S8]. The miniquad part
carries `window_title`, `window_width` (800), `window_height` (600), `high_dpi` (false),
`fullscreen` (false), `sample_count` (1), `window_resizable`, `icon`, `platform` [S9].

### 1.2 Camera2D

Plain struct, public fields [S10]:

```rust
pub struct Camera2D { pub rotation: f32 /*degrees*/, pub zoom: Vec2, pub target: Vec2,
                      pub offset: Vec2, pub render_target: Option<RenderTarget>,
                      pub viewport: Option<(i32, i32, i32, i32)> }
```

`Camera2D::from_display_rect(rect)` maps camera space onto a world rect; `world_to_screen` /
`screen_to_world` handle HUD placement and mouse picking; `set_camera(&cam)` switches target
and flushes pending draw calls; `set_default_camera()` returns to pixel space [S10].

**Y-axis gotcha:** `Camera2D::matrix()` multiplies `zoom.y` by `-1.0` for the screen but
`+1.0` when `render_target.is_some()` [S10, lines 94–104]. A screen-tuned camera renders
**vertically mirrored** into a render target: use `vec2(2.0/w, -2.0/h)` on screen and
`vec2(2.0/w, 2.0/h)` offscreen. Derived from source — confirm with an asymmetric scene.

### 1.3 Primitives

| need | call [S11] |
|---|---|
| filled circle / outline | `draw_circle(x, y, r, color)` / `draw_circle_lines(x, y, r, thickness, color)` |
| thick line | `draw_line(x1, y1, x2, y2, thickness, color)` |
| filled triangle / outline | `draw_triangle(v1: Vec2, v2: Vec2, v3: Vec2, color)` / `draw_triangle_lines(..., thickness, color)` |
| **regular** n-gon only | `draw_poly(x, y, sides: u8, radius, rotation, color)` / `draw_poly_lines(..., thickness, color)` |
| rect (+rotation) | `draw_rectangle`, `draw_rectangle_lines`, `draw_rectangle_ex(..., DrawRectangleParams)` |

There is **no** arbitrary-polygon call: triangulate and emit `draw_triangle`s, or stroke with
`draw_line` plus `draw_circle` at joints for round joins.

Text [S12]: `draw_text(text, x, y, font_size, color) -> TextDimensions`; `y` is the
**baseline**, not the top — glyphs land in
`Rect::new(x, y - dimensions.offset_y, dimensions.width, dimensions.height)` [S12].
`measure_text(text, font, font_size, font_scale)` gives width/height for layout [S12].

### 1.4 Screen capture to PNG

Screen readback, `get_screen_data() -> Image` [S13] — flushes, allocates a screen-sized
texture, `grab_screen()`, copies to CPU [S14]. Fine once, too costly per frame:

```rust
let img = get_screen_data();
img.export_png("frame.png");
```

Offscreen target — window-size independent, deterministic [S15]:

```rust
let rt = render_target(1280, 720);                       // sample_count 1, no depth
rt.texture.set_filter(FilterMode::Nearest);
set_camera(&Camera2D { target: vec2(cx, cy), zoom: vec2(2.0/1280.0, 2.0/720.0), // +y offscreen
                       render_target: Some(rt.clone()), ..Default::default() });
draw_scene();
set_default_camera();                                    // flushes the render pass
let img: Image = rt.texture.get_texture_data();
img.export_png("frame.png");
```

`RenderTarget` has public `texture: Texture2D` and `render_pass` fields [S14]; constructors
are `render_target`, `render_target_msaa`, `render_target_ex` [S15];
`Texture2D::get_texture_data() -> Image` reads back [S16].

`Image::export_png(&self, path: &str)` returns `()`, **panics** on I/O error (`unwrap()` on
`image::save_buffer`), and is documented as unsupported on web [S17]. It flips rows
vertically before writing [S17], so `get_screen_data()` output is already corrected, but an
`Image` you fill yourself is not.

### 1.5 Headless

* **No headless switch.** `miniquad::conf::Platform` lists only windowing backends
  (`linux_x11_gl`, `linux_backend`, `apple_gfx_api`, `swap_interval`, `blocking_event_loop`
  …) — no offscreen-only mode [S18], and window creation precedes `main` [S6].
* Without X11/Wayland on Linux it dies with `XOpenDisplay() failed!` [S19]; workaround
  `xvfb-run -a cargo run` in CI. On macOS a GUI login session is needed for a Metal/GL
  context [INFERENCE — unverified].
* Therefore **split the crate**: `sim` (pure Rust: world, physics, sensors, fitness,
  evolution, all tests) + `app` (macroquad viewer/renderer). Evolution runs headless.
* `blocking_event_loop = true` makes `next_frame().await` block until `schedule_update()`
  [S8, S20] — don't enable it in a one-frame-per-iteration capture loop.

## 2. Determinism

### 2.1 RNG

`StdRng` is **explicitly non-portable** ("any future library version may replace the
algorithm and results may be platform-dependent") [S21], and so is `SmallRng`; the named
portable generators are `ChaCha8/12/20Rng` and `Xoshiro128/256PlusPlus` [S22]. Use
`ChaCha12Rng` — today's `StdRng` algorithm, but with the reproducibility guarantee
[S21, S22, S23]:

```rust
use rand::{Rng, SeedableRng, rngs::ChaCha12Rng};
let mut rng = ChaCha12Rng::seed_from_u64(0xC0FF_EE00);
let track_seed: u64 = rng.random();
```

`SeedableRng::from_seed`: "given a fixed `seed`, the same sequence of output should be
produced on all runs, library versions and architectures" [S24]. Pin `rand` exactly and
commit `Cargo.lock`: portable items are stable across platforms and **patch** releases, but
*minor* releases may break values, and `usize` width differs per target [S25].

### 2.2 `f32`

* `+ - * /` on `f32` are IEEE-754 binary32 with `roundTiesToEven` → reproducible on any
  IEEE target [S26].
* **Transcendentals are not**: `sin`, `cos`, `tan`, `atan2`, `powf`, `exp` "varies by
  platform, Rust version, and can even differ within the same execution from one invocation
  to the next"; `f32::algebraic_*` is likewise non-deterministic [S26].
* NaN payload/sign is non-deterministic across operations [S26]; compare with `total_cmp`,
  never `==`.
* `mul_add` is guaranteed correctly-rounded-fused [S26] and differs from `a*b + c` — pick
  one, never mix.
* Rules: fixed `dt` (e.g. `1.0/120.0`), never `get_frame_time()` (wall-clock frame delta
  [S27]); no `HashMap` iteration in the sim; precompute ray-angle `sin`/`cos` from a table.
  Bit-exactness is per binary+target; across machines results are only statistically equal.

## 3. Vehicle model

### 3.1 Kinematic bicycle, rear-axle reference

State `(x, y, θ, v)`, inputs `(δ, a)`, wheelbase `L = l_f + l_r` [S28]:

```text
ẋ = v·cos θ        ẏ = v·sin θ        θ̇ = v·tan δ / L        v̇ = a
R = L / tan δ                          (turning radius; δ→0 ⇒ R→∞)
```

### 3.2 Centre-of-gravity variant (Kong et al. 2015, eq. 1)

`l_f`, `l_r` = CoG→front/rear axle; `β` = CoG slip angle [S29]:

```text
ẋ = v·cos(ψ + β)          ẏ = v·sin(ψ + β)
ψ̇ = (v / l_r)·sin β       v̇ = a
β = arctan( (l_r / (l_f + l_r))·tan δ_f )
```

Setting `l_f = 0`, `l_r = L` gives `β = 0`, `ψ̇ = v·tan δ / L` — identical to §3.1, same
model with a different reference point.

### 3.3 Exact discrete update (recommended)

Integrating §3.1 analytically per step (constant `v`, `δ`) removes Euler's speed-dependent
heading bias and is stable at 60–120 Hz:

```text
Δθ = v·tan δ / L · dt
if |tan δ| > 1e-4:  R = L / tan δ
                    x += R·(sin(θ + Δθ) − sin θ)
                    y += R·(cos θ − cos(θ + Δθ))
else:               x += v·dt·cos θ ;  y += v·dt·sin θ
θ += Δθ ;  v = clamp(v + a·dt, 0, v_max)
```

Check: `Δθ→0` ⇒ `R·Δθ·cos θ = v·dt·cos θ`. (Integrated from §3.1.)

```text
δ   = clamp(δ + δ̇·dt, −δ_max, δ_max)
a_y = v²·tan δ / L = v² / R                  (centripetal)
|a_y| ≤ μ·g   ⇒   v_max(R) = sqrt(μ·g·R)     (friction-circle limit) [S30]
```

For arcade feel, apply the `a_y ≤ μg` bound as a soft clamp on `v` in corners instead of
modelling tyres.

### 3.4 Slip / drift

Dynamic bicycle with linear tyres, Kong et al. eqs. (2)–(3) [S29]:

```text
v̇_x =  v_y·ψ̇ + a_x
v̇_y = −v_x·ψ̇ + (F_yf·cos δ + F_yr) / m
ψ̈   = (l_f·F_yf·cos δ − l_r·F_yr) / I_z
F_yi = C_i·α_i
α_f  = δ − arctan((v_y + l_f·ψ̇) / v_x)
α_r  = −arctan((v_y − l_r·ψ̇) / v_x)
Ẋ = v_x·cos ψ − v_y·sin ψ        Ẏ = v_x·sin ψ + v_y·cos ψ
```

Singular as `v_x → 0` (slip angles divide by `v_x`) — exactly why Kong et al. recommend the
kinematic model at low speed [S29]. If used: clamp `v_x ≥ 0.5`, integrate at `dt ≤ 1/240`.

Cheap drift without tyres: keep §3.1 with `κ_eff = (tan δ / L)·grip(v)`,
`grip(v) = clamp(1 − k·v², 0.4, 1)`, then lerp the velocity vector toward the heading each
step. Engineering choice, not literature.

## 4. Range sensors

### 4.1 Ray–segment intersection

Ray `P + t·d` with `|d| = 1`, segment `A + u·s`, `s = B − A`, `q = A − P` [S31]:

```text
den = d × s = d.x·s.y − d.y·s.x
if |den| < 1e-9 → no hit (parallel)
t = (q × s) / den        // metric distance because |d| = 1
u = (q × d) / den
hit ⇔ t ≥ 0 and 0 ≤ u ≤ 1
```

Cross-product form of Bourke's two-segment solution (`ua`/`ub` share the denominator;
intersection iff both in `[0,1]`) [S31]. Range = `min(t)` over all wall segments.

Cost: `N` rays × `M` walls per car per step. Bucket segments into a uniform grid (cell
≈ 4 m) once and test only cells the ray crosses — ~10 µs vs ~1 ms per step for 19 rays.

### 4.2 Layout & normalisation

`φ_i = −90° + i·10°`, `i = 0..18` (19 rays, forward 180°); world direction
`d_i = (cos(θ+φ_i), sin(θ+φ_i))`.

```text
r_i = clamp(1 − t_i / d_max, 0, 1)      // 1 = wall adjacent; d_max ≈ 2–3 × track width
```

Bounded, monotone, saturating; scaling inputs to a fixed symmetric range (zero mean / unit
variance) is the standard preprocessing rule [S32]. Add `v / v_max`, `sin(Δψ)`, `cos(Δψ)`
with `Δψ` the heading error to the track tangent — `sin`/`cos` avoids the ±π wrap.

Egocentric evidence: Togelius & Lucas found only the controller using **first-person state
inputs** evolved good behaviour; global-state and open-loop controllers failed [S33].

## 5. Neuroevolution

### 5.1 NEAT (Stanley & Miikkulainen 2002)

*Evolving Neural Networks through Augmenting Topologies*, Evol. Comput. 10(2):99–127,
DOI 10.1162/106365602320169811 [S34]. Three mechanisms: innovation-number gene matching for
crossover, speciation with fitness sharing to protect new structure, growth from minimal
topology (zero hidden nodes) [S34]. Published settings (paper §4.1) [S34]:

```text
population 150 (1000 for hardest pole task)
δ = c1·E/N + c2·D/N + c3·W̄,  c1 = c2 = 1.0, c3 = 0.4 ;  speciation threshold t = 3.0
stagnation 15 gens without species max-fitness gain ⇒ no offspring
elitism: champion of each species with >5 members copied unchanged
weight mutation 80% of genomes; per weight 90% uniform perturb, 10% new random
structural: p(add node) = 0.03, p(add link) = 0.05 (0.3 link in the 1000-pop runs)
asexual offspring 25% ; interspecies mating 0.001 ; re-enable disabled gene 75%
activation 1 / (1 + e^(−4.9x))
```

### 5.2 Racing applications

* Cardamone, Loiacono & Lanzi 2010 [S35]: online NEAT/rtNEAT evolved TORCS drivers from
  scratch, transferred knowledge between tracks and generalised to unseen tracks; fitness is
  updated *within* a lap, not only per generation.
* Togelius & Lucas 2005 [S33]: compared nets, force fields and action sequences; only
  egocentric-state nets evolved well.
* Practical: score the whole population on **one shared track per generation** (comparable
  fitness) and add a held-out track every N generations — the [S35] generalisation test.

### 5.3 Fixed-topology GA vs NEAT

For 22 inputs → 2 outputs, a fixed `22 → 8 → 2` tanh MLP with a plain GA is simpler and
converges faster per wall-clock second; NEAT pays off for minimal networks or recurrence.
Engineering defaults (not from literature): pop 100–200, elitism 2, tournament `k = 3`,
per-weight mutation 0.1 with `σ = 0.2`, crossover 0.75, `σ` annealed 0.3 → 0.05.

Inputs: the 19 ranges + speed + heading error. Outputs: `tanh` steering → `δ = δ_max·out_0`;
throttle/brake → `a = a_max·out_1` (negative = brake). Two outputs beat one: braking into a
corner separates a fast lap from a survivable one.

## 6. Track generation

### 6.1 Closed Catmull–Rom centreline

`n` control points `P_0..P_{n-1}` CCW, indices mod `n`; segment `i` uses `p0 = P_{i-1}`,
`p1 = P_i`, `p2 = P_{i+1}`, `p3 = P_{i+2}`. Centripetal (`α = 0.5`) is the family member
guaranteed free of cusps and self-intersections within a segment, and the paper shows how to
build globally intersection-free CR curves from it [S36]; uniform (`α = 0`) visibly
overshoots and can loop.

```text
d1 = |p1 − p0| , d2 = |p2 − p1| , d3 = |p3 − p2|
t0 = 0 , t1 = d1^α , t2 = t1 + d2^α , t3 = t2 + d3^α , h = t2 − t1
m1 = (p1−p0)/(t1−t0) − (p2−p0)/(t2−t0) + (p2−p1)/(t2−t1)
m2 = (p2−p1)/(t2−t1) − (p3−p1)/(t3−t1) + (p3−p2)/(t3−t2)
h00 = 2u³−3u²+1   h10 = u³−2u²+u   h01 = −2u³+3u²   h11 = u³−u²
C(u)  = h00·p1 + h10·(h·m1) + h01·p2 + h11·(h·m2)
C'(u) = (6u²−6u)·p1 + (3u²−4u+1)·(h·m1) + (−6u²+6u)·p2 + (3u²−2u)·(h·m2)
B0 = p1 , B1 = p1 + (h/3)·m1 , B2 = p2 − (h/3)·m2 , B3 = p2
```

Tangents are the Barry–Goldman formulation [S37]; Hermite→Bézier is `B1 = P1 + T1/3`,
`B2 = P2 − T2/3`. Shared tangents at joints make the curve C1; wrapped indices make the
**seam** C1 too (do not duplicate endpoints — that leaves a corner at the start line).

### 6.2 Arc-length resampling

No closed form; tabulate from dense samples (32 per segment at metre scale):

```text
for each i, k in 1..=N:  s[i][k] = s[i][k-1] + |C_i(k/N) − C_i((k-1)/N)|
L = s[n-1][N]
```

Resample at `ds`: walk the cumulative table, binary search `s_target = j·ds`, linearly
interpolate `u` between bracketing samples (piecewise-linear `∫|C'|du`, `O(h²)` error).
Store per sample: `C`, `T = C'/|C'|`, `N = (−T.y, T.x)`, `κ = |C' × C''| / |C'|³`, `s`.

### 6.3 Width & boundaries

Per-control-point `w_i`, interpolated along `u`:

```text
left(s)  = C(s) + (w(s)/2)·N(s)
right(s) = C(s) − (w(s)/2)·N(s)
```

An offset of a C1 curve is C1 only where the radius of curvature exceeds the offset; sharp
corners self-intersect on the inside. Guard `κ(s)·w(s)/2 < 1` and shrink `w` there. Keep
`w ∈ [0.8, 1.4]·w_0`; a width step at a joint kinks visibly even though `C` stays C1.

### 6.4 Rejection checks

Centripetal CR is intersection-free *within* a segment [S36], not between distant ones:

```text
1. Sample the centreline at ds ≈ 0.5 m.
2. For pairs (i, j) with circular segment distance ≥ 2:
       d = min pairwise point-to-segment distance ; require d ≥ w_max + margin (≈ 0.5·w_0)
3. Drivability: max κ(s) ≤ 1/R_min ,  R_min = max(L / tan δ_max, v_ref² / (μ·g))
4. On failure jitter control points (radius/angle) and regenerate; cap ~200 tries,
   then relax width.
```

Point-to-segment distance is the standard clamped-projection primitive [S38]; pair it with
the exact segment test of §4.1 to catch crossings.

## 7. Fitness & collisions

### 7.1 Centreline projection

Start from the last known sample and search ±20 samples for `O(1)` amortised cost:

```text
ab = b − a
t  = clamp(((P − a)·ab) / (ab·ab), 0, 1)
Q  = a + t·ab
s  = s_a + t·|ab|                       (arc length)
e  = cross(ab, P − a) / |ab|            (signed lateral error, +=left)
```

Same projection Kong et al. use to build their reference path [S29]; `(s, e)` is the
Frenet-frame longitudinal/lateral split used for road trajectories [S39].

### 7.2 Lap counting without exploits

```text
ds = s − s_prev
if ds >  L/2 { ds -= L }          // wrapped forward across the line
if ds < −L/2 { ds += L }
s_unwrapped += ds
progress = max(progress, s_unwrapped)      // monotone: reversing earns nothing
laps     = floor(progress / L)
```

The `max` matters: `Σ|ds|` or `Σ max(ds,0)` lets an agent farm fitness by oscillating over
the line.

### 7.3 Fitness

Fixed time budget per episode (e.g. 90 s simulated), so speed pays implicitly:

```text
F = 100·laps + 100·(progress mod L)/L − 5·mean(|e| / (w/2))
    − 50·crashes − 10·(time_used / time_budget)
```

With "crash ends episode" `crashes ∈ {0,1}`. Also terminate at `laps == 2` and convert the
leftover time into a bonus — that pushes the population toward fast lines, not survival.

### 7.4 Collision

Approximate the car by 3 circles at local `x = −l/2, 0, +l/2`, radius `w_car/2` (capsule).
For each centre `P` and wall segment compute `Q`, `d = |P − Q|` as in 7.1.

```text
A. Terminate (recommended for evolution)
   if any d < r: episode over, fitness frozen at current progress.
B. Slide (human-playable mode)
   n = (P − Q)/d ;  P += n·(r − d) ;  vn = v·n
   if vn < 0 { v −= (1 + e)·vn·n }        // e = 0
   v *= 0.9 ;  crashes += 1
```

Positional correction plus velocity projection along the contact normal is the standard
penetration-resolution scheme [S38]. Never scale the impulse by penetration depth with a
large `dt` — it injects energy and breaks reproducibility.

## 8. Parallel evaluation (rayon)

`par_iter()` from `IntoParallelRefIterator`, `into_par_iter()` from
`IntoParallelIterator` [S40]:

```rust
use rayon::prelude::*;
let fitness: Vec<f32> = genomes
    .par_iter().enumerate()
    .map(|(i, g)| evaluate(g, track, base_seed ^ (i as u64).wrapping_mul(0x9E37_79B9)))
    .collect();
```

Reproducible because each episode owns a `ChaCha12Rng` seeded from `(base_seed, i)` — never
`rand::rng()`, which is thread-local and non-deterministic [S22]; `collect()` on an indexed
iterator preserves input order, so `fitness[i]` belongs to `genomes[i]`; `evaluate` takes
`&Track` and returns a value (no shared mutable state). Size the pool once if the renderer
also spawns threads: `rayon::ThreadPoolBuilder::new().num_threads(8).build_global()`.

**Use rayon, not `std::thread`.** Episodes are independent, similarly sized, and number in
the hundreds; work-stealing removes manual chunking and absorbs stragglers (a car that
survives longer). `std::thread` only for a hand-rolled long-lived pool with pinned state —
and rayon's `map_init` already covers thread-local state. Never evaluate on the render thread
inside `next_frame().await`: the window freezes for a whole generation; run it in a task and
poll a channel.

## 9. Recommended stack

| layer | choice | why |
|---|---|---|
| render | `macroquad` **0.4.16** | immediate-mode 2D; `render_target` + `Image::export_png` for deterministic captures [S1, S15, S17] |
| math | `glam` **0.27** via `macroquad::math` | already a macroquad dep (scalar-math), no duplicate glam version [S2] |
| RNG | `rand` **0.10.3** + `chacha` → `ChaCha12Rng` | only portable generators carry a reproducibility contract [S22, S24, S25] |
| parallel | `rayon` **1.12.0** | `par_iter().map().collect()` over the population, order-preserving [S40] |
| persist | `serde` + `serde_json` (optional) | champion genomes; keep out of the sim hot path |
| crates | `sim` (no graphics, all tests) + `app` (viewer) | macroquad cannot run headless [S19] |

Flow: `seed → track generator (CR + rejection) → sim (fixed dt) → sensors (19 rays) → MLP →
(δ, a) → fitness (progress, |e|, crash) → GA/NEAT → next generation`; the `app` crate renders
champion episodes and exports PNG frames.

## 10. Unresolved / uncertain

1. **`Camera2D` y-flip offscreen** — read from `camera.rs` 94–104 (`invert_y = 1.0` when
   `render_target.is_some()`), not from prose docs, not executed. Verify with an asymmetric
   scene before trusting offscreen screenshots.
2. **macOS headless** — unstated in macroquad docs; only the Linux failure is documented
   [S19]. [INFERENCE]
3. **`get_screen_data()` with MSAA** — the readback renders into a plain texture [S14]; no
   doc guarantees resolve behaviour. Prefer `render_target` with `sample_count: 1`.
4. **`export_png` errors** — returns `()` and panics via `unwrap()` [S17]; `catch_unwind` if
   a bad path must not kill the app.
5. **Kong et al. eq. (2) signs** — the public mirror's Greek glyphs are corrupted in text
   extraction; §3.4 follows the standard SAE body-frame convention, but check signs against
   the IEEE version [S29].
6. **NEAT parameters** — §5.1 numbers come from pole-balancing/XOR (§4.1), not racing [S34].
7. **Arc-length `O(h²)`** — standard piecewise-linear convergence; no specific citation.

## Sources

| # | claim | URL |
|---|---|---|
| S1 | macroquad latest stable 0.4.16, published 2026-07-30 | https://crates.io/api/v1/crates/macroquad |
| S2 | macroquad 0.4.16 deps: glam 0.27 (`scalar-math`), image 0.24 (png, tga), miniquad `=0.4.11` | https://docs.rs/crate/macroquad/0.4.16/source/Cargo.toml |
| S3 | rand latest stable 0.10.3; `chacha` non-default feature | https://crates.io/api/v1/crates/rand |
| S4 | rand_chacha latest stable 0.10.0 | https://crates.io/api/v1/crates/rand_chacha |
| S5 | rayon latest stable 1.12.0, MSRV 1.80 | https://crates.io/api/v1/crates/rayon |
| S6 | entry point runs through `miniquad::start` before user code | https://docs.rs/macroquad/0.4.16/src/macroquad/lib.rs.html#930-985 |
| S7 | `window_conf() -> Conf` pattern | https://docs.rs/macroquad/0.4.16/macroquad/attr.main.html |
| S8 | `macroquad::conf::Conf` fields, `blocking_event_loop` note | https://docs.rs/macroquad/0.4.16/macroquad/conf/struct.Conf.html |
| S9 | miniquad `Conf` window fields and defaults | https://docs.rs/miniquad/0.4.11/miniquad/conf/struct.Conf.html |
| S10 | `Camera2D` fields, `from_display_rect`, `set_camera`/`set_default_camera`, y-inversion in `matrix()` | https://docs.rs/macroquad/0.4.16/src/macroquad/camera.rs.html |
| S11 | `draw_circle(_lines)`, `draw_line`, `draw_triangle(_lines)`, `draw_poly(_lines)`, rect params | https://docs.rs/macroquad/0.4.16/macroquad/shapes/index.html |
| S12 | `draw_text` baseline semantics, `measure_text`, `TextDimensions.offset_y` | https://docs.rs/macroquad/0.4.16/src/macroquad/text.rs.html |
| S13 | `get_screen_data() -> Image` | https://docs.rs/macroquad/0.4.16/macroquad/texture/fn.get_screen_data.html |
| S14 | `get_screen_data` impl (flush → texture → grab_screen → readback); `RenderTarget { texture, render_pass }` | https://docs.rs/macroquad/0.4.16/src/macroquad/texture.rs.html |
| S15 | `render_target`, `render_target_msaa`, `render_target_ex` | https://docs.rs/macroquad/0.4.16/macroquad/texture/index.html |
| S16 | `Texture2D::get_texture_data() -> Image`, `grab_screen` | https://docs.rs/macroquad/0.4.16/macroquad/texture/struct.Texture2D.html |
| S17 | `Image::export_png` — `()`, flips rows, `unwrap()`, panics on web | https://docs.rs/macroquad/0.4.16/src/macroquad/texture.rs.html |
| S18 | miniquad `Platform` — window-system backends only, no headless mode | https://docs.rs/miniquad/0.4.11/miniquad/conf/struct.Platform.html |
| S19 | macroquad fails without a display (`XOpenDisplay() failed!`) | https://github.com/not-fl3/macroquad/issues/306 |
| S20 | `blocking_event_loop` + `schedule_update` | https://docs.rs/miniquad/0.4.11/miniquad/conf/struct.Platform.html |
| S21 | `StdRng` non-portable; current algorithm ChaCha with 12 rounds | https://docs.rs/rand/0.10.3/rand/rngs/struct.StdRng.html |
| S22 | non-portable vs named-portable generators; `rand::rng()` thread-local | https://docs.rs/rand/0.10.3/rand/rngs/index.html |
| S23 | `ChaCha12Rng` (64-bit counter/stream, 32-byte seed) | https://docs.rs/rand/0.10.3/rand/rngs/struct.ChaCha12Rng.html |
| S24 | `SeedableRng::from_seed` reproducibility contract | https://docs.rs/rand/0.10.3/rand/trait.SeedableRng.html |
| S25 | portable items stable across platforms/patch releases; minor releases may break values; `usize` non-portable | https://rust-random.github.io/book/crate-reprod.html |
| S26 | `f32` IEEE-754 binary32, roundTiesToEven; NaN non-determinism; transcendentals non-deterministic; `mul_add` guarantee | https://doc.rust-lang.org/std/primitive.f32.html |
| S27 | `get_frame_time()` = wall-clock duration of last frame | https://docs.rs/macroquad/0.4.16/macroquad/time/index.html |
| S28 | rear-axle kinematic bicycle `θ̇ = v tan δ / L`, `R = L / tan δ` | https://thomasfermi.github.io/Algorithms-for-Automated-Driving/Control/BicycleModel.html |
| S29 | Kong et al. 2015 (IV 2015) — kinematic eq. (1), dynamic eqs. (2)–(3), low-speed singularity, centreline projection | https://doi.org/10.1109/IVS.2015.7225830 · PDF: https://nuhuo08.github.io/control/IV_KinematicMPC_jason.pdf |
| S30 | Rajamani, *Vehicle Dynamics and Control* 2nd ed. — tyre/road friction limit (`a_y ≤ μg`) | https://doi.org/10.1007/978-1-4614-1433-9 |
| S31 | two-segment intersection: parametric `ua`/`ub`, shared denominator, `[0,1]` tests | https://paulbourke.net/geometry/pointlineplane/ |
| S32 | input normalisation to fixed symmetric range / unit variance | https://doi.org/10.1007/978-3-642-35289-8_3 |
| S33 | Togelius & Lucas 2005 — only egocentric-state neural controllers evolved good racing behaviour | https://doi.org/10.1109/CEC.2005.1554920 |
| S34 | Stanley & Miikkulainen 2002, Evol. Comput. 10(2):99–127 — NEAT mechanisms, §4.1 parameters | https://doi.org/10.1162/106365602320169811 · PDF: http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf |
| S35 | Cardamone, Loiacono & Lanzi 2010 — online neuroevolution (NEAT/rtNEAT) for TORCS, generalisation to unseen tracks | https://doi.org/10.1109/TCIAIG.2010.2052102 |
| S36 | Yuksel, Schaefer & Keyser 2011 — centripetal CR avoids cusps/self-intersections; CR→Bézier | https://doi.org/10.1016/j.cad.2010.08.008 · https://www.cemyuksel.com/research/catmullrom_param/ |
| S37 | Barry & Goldman 1988, SIGGRAPH '88 pp. 199–204 — recursive evaluation, CR tangents | https://doi.org/10.1145/378456.378511 |
| S38 | Ericson, *Real-Time Collision Detection* (CRC 2004) — closest point on segment, contact-normal resolution | https://doi.org/10.1201/b14581 |
| S39 | Werling et al. 2010 — Frenet-frame longitudinal/lateral decomposition | https://doi.org/10.1109/ROBOT.2010.5509799 |
| S40 | rayon `par_iter` / `into_par_iter` | https://docs.rs/rayon/1.12.0/rayon/iter/trait.IntoParallelRefIterator.html · https://docs.rs/rayon/1.12.0/rayon/iter/trait.IntoParallelIterator.html |
