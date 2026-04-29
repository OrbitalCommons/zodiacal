//! Benchmark: solve time distribution with and without attitude hints.
//!
//! Generates synthetic star fields at varied sky positions and rotations,
//! then solves each under three conditions:
//!   - No attitude hint (blind solve)
//!   - 10-degree accurate hint
//!   - 1-degree accurate hint
//!
//! Outputs JSON to stdout for plotting with `scripts/plot_attitude_bench.py`.
//!
//! Usage:
//!   cargo bench --bench attitude_hint -- --trials 50

use std::f64::consts::PI;
use std::path::PathBuf;
use std::time::Instant;

use starfield::Equatorial;
use zodiacal::extraction::DetectedSource;
use zodiacal::geom::sphere::radec_to_xyz;
use zodiacal::geom::tan::TanWcs;
use zodiacal::index::Index;
use zodiacal::index::builder::{IndexBuilderConfig, build_index};
use zodiacal::solver::{SkyRegion, SolveStats, SolverConfig, solve};
use zodiacal::verify::VerifyConfig;

/// Simple xorshift PRNG for deterministic benchmarks.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 as f64) / (u64::MAX as f64)
    }

    /// Uniform in [lo, hi).
    fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        lo + self.next_f64() * (hi - lo)
    }
}

fn make_test_wcs(
    crval: [f64; 2],
    pixel_scale_arcsec: f64,
    rotation_rad: f64,
    image_size: (f64, f64),
) -> TanWcs {
    let scale = (pixel_scale_arcsec / 3600.0).to_radians();
    let c = rotation_rad.cos() * scale;
    let s = rotation_rad.sin() * scale;
    TanWcs {
        crval,
        crpix: [image_size.0 / 2.0, image_size.1 / 2.0],
        cd: [[c, -s], [s, c]],
        image_size: [image_size.0, image_size.1],
    }
}

struct Scenario {
    sources: Vec<DetectedSource>,
    index: Index,
    wcs: TanWcs,
}

/// Generate a scenario with a wide-field catalog and a narrow field of view.
///
/// The catalog covers a large sky area (~20 degree radius) with many stars,
/// but the image FOV only sees a small subset. This creates realistic
/// ambiguity for the solver — many potential quad matches to sift through.
fn make_scenario(rng: &mut Rng, n_field_stars: usize) -> Scenario {
    let image_size = (512.0, 512.0);
    let pixel_scale_arcsec = 2.0;

    // Random sky position and rotation
    let ra = rng.uniform(0.0, 2.0 * PI);
    let dec = rng.uniform(-PI / 3.0, PI / 3.0); // avoid poles
    let rotation = rng.uniform(0.0, 2.0 * PI);

    let wcs = make_test_wcs([ra, dec], pixel_scale_arcsec, rotation, image_size);

    let scale_rad = (pixel_scale_arcsec / 3600.0).to_radians();
    let field_diag = (image_size.0 * image_size.0 + image_size.1 * image_size.1).sqrt();
    let field_radius = field_diag * scale_rad / 2.0;

    // Generate stars visible in the image
    let mut catalog = Vec::new();
    let mut sources = Vec::new();
    let mut star_id: u64 = 0;

    for i in 0..n_field_stars {
        let px = 30.0 + rng.next_f64() * 452.0;
        let py = 30.0 + rng.next_f64() * 452.0;
        let (star_ra, star_dec) = wcs.pixel_to_radec(px, py);
        catalog.push((star_id, star_ra, star_dec, i as f64));
        sources.push(DetectedSource {
            x: px,
            y: py,
            flux: 1000.0 - i as f64 * 10.0,
        });
        star_id += 1;
    }

    // Add catalog stars outside the field of view (within ~20 degrees).
    // These create false quad matches that the solver must reject.
    let n_background = n_field_stars * 20;
    for i in 0..n_background {
        // Random offset from center, between field_radius and 20 degrees
        let offset_angle = rng.uniform(field_radius, 20.0_f64.to_radians());
        let position_angle = rng.uniform(0.0, 2.0 * PI);

        // Offset in tangent plane, then project back
        let dx = offset_angle * position_angle.cos();
        let dy = offset_angle * position_angle.sin();

        // Simple tangent-plane offset (good enough for benchmark)
        let cos_dec = dec.cos().max(0.1);
        let star_ra = ra + dx / cos_dec;
        let star_dec = (dec + dy).clamp(-PI / 2.0, PI / 2.0);

        catalog.push((star_id, star_ra, star_dec, (n_field_stars + i) as f64));
        star_id += 1;
    }

    let max_angle = 20.0_f64.to_radians();

    let index_config = IndexBuilderConfig {
        scale_lower: scale_rad * 10.0,
        scale_upper: max_angle,
        max_stars: catalog.len(),
        max_quads: 100_000,
    };

    let index = build_index(&catalog, &index_config);
    Scenario {
        sources,
        index,
        wcs,
    }
}

#[derive(serde::Serialize)]
struct TrialResult {
    trial: usize,
    ra_deg: f64,
    dec_deg: f64,
    rotation_deg: f64,
    no_hint_us: u64,
    no_hint_solved: bool,
    hint_10deg_us: u64,
    hint_10deg_solved: bool,
    hint_1deg_us: u64,
    hint_1deg_solved: bool,
}

fn base_config() -> SolverConfig {
    SolverConfig {
        max_field_stars: 25,
        code_tolerance: 0.002,
        verify: VerifyConfig {
            match_radius_pix: 3.0,
            log_odds_accept: 10.0,
            min_matches: 3,
            ..VerifyConfig::default()
        },
        ..SolverConfig::default()
    }
}

fn solve_timed_multi(
    sources: &[DetectedSource],
    indexes: &[&Index],
    image_size: (f64, f64),
    config: &SolverConfig,
) -> (bool, u64, SolveStats) {
    let start = Instant::now();
    let (solution, stats) = solve(sources, indexes, image_size, config);
    let elapsed_us = start.elapsed().as_micros() as u64;
    (solution.is_some(), elapsed_us, stats)
}

/// Pick `n` brightest stars from the loaded index that fall inside the
/// given truth WCS image, projected to pixel coordinates. Returns None
/// if the field doesn't have enough catalog stars.
fn synthesize_sources_from_real_index(
    index: &Index,
    wcs: &TanWcs,
    image_size: (f64, f64),
    n: usize,
) -> Option<Vec<DetectedSource>> {
    let (center_ra, center_dec) = wcs.field_center();
    let center_xyz = radec_to_xyz(center_ra, center_dec);
    // Conservatively use full diagonal as field radius.
    let field_diag_px = (image_size.0 * image_size.0 + image_size.1 * image_size.1).sqrt();
    let pixel_scale_rad = wcs.pixel_scale().to_radians();
    let field_radius_rad = field_diag_px * pixel_scale_rad / 2.0;
    let radius_sq = 2.0 * (1.0 - field_radius_rad.cos());

    let nearby = index.star_tree.range_search(&center_xyz, radius_sq);
    if nearby.len() < n {
        return None;
    }

    let mut by_brightness: Vec<_> = nearby.into_iter().collect();
    by_brightness.sort_by(|a, b| {
        index.stars[a.index]
            .mag
            .partial_cmp(&index.stars[b.index].mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut sources = Vec::with_capacity(n);
    for (i, n_match) in by_brightness.iter().enumerate() {
        if sources.len() >= n {
            break;
        }
        let star = &index.stars[n_match.index];
        let xyz = radec_to_xyz(star.ra, star.dec);
        if let Some((px, py)) = wcs.xyz_to_pixel(xyz)
            && px >= 0.0
            && px < image_size.0
            && py >= 0.0
            && py < image_size.1
        {
            sources.push(DetectedSource {
                x: px,
                y: py,
                flux: 1000.0 - i as f64 * 10.0,
            });
        }
    }

    if sources.len() < n / 2 {
        return None;
    }
    Some(sources)
}

fn main() {
    let mut n_trials: usize = 50;
    let mut n_stars: usize = 25;
    let mut indexes_dir: Option<PathBuf> = None;
    let mut pixel_scale_arcsec: f64 = 1.0;

    // Minimal arg parsing
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--trials" => {
                i += 1;
                n_trials = args[i].parse().expect("invalid --trials value");
            }
            "--stars" => {
                i += 1;
                n_stars = args[i].parse().expect("invalid --stars value");
            }
            "--indexes-dir" => {
                i += 1;
                indexes_dir = Some(PathBuf::from(&args[i]));
            }
            "--pixel-scale" => {
                i += 1;
                pixel_scale_arcsec = args[i].parse().expect("invalid --pixel-scale value");
            }
            // Ignore unknown args (cargo bench passes --bench etc.)
            _ => {}
        }
        i += 1;
    }

    let image_size = (512.0, 512.0);
    let mut rng = Rng::new(42);
    let mut results = Vec::new();

    let real_indexes: Option<Vec<Index>> = if let Some(dir) = indexes_dir.as_ref() {
        let mut paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .expect("indexes dir read")
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("zdcl"))
            .collect();
        paths.sort();
        eprintln!("Loading {} indexes from {}...", paths.len(), dir.display());
        let load_start = Instant::now();
        let loaded: Vec<Index> = paths
            .iter()
            .map(|p| {
                let t = Instant::now();
                let idx = Index::load(p).expect("index load");
                eprintln!(
                    "  {} ({} stars, {} quads, scale {:.0}-{:.0}\") in {:.1}s",
                    p.file_name().unwrap().to_string_lossy(),
                    idx.stars.len(),
                    idx.quads.len(),
                    idx.scale_lower.to_degrees() * 3600.0,
                    idx.scale_upper.to_degrees() * 3600.0,
                    t.elapsed().as_secs_f64(),
                );
                idx
            })
            .collect();
        eprintln!(
            "All indexes loaded in {:.1}s",
            load_start.elapsed().as_secs_f64()
        );
        Some(loaded)
    } else {
        None
    };

    eprintln!(
        "Running {n_trials} trials, {n_stars} stars/scenario, mode={}, pixel_scale={pixel_scale_arcsec}\"...",
        if real_indexes.is_some() {
            "real"
        } else {
            "synthetic"
        }
    );

    for trial in 0..n_trials {
        // ---- build the scenario ----
        let (sources, _indexes_owned, indexes_borrow): (
            Vec<DetectedSource>,
            Option<Index>,
            Vec<&Index>,
        );
        let wcs;

        if let Some(ref real) = real_indexes {
            // Try multiple times to find a pointing with enough stars.
            let mut attempt = 0;
            let scenario = loop {
                let ra = rng.uniform(0.0, 2.0 * PI);
                let dec = rng.uniform(-PI / 3.0, PI / 3.0);
                let rotation = rng.uniform(0.0, 2.0 * PI);
                let trial_wcs = make_test_wcs([ra, dec], pixel_scale_arcsec, rotation, image_size);

                if let Some(srcs) =
                    synthesize_sources_from_real_index(&real[0], &trial_wcs, image_size, n_stars)
                {
                    break (srcs, trial_wcs);
                }
                attempt += 1;
                if attempt > 50 {
                    panic!("could not find a star-rich pointing in 50 attempts");
                }
            };
            sources = scenario.0;
            wcs = scenario.1;
            _indexes_owned = None;
            indexes_borrow = real.iter().collect();
        } else {
            let s = make_scenario(&mut rng, n_stars);
            sources = s.sources;
            wcs = s.wcs;
            _indexes_owned = Some(s.index);
            indexes_borrow = vec![_indexes_owned.as_ref().unwrap()];
        }

        let (center_ra, center_dec) = wcs.field_center();

        // No hint
        let config_none = base_config();
        let (solved_none, us_none, _) =
            solve_timed_multi(&sources, &indexes_borrow, image_size, &config_none);

        // 10-degree hint
        let mut config_10 = base_config();
        config_10.within = Some(SkyRegion::from_degrees(
            Equatorial::new(center_ra, center_dec),
            10.0,
        ));
        let (solved_10, us_10, _) =
            solve_timed_multi(&sources, &indexes_borrow, image_size, &config_10);

        // 1-degree hint
        let mut config_1 = base_config();
        config_1.within = Some(SkyRegion::from_degrees(
            Equatorial::new(center_ra, center_dec),
            1.0,
        ));
        let (solved_1, us_1, _) =
            solve_timed_multi(&sources, &indexes_borrow, image_size, &config_1);

        let rotation_deg = f64::atan2(wcs.cd[1][0], wcs.cd[0][0]).to_degrees();

        results.push(TrialResult {
            trial,
            ra_deg: center_ra.to_degrees(),
            dec_deg: center_dec.to_degrees(),
            rotation_deg,
            no_hint_us: us_none,
            no_hint_solved: solved_none,
            hint_10deg_us: us_10,
            hint_10deg_solved: solved_10,
            hint_1deg_us: us_1,
            hint_1deg_solved: solved_1,
        });

        if (trial + 1) % 10 == 0 {
            eprintln!("  completed {}/{n_trials}", trial + 1);
        }
    }

    // Summary to stderr
    let solved_counts = (
        results.iter().filter(|r| r.no_hint_solved).count(),
        results.iter().filter(|r| r.hint_10deg_solved).count(),
        results.iter().filter(|r| r.hint_1deg_solved).count(),
    );
    eprintln!(
        "\nSolved: no_hint={}/{n_trials}, 10deg={}/{n_trials}, 1deg={}/{n_trials}",
        solved_counts.0, solved_counts.1, solved_counts.2
    );

    // JSON to stdout
    println!("{}", serde_json::to_string_pretty(&results).unwrap());
}
