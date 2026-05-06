//! `bench-indexes` — sweep the 1000-case test corpus against a list
//! of pre-built single-band `.zdcl` indexes (the old multi-file
//! layout).
//!
//! Mirrors `bench-bundle`'s output schema so the two can be compared
//! directly. The indexes are full-sky and loaded once at startup; per
//! test case we just project sources and run `solve()` against the
//! shared `Vec<&Index>`.

use std::fs;
use std::io;
use std::path::PathBuf;
use std::time::{Duration, Instant};

use serde::Deserialize;

use zodiacal::extraction::DetectedSource;
use zodiacal::geom::sphere::{angular_distance, radec_to_xyz};
use zodiacal::index::Index;
use zodiacal::solver::{SolverConfig, solve};
use zodiacal::verify::VerifyConfig;

#[derive(Debug, Deserialize)]
struct TestCase {
    image_width: f64,
    image_height: f64,
    ra_deg: f64,
    dec_deg: f64,
    plate_scale_arcsec: f64,
    sources: Vec<TestCaseSource>,
}

#[derive(Debug, Deserialize)]
struct TestCaseSource {
    x: f64,
    y: f64,
    flux: f64,
}

pub struct BenchIndexesConfig {
    pub index_files: Vec<PathBuf>,
    pub test_cases_dir: PathBuf,
    pub limit: Option<usize>,
    pub scale_hint: bool,
    pub timeout_secs: u64,
}

pub fn run(cfg: &BenchIndexesConfig) -> io::Result<()> {
    if cfg.index_files.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "at least one --index-file required",
        ));
    }

    eprintln!("Loading {} index file(s)...", cfg.index_files.len());
    let load_start = Instant::now();
    let indexes: Vec<Index> = cfg
        .index_files
        .iter()
        .map(|p| {
            eprint!("  {}: ", p.display());
            let s = Instant::now();
            let idx = Index::load(p)?;
            eprintln!(
                "{} stars / {} quads in {:.1}s",
                idx.stars.len(),
                idx.quads.len(),
                s.elapsed().as_secs_f64()
            );
            Ok::<_, io::Error>(idx)
        })
        .collect::<io::Result<Vec<_>>>()?;
    let total_quads: usize = indexes.iter().map(|i| i.quads.len()).sum();
    let total_stars: usize = indexes.iter().map(|i| i.stars.len()).sum();
    eprintln!(
        "Loaded {} indexes in {:.1}s — {} stars, {} quads total",
        indexes.len(),
        load_start.elapsed().as_secs_f64(),
        total_stars,
        total_quads,
    );
    let index_refs: Vec<&Index> = indexes.iter().collect();

    let mut paths: Vec<PathBuf> = fs::read_dir(&cfg.test_cases_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    paths.sort();
    if let Some(n) = cfg.limit {
        paths.truncate(n);
    }
    eprintln!(
        "Running {} test case(s), scale_hint={}",
        paths.len(),
        cfg.scale_hint,
    );

    println!(
        "case,solved,solve_ms,error_arcsec,n_total_quads,n_total_stars,n_verified,best_rejected_log_odds,best_rejected_n_matched,best_rejected_error_arcsec,timed_out"
    );

    let mut n_solved = 0usize;
    let mut total_solve_ms = 0.0f64;
    let bench_start = Instant::now();

    for (i, p) in paths.iter().enumerate() {
        let raw = fs::read_to_string(p)?;
        let tc: TestCase = serde_json::from_str(&raw).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", p.display()))
        })?;

        let sources: Vec<DetectedSource> = tc
            .sources
            .iter()
            .map(|s| DetectedSource {
                x: s.x,
                y: s.y,
                flux: s.flux,
            })
            .collect();

        let mut solver_cfg = SolverConfig {
            max_field_stars: 50,
            code_tolerance: 0.002,
            verify: VerifyConfig {
                match_radius_pix: 3.0,
                log_odds_accept: 20.0,
                min_matches: 5,
                ..VerifyConfig::default()
            },
            ..SolverConfig::default()
        };
        if cfg.scale_hint {
            solver_cfg.scale_range =
                Some((tc.plate_scale_arcsec * 0.75, tc.plate_scale_arcsec * 1.25));
        }
        if cfg.timeout_secs > 0 {
            solver_cfg.timeout = Some(Duration::from_secs(cfg.timeout_secs));
        }

        let solve_start = Instant::now();
        let (solution, stats) = solve(
            &sources,
            &index_refs,
            (tc.image_width, tc.image_height),
            &solver_cfg,
        );
        let solve_ms = solve_start.elapsed().as_secs_f64() * 1000.0;

        let truth_xyz = radec_to_xyz(tc.ra_deg.to_radians(), tc.dec_deg.to_radians());
        let wcs_error_arcsec = |wcs: &zodiacal::geom::tan::TanWcs| -> f64 {
            let (got_ra, got_dec) = wcs.field_center();
            let got_xyz = radec_to_xyz(got_ra, got_dec);
            angular_distance(truth_xyz, got_xyz).to_degrees() * 3600.0
        };

        let (solved, error_arcsec) = match &solution {
            Some(sol) => (true, wcs_error_arcsec(&sol.wcs)),
            None => (false, f64::NAN),
        };
        let (best_rej_log_odds, best_rej_n_matched, best_rej_error) =
            match (stats.best_rejected, stats.best_rejected_wcs.as_ref()) {
                (Some((lo, nm)), Some(wcs)) => (lo, nm as i64, wcs_error_arcsec(wcs)),
                _ => (f64::NAN, -1i64, f64::NAN),
            };

        if solved {
            n_solved += 1;
        }
        total_solve_ms += solve_ms;

        let case_name = p.file_stem().unwrap().to_string_lossy();
        let fmt_f = |v: f64| -> String {
            if v.is_nan() {
                String::new()
            } else {
                format!("{:.3}", v)
            }
        };
        let fmt_i = |v: i64| -> String { if v < 0 { String::new() } else { v.to_string() } };
        println!(
            "{},{},{:.1},{},{},{},{},{},{},{},{}",
            case_name,
            solved as u8,
            solve_ms,
            fmt_f(error_arcsec),
            total_quads,
            total_stars,
            stats.n_verified,
            fmt_f(best_rej_log_odds),
            fmt_i(best_rej_n_matched),
            fmt_f(best_rej_error),
            stats.timed_out as u8,
        );

        if (i + 1) % 50 == 0 {
            eprintln!(
                "  [{:>4}/{}] solved={} ({:.1}%) solve_avg={:.0}ms",
                i + 1,
                paths.len(),
                n_solved,
                100.0 * n_solved as f64 / (i + 1) as f64,
                total_solve_ms / (i + 1) as f64,
            );
        }
    }

    let n = paths.len() as f64;
    let total_elapsed = bench_start.elapsed().as_secs_f64();
    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("Total cases:  {}", paths.len());
    eprintln!(
        "Solved:       {} ({:.1}%)",
        n_solved,
        100.0 * n_solved as f64 / n,
    );
    eprintln!("Avg solve:    {:.1} ms", total_solve_ms / n);
    eprintln!("Wall-clock:   {:.1} s", total_elapsed);

    Ok(())
}
