//! Solve test_cases/*.json against a prebuilt index.
//!
//! Usage:
//!   cargo run --example solve_test_cases --release -- --index scratch/gaia_d8_index.zdcl

use std::path::PathBuf;
use std::time::Instant;

use zodiacal::extraction::read_sources_json;
use zodiacal::index::Index;
use zodiacal::solver::{SolverConfig, solve};
use zodiacal::verify::VerifyConfig;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut index_path = PathBuf::from("scratch/gaia_d8_index.zdcl");
    let mut test_dir = PathBuf::from("test_cases");
    let mut max_cases: Option<usize> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--index" => {
                i += 1;
                index_path = PathBuf::from(&args[i]);
            }
            "--dir" => {
                i += 1;
                test_dir = PathBuf::from(&args[i]);
            }
            "--max" => {
                i += 1;
                max_cases = Some(args[i].parse().expect("invalid --max"));
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("Loading index from {}...", index_path.display());
    let t0 = Instant::now();
    let index = Index::load(&index_path).unwrap_or_else(|e| {
        eprintln!("Failed to load index: {e}");
        std::process::exit(1);
    });
    eprintln!(
        "Loaded index in {:.1}s: {} stars, {} quads",
        t0.elapsed().as_secs_f64(),
        index.stars.len(),
        index.quads.len()
    );

    let mut files: Vec<PathBuf> = std::fs::read_dir(&test_dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().is_some_and(|e| e == "json"))
        .collect();
    files.sort();
    if let Some(max) = max_cases {
        files.truncate(max);
    }

    eprintln!("Solving {} test cases...\n", files.len());

    let mut config = SolverConfig::default();
    config.code_tolerance = 0.01;
    config.timeout = Some(std::time::Duration::from_secs(5));
    config.verify = VerifyConfig {
        match_radius_pix: 10.0,
        log_odds_accept: 15.0,
        min_matches: 6,
        ..VerifyConfig::default()
    };

    let mut n_solved = 0;
    let mut n_failed = 0;
    let mut solve_times = Vec::new();

    for (i, file) in files.iter().enumerate() {
        let name = file.file_name().unwrap().to_string_lossy();
        let reader = std::fs::File::open(file).unwrap();
        let sj = read_sources_json(reader).unwrap();

        let t1 = Instant::now();
        let (result, stats) = solve(
            &sj.sources,
            &[&index],
            (sj.image_width, sj.image_height),
            &config,
        );
        let elapsed = t1.elapsed().as_secs_f64();

        match result {
            Some(solution) => {
                let (ra, dec) = solution.wcs.field_center();
                n_solved += 1;
                solve_times.push(elapsed);
                eprintln!(
                    "[{:3}/{}] {} SOLVED {:.2}s  RA={:.4}° Dec={:+.4}°  matched={}  verified={}",
                    i + 1,
                    files.len(),
                    name,
                    elapsed,
                    ra.to_degrees(),
                    dec.to_degrees(),
                    solution.verify_result.n_matched,
                    stats.n_verified,
                );
            }
            None => {
                n_failed += 1;
                eprintln!(
                    "[{:3}/{}] {} FAILED {:.2}s  verified={}  best_rejected={:?}",
                    i + 1,
                    files.len(),
                    name,
                    elapsed,
                    stats.n_verified,
                    stats.best_rejected,
                );
            }
        }
    }

    eprintln!("\n=== Results ===");
    eprintln!("Solved: {n_solved}/{}", files.len());
    eprintln!("Failed: {n_failed}/{}", files.len());
    if !solve_times.is_empty() {
        solve_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = solve_times[solve_times.len() / 2];
        let mean: f64 = solve_times.iter().sum::<f64>() / solve_times.len() as f64;
        let p95 = solve_times[(solve_times.len() as f64 * 0.95) as usize];
        eprintln!("Solve times: median={median:.3}s, mean={mean:.3}s, p95={p95:.3}s");
    }
}
