//! Runtime checks for `wasm32-unknown-unknown`, run under Node by `wasm-bindgen-test-runner`.
//!
//! A `cargo check` for wasm32 only proves the crate compiles. These tests run the index builder and
//! the solver with a timeout set, so clock reads (`Instant::now`) happen in the JS environment,
//! where `std::time::Instant` would panic.
#![cfg(target_arch = "wasm32")]

use std::f64::consts::PI;
use std::time::Duration;

use wasm_bindgen_test::wasm_bindgen_test;
use zodiacal::extraction::DetectedSource;
use zodiacal::geom::tan::TanWcs;
use zodiacal::index::Index;
use zodiacal::index::builder::{IndexBuilderConfig, build_index};
use zodiacal::solver::{SolverConfig, solve};
use zodiacal::verify::VerifyConfig;

const SIZE: f64 = 512.0;
const PIXEL_SCALE_ARCSEC: f64 = 2.0;

fn scenario() -> (Vec<DetectedSource>, Index, TanWcs) {
    let scale = (PIXEL_SCALE_ARCSEC / 3600.0).to_radians();
    let wcs = TanWcs {
        crval: [1.0, 0.5],
        crpix: [SIZE / 2.0, SIZE / 2.0],
        cd: [[scale, 0.0], [0.0, scale]],
        image_size: [SIZE, SIZE],
    };

    // Deterministic xorshift positions; regular grids cause code collisions.
    let mut state: u64 = 314159265;
    let mut rng = || -> f64 {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        (state as f64) / (u64::MAX as f64)
    };
    let mut catalog = Vec::new();
    let mut sources = Vec::new();
    for i in 0..25 {
        let (px, py) = (30.0 + rng() * 452.0, 30.0 + rng() * 452.0);
        let (ra, dec) = wcs.pixel_to_radec(px, py);
        catalog.push((i as u64, ra, dec, i as f64));
        sources.push(DetectedSource {
            x: px,
            y: py,
            flux: 1000.0 - i as f64 * 10.0,
        });
    }

    let diag = (2.0 * SIZE * SIZE).sqrt() * scale;
    let index = build_index(
        &catalog,
        &IndexBuilderConfig {
            scale_lower: scale * 10.0,
            scale_upper: diag,
            max_stars: 25,
            max_quads: 50_000,
        },
    );
    (sources, index, wcs)
}

fn config(timeout: Option<Duration>) -> SolverConfig {
    SolverConfig {
        scale_range: None,
        max_field_stars: 25,
        code_tolerance: 0.002,
        timeout,
        verify: VerifyConfig {
            match_radius_pix: 3.0,
            log_odds_accept: 10.0,
            min_matches: 3,
            ..VerifyConfig::default()
        },
        ..SolverConfig::default()
    }
}

#[wasm_bindgen_test]
fn solves_a_synthetic_field_with_a_timeout() {
    let (sources, index, truth) = scenario();
    let (solution, _) = solve(
        &sources,
        &[&index],
        (SIZE, SIZE),
        &config(Some(Duration::from_secs(10))),
    );
    let solution = solution.expect("solver should find the synthetic field");

    let (ra, dec) = solution.wcs.field_center();
    let (true_ra, true_dec) = truth.field_center();
    let arcsec = PI / (180.0 * 3600.0);
    assert!(
        (ra - true_ra).abs() < 30.0 * arcsec,
        "RA off by {} arcsec",
        (ra - true_ra).abs() / arcsec
    );
    assert!(
        (dec - true_dec).abs() < 30.0 * arcsec,
        "Dec off by {} arcsec",
        (dec - true_dec).abs() / arcsec
    );
}

#[wasm_bindgen_test]
fn an_expired_timeout_is_checked_without_panicking() {
    let (sources, index, _) = scenario();
    // The deadline is already due at the first check. Browser timers are coarse, so a fast solve
    // may still finish first; the point is that every deadline check reads the clock safely.
    let _ = solve(
        &sources,
        &[&index],
        (SIZE, SIZE),
        &config(Some(Duration::ZERO)),
    );
}
