//! Main blind plate solver that ties together quad building, code matching,
//! WCS fitting, and verification.

use std::time::{Duration, Instant};

use crate::extraction::DetectedSource;
use crate::fitting::{FitError, fit_tan_wcs};
use crate::geom::sphere::radec_to_xyz;
use crate::geom::tan::TanWcs;
use crate::index::Index;
use crate::quads::{DIMCODES, DIMQUADS};
use crate::verify::{VerifyConfig, VerifyResult, verify_solution};

/// Configuration for the solver.
pub struct SolverConfig {
    /// Pixel scale range to search (arcsec/pixel). If None, try all scales.
    pub scale_range: Option<(f64, f64)>,
    /// Maximum number of field stars to use for quad building.
    pub max_field_stars: usize,
    /// Code matching tolerance (squared L2 distance in code space).
    pub code_tolerance: f64,
    /// Verification configuration.
    pub verify: VerifyConfig,
    /// Maximum time to spend solving before giving up.
    pub timeout: Option<Duration>,
}

impl Default for SolverConfig {
    fn default() -> Self {
        Self {
            scale_range: None,
            max_field_stars: 50,
            code_tolerance: 0.01,
            verify: VerifyConfig::default(),
            timeout: None,
        }
    }
}

/// A successful plate solve solution.
#[derive(Debug, Clone)]
pub struct Solution {
    pub wcs: TanWcs,
    pub verify_result: VerifyResult,
    pub quad_match: QuadMatch,
}

/// Statistics collected during a solve attempt.
#[derive(Debug, Clone)]
pub struct SolveStats {
    /// Total number of WCS candidates that passed cheap filters and reached verification.
    pub n_verified: usize,
    /// Best rejected candidate: (log_odds, n_matched). None if no rejections.
    pub best_rejected: Option<(f64, usize)>,
    /// Log-odds of the accepted candidate (if any).
    pub accepted_log_odds: Option<f64>,
    /// Whether the solve timed out.
    pub timed_out: bool,
}

/// Information about the quad that produced the solution.
#[derive(Debug, Clone)]
pub struct QuadMatch {
    /// Indices into the field source list.
    pub field_indices: [usize; DIMQUADS],
    /// Indices into the index star list.
    pub index_indices: [usize; DIMQUADS],
}

/// Compute field quad codes from 4 pixel positions.
///
/// Replicates the rotation/scale formula from `compute_code` in quads.rs.
/// Because the WCS transformation may involve a reflection (parity flip)
/// between pixel coordinates and the tangent-plane coordinates used by
/// `compute_code`, we compute codes for both orientations and return both.
fn compute_field_codes(
    positions: &[(f64, f64); DIMQUADS],
) -> [([f64; DIMCODES], [usize; DIMQUADS], bool); 2] {
    let mut results = [([0.0f64; DIMCODES], [0usize; DIMQUADS], false); 2];

    for (variant, swap) in [(false), (true)].iter().enumerate() {
        let coords: [(f64, f64); DIMQUADS] = if *swap {
            std::array::from_fn(|i| (positions[i].1, positions[i].0))
        } else {
            *positions
        };

        let (a_x, a_y) = coords[0];
        let (b_x, b_y) = coords[1];

        let ab_x = b_x - a_x;
        let ab_y = b_y - a_y;
        let scale = ab_x * ab_x + ab_y * ab_y;
        let invscale = 1.0 / scale;
        let costheta = (ab_y + ab_x) * invscale;
        let sintheta = (ab_y - ab_x) * invscale;

        let mut code = [0.0f64; DIMCODES];
        for i in 2..DIMQUADS {
            let (d_x, d_y) = coords[i];
            let ad_x = d_x - a_x;
            let ad_y = d_y - a_y;
            let x = ad_x * costheta + ad_y * sintheta;
            let y = -ad_x * sintheta + ad_y * costheta;
            code[2 * (i - 2)] = x;
            code[2 * (i - 2) + 1] = y;
        }

        let star_ids: [usize; DIMQUADS] = [0, 1, 2, 3];
        results[variant] = crate::quads::enforce_invariants(code, star_ids);
    }

    results
}

/// Try a single field quad (a, b, c, d) against all indexes.
/// Returns Some(Solution) if a verified match is found.
#[allow(clippy::too_many_arguments)]
fn try_quad(
    sorted: &[(usize, &DetectedSource)],
    a: usize,
    b: usize,
    c: usize,
    d: usize,
    indexes: &[&Index],
    sources: &[DetectedSource],
    image_size: (f64, f64),
    config: &SolverConfig,
    stats: &mut SolveStats,
) -> Option<Solution> {
    let (a_orig, sa) = sorted[a];
    let (b_orig, sb) = sorted[b];
    let (c_orig, sc) = sorted[c];
    let (d_orig, sd) = sorted[d];

    let positions: [(f64, f64); DIMQUADS] =
        [(sa.x, sa.y), (sb.x, sb.y), (sc.x, sc.y), (sd.x, sd.y)];
    let codes = compute_field_codes(&positions);
    let orig_indices = [a_orig, b_orig, c_orig, d_orig];

    for (field_code, reordered, _) in &codes {
        let reordered_orig: [usize; DIMQUADS] = std::array::from_fn(|i| orig_indices[reordered[i]]);
        let reordered_positions: [(f64, f64); DIMQUADS] =
            std::array::from_fn(|i| positions[reordered[i]]);

        for index in indexes {
            let matches = index
                .code_tree
                .range_search(field_code, config.code_tolerance);

            for code_match in &matches {
                let quad = &index.quads[code_match.index];

                let star_xyz: [[f64; 3]; DIMQUADS] = std::array::from_fn(|i| {
                    let s = &index.stars[quad.star_ids[i]];
                    radec_to_xyz(s.ra, s.dec)
                });

                let field_xy: [(f64, f64); DIMQUADS] = reordered_positions;

                let fit_result = fit_tan_wcs(star_xyz.as_slice(), field_xy.as_slice(), image_size);

                let wcs = match fit_result {
                    Ok(wcs) => wcs,
                    Err(FitError::TooFewCorrespondences)
                    | Err(FitError::SingularMatrix)
                    | Err(FitError::ProjectionFailed) => continue,
                };

                if let Some((lo, hi)) = config.scale_range {
                    let scale_arcsec = wcs.pixel_scale() * 3600.0;
                    if scale_arcsec < lo || scale_arcsec > hi {
                        continue;
                    }
                }

                let quad_residual_limit = 10.0;
                let max_residual_sq = quad_residual_limit * quad_residual_limit;
                let quad_ok = (0..DIMQUADS).all(|i| {
                    if let Some((px, py)) = wcs.xyz_to_pixel(star_xyz[i]) {
                        let dx = px - field_xy[i].0;
                        let dy = py - field_xy[i].1;
                        dx * dx + dy * dy < max_residual_sq
                    } else {
                        false
                    }
                });
                if !quad_ok {
                    continue;
                }

                let verify_result = verify_solution(&wcs, sources, index, &config.verify);
                stats.n_verified += 1;

                if verify_result.is_accepted(&config.verify) {
                    stats.accepted_log_odds = Some(verify_result.log_odds);
                    return Some(Solution {
                        wcs,
                        verify_result,
                        quad_match: QuadMatch {
                            field_indices: reordered_orig,
                            index_indices: quad.star_ids,
                        },
                    });
                } else {
                    let lo = verify_result.log_odds;
                    let nm = verify_result.n_matched;
                    match stats.best_rejected {
                        None => stats.best_rejected = Some((lo, nm)),
                        Some((best_lo, _)) if lo > best_lo => {
                            stats.best_rejected = Some((lo, nm));
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    None
}

/// Attempt to blindly solve a field of detected sources against one or more indexes.
///
/// This is the main entry point for plate solving. It:
/// 1. Sorts field sources by brightness
/// 2. Builds field quads from the brightest sources
/// 3. Matches field quad codes against each index's code tree
/// 4. For each match, fits a TAN WCS and verifies it
/// 5. Returns the first solution that passes verification
pub fn solve(
    sources: &[DetectedSource],
    indexes: &[&Index],
    image_size: (f64, f64),
    config: &SolverConfig,
) -> (Option<Solution>, SolveStats) {
    let mut stats = SolveStats {
        n_verified: 0,
        best_rejected: None,
        accepted_log_odds: None,
        timed_out: false,
    };

    if sources.len() < DIMQUADS || indexes.is_empty() {
        return (None, stats);
    }

    let deadline = config.timeout.map(|d| Instant::now() + d);

    // 1. Sort sources by flux (descending = brightest first).
    let mut sorted: Vec<(usize, &DetectedSource)> = sources.iter().enumerate().collect();
    sorted.sort_by(|a, b| {
        b.1.flux
            .partial_cmp(&a.1.flux)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    sorted.truncate(config.max_field_stars);

    // Precompute pixel scale bounds in radians if provided.
    let scale_rad = config.scale_range.map(|(lo, hi)| {
        let lo_rad = lo * std::f64::consts::PI / (180.0 * 3600.0);
        let hi_rad = hi * std::f64::consts::PI / (180.0 * 3600.0);
        (lo_rad, hi_rad)
    });

    // Helper: check if backbone distance passes scale filtering.
    let ab_scale_ok = |dist: f64| -> bool {
        if let Some((pix_lo_rad, pix_hi_rad)) = scale_rad {
            for index in indexes {
                let ang_lo = dist * pix_lo_rad;
                let ang_hi = dist * pix_hi_rad;
                if ang_lo <= index.scale_upper && ang_hi >= index.scale_lower {
                    return true;
                }
            }
            false
        } else {
            true
        }
    };

    // Helper: check if star i is inside the bounding circle for backbone (a, b).
    let in_bbox = |sorted: &[(usize, &DetectedSource)],
                   i: usize,
                   mid_x: f64,
                   mid_y: f64,
                   dist_sq: f64|
     -> bool {
        let (_, si) = sorted[i];
        let cdx = si.x - mid_x;
        let cdy = si.y - mid_y;
        cdx * cdx + cdy * cdy <= dist_sq
    };

    // 2. Incremental "new star" loop following astrometry.net's two-phase approach.
    //    Star n must participate in every quad tried at this iteration.
    //    This ensures each unique quad is tried exactly once.
    for n in 2..sorted.len() {
        if let Some(dl) = deadline
            && Instant::now() > dl
        {
            stats.timed_out = true;
            return (None, stats);
        }

        // Phase 1: n is on the backbone (B = n).
        // Try all A < n with B = n, then find C, D from [0..n).
        for a in 0..n {
            let b = n;
            let (_, sa) = sorted[a];
            let (_, sb) = sorted[b];

            let dx = sb.x - sa.x;
            let dy = sb.y - sa.y;
            let dist_sq = dx * dx + dy * dy;
            let dist = dist_sq.sqrt();
            if dist < 1e-10 {
                continue;
            }
            if !ab_scale_ok(dist) {
                continue;
            }

            let mid_x = (sa.x + sb.x) / 2.0;
            let mid_y = (sa.y + sb.y) / 2.0;

            // C and D candidates from [0..n), excluding a.
            let candidates: Vec<usize> = (0..n)
                .filter(|&i| i != a && in_bbox(&sorted, i, mid_x, mid_y, dist_sq))
                .collect();
            if candidates.len() < 2 {
                continue;
            }

            for ci in 0..candidates.len() {
                for di in (ci + 1)..candidates.len() {
                    let c = candidates[ci];
                    let d = candidates[di];
                    if let Some(sol) = try_quad(
                        &sorted, a, b, c, d, indexes, sources, image_size, config, &mut stats,
                    ) {
                        return (Some(sol), stats);
                    }
                }
            }
        }

        // Phase 2: n is off-diagonal (C = n).
        // Try all backbone pairs A < B < n, with C = n and D from [0..n).
        for a in 0..n {
            if let Some(dl) = deadline
                && Instant::now() > dl
            {
                stats.timed_out = true;
                return (None, stats);
            }
            for b in (a + 1)..n {
                let (_, sa) = sorted[a];
                let (_, sb) = sorted[b];

                let dx = sb.x - sa.x;
                let dy = sb.y - sa.y;
                let dist_sq = dx * dx + dy * dy;
                let dist = dist_sq.sqrt();
                if dist < 1e-10 {
                    continue;
                }
                if !ab_scale_ok(dist) {
                    continue;
                }

                let mid_x = (sa.x + sb.x) / 2.0;
                let mid_y = (sa.y + sb.y) / 2.0;

                // Check if n is in the bounding circle for this backbone.
                if !in_bbox(&sorted, n, mid_x, mid_y, dist_sq) {
                    continue;
                }

                // C = n, D from [0..n) excluding a, b.
                let c = n;
                for d in 0..n {
                    if d == a || d == b {
                        continue;
                    }
                    if !in_bbox(&sorted, d, mid_x, mid_y, dist_sq) {
                        continue;
                    }
                    if let Some(sol) = try_quad(
                        &sorted, a, b, c, d, indexes, sources, image_size, config, &mut stats,
                    ) {
                        return (Some(sol), stats);
                    }
                }
            }
        }
    }

    (None, stats)
}

/// Solve from an image directly.
///
/// Extracts sources, then runs the solver.
pub fn solve_image(
    image: &ndarray::Array2<f32>,
    indexes: &[&Index],
    extraction_config: &crate::extraction::ExtractionConfig,
    solver_config: &SolverConfig,
) -> (Option<Solution>, SolveStats) {
    let sources = crate::extraction::extract_sources(image, extraction_config);
    let (h, w) = image.dim();
    solve(&sources, indexes, (w as f64, h as f64), solver_config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::tan::TanWcs;
    use crate::index::builder::{IndexBuilderConfig, build_index};
    use std::f64::consts::PI;

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

    fn make_synthetic_scenario() -> (Vec<DetectedSource>, Index, TanWcs) {
        let image_size = (512.0, 512.0);
        let pixel_scale_arcsec = 2.0;
        let wcs = make_test_wcs([1.0, 0.5], pixel_scale_arcsec, 0.0, image_size);

        // Use a deterministic pseudo-random generator to place stars
        // in non-regular positions (regular grids cause pathological
        // code collisions).
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
            let px = 30.0 + rng() * 452.0;
            let py = 30.0 + rng() * 452.0;
            let (ra, dec) = wcs.pixel_to_radec(px, py);
            catalog.push((i as u64, ra, dec, i as f64));
            sources.push(DetectedSource {
                x: px,
                y: py,
                flux: 1000.0 - i as f64 * 10.0,
            });
        }

        // Compute the angular scale of the field to set index scale bounds.
        let scale_rad = (pixel_scale_arcsec / 3600.0).to_radians();
        let field_diag = (image_size.0 * image_size.0 + image_size.1 * image_size.1).sqrt();
        let max_angle = field_diag * scale_rad;

        let index_config = IndexBuilderConfig {
            scale_lower: scale_rad * 10.0,
            scale_upper: max_angle,
            max_stars: 25,
            max_quads: 5000,
        };

        let index = build_index(&catalog, &index_config);

        (sources, index, wcs)
    }

    #[test]
    fn synthetic_solve() {
        let (sources, index, known_wcs) = make_synthetic_scenario();

        let config = SolverConfig {
            scale_range: None,
            max_field_stars: 25,
            code_tolerance: 0.002,
            verify: VerifyConfig {
                match_radius_pix: 3.0,
                log_odds_accept: 10.0,
                min_matches: 3,
                ..VerifyConfig::default()
            },
            ..SolverConfig::default()
        };

        let (solution, stats) = solve(&sources, &[&index], (512.0, 512.0), &config);
        assert!(solution.is_some(), "solver should find a solution");
        assert!(stats.accepted_log_odds.is_some());

        let solution = solution.unwrap();

        // Verify the solved WCS closely matches the original.
        let (solved_ra, solved_dec) = solution.wcs.field_center();
        let (known_ra, known_dec) = known_wcs.field_center();

        let arcsec = PI / (180.0 * 3600.0);
        let ra_diff = (solved_ra - known_ra).abs();
        let dec_diff = (solved_dec - known_dec).abs();

        assert!(
            ra_diff < 30.0 * arcsec,
            "RA difference too large: {} arcsec",
            ra_diff / arcsec
        );
        assert!(
            dec_diff < 30.0 * arcsec,
            "Dec difference too large: {} arcsec",
            dec_diff / arcsec
        );

        // Verify pixel scale is within 5%.
        let solved_scale = solution.wcs.pixel_scale();
        let known_scale = known_wcs.pixel_scale();
        let scale_ratio = solved_scale / known_scale;
        assert!(
            (0.95..=1.05).contains(&scale_ratio),
            "pixel scale ratio {} outside [0.95, 1.05]",
            scale_ratio
        );

        // Verify the solution was accepted.
        assert!(solution.verify_result.is_accepted(&config.verify));
    }

    #[test]
    fn no_solution_random_sources() {
        // Build a small index with few quads to avoid saturating the code space.
        let mut catalog = Vec::new();
        let mut state: u64 = 999999;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };
        for i in 0..10 {
            let ra = 3.0 + rng() * 0.01;
            let dec = -0.5 + rng() * 0.01;
            catalog.push((i as u64, ra, dec, i as f64));
        }
        let index_config = IndexBuilderConfig {
            scale_lower: 0.001,
            scale_upper: 0.02,
            max_stars: 10,
            max_quads: 100,
        };
        let index = build_index(&catalog, &index_config);

        // Random pixel positions that don't correspond to index stars.
        let sources: Vec<DetectedSource> = (0..15)
            .map(|_| DetectedSource {
                x: rng() * 1024.0,
                y: rng() * 1024.0,
                flux: rng() * 1000.0,
            })
            .collect();

        let config = SolverConfig {
            max_field_stars: 15,
            code_tolerance: 0.001,
            verify: VerifyConfig {
                match_radius_pix: 3.0,
                log_odds_accept: 20.0,
                min_matches: 3,
                ..VerifyConfig::default()
            },
            ..SolverConfig::default()
        };

        let (solution, stats) = solve(&sources, &[&index], (1024.0, 1024.0), &config);
        assert!(solution.is_none(), "random sources should not solve");
        assert!(stats.accepted_log_odds.is_none());
    }

    #[test]
    fn scale_filtering_rejects() {
        let (sources, index, _) = make_synthetic_scenario();

        // Set a scale range that is way off from the actual pixel scale (2 arcsec/pixel).
        let config = SolverConfig {
            scale_range: Some((100.0, 200.0)),
            max_field_stars: 30,
            code_tolerance: 0.01,
            verify: VerifyConfig {
                match_radius_pix: 10.0,
                log_odds_accept: 10.0,
                min_matches: 3,
                ..VerifyConfig::default()
            },
            ..SolverConfig::default()
        };

        let (solution, _stats) = solve(&sources, &[&index], (512.0, 512.0), &config);
        assert!(
            solution.is_none(),
            "wrong scale range should prevent solving"
        );
    }

    #[test]
    fn multiple_indexes() {
        let (sources, correct_index, known_wcs) = make_synthetic_scenario();

        // Build a decoy index from stars in a totally different part of the sky.
        let mut decoy_catalog = Vec::new();
        for i in 0..20 {
            let ra = 4.0 + (i % 5) as f64 * 0.005;
            let dec = -1.0 + (i / 5) as f64 * 0.005;
            decoy_catalog.push((100 + i as u64, ra, dec, i as f64));
        }
        let decoy_config = IndexBuilderConfig {
            scale_lower: 0.001,
            scale_upper: 0.05,
            max_stars: 20,
            max_quads: 1000,
        };
        let decoy_index = build_index(&decoy_catalog, &decoy_config);

        let config = SolverConfig {
            scale_range: None,
            max_field_stars: 30,
            code_tolerance: 0.01,
            verify: VerifyConfig {
                match_radius_pix: 10.0,
                log_odds_accept: 10.0,
                min_matches: 3,
                ..VerifyConfig::default()
            },
            ..SolverConfig::default()
        };

        let (solution, _stats) = solve(
            &sources,
            &[&decoy_index, &correct_index],
            (512.0, 512.0),
            &config,
        );
        assert!(
            solution.is_some(),
            "should solve with correct index among multiple"
        );

        let solution = solution.unwrap();
        let (solved_ra, solved_dec) = solution.wcs.field_center();
        let (known_ra, known_dec) = known_wcs.field_center();
        let arcsec = PI / (180.0 * 3600.0);
        assert!((solved_ra - known_ra).abs() < 30.0 * arcsec, "RA mismatch");
        assert!(
            (solved_dec - known_dec).abs() < 30.0 * arcsec,
            "Dec mismatch"
        );
    }

    #[test]
    fn solver_config_defaults() {
        let config = SolverConfig::default();
        assert!(config.scale_range.is_none());
        assert_eq!(config.max_field_stars, 50);
        assert!((config.code_tolerance - 0.01).abs() < 1e-15);
        assert!((config.verify.match_radius_pix - 5.0).abs() < 1e-15);
        assert!((config.verify.distractor_fraction - 0.25).abs() < 1e-15);
        assert!((config.verify.log_odds_accept - 20.0).abs() < 1e-15);
        assert!((config.verify.log_odds_bail - (-20.0)).abs() < 1e-15);
        assert_eq!(config.verify.min_matches, 10);
    }

    #[test]
    fn too_few_sources() {
        let (_, index, _) = make_synthetic_scenario();

        let sources = vec![
            DetectedSource {
                x: 100.0,
                y: 100.0,
                flux: 100.0,
            },
            DetectedSource {
                x: 200.0,
                y: 200.0,
                flux: 90.0,
            },
            DetectedSource {
                x: 300.0,
                y: 300.0,
                flux: 80.0,
            },
        ];

        let config = SolverConfig::default();
        let (solution, _stats) = solve(&sources, &[&index], (512.0, 512.0), &config);
        assert!(solution.is_none(), "fewer than 4 sources should not solve");
    }

    #[test]
    fn no_indexes() {
        let (sources, _, _) = make_synthetic_scenario();
        let config = SolverConfig::default();
        let (solution, _stats) = solve(&sources, &[], (512.0, 512.0), &config);
        assert!(solution.is_none(), "no indexes should return None");
    }

    #[test]
    fn field_code_invariants() {
        let positions: [(f64, f64); DIMQUADS] = [
            (100.0, 100.0),
            (300.0, 100.0),
            (200.0, 150.0),
            (150.0, 80.0),
        ];

        let codes = compute_field_codes(&positions);

        for (code, _, _) in &codes {
            // cx <= dx
            assert!(
                code[0] <= code[2] + 1e-15,
                "cx ({}) > dx ({})",
                code[0],
                code[2]
            );

            // mean_x <= 0.5
            let mean_x = (code[0] + code[2]) / 2.0;
            assert!(mean_x <= 0.5 + 1e-15, "mean_x ({}) > 0.5", mean_x);
        }
    }

    #[test]
    fn field_code_scale_invariant() {
        let positions1: [(f64, f64); DIMQUADS] = [
            (100.0, 100.0),
            (200.0, 100.0),
            (150.0, 150.0),
            (120.0, 80.0),
        ];

        // Same geometry scaled by 2x.
        let positions2: [(f64, f64); DIMQUADS] = [
            (200.0, 200.0),
            (400.0, 200.0),
            (300.0, 300.0),
            (240.0, 160.0),
        ];

        let codes1 = compute_field_codes(&positions1);
        let codes2 = compute_field_codes(&positions2);

        // Both parity variants should be scale-invariant independently.
        for v in 0..2 {
            for i in 0..DIMCODES {
                assert!(
                    (codes1[v].0[i] - codes2[v].0[i]).abs() < 1e-10,
                    "variant {v} code[{}]: {} vs {} (diff = {})",
                    i,
                    codes1[v].0[i],
                    codes2[v].0[i],
                    (codes1[v].0[i] - codes2[v].0[i]).abs()
                );
            }
        }
    }

    #[test]
    fn field_code_matches_index_code() {
        let image_size = (512.0, 512.0);
        let pixel_scale_arcsec = 2.0;
        let wcs = make_test_wcs([1.0, 0.5], pixel_scale_arcsec, 0.0, image_size);

        // Pick 4 pixel positions and compute their sky positions.
        let pixel_positions: [(f64, f64); DIMQUADS] = [
            (100.0, 100.0),
            (300.0, 100.0),
            (200.0, 200.0),
            (150.0, 150.0),
        ];

        let sky_xyz: [[f64; 3]; DIMQUADS] =
            std::array::from_fn(|i| wcs.pixel_to_xyz(pixel_positions[i].0, pixel_positions[i].1));

        // Compute code from sky positions (as the index builder would).
        let (sky_code, _sky_ids, _) = crate::quads::compute_canonical_code(&sky_xyz, [0, 1, 2, 3]);

        // Compute codes from pixel positions (both parity variants).
        let field_codes = compute_field_codes(&pixel_positions);

        // At least one of the two parity variants should match the sky code.
        let match_found = field_codes
            .iter()
            .any(|(fc, _, _)| (0..DIMCODES).all(|i| (sky_code[i] - fc[i]).abs() < 0.01));

        assert!(
            match_found,
            "neither parity variant matched sky code {:?}, got {:?} and {:?}",
            sky_code, field_codes[0].0, field_codes[1].0
        );
    }

    #[test]
    fn synthetic_solve_with_rotation() {
        let image_size = (512.0, 512.0);
        let pixel_scale_arcsec = 2.0;
        let rotation = PI / 6.0;
        let wcs = make_test_wcs([1.0, 0.5], pixel_scale_arcsec, rotation, image_size);

        // Use random positions to avoid grid symmetry artifacts.
        let mut state: u64 = 271828182;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        let mut catalog = Vec::new();
        let mut sources = Vec::new();

        for i in 0..30 {
            let px = 30.0 + rng() * 452.0;
            let py = 30.0 + rng() * 452.0;
            let (ra, dec) = wcs.pixel_to_radec(px, py);
            catalog.push((i as u64, ra, dec, i as f64));
            sources.push(DetectedSource {
                x: px,
                y: py,
                flux: 1000.0 - i as f64 * 10.0,
            });
        }

        let scale_rad = (pixel_scale_arcsec / 3600.0).to_radians();
        let field_diag = (image_size.0 * image_size.0 + image_size.1 * image_size.1).sqrt();
        let max_angle = field_diag * scale_rad;

        let index_config = IndexBuilderConfig {
            scale_lower: scale_rad * 10.0,
            scale_upper: max_angle,
            max_stars: 30,
            max_quads: 10000,
        };

        let index = build_index(&catalog, &index_config);

        let config = SolverConfig {
            scale_range: None,
            max_field_stars: 30,
            code_tolerance: 0.002,
            verify: VerifyConfig {
                match_radius_pix: 3.0,
                log_odds_accept: 10.0,
                min_matches: 3,
                ..VerifyConfig::default()
            },
            ..SolverConfig::default()
        };

        let (solution, _stats) = solve(&sources, &[&index], image_size, &config);
        assert!(
            solution.is_some(),
            "solver should find a solution with rotated WCS"
        );

        let solution = solution.unwrap();
        let solved_scale = solution.wcs.pixel_scale();
        let known_scale = wcs.pixel_scale();
        let scale_ratio = solved_scale / known_scale;
        assert!(
            (0.95..=1.05).contains(&scale_ratio),
            "pixel scale ratio {} outside [0.95, 1.05]",
            scale_ratio
        );
    }
}
