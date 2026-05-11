//! Bayesian verification of candidate WCS solutions.
//!
//! Given a candidate TAN WCS, projects reference stars from the index into
//! pixel space and checks how many observed field sources land near a
//! projected reference star. Uses log-odds scoring: each match increases
//! confidence, each miss decreases it.

use std::f64::consts::PI;

use crate::extraction::DetectedSource;
use crate::geom::sphere::radec_to_xyz;
use crate::geom::tan::TanWcs;
use crate::index::Index;
use crate::kdtree::KdTree;

/// Configuration for verification.
pub struct VerifyConfig {
    /// Positional tolerance in pixels -- how close a field source must be
    /// to a projected reference star to count as a match.
    pub match_radius_pix: f64,
    /// Expected fraction of field sources that are noise/artifacts (0.0 to 1.0).
    pub distractor_fraction: f64,
    /// Log-odds threshold to accept a solution.
    pub log_odds_accept: f64,
    /// Log-odds threshold to bail early (definitely wrong).
    pub log_odds_bail: f64,
    /// Minimum number of matched stars to accept a solution.
    pub min_matches: usize,
}

impl Default for VerifyConfig {
    fn default() -> Self {
        Self {
            match_radius_pix: 5.0,
            distractor_fraction: 0.25,
            log_odds_accept: 20.0,
            log_odds_bail: -20.0,
            min_matches: 10,
        }
    }
}

/// Result of verification.
#[derive(Debug, Clone)]
pub struct VerifyResult {
    /// Log-odds score: positive = likely real, negative = likely spurious.
    pub log_odds: f64,
    /// Number of field sources matched to reference stars.
    pub n_matched: usize,
    /// Number of field sources with no match (distractors).
    pub n_distractor: usize,
    /// Number of field sources near multiple reference stars (conflicts).
    pub n_conflict: usize,
    /// Matched pairs: (field_source_index, reference_star_index).
    pub matched_pairs: Vec<(usize, usize)>,
}

impl VerifyResult {
    /// Check whether a verify result represents an accepted solution.
    pub fn is_accepted(&self, config: &VerifyConfig) -> bool {
        self.log_odds >= config.log_odds_accept && self.n_matched >= config.min_matches
    }
}

/// Verify a candidate WCS solution by checking how well it explains
/// the observed field sources.
///
/// Projects all reference stars in the field of view onto pixel coordinates
/// using the candidate WCS, then checks how many field sources have a
/// nearby projected reference star.
///
/// Uses a Bayesian log-odds scoring: each matched source increases the
/// score (evidence for correct solution), each unmatched source decreases
/// it (evidence for spurious solution).
pub fn verify_solution(
    wcs: &TanWcs,
    field_sources: &[DetectedSource],
    index: &Index,
    config: &VerifyConfig,
    obs_epoch: Option<&starfield::time::Time>,
) -> VerifyResult {
    if field_sources.is_empty() {
        return VerifyResult {
            log_odds: 0.0,
            n_matched: 0,
            n_distractor: 0,
            n_conflict: 0,
            matched_pairs: Vec::new(),
        };
    }

    // 1. Find reference stars in field of view.
    let (center_ra, center_dec) = wcs.field_center();
    let center_xyz = radec_to_xyz(center_ra, center_dec);
    let field_radius = wcs.field_radius();
    let radius_sq = 2.0 * (1.0 - field_radius.cos());

    let nearby_results = index.star_tree.range_search(&center_xyz, radius_sq);

    // 2. Project reference stars to pixels, keeping only those within bounds.
    let margin = config.match_radius_pix;
    let mut proj_points: Vec<[f64; 2]> = Vec::new();
    let mut proj_indices: Vec<usize> = Vec::new();

    for result in &nearby_results {
        let star = &index.stars[result.index];
        let pos = match (obs_epoch, &star.proper_motion) {
            (Some(obs), Some(pm)) => {
                crate::geom::sphere::propagate_pm(star.position, *pm, star.ref_epoch.as_time(), obs)
            }
            _ => star.position,
        };
        if let Some((px, py)) = wcs.radec_to_pixel(pos.ra, pos.dec)
            && px >= -margin
            && px <= wcs.image_size[0] + margin
            && py >= -margin
            && py <= wcs.image_size[1] + margin
        {
            proj_points.push([px, py]);
            proj_indices.push(result.index);
        }
    }

    // 3. Build a 2D KD-tree over projected reference star pixel positions.
    let n_ref = proj_points.len() as f64;
    if n_ref < 1.0 {
        return VerifyResult {
            log_odds: f64::NEG_INFINITY,
            n_matched: 0,
            n_distractor: field_sources.len(),
            n_conflict: 0,
            matched_pairs: Vec::new(),
        };
    }
    let ref_tree = KdTree::<2>::build(proj_points, proj_indices);

    // 4. Score each field source following astrometry.net's verify.c.
    //
    // Every source contributes to the log-odds:
    //   match:    contribution = max(log_fg, log_distractor) - log_bg
    //   no match: contribution = log_distractor - log_bg  (negative)
    //
    // log_fg = log((1-d)/(2πσ²·NR)) - dist²/(2σ²)     [Gaussian]
    // log_distractor = log(d + (1-d)·mu/NR) + log_bg    [dynamic]
    // log_bg = log(1/image_area)                         [uniform]
    let image_area = wcs.image_size[0] * wcs.image_size[1];
    let sigma_sq = config.match_radius_pix * config.match_radius_pix / 4.0;
    let match_radius_sq = config.match_radius_pix * config.match_radius_pix;
    let distractors = config.distractor_fraction;
    let log_gauss_peak = ((1.0 - distractors) / (2.0 * PI * sigma_sq * n_ref)).ln();
    let log_bg = (1.0_f64 / image_area).ln();

    let mut log_odds = 0.0;
    let mut n_matched = 0;
    let mut n_distractor = 0;
    let mut n_conflict = 0;
    let mut matched_pairs = Vec::new();

    for (field_idx, source) in field_sources.iter().enumerate() {
        let query = [source.x, source.y];
        let matches = ref_tree.range_search(&query, match_radius_sq);

        // Distractor log-density (updated with matches found so far).
        let log_distractor =
            (distractors + (1.0 - distractors) * n_matched as f64 / n_ref).ln() + log_bg;

        if matches.is_empty() {
            // Non-match: evidence against correct WCS.
            n_distractor += 1;
            log_odds += log_distractor - log_bg;
        } else {
            if matches.len() > 1 {
                n_conflict += 1;
            }

            let best = matches
                .iter()
                .min_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap())
                .unwrap();

            // Foreground Gaussian log-density at this distance.
            let log_fg = log_gauss_peak - best.dist_sq / (2.0 * sigma_sq);

            if log_fg >= log_distractor {
                // Treat as real match.
                n_matched += 1;
                matched_pairs.push((field_idx, best.index));
                log_odds += log_fg - log_bg;
            } else {
                // Gaussian weaker than distractor — treat as distractor.
                n_distractor += 1;
                log_odds += log_distractor - log_bg;
            }
        }

        // Early termination.
        if log_odds <= config.log_odds_bail {
            // Remaining unexamined sources are counted as distractors for bookkeeping.
            n_distractor += field_sources.len() - field_idx - 1;
            break;
        }
        if log_odds >= config.log_odds_accept && n_matched >= config.min_matches {
            // Count remaining as distractors for bookkeeping (conservative).
            n_distractor += field_sources.len() - field_idx - 1;
            break;
        }
    }

    VerifyResult {
        log_odds,
        n_matched,
        n_distractor,
        n_conflict,
        matched_pairs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::DetectedSource;
    use crate::geom::sphere::radec_to_xyz;
    use crate::geom::tan::TanWcs;
    use crate::index::{Index, IndexStar};
    use crate::kdtree::KdTree;
    use crate::quads::{DIMCODES, Quad};
    use starfield::Equatorial;
    use std::f64::consts::PI;

    fn make_test_wcs() -> TanWcs {
        let arcsec_rad = (1.0_f64 / 3600.0).to_radians();
        TanWcs {
            crval: [PI, 0.25],
            crpix: [512.0, 512.0],
            cd: [[arcsec_rad, 0.0], [0.0, arcsec_rad]],
            image_size: [1024.0, 1024.0],
        }
    }

    fn make_test_index(stars: Vec<IndexStar>) -> Index {
        let points: Vec<[f64; 3]> = stars
            .iter()
            .map(|s| radec_to_xyz(s.position.ra, s.position.dec))
            .collect();
        let indices: Vec<usize> = (0..stars.len()).collect();
        let star_tree = KdTree::<3>::build(points, indices);

        let code_tree = KdTree::<{ DIMCODES }>::build(vec![], vec![]);
        let quads: Vec<Quad> = vec![];

        Index {
            star_tree,
            stars,
            code_tree,
            quads,
            scale_lower: 0.0,
            scale_upper: 1.0,
            metadata: None,
        }
    }

    fn stars_from_wcs(wcs: &TanWcs, pixel_positions: &[(f64, f64)]) -> Vec<IndexStar> {
        pixel_positions
            .iter()
            .enumerate()
            .map(|(i, &(px, py))| {
                let (ra, dec) = wcs.pixel_to_radec(px, py);
                IndexStar::without_pm(i as u64, Equatorial::new(ra, dec), 10.0)
            })
            .collect()
    }

    #[test]
    fn perfect_match() {
        let wcs = make_test_wcs();

        let pixel_positions: Vec<(f64, f64)> = (0..20)
            .map(|i| {
                let x = 100.0 + (i % 5) as f64 * 200.0;
                let y = 100.0 + (i / 5) as f64 * 200.0;
                (x, y)
            })
            .collect();

        let catalog_stars = stars_from_wcs(&wcs, &pixel_positions);
        let index = make_test_index(catalog_stars);

        let field_sources: Vec<DetectedSource> = pixel_positions
            .iter()
            .map(|&(x, y)| DetectedSource { x, y, flux: 100.0 })
            .collect();

        let config = VerifyConfig::default();
        let result = verify_solution(&wcs, &field_sources, &index, &config, None);

        assert!(
            result.log_odds > 0.0,
            "Expected positive log-odds, got {}",
            result.log_odds
        );
        assert!(
            result.n_matched >= 2,
            "Expected at least 2 matches, got {}",
            result.n_matched
        );
        assert_eq!(result.n_conflict, 0);
        assert!(result.is_accepted(&config));
    }

    #[test]
    fn perfect_match_no_early_exit() {
        let wcs = make_test_wcs();

        let pixel_positions: Vec<(f64, f64)> = (0..20)
            .map(|i| {
                let x = 100.0 + (i % 5) as f64 * 200.0;
                let y = 100.0 + (i / 5) as f64 * 200.0;
                (x, y)
            })
            .collect();

        let catalog_stars = stars_from_wcs(&wcs, &pixel_positions);
        let index = make_test_index(catalog_stars);

        let field_sources: Vec<DetectedSource> = pixel_positions
            .iter()
            .map(|&(x, y)| DetectedSource { x, y, flux: 100.0 })
            .collect();

        let config = VerifyConfig {
            log_odds_accept: f64::INFINITY,
            log_odds_bail: f64::NEG_INFINITY,
            ..VerifyConfig::default()
        };
        let result = verify_solution(&wcs, &field_sources, &index, &config, None);

        assert_eq!(result.n_matched, 20);
        assert_eq!(result.n_distractor, 0);
        assert!(result.log_odds > 0.0);
    }

    #[test]
    fn no_match_wrong_sky() {
        let wcs = make_test_wcs();

        let pixel_positions: Vec<(f64, f64)> = (0..10)
            .map(|i| {
                let x = 200.0 + (i % 5) as f64 * 100.0;
                let y = 200.0 + (i / 5) as f64 * 100.0;
                (x, y)
            })
            .collect();

        let catalog_stars = stars_from_wcs(&wcs, &pixel_positions);
        let index = make_test_index(catalog_stars);

        let wrong_wcs = TanWcs {
            crval: [0.5, -0.5],
            ..make_test_wcs()
        };

        let field_sources: Vec<DetectedSource> = pixel_positions
            .iter()
            .map(|&(x, y)| DetectedSource { x, y, flux: 100.0 })
            .collect();

        let config = VerifyConfig::default();
        let result = verify_solution(&wrong_wcs, &field_sources, &index, &config, None);

        assert!(
            result.log_odds <= 0.0,
            "Expected non-positive log-odds, got {}",
            result.log_odds
        );
        assert_eq!(result.n_matched, 0);
        assert!(!result.is_accepted(&config));
    }

    #[test]
    fn partial_match() {
        let wcs = make_test_wcs();

        let pixel_positions: Vec<(f64, f64)> = (0..10)
            .map(|i| {
                let x = 200.0 + (i % 5) as f64 * 100.0;
                let y = 200.0 + (i / 5) as f64 * 300.0;
                (x, y)
            })
            .collect();

        let catalog_stars = stars_from_wcs(&wcs, &pixel_positions);
        let index = make_test_index(catalog_stars);

        let mut field_sources: Vec<DetectedSource> = pixel_positions
            .iter()
            .map(|&(x, y)| DetectedSource { x, y, flux: 100.0 })
            .collect();

        let mut state: u64 = 42;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        for _ in 0..10 {
            field_sources.push(DetectedSource {
                x: rng() * 1024.0,
                y: rng() * 1024.0,
                flux: 50.0,
            });
        }

        let config = VerifyConfig {
            min_matches: 3,
            ..VerifyConfig::default()
        };
        let result = verify_solution(&wcs, &field_sources, &index, &config, None);

        assert!(
            result.n_matched >= 3,
            "Expected at least 3 matches, got {}",
            result.n_matched
        );
        assert!(
            result.log_odds > 0.0,
            "Expected positive log-odds for partial match, got {}",
            result.log_odds
        );
        assert!(result.is_accepted(&config));
    }

    #[test]
    fn all_distractors() {
        let wcs = make_test_wcs();

        let pixel_positions: Vec<(f64, f64)> = (0..5)
            .map(|i| {
                let x = 200.0 + (i % 5) as f64 * 100.0;
                let y = 200.0 + (i / 5) as f64 * 100.0;
                (x, y)
            })
            .collect();

        let catalog_stars = stars_from_wcs(&wcs, &pixel_positions);
        let index = make_test_index(catalog_stars);

        let wrong_wcs = TanWcs {
            crval: [1.0, -1.0],
            ..make_test_wcs()
        };

        let mut state: u64 = 99;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        let field_sources: Vec<DetectedSource> = (0..20)
            .map(|_| DetectedSource {
                x: rng() * 1024.0,
                y: rng() * 1024.0,
                flux: 50.0,
            })
            .collect();

        let config = VerifyConfig::default();
        let result = verify_solution(&wrong_wcs, &field_sources, &index, &config, None);

        assert!(
            result.log_odds <= 0.0 || result.n_matched == 0,
            "Expected negative log-odds or no matches for all distractors"
        );
        assert!(!result.is_accepted(&config));
    }

    #[test]
    fn empty_field() {
        let wcs = make_test_wcs();

        let stars = vec![IndexStar::without_pm(0, Equatorial::new(PI, 0.25), 10.0)];
        let index = make_test_index(stars);

        let field_sources: Vec<DetectedSource> = vec![];

        let config = VerifyConfig::default();
        let result = verify_solution(&wcs, &field_sources, &index, &config, None);

        assert_eq!(result.log_odds, 0.0);
        assert_eq!(result.n_matched, 0);
        assert_eq!(result.n_distractor, 0);
        assert_eq!(result.n_conflict, 0);
        assert!(result.matched_pairs.is_empty());
    }

    #[test]
    fn is_accepted_thresholds() {
        let config = VerifyConfig {
            log_odds_accept: 10.0,
            ..VerifyConfig::default()
        };

        let accepted = VerifyResult {
            log_odds: 15.0,
            n_matched: 10,
            n_distractor: 2,
            n_conflict: 0,
            matched_pairs: vec![],
        };
        assert!(accepted.is_accepted(&config));

        let borderline = VerifyResult {
            log_odds: 10.0,
            n_matched: 10,
            n_distractor: 5,
            n_conflict: 0,
            matched_pairs: vec![],
        };
        assert!(borderline.is_accepted(&config));

        let rejected = VerifyResult {
            log_odds: 9.99,
            n_matched: 5,
            n_distractor: 5,
            n_conflict: 0,
            matched_pairs: vec![],
        };
        assert!(!rejected.is_accepted(&config));

        let negative = VerifyResult {
            log_odds: -5.0,
            n_matched: 0,
            n_distractor: 10,
            n_conflict: 0,
            matched_pairs: vec![],
        };
        assert!(!negative.is_accepted(&config));
    }

    #[test]
    fn early_bail() {
        let wcs = make_test_wcs();

        let pixel_positions: Vec<(f64, f64)> = (0..5)
            .map(|i| {
                let x = 200.0 + i as f64 * 100.0;
                let y = 200.0;
                (x, y)
            })
            .collect();

        let catalog_stars = stars_from_wcs(&wcs, &pixel_positions);
        let index = make_test_index(catalog_stars);

        let wrong_wcs = TanWcs {
            crval: [0.0, -1.2],
            ..make_test_wcs()
        };

        let num_sources = 500;
        let mut state: u64 = 77;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        let field_sources: Vec<DetectedSource> = (0..num_sources)
            .map(|_| DetectedSource {
                x: rng() * 1024.0,
                y: rng() * 1024.0,
                flux: 50.0,
            })
            .collect();

        let config = VerifyConfig {
            log_odds_bail: -10.0,
            ..VerifyConfig::default()
        };

        let result = verify_solution(&wrong_wcs, &field_sources, &index, &config, None);

        let total_examined = result.n_matched + result.n_distractor;
        assert!(
            total_examined <= num_sources,
            "Should not examine more sources than available"
        );
        assert!(
            result.log_odds <= config.log_odds_bail || total_examined == num_sources,
            "Expected early bail or all sources examined"
        );
        assert!(!result.is_accepted(&config));
    }

    /// Regression for #125: a catalog star with non-trivial proper motion
    /// must match its detection only when `obs_epoch` is set and
    /// `propagate_pm` is applied.
    ///
    /// Scenario: detection sits at pixel (300, 400). The catalog star's
    /// position at the *observation* epoch projects to that pixel. We
    /// then back-extrapolate the catalog position to a Gaia 2016
    /// reference epoch by applying the *negative* of the proper motion;
    /// that's the (ra, dec) we store in the index. Without
    /// `obs_epoch`, the verifier projects the index's 2016 position and
    /// misses. With `obs_epoch = obs`, it forward-propagates back to
    /// the obs position and matches.
    #[test]
    fn obs_epoch_recovers_high_pm_star() {
        let wcs = make_test_wcs();

        // Detection at pixel (300, 400).
        let det_px = (300.0_f64, 400.0_f64);
        let (det_ra, det_dec) = wcs.pixel_to_radec(det_px.0, det_px.1);
        let field_sources = vec![DetectedSource {
            x: det_px.0,
            y: det_px.1,
            flux: 100.0,
        }];

        // PM in mas/yr. 50 mas/yr × 14 yr = 700 mas drift in RA. At the
        // synthetic 1″/px plate scale, that's 0.7 px — comfortably
        // inside the default 5 px match radius, so the test must fail
        // for the right reason (PM not applied), not because the index
        // is way off-image.
        //
        // Push it harder: 500 mas/yr × 14 yr = 7000 mas = 7″ = 7 px,
        // outside the default tolerance.
        let pm = crate::geom::ProperMotion {
            pmra: 500.0,
            pmdec: 200.0,
        };
        let ts = starfield::time::Timescale::default();
        let ref_epoch = ts.j(2016.0);
        let obs_epoch = ts.j(2002.0);
        let dt = obs_epoch.j() - ref_epoch.j(); // negative

        const MAS_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3600.0 * 1000.0);
        // Reverse the PM extrapolation: where the star was at ref_epoch
        // such that at obs_epoch (with the recorded PM) it ends up at
        // (det_ra, det_dec).
        let cos_dec = det_dec.cos();
        let dra = pm.pmra * dt * MAS_TO_RAD / cos_dec;
        let ddec = pm.pmdec * dt * MAS_TO_RAD;
        // Store position at ref_epoch = det position - PM*dt (rearranged
        // from propagate_pm).
        let star_with_pm = IndexStar {
            catalog_id: 1,
            position: Equatorial::new(det_ra - dra, det_dec - ddec),
            mag: 10.0,
            proper_motion: Some(pm),
            ref_epoch: crate::index::RefEpoch::new(ref_epoch),
        };
        let index = make_test_index(vec![star_with_pm]);

        let config = VerifyConfig::default();

        // Without obs_epoch: the catalog projects to its 2016 position,
        // ~7 px away from the detection — outside the default 5 px
        // match radius. So this should be a miss.
        let r_no_pm = verify_solution(&wcs, &field_sources, &index, &config, None);
        assert_eq!(
            r_no_pm.n_matched, 0,
            "Without obs_epoch the high-PM star should miss; got {} matches",
            r_no_pm.n_matched
        );

        // With obs_epoch: propagate_pm forwards the catalog position to
        // 2002, landing on the detection. Now it's a hit.
        let r_with_pm = verify_solution(&wcs, &field_sources, &index, &config, Some(&obs_epoch));
        assert_eq!(
            r_with_pm.n_matched, 1,
            "With obs_epoch the high-PM star should match; got {} matches",
            r_with_pm.n_matched
        );
    }
}
