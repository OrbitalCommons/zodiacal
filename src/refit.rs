//! Robust TAN refit from verified correspondences.
//!
//! The solver's WCS is fitted from the four stars of the matching quad. Over a wide field that
//! initial fit can be off enough that verification pairs some sources with nearby wrong stars.
//! This module refits the WCS from all verified correspondences, rejecting outliers with a
//! bounded, deterministic RANSAC, and re-verifies the result.
//!
//! A TAN-to-TAN mapping between two tangent planes is projective, so each fit solves for the eight
//! coefficients of a homography from normalised pixel coordinates to the tangent plane of the
//! current solution, then expresses the recovered camera basis as a TAN WCS centred on the image.
//! Only the solver's own correspondences are used; no external pointing or truth is involved.

use std::collections::BTreeMap;

use nalgebra::{DMatrix, DVector};

use crate::extraction::DetectedSource;
use crate::geom::sphere::{propagate_pm, radec_to_xyz, star_coords, xyz_to_radec};
use crate::geom::tan::TanWcs;
use crate::index::Index;
use crate::solver::Solution;
use crate::verify::{VerifyConfig, verify_solution};

/// Settings for [`robust_refit_tan`] and [`refine_solution`].
#[derive(Debug, Clone)]
pub struct RefitConfig {
    /// A correspondence is an inlier if the refitted WCS projects its star within this many pixels
    /// of its source.
    pub inlier_radius_pix: f64,
    /// Random four-pair samples tried after the initial all-pairs fit.
    pub trials: usize,
    /// Fewest inliers needed to accept a refit.
    pub min_inliers: usize,
    /// Seed for the xorshift sampler, so results are reproducible.
    pub seed: u32,
    /// Refit and re-verify rounds in [`refine_solution`]; each round starts from the previous fit.
    pub rounds: usize,
}

impl Default for RefitConfig {
    fn default() -> Self {
        Self {
            inlier_radius_pix: 2.5,
            trials: 128,
            min_inliers: 6,
            seed: 0x9e37_79b9,
            rounds: 3,
        }
    }
}

/// Index star position at `obs_epoch`, propagated exactly as [`verify_solution`] does; the
/// catalogue position when `obs_epoch` is `None`.
fn star_position(index: &Index, si: usize, obs_epoch: Option<f64>) -> (f64, f64) {
    let star = &index.stars[si];
    match obs_epoch {
        Some(obs) => propagate_pm(
            star.ra,
            star.dec,
            star.pmra,
            star.pmdec,
            star.ref_epoch,
            obs,
        ),
        None => (star.ra, star.dec),
    }
}

/// Least-squares projective TAN fit to `pairs` of (source index, index star index).
///
/// Star positions are taken at `obs_epoch` (see [`verify_solution`]); pass the same epoch used to
/// verify the pairs. Returns `None` for fewer than four pairs, out-of-range indices, non-finite
/// positions or image size, a solution centred exactly on a celestial pole (where the tangent
/// basis is degenerate), or a rank-deficient system such as collinear correspondences.
pub fn refit_tan(
    wcs: &TanWcs,
    pairs: &[(usize, usize)],
    sources: &[DetectedSource],
    index: &Index,
    obs_epoch: Option<f64>,
) -> Option<TanWcs> {
    if pairs.len() < 4 || !valid_inputs(wcs, pairs, sources, index, obs_epoch) {
        return None;
    }
    let w = wcs.image_size[0];
    let h = wcs.image_size[1];
    let reference = radec_to_xyz(wcs.crval[0], wcs.crval[1]);
    if reference[2].abs() == 1.0 {
        return None;
    }
    // Pixel coordinates are centred and divided by the width to keep the system well scaled.
    let mut a = DMatrix::<f64>::zeros(2 * pairs.len(), 8);
    let mut rhs = DVector::<f64>::zeros(2 * pairs.len());
    for (i, &(fi, si)) in pairs.iter().enumerate() {
        let p = &sources[fi];
        let (ra, dec) = star_position(index, si, obs_epoch);
        let (x, y) = star_coords(radec_to_xyz(ra, dec), reference)?;
        let u = (p.x - w / 2.0) / w;
        let v = (p.y - h / 2.0) / w;
        for (j, value) in [u, v, 1.0, 0.0, 0.0, 0.0, -x * u, -x * v]
            .iter()
            .enumerate()
        {
            a[(2 * i, j)] = *value;
        }
        for (j, value) in [0.0, 0.0, 0.0, u, v, 1.0, -y * u, -y * v]
            .iter()
            .enumerate()
        {
            a[(2 * i + 1, j)] = *value;
        }
        rhs[2 * i] = x;
        rhs[2 * i + 1] = y;
    }
    // Extreme inputs (a tiny image width, say) can overflow the normalised system.
    if !a.iter().chain(rhs.iter()).all(|x| x.is_finite()) {
        return None;
    }
    // The SVD solve is a pseudoinverse; reject rank-deficient geometry instead of returning the
    // minimum-norm fit to it.
    let svd = a.svd(true, true);
    let largest = svd.singular_values.max();
    if !largest.is_finite() || largest <= 0.0 || svd.singular_values.min() <= largest * 1e-12 {
        return None;
    }
    let q = svd.solve(&rhs, 1e-12).ok()?;

    let (ra, dec) = (wcs.crval[0], wcs.crval[1]);
    let east = [-ra.sin(), ra.cos(), 0.0];
    let north = [-dec.sin() * ra.cos(), -dec.sin() * ra.sin(), dec.cos()];
    let combine = |x: f64, y: f64, z: f64| {
        std::array::from_fn::<_, 3, _>(|i| east[i] * x + north[i] * y + reference[i] * z)
    };
    // The image centre maps to (q2, q5, 1) in the old tangent basis.
    let center = combine(q[2], q[5], 1.0);
    let norm = center.iter().map(|x| x * x).sum::<f64>().sqrt();
    if !norm.is_finite() || norm <= 0.0 {
        return None;
    }
    let center = center.map(|x| x / norm);
    if center[2].abs() == 1.0 {
        return None;
    }
    let (ra, dec) = xyz_to_radec(center);
    let new_east = [-ra.sin(), ra.cos(), 0.0];
    let new_north = [-dec.sin() * ra.cos(), -dec.sin() * ra.sin(), dec.cos()];
    let u = combine(q[0], q[3], q[6]);
    let v = combine(q[1], q[4], q[7]);
    let dot = |a: [f64; 3], b: [f64; 3]| a.iter().zip(b).map(|(x, y)| x * y).sum::<f64>();
    let cd = [
        [dot(u, new_east) / (w * norm), dot(v, new_east) / (w * norm)],
        [
            dot(u, new_north) / (w * norm),
            dot(v, new_north) / (w * norm),
        ],
    ];
    let det = cd[0][0] * cd[1][1] - cd[0][1] * cd[1][0];
    if !cd.iter().flatten().all(|x| x.is_finite()) || !det.is_finite() || det == 0.0 {
        return None;
    }
    Some(TanWcs {
        crval: [ra, dec],
        crpix: [w / 2.0, h / 2.0],
        cd,
        image_size: [w, h],
    })
}

/// Projective TAN refit with deterministic RANSAC outlier rejection.
///
/// Tries the fit to all pairs, then `config.trials` random four-pair samples, keeping the fit
/// with the most inliers (ties broken by smaller summed squared residual). Each index star
/// counts once, through its closest source. The final WCS is refitted to the best inlier set.
/// Returns `None` if fewer than `config.min_inliers` pairs survive or the inputs are invalid.
pub fn robust_refit_tan(
    wcs: &TanWcs,
    pairs: &[(usize, usize)],
    sources: &[DetectedSource],
    index: &Index,
    config: &RefitConfig,
    obs_epoch: Option<f64>,
) -> Option<TanWcs> {
    if pairs.len() < 4
        || !valid_inputs(wcs, pairs, sources, index, obs_epoch)
        || !config.inlier_radius_pix.is_finite()
        || config.inlier_radius_pix <= 0.0
    {
        return None;
    }
    let radius_sq = config.inlier_radius_pix * config.inlier_radius_pix;
    // xorshift has a fixed point at zero, which would never yield four distinct samples.
    let mut seed = if config.seed == 0 {
        RefitConfig::default().seed
    } else {
        config.seed
    };
    let mut best: Vec<(usize, usize)> = Vec::new();
    let mut best_error = f64::INFINITY;
    for attempt in 0..=config.trials {
        let sample = if attempt == 0 {
            pairs.to_vec()
        } else {
            let mut ids = Vec::new();
            while ids.len() < 4 {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                let i = seed as usize % pairs.len();
                if !ids.contains(&i) {
                    ids.push(i);
                }
            }
            ids.iter().map(|i| pairs[*i]).collect()
        };
        let Some(candidate) = refit_tan(wcs, &sample, sources, index, obs_epoch) else {
            continue;
        };
        let mut unique = BTreeMap::new();
        for &(fi, si) in pairs {
            let (ra, dec) = star_position(index, si, obs_epoch);
            if let Some((x, y)) = candidate.radec_to_pixel(ra, dec) {
                let p = &sources[fi];
                let d = (x - p.x).powi(2) + (y - p.y).powi(2);
                if d < radius_sq {
                    let entry = unique.entry(si).or_insert((fi, d));
                    if d < entry.1 {
                        *entry = (fi, d);
                    }
                }
            }
        }
        let error: f64 = unique.values().map(|(_, distance)| distance).sum();
        let inliers: Vec<_> = unique.into_iter().map(|(si, (fi, _))| (fi, si)).collect();
        if inliers.len() > best.len() || (inliers.len() == best.len() && error < best_error) {
            best = inliers;
            best_error = error;
        }
        if best.len() == pairs.len() && best_error < 1e-6 {
            break;
        }
    }
    if best.len() < config.min_inliers.max(4) {
        return None;
    }
    refit_tan(wcs, &best, sources, index, obs_epoch)
}

/// Refine a solver solution in place.
///
/// Re-verifies the solution with [`VerifyConfig::exhaustive`] to gather every correspondence,
/// then runs up to `config.rounds` rounds of [`robust_refit_tan`]. A round's refit is kept only
/// if its own exhaustive verification passes `accept` (the caller's decision thresholds, such as
/// the solver's `verify` config). Returns whether at least one round was kept; if none was, the
/// solution's WCS is unchanged and its `verify_result` holds the exhaustive verification.
///
/// `obs_epoch` is used for every verification and refit, so star positions stay consistent; pass
/// `None` to work with catalogue positions throughout.
pub fn refine_solution(
    solution: &mut Solution,
    sources: &[DetectedSource],
    index: &Index,
    accept: &VerifyConfig,
    config: &RefitConfig,
    obs_epoch: Option<f64>,
) -> bool {
    let gather = accept.exhaustive();
    solution.verify_result = verify_solution(&solution.wcs, sources, index, &gather, obs_epoch);
    let mut refined = false;
    for _ in 0..config.rounds {
        let Some(wcs) = robust_refit_tan(
            &solution.wcs,
            &solution.verify_result.matched_pairs,
            sources,
            index,
            config,
            obs_epoch,
        ) else {
            break;
        };
        let verification = verify_solution(&wcs, sources, index, &gather, obs_epoch);
        if !verification.is_accepted(accept) {
            break;
        }
        solution.wcs = wcs;
        solution.verify_result = verification;
        refined = true;
    }
    refined
}

fn valid_inputs(
    wcs: &TanWcs,
    pairs: &[(usize, usize)],
    sources: &[DetectedSource],
    index: &Index,
    obs_epoch: Option<f64>,
) -> bool {
    let [w, h] = wcs.image_size;
    w.is_finite()
        && h.is_finite()
        && w > 0.0
        && h > 0.0
        && wcs.crval.iter().all(|x| x.is_finite())
        && obs_epoch.is_none_or(f64::is_finite)
        && pairs.iter().all(|&(fi, si)| {
            if fi >= sources.len() || si >= index.stars.len() {
                return false;
            }
            let (ra, dec) = star_position(index, si, obs_epoch);
            sources[fi].x.is_finite()
                && sources[fi].y.is_finite()
                && ra.is_finite()
                && dec.is_finite()
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::builder::{IndexBuilderConfig, build_index};
    use crate::solver::{SolverConfig, solve};

    const SIZE: f64 = 1024.0;

    fn truth() -> TanWcs {
        // 20 degrees across, rotated 15 degrees.
        let scale = (20.0f64 / SIZE).to_radians();
        let (s, c) = 15.0f64.to_radians().sin_cos();
        TanWcs {
            crval: [1.2, 0.4],
            crpix: [SIZE / 2.0, SIZE / 2.0],
            cd: [[c * scale, -s * scale], [s * scale, c * scale]],
            image_size: [SIZE, SIZE],
        }
    }

    /// Sources at the true positions of an index's stars, plus the identity pairs.
    fn scenario(n: usize) -> (Vec<DetectedSource>, Index, Vec<(usize, usize)>) {
        let wcs = truth();
        let mut state: u64 = 271828183;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };
        let mut catalog = Vec::new();
        let mut sources = Vec::new();
        for i in 0..n {
            let (px, py) = (40.0 + rng() * (SIZE - 80.0), 40.0 + rng() * (SIZE - 80.0));
            let (ra, dec) = wcs.pixel_to_radec(px, py);
            catalog.push((i as u64, ra, dec, i as f64));
            sources.push(DetectedSource {
                x: px,
                y: py,
                flux: 1000.0 - i as f64,
            });
        }
        let scale = wcs.pixel_scale().to_radians();
        let index = build_index(
            &catalog,
            &IndexBuilderConfig {
                scale_lower: scale * 60.0,
                scale_upper: scale * SIZE * 1.5,
                max_stars: n,
                max_quads: 20_000,
            },
        );
        // build_index keeps every star; pair each source with the star at its position.
        let pairs = (0..n)
            .map(|fi| {
                let (ra, dec) = wcs.pixel_to_radec(sources[fi].x, sources[fi].y);
                let si = index
                    .stars
                    .iter()
                    .position(|s| (s.ra - ra).abs() < 1e-12 && (s.dec - dec).abs() < 1e-12)
                    .unwrap();
                (fi, si)
            })
            .collect();
        (sources, index, pairs)
    }

    /// The initial WCS a refit starts from: the truth, shifted and slightly mis-scaled.
    fn perturbed() -> TanWcs {
        let mut wcs = truth();
        wcs.crval[0] += 0.002;
        wcs.crval[1] -= 0.001;
        wcs.cd = wcs.cd.map(|row| row.map(|x| x * 1.003));
        wcs
    }

    fn center_error_arcsec(a: &TanWcs, b: &TanWcs) -> f64 {
        let (ra1, dec1) = a.field_center();
        let (ra2, dec2) = b.field_center();
        let p = radec_to_xyz(ra1, dec1);
        let q = radec_to_xyz(ra2, dec2);
        let dot: f64 = p.iter().zip(q).map(|(x, y)| x * y).sum();
        dot.clamp(-1.0, 1.0).acos().to_degrees() * 3600.0
    }

    #[test]
    fn refit_recovers_the_true_wcs_from_exact_pairs() {
        let (sources, index, pairs) = scenario(40);
        let fit = refit_tan(&perturbed(), &pairs, &sources, &index, None).unwrap();
        assert!(center_error_arcsec(&fit, &truth()) < 0.01);
        assert!((fit.pixel_scale() / truth().pixel_scale() - 1.0).abs() < 1e-9);
    }

    #[test]
    fn robust_refit_ignores_a_wrong_correspondence() {
        let (sources, index, mut pairs) = scenario(40);
        // Pair the first source with a star elsewhere in the field.
        pairs[0].1 = pairs[20].1;
        let plain = refit_tan(&perturbed(), &pairs, &sources, &index, None).unwrap();
        let robust = robust_refit_tan(
            &perturbed(),
            &pairs,
            &sources,
            &index,
            &RefitConfig::default(),
            None,
        )
        .unwrap();
        assert!(center_error_arcsec(&robust, &truth()) < 0.01);
        assert!(center_error_arcsec(&plain, &truth()) > center_error_arcsec(&robust, &truth()));
    }

    #[test]
    fn a_zero_seed_still_terminates() {
        let (sources, index, mut pairs) = scenario(12);
        pairs[0].1 = pairs[5].1;
        let config = RefitConfig {
            seed: 0,
            ..RefitConfig::default()
        };
        assert!(robust_refit_tan(&perturbed(), &pairs, &sources, &index, &config, None).is_some());
    }

    #[test]
    fn collinear_pairs_are_rejected() {
        let wcs = truth();
        let sources: Vec<DetectedSource> = (0..6)
            .map(|i| DetectedSource {
                x: 100.0 + i as f64 * 150.0,
                y: 100.0 + i as f64 * 150.0,
                flux: 1.0,
            })
            .collect();
        let catalog: Vec<_> = sources
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let (ra, dec) = wcs.pixel_to_radec(s.x, s.y);
                (i as u64, ra, dec, i as f64)
            })
            .collect();
        let index = build_index(
            &catalog,
            &IndexBuilderConfig {
                scale_lower: 0.0,
                scale_upper: 1.0,
                max_stars: 6,
                max_quads: 10,
            },
        );
        let pairs: Vec<_> = (0..6).map(|i| (i, i)).collect();
        assert!(refit_tan(&wcs, &pairs, &sources, &index, None).is_none());
    }

    #[test]
    fn star_positions_follow_the_observation_epoch() {
        // The sources are where the stars were in 2000; the index records them in 2016 after
        // a large, uniform proper motion.
        let (sources, mut index, pairs) = scenario(30);
        let (pmra, pmdec) = (40_000.0, -25_000.0);
        for star in &mut index.stars {
            let (ra, dec) = propagate_pm(star.ra, star.dec, pmra, pmdec, 2000.0, 2016.0);
            // propagate_pm divides by cos(dec) at the starting epoch; rescale so propagating
            // back from 2016 lands on the 2000 position.
            star.pmra = pmra * dec.cos() / star.dec.cos();
            star.ra = ra;
            star.dec = dec;
            star.pmdec = pmdec;
            star.ref_epoch = 2016.0;
        }
        let at_epoch = refit_tan(&perturbed(), &pairs, &sources, &index, Some(2000.0)).unwrap();
        let catalogue = refit_tan(&perturbed(), &pairs, &sources, &index, None).unwrap();
        assert!(center_error_arcsec(&at_epoch, &truth()) < 0.05);
        assert!(center_error_arcsec(&catalogue, &truth()) > 100.0);
    }

    #[test]
    fn out_of_range_and_non_finite_inputs_are_rejected() {
        let (sources, mut index, pairs) = scenario(12);
        let mut bad = pairs.clone();
        bad[0].0 = sources.len();
        assert!(refit_tan(&truth(), &bad, &sources, &index, None).is_none());
        let mut bad = pairs.clone();
        bad[0].1 = index.stars.len();
        assert!(refit_tan(&truth(), &bad, &sources, &index, None).is_none());
        let mut wcs = truth();
        wcs.image_size[0] = f64::NAN;
        assert!(refit_tan(&wcs, &pairs, &sources, &index, None).is_none());
        assert!(refit_tan(&truth(), &pairs, &sources, &index, Some(f64::NAN)).is_none());
        let config = RefitConfig {
            inlier_radius_pix: 0.0,
            ..RefitConfig::default()
        };
        assert!(robust_refit_tan(&truth(), &pairs, &sources, &index, &config, None).is_none());
        index.stars[pairs[0].1].ra = f64::NAN;
        assert!(refit_tan(&truth(), &pairs, &sources, &index, None).is_none());
    }

    #[test]
    fn exhaustive_keeps_matching_settings_and_disables_early_exit() {
        let base = VerifyConfig {
            match_radius_pix: 3.0,
            min_matches: 7,
            ..VerifyConfig::default()
        };
        let all = base.exhaustive();
        assert_eq!(all.match_radius_pix, 3.0);
        assert_eq!(all.min_matches, 7);
        assert_eq!(all.distractor_fraction, base.distractor_fraction);
        assert!(all.log_odds_accept.is_infinite() && all.log_odds_accept > 0.0);
        assert!(all.log_odds_bail.is_infinite() && all.log_odds_bail < 0.0);
    }

    #[test]
    fn refine_solution_tightens_a_solver_result() {
        let (sources, index, _) = scenario(40);
        let config = SolverConfig {
            scale_range: None,
            max_field_stars: 24,
            code_tolerance: 0.002,
            ..SolverConfig::default()
        };
        let (solution, _) = solve(&sources, &[&index], (SIZE, SIZE), &config);
        let mut solution = solution.expect("solver should find the synthetic field");
        let before = center_error_arcsec(&solution.wcs, &truth());
        assert!(refine_solution(
            &mut solution,
            &sources,
            &index,
            &config.verify,
            &RefitConfig::default(),
            None,
        ));
        let after = center_error_arcsec(&solution.wcs, &truth());
        assert!(
            after <= before + 1e-6,
            "refit made the centre worse: {before} -> {after}"
        );
        assert!(after < 0.05, "centre still {after} arcsec off");
        assert!(solution.verify_result.is_accepted(&config.verify));
    }
}
