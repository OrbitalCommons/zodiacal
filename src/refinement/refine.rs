//! Iterative refinement of a tweaked WCS to ~10 mas absolute astrometry.
//!
//! Takes a `SipWcs` from `tweak_solution` plus a catalog of full Gaia
//! astrometry and an observation context, and returns an updated WCS
//! where matched catalog stars' **apparent** positions are used as the
//! sky-side truth instead of their catalog (ICRS @ ref epoch) positions.
//!
//! SIP distortion coefficients are preserved across refinement (SIP
//! captures optical distortion in pixel space, which does not change
//! when the apparent-place correction is applied). Only the TAN
//! parameters — CRVAL, CD matrix — are re-fit.

use crate::extraction::DetectedSource;
use crate::fitting::fit_tan_wcs;
use crate::geom::sip::SipWcs;
use crate::geom::sphere::radec_to_xyz;
use crate::index::Index;
use crate::kdtree::KdTree;

use super::apparent::apparent_radec;
use super::types::{
    ObservationContext, RefinedMatch, RefinedSolution, RefinementCatalog, RefinementConfig,
    RefinementError,
};

/// Refine a tweaked WCS by applying apparent-place corrections to matched
/// catalog stars and re-fitting the TAN parameters.
///
/// The SIP distortion coefficients on `initial` are preserved unchanged.
pub fn refine_solution(
    initial: &SipWcs,
    field_sources: &[DetectedSource],
    index: &Index,
    catalog: &RefinementCatalog,
    obs: &ObservationContext,
    config: &RefinementConfig,
) -> Result<RefinedSolution, RefinementError> {
    let field_points: Vec<[f64; 2]> = field_sources.iter().map(|s| [s.x, s.y]).collect();
    let field_indices: Vec<usize> = (0..field_sources.len()).collect();
    let field_tree = KdTree::<2>::build(field_points, field_indices);

    let match_radius_sq = config.match_radius_pix * config.match_radius_pix;
    let image_size = (initial.tan.image_size[0], initial.tan.image_size[1]);

    let mut current = initial.clone();
    let mut prev_crval = current.tan.crval;
    let mut prev_cd = current.tan.cd;

    let mut matches: Vec<RefinedMatch> = Vec::new();
    let mut residual_rms_pix = f64::NAN;
    let mut residual_rms_mas = f64::NAN;
    let mut iterations_run = 0;

    for iter in 0..config.max_iterations {
        iterations_run = iter + 1;

        // 1. Find candidate catalog stars in the field of view.
        let (center_ra, center_dec) = current.field_center();
        let center_xyz = radec_to_xyz(center_ra, center_dec);
        let field_radius = current.field_radius();
        let radius_sq = 2.0 * (1.0 - field_radius.cos());
        let nearby = index.star_tree.range_search(&center_xyz, radius_sq);

        // 2. For each catalog star with Gaia astrometry, compute apparent
        //    position and attempt to match to a field source.
        let mut matched_app_xyz: Vec<[f64; 3]> = Vec::new();
        let mut matched_field_xy: Vec<(f64, f64)> = Vec::new();
        let mut local_matches: Vec<RefinedMatch> = Vec::new();

        for result in &nearby {
            let star = &index.stars[result.index];
            let Some(astro) = catalog.get(star.catalog_id) else {
                continue;
            };

            let (apparent_ra, apparent_dec) = apparent_radec(astro, obs)
                .map_err(|e| RefinementError::Starfield(format!("{e}")))?;
            let app_xyz = radec_to_xyz(apparent_ra, apparent_dec);

            let predicted_pixel = current.radec_to_pixel(apparent_ra, apparent_dec);
            let Some((pred_x, pred_y)) = predicted_pixel else {
                continue;
            };

            let margin = 10.0;
            if pred_x < -margin
                || pred_x > image_size.0 + margin
                || pred_y < -margin
                || pred_y > image_size.1 + margin
            {
                continue;
            }

            if let Some(nearest) = field_tree.nearest(&[pred_x, pred_y])
                && nearest.dist_sq <= match_radius_sq
            {
                let fs = &field_sources[nearest.index];
                let dx = pred_x - fs.x;
                let dy = pred_y - fs.y;
                let residual_pix = (dx * dx + dy * dy).sqrt();

                matched_app_xyz.push(app_xyz);
                matched_field_xy.push((fs.x, fs.y));

                let weight = gaia_weight(astro, obs);
                local_matches.push(RefinedMatch {
                    catalog_id: star.catalog_id,
                    field_source_idx: nearest.index,
                    apparent_ra_deg: apparent_ra.to_degrees(),
                    apparent_dec_deg: apparent_dec.to_degrees(),
                    residual_mas: residual_pix_to_mas(residual_pix, &current),
                    weight,
                });
            }
        }

        let n_matches = matched_app_xyz.len();
        if n_matches < config.min_matches {
            return Err(RefinementError::InsufficientMatches {
                required: config.min_matches,
                actual: n_matches,
            });
        }

        // 3. Re-fit TAN using the apparent positions as the sky truth.
        let new_tan = fit_tan_wcs(&matched_app_xyz, &matched_field_xy, image_size)
            .map_err(|_| RefinementError::Starfield("fit_tan_wcs failed".into()))?;

        // Preserve SIP distortion; only TAN parameters change across refinement.
        current = SipWcs {
            tan: new_tan,
            a: current.a,
            b: current.b,
            a_order: current.a_order,
            b_order: current.b_order,
            ap: current.ap,
            bp: current.bp,
            ap_order: current.ap_order,
            bp_order: current.bp_order,
        };

        // 4. Compute residual RMS and convergence diagnostics.
        matches = local_matches;
        residual_rms_pix = rms_pixel(&matched_app_xyz, &matched_field_xy, &current);
        residual_rms_mas = residual_rms_pix * pixel_scale_mas(&current);

        if wcs_changed_less_than(prev_crval, prev_cd, &current.tan, config.convergence_pix) {
            break;
        }
        prev_crval = current.tan.crval;
        prev_cd = current.tan.cd;
    }

    Ok(RefinedSolution {
        wcs: current,
        n_iterations: iterations_run,
        residual_rms_mas,
        residual_rms_pix,
        matched: matches,
    })
}

/// Per-source weight from Gaia uncertainties propagated to the observation
/// epoch, treating RA and Dec errors as uncorrelated (see PLAN.md §11.4).
fn gaia_weight(astro: &super::types::GaiaAstrometry, obs: &ObservationContext) -> f64 {
    let dt_years = (obs.time.tt() - jyear_to_tt_jd(astro.ref_epoch_jyear)) / 365.25;
    let sigma_ra_sq =
        astro.sigma_ra_mas.powi(2) + (dt_years * astro.sigma_pmra_mas_per_year).powi(2);
    let sigma_dec_sq =
        astro.sigma_dec_mas.powi(2) + (dt_years * astro.sigma_pmdec_mas_per_year).powi(2);
    let sigma_pos_sq = (sigma_ra_sq + sigma_dec_sq) / 2.0;
    if sigma_pos_sq > 0.0 {
        1.0 / sigma_pos_sq
    } else {
        1.0
    }
}

fn jyear_to_tt_jd(jyear: f64) -> f64 {
    2_451_545.0 + (jyear - 2000.0) * 365.25
}

fn rms_pixel(sky_xyz: &[[f64; 3]], field_xy: &[(f64, f64)], wcs: &SipWcs) -> f64 {
    let mut sum_sq = 0.0;
    let mut n = 0usize;
    for (xyz, (fx, fy)) in sky_xyz.iter().zip(field_xy.iter()) {
        if let Some((px, py)) = wcs.tan.xyz_to_pixel(*xyz) {
            let dx = px - *fx;
            let dy = py - *fy;
            sum_sq += dx * dx + dy * dy;
            n += 1;
        }
    }
    if n > 0 {
        (sum_sq / n as f64).sqrt()
    } else {
        f64::NAN
    }
}

fn pixel_scale_mas(wcs: &SipWcs) -> f64 {
    // Pixel scale in degrees/pixel → mas/pixel.
    wcs.tan.pixel_scale() * 3_600_000.0
}

fn residual_pix_to_mas(residual_pix: f64, wcs: &SipWcs) -> f64 {
    residual_pix * pixel_scale_mas(wcs)
}

fn wcs_changed_less_than(
    prev_crval: [f64; 2],
    prev_cd: [[f64; 2]; 2],
    tan: &crate::geom::tan::TanWcs,
    threshold_pix: f64,
) -> bool {
    // Approximate: test a ~1 pixel movement in crval against pixel scale,
    // plus Frobenius norm of CD delta expressed in pixel-equivalent units.
    let d_ra = (tan.crval[0] - prev_crval[0]).abs();
    let d_dec = (tan.crval[1] - prev_crval[1]).abs();
    let scale_rad_per_pix = tan.pixel_scale().to_radians();
    let crval_shift_pix =
        ((d_ra * tan.crval[1].cos()).powi(2) + d_dec.powi(2)).sqrt() / scale_rad_per_pix;

    let mut cd_delta_sq = 0.0;
    for (cd_row, prev_row) in tan.cd.iter().zip(prev_cd.iter()) {
        for (cd_v, prev_v) in cd_row.iter().zip(prev_row.iter()) {
            cd_delta_sq += (cd_v - prev_v).powi(2);
        }
    }
    let cd_delta = cd_delta_sq.sqrt() / scale_rad_per_pix.max(f64::EPSILON);

    crval_shift_pix < threshold_pix && cd_delta < threshold_pix
}
