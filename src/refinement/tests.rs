//! End-to-end test: synthetic catalog → PM-displaced field → refinement
//! recovers the true WCS.
//!
//! Setup:
//! - 20 catalog stars at ref epoch J2016.0 with per-star proper motions
//!   of ~50 mas/yr (chosen randomly per star).
//! - Observation at J2026.0 (10 years later).
//! - Observer at the solar system barycenter (zero velocity), so no
//!   parallax or aberration effects — only PM propagation.
//! - "Truth" WCS projects each star's APPARENT (J2026) position to a
//!   pixel; those pixels are the field sources.
//! - "Initial" WCS is what the blind solver would produce: we fit
//!   fit_tan_wcs(catalog_xyz_at_ref_epoch, field_xy), which has
//!   systematic bias because the catalog positions don't match the
//!   field pixels.
//! - Refinement should recover the truth WCS to sub-mas residuals.

use super::*;
use crate::extraction::DetectedSource;
use crate::fitting::fit_tan_wcs;
use crate::geom::sip::SipWcs;
use crate::geom::sphere::radec_to_xyz;
use crate::geom::tan::TanWcs;
use crate::index::{Index, IndexStar};
use crate::kdtree::KdTree;
use crate::quads::{DIMCODES, Quad};

const PIXEL_SCALE_ARCSEC: f64 = 2.0;
const IMAGE_SIZE: f64 = 512.0;
const REF_EPOCH: f64 = 2016.0;
const OBS_EPOCH: f64 = 2026.0;
const N_STARS: usize = 20;

fn truth_wcs() -> TanWcs {
    let arcsec_rad = (PIXEL_SCALE_ARCSEC / 3600.0).to_radians();
    TanWcs {
        crval: [1.0, 0.5],
        crpix: [IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0],
        cd: [[arcsec_rad, 0.0], [0.0, arcsec_rad]],
        image_size: [IMAGE_SIZE, IMAGE_SIZE],
    }
}

fn deterministic_pm(i: usize) -> (f64, f64) {
    // Per-star PMs in mas/yr, chosen to be nontrivial but not pathological.
    let state = (i as u64).wrapping_mul(2_654_435_761);
    let a = ((state % 200) as f64) - 100.0; // -100..+99 mas/yr
    let b = (((state / 200) % 200) as f64) - 100.0;
    (a, b)
}

fn make_scenario() -> (
    Vec<DetectedSource>,
    Index,
    RefinementCatalog,
    TanWcs,          // truth
    SipWcs,          // initial (biased)
    ObservationContext,
) {
    let wcs_truth = truth_wcs();
    let ts = starfield::time::Timescale::default();
    let obs = ObservationContext {
        time: ts.tt_jd(jyear_to_tt_jd(OBS_EPOCH), None),
        observer: ObserverState::Barycentric {
            position_au: [0.0, 0.0, 0.0],
            velocity_au_per_day: [0.0, 0.0, 0.0],
        },
    };

    let mut index_stars: Vec<IndexStar> = Vec::new();
    let mut field_sources: Vec<DetectedSource> = Vec::new();
    let mut catalog = RefinementCatalog::new();
    let mut ref_xyz: Vec<[f64; 3]> = Vec::new();
    let mut field_xy: Vec<(f64, f64)> = Vec::new();

    let side = (N_STARS as f64).sqrt().ceil() as usize;

    for i in 0..N_STARS {
        let ix = i % side;
        let iy = i / side;
        let px = IMAGE_SIZE * 0.1
            + IMAGE_SIZE * 0.8 * (ix as f64) / (side as f64 - 1.0).max(1.0);
        let py = IMAGE_SIZE * 0.1
            + IMAGE_SIZE * 0.8 * (iy as f64) / (side as f64 - 1.0).max(1.0);

        // The pixel is the target *apparent* position. Work backwards to find
        // the apparent (RA, Dec), then back-propagate via PM to find the
        // catalog (J2016) position.
        let (apparent_ra, apparent_dec) = wcs_truth.pixel_to_radec(px, py);

        let (pmra_mas_per_year, pmdec_mas_per_year) = deterministic_pm(i);
        let dt_years = OBS_EPOCH - REF_EPOCH;

        // Back-propagate to get the catalog RA/Dec at J2016.0.
        // PM is given in RA*cos(dec) and Dec, so RA shift is divided by cos(dec).
        let mas_per_rad = 180.0 * 3_600_000.0 / std::f64::consts::PI;
        let d_ra = (pmra_mas_per_year * dt_years) / mas_per_rad / apparent_dec.cos();
        let d_dec = (pmdec_mas_per_year * dt_years) / mas_per_rad;
        let ref_ra = apparent_ra - d_ra;
        let ref_dec = apparent_dec - d_dec;

        index_stars.push(IndexStar {
            catalog_id: i as u64,
            ra: ref_ra,
            dec: ref_dec,
            mag: i as f64,
        });

        field_sources.push(DetectedSource {
            x: px,
            y: py,
            flux: 1000.0 - i as f64,
        });

        catalog.insert(
            i as u64,
            GaiaAstrometry {
                ra_deg: ref_ra.to_degrees(),
                dec_deg: ref_dec.to_degrees(),
                pmra_mas_per_year,
                pmdec_mas_per_year,
                parallax_mas: 0.0,
                radial_km_per_s: 0.0,
                ref_epoch_jyear: REF_EPOCH,
                sigma_ra_mas: 0.5,
                sigma_dec_mas: 0.5,
                sigma_pmra_mas_per_year: 0.1,
                sigma_pmdec_mas_per_year: 0.1,
                sigma_parallax_mas: 0.1,
            },
        );

        ref_xyz.push(radec_to_xyz(ref_ra, ref_dec));
        field_xy.push((px, py));
    }

    // Build Index.
    let star_points: Vec<[f64; 3]> = index_stars
        .iter()
        .map(|s| radec_to_xyz(s.ra, s.dec))
        .collect();
    let star_indices: Vec<usize> = (0..index_stars.len()).collect();
    let star_tree = KdTree::<3>::build(star_points, star_indices);
    let code_tree = KdTree::<{ DIMCODES }>::build(vec![], vec![]);
    let index = Index {
        star_tree,
        stars: index_stars,
        code_tree,
        quads: Vec::<Quad>::new(),
        scale_lower: 0.0,
        scale_upper: std::f64::consts::PI,
        metadata: None,
    };

    // Biased initial WCS: fit using catalog (J2016.0) positions against
    // observed pixels (which correspond to J2026.0 apparent positions).
    let biased_tan = fit_tan_wcs(&ref_xyz, &field_xy, (IMAGE_SIZE, IMAGE_SIZE)).unwrap();
    let initial = SipWcs::from_tan(biased_tan, 2);

    (field_sources, index, catalog, wcs_truth, initial, obs)
}

fn jyear_to_tt_jd(jyear: f64) -> f64 {
    2_451_545.0 + (jyear - 2000.0) * 365.25
}

/// The biased initial WCS (fit against catalog J2016 positions while the
/// field sources are at apparent J2026 positions) should have observable
/// residuals. If this test fails, the scenario isn't exercising refinement.
#[test]
fn biased_initial_wcs_has_nontrivial_residuals() {
    let (field_sources, _index, catalog, truth, initial, obs) = make_scenario();

    // Compute residual of initial WCS at each field source.
    let mut max_residual_mas = 0.0f64;
    for (i, fs) in field_sources.iter().enumerate() {
        let astro = catalog.get(i as u64).unwrap();
        let (app_ra, app_dec) = apparent_radec(astro, &obs).unwrap();
        if let Some((px, py)) = initial.radec_to_pixel(app_ra, app_dec) {
            let dx = px - fs.x;
            let dy = py - fs.y;
            let r_pix = (dx * dx + dy * dy).sqrt();
            let r_mas = r_pix * truth.pixel_scale() * 3_600_000.0;
            if r_mas > max_residual_mas {
                max_residual_mas = r_mas;
            }
        }
    }
    assert!(
        max_residual_mas > 10.0,
        "scenario not exercising refinement: max initial residual only {} mas",
        max_residual_mas
    );
}

#[test]
fn refinement_via_sidecar_recovers_truth_wcs() {
    // Same scenario as `refinement_recovers_truth_wcs`, but route the
    // catalog through a persisted sidecar file + load_sidecar_filtered.
    let (field_sources, index, catalog, _truth, initial, obs) = make_scenario();

    // Write the in-memory catalog to a sidecar.
    let mut sidecar_path = std::env::temp_dir();
    sidecar_path.push(format!(
        "zodiacal-refine-e2e-{}.gaia",
        std::process::id()
    ));
    let records: Vec<SidecarRecord> = catalog
        .sources
        .iter()
        .map(|(source_id, astro)| SidecarRecord {
            source_id: *source_id,
            ref_epoch: astro.ref_epoch_jyear,
            ra: astro.ra_deg,
            dec: astro.dec_deg,
            pmra: astro.pmra_mas_per_year,
            pmdec: astro.pmdec_mas_per_year,
            parallax: if astro.parallax_mas == 0.0 {
                f64::NAN
            } else {
                astro.parallax_mas
            },
            radial_velocity: if astro.radial_km_per_s == 0.0 {
                f64::NAN
            } else {
                astro.radial_km_per_s
            },
            sigma_ra: astro.sigma_ra_mas as f32,
            sigma_dec: astro.sigma_dec_mas as f32,
            sigma_pmra: astro.sigma_pmra_mas_per_year as f32,
            sigma_pmdec: astro.sigma_pmdec_mas_per_year as f32,
            sigma_parallax: astro.sigma_parallax_mas as f32,
            flags: 0,
        })
        .collect();
    write_sidecar(&sidecar_path, records, 8).unwrap();

    // Load only the rows for the index's stars (here: all of them).
    let source_ids: Vec<u64> = index.stars.iter().map(|s| s.catalog_id).collect();
    let reloaded =
        RefinementCatalog::load_sidecar_filtered(&sidecar_path, &source_ids).unwrap();
    assert_eq!(reloaded.len(), catalog.len());

    // Run refinement through the reloaded catalog.
    let config = RefinementConfig {
        match_radius_pix: 5.0,
        max_iterations: 5,
        convergence_pix: 0.001,
        min_matches: 10,
    };
    let refined =
        refine_solution(&initial, &field_sources, &index, &reloaded, &obs, &config)
            .expect("refinement via sidecar should succeed");

    assert!(
        refined.residual_rms_mas < 1.0,
        "residual RMS via sidecar {} mas",
        refined.residual_rms_mas
    );
    assert_eq!(refined.matched.len(), N_STARS);

    let _ = std::fs::remove_file(&sidecar_path);
}

#[test]
fn refinement_recovers_truth_wcs() {
    let (field_sources, index, catalog, truth, initial, obs) = make_scenario();

    let config = RefinementConfig {
        match_radius_pix: 5.0,
        max_iterations: 5,
        convergence_pix: 0.001,
        min_matches: 10,
    };

    let refined = refine_solution(&initial, &field_sources, &index, &catalog, &obs, &config)
        .expect("refinement should succeed");

    assert!(
        refined.residual_rms_mas < 1.0,
        "residual RMS {} mas exceeds 1 mas (expected near-zero for noise-free synthetic)",
        refined.residual_rms_mas
    );

    // Compare the pixel→sky mapping at the image center rather than CRVAL
    // directly: fit_tan_wcs re-parameterizes the tangent point at the
    // centroid of stars, so CRVAL differs from the truth CRVAL even though
    // the mapping is equivalent.
    let (truth_ra, truth_dec) = wcs_pixel_to_radec(&truth, IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0);
    let (refined_ra, refined_dec) =
        wcs_pixel_to_radec_sip(&refined.wcs, IMAGE_SIZE / 2.0, IMAGE_SIZE / 2.0);

    let arcsec = std::f64::consts::PI / (180.0 * 3600.0);
    let mas = arcsec / 1000.0;

    // Angular separation in mas.
    let cos_dec = truth_dec.cos();
    let d_ra_mas = ((refined_ra - truth_ra) * cos_dec) / mas;
    let d_dec_mas = (refined_dec - truth_dec) / mas;
    let sep_mas = (d_ra_mas * d_ra_mas + d_dec_mas * d_dec_mas).sqrt();

    assert!(
        sep_mas < 1.0,
        "image-center sky position recovered to {sep_mas} mas (should be <1 mas)",
    );

    assert_eq!(
        refined.matched.len(),
        N_STARS,
        "expected all {} stars to match",
        N_STARS
    );
}

fn wcs_pixel_to_radec(tan: &TanWcs, px: f64, py: f64) -> (f64, f64) {
    tan.pixel_to_radec(px, py)
}

fn wcs_pixel_to_radec_sip(sip: &SipWcs, px: f64, py: f64) -> (f64, f64) {
    sip.pixel_to_radec(px, py)
}
