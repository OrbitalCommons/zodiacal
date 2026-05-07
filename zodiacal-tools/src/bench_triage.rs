//! `bench-triage` — diagnose why specific bench cases fail.
//!
//! For each requested case (typically the failures from a prior
//! `bench-bundle` run), build the truth `TanWcs` from the test JSON,
//! load the bundle's region around truth, and answer two questions:
//!
//! 1. **Catalog coverage**: of the test image's brightest detected
//!    sources, how many have a catalog star within `match_radius_pix`
//!    when projected through the truth WCS?
//! 2. **Findable quads**: how many of the bundle's quads in this
//!    region project (via the truth WCS) to four image-internal pixel
//!    positions, each within `match_radius_pix` of *some* detected
//!    source — i.e. quads the solver could in principle find from
//!    the brightest detections?
//!
//! If (1) is low, the catalog doesn't agree with the rendered image at
//! the truth pointing — the failure isn't a solver issue.
//! If (1) is fine but (2) is zero, no cataloged 4-tuple covers the
//! detected geometry — bundle-side gap.
//! If both are fine but the solver still missed it, it's a solver
//! tuning issue (max_field_stars, code_tolerance, use_count caps).

use std::collections::HashSet;
use std::fs;
use std::io::{self, Write};
use std::path::PathBuf;

use serde::{Deserialize, Serialize};
use starfield::Equatorial;

use zodiacal::bundle::reader::ZdclBundle;
use zodiacal::geom::tan::TanWcs;
use zodiacal::solver::SkyRegion;

#[derive(Debug, Deserialize)]
struct TestCase {
    image_width: f64,
    image_height: f64,
    ra_deg: f64,
    dec_deg: f64,
    plate_scale_arcsec: f64,
    sources: Vec<TestCaseSource>,
}

#[derive(Debug, Deserialize, Clone, Copy)]
struct TestCaseSource {
    x: f64,
    y: f64,
    flux: f64,
}

#[derive(Debug, Serialize)]
struct TriageRow {
    case: String,
    n_detected: usize,
    n_top_field: usize,
    n_catalog_in_region: usize,
    n_catalog_on_image: usize,
    n_top_with_match_3px: usize,
    n_top_with_match_10px: usize,
    median_top_residual_px: f64,
    p95_top_residual_px: f64,
    n_quads_in_region: usize,
    n_quads_on_image: usize,
    n_findable_quads: usize,
    findable_per_band: String,
}

pub struct BenchTriageConfig {
    pub bundle_path: PathBuf,
    pub test_cases_dir: PathBuf,
    /// Optional CSV path. If set, every row with `solved=0` is triaged.
    pub csv: Option<PathBuf>,
    /// Optional explicit case-id list. Takes precedence over `csv` when both set.
    pub case_ids: Vec<String>,
    /// Region radius in degrees (must match the bench's radius for consistency).
    pub radius_deg: f64,
    /// Pixel-radius for catalog-vs-detection match.
    pub match_radius_pix: f64,
    /// Cap on bright field stars considered (mirrors `solver_cfg.max_field_stars`).
    pub max_field_stars: usize,
    /// Optional cap on cases — useful for spot-checking.
    pub limit: Option<usize>,
    /// If true, sweep the 8 standard CD orientations on each case and
    /// print the match counts for each. Used to figure out the
    /// renderer's WCS convention. Skips the normal triage CSV output.
    pub probe_cd: bool,
}

pub fn run(cfg: &BenchTriageConfig) -> io::Result<()> {
    let bundle = ZdclBundle::open(&cfg.bundle_path)?;
    eprintln!(
        "Opened bundle: cell_depth={} bands={} populated={}",
        bundle.cell_depth(),
        bundle.bands().len(),
        bundle.manifest().gaia.populated_cells,
    );

    let case_ids = collect_case_ids(cfg)?;
    eprintln!("Triaging {} case(s)…", case_ids.len());

    if cfg.probe_cd {
        return probe_cd_orientations(&bundle, &case_ids, cfg);
    }

    // CSV header.
    println!(
        "case,n_detected,n_top_field,n_catalog_in_region,n_catalog_on_image,n_top_with_match_3px,n_top_with_match_10px,median_top_residual_px,p95_top_residual_px,n_quads_in_region,n_quads_on_image,n_findable_quads,findable_per_band"
    );

    for (i, id) in case_ids.iter().enumerate() {
        let path = cfg.test_cases_dir.join(format!("{id}.json"));
        let raw = fs::read_to_string(&path).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        })?;
        let tc: TestCase = serde_json::from_str(&raw).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        })?;

        let row = triage_case(&bundle, id, &tc, cfg)?;
        let line = format!(
            "{},{},{},{},{},{},{},{:.3},{:.3},{},{},{},{}",
            row.case,
            row.n_detected,
            row.n_top_field,
            row.n_catalog_in_region,
            row.n_catalog_on_image,
            row.n_top_with_match_3px,
            row.n_top_with_match_10px,
            row.median_top_residual_px,
            row.p95_top_residual_px,
            row.n_quads_in_region,
            row.n_quads_on_image,
            row.n_findable_quads,
            row.findable_per_band,
        );
        println!("{line}");
        io::stdout().flush().ok();

        if (i + 1) % 10 == 0 || i + 1 == case_ids.len() {
            eprintln!("  [{}/{}] last: {}", i + 1, case_ids.len(), line);
        }
    }

    Ok(())
}

/// Collect case-IDs to triage. Priority: explicit list > CSV failures.
fn collect_case_ids(cfg: &BenchTriageConfig) -> io::Result<Vec<String>> {
    if !cfg.case_ids.is_empty() {
        let mut ids = cfg.case_ids.clone();
        if let Some(n) = cfg.limit {
            ids.truncate(n);
        }
        return Ok(ids);
    }
    let csv = cfg.csv.as_ref().ok_or_else(|| {
        io::Error::new(io::ErrorKind::InvalidInput, "must pass --csv or --case-ids")
    })?;
    let raw = fs::read_to_string(csv)?;
    let mut lines = raw.lines();
    let header = lines
        .next()
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "empty CSV"))?;
    let cols: Vec<&str> = header.split(',').collect();
    let case_idx = cols
        .iter()
        .position(|c| *c == "case")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no `case` column"))?;
    let solved_idx = cols
        .iter()
        .position(|c| *c == "solved")
        .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "no `solved` column"))?;
    let mut out = Vec::new();
    for line in lines {
        let parts: Vec<&str> = line.split(',').collect();
        if parts.get(solved_idx) == Some(&"0")
            && let Some(case) = parts.get(case_idx)
        {
            out.push((*case).to_string());
        }
    }
    if let Some(n) = cfg.limit {
        out.truncate(n);
    }
    Ok(out)
}

/// Build a TAN WCS from the test case's truth (ra, dec, plate_scale)
/// using the supplied CD layout. The eight standard FITS orientations
/// are: 4 diagonal sign combinations × 2 swap-axes.
fn truth_wcs_with_cd(tc: &TestCase, cd: [[f64; 2]; 2]) -> TanWcs {
    TanWcs {
        crval: [tc.ra_deg.to_radians(), tc.dec_deg.to_radians()],
        crpix: [tc.image_width / 2.0, tc.image_height / 2.0],
        cd,
        image_size: [tc.image_width, tc.image_height],
    }
}

/// Default WCS — empirically determined to match the
/// `focalplane motion_simulator` renderer (see `--probe-cd`):
/// `cd = [[+scale, 0], [0, -scale]]`. Image y-axis is "down" with
/// north at top, x-axis is "right" with east toward higher pixel x
/// in the tangent-plane intermediate coordinates.
fn truth_wcs(tc: &TestCase) -> TanWcs {
    let scale_rad = tc.plate_scale_arcsec * std::f64::consts::PI / (180.0 * 3600.0);
    truth_wcs_with_cd(tc, [[scale_rad, 0.0], [0.0, -scale_rad]])
}

fn triage_case(
    bundle: &ZdclBundle,
    case_id: &str,
    tc: &TestCase,
    cfg: &BenchTriageConfig,
) -> io::Result<TriageRow> {
    let region = SkyRegion::from_degrees(
        Equatorial::new(tc.ra_deg.to_radians(), tc.dec_deg.to_radians()),
        cfg.radius_deg,
    );
    let multi = bundle.load_region(&region)?;
    let n_catalog_in_region = multi.gaia_records.len();

    let wcs = truth_wcs(tc);
    let w = tc.image_width;
    let h = tc.image_height;

    // Project every catalog star into image pixels via truth WCS.
    // None ⇒ behind the tangent plane; out-of-bounds ⇒ off-image.
    let projections: Vec<Option<(f64, f64)>> = multi
        .gaia_records
        .iter()
        .map(|g| wcs.radec_to_pixel(g.ra.to_radians(), g.dec.to_radians()))
        .collect();
    let n_catalog_on_image = projections
        .iter()
        .filter(|p| match p {
            Some((x, y)) => *x >= 0.0 && *x < w && *y >= 0.0 && *y < h,
            None => false,
        })
        .count();

    // Top-N brightest detected sources (mirrors solver_cfg.max_field_stars).
    let mut top: Vec<TestCaseSource> = tc.sources.clone();
    top.sort_by(|a, b| {
        b.flux
            .partial_cmp(&a.flux)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    top.truncate(cfg.max_field_stars);
    let n_top_field = top.len();

    // For each top detection, find nearest catalog projection.
    let mut residuals_px = Vec::with_capacity(top.len());
    let mut n_match_3 = 0usize;
    let mut n_match_10 = 0usize;
    for src in &top {
        let mut best = f64::INFINITY;
        for &(px, py) in projections.iter().flatten() {
            let dx = px - src.x;
            let dy = py - src.y;
            let d = (dx * dx + dy * dy).sqrt();
            if d < best {
                best = d;
            }
        }
        residuals_px.push(best);
        if best < 3.0 {
            n_match_3 += 1;
        }
        if best < 10.0 {
            n_match_10 += 1;
        }
    }
    let median_top_residual_px = percentile(&mut residuals_px.clone(), 0.5);
    let p95_top_residual_px = percentile(&mut residuals_px.clone(), 0.95);

    // Build the set of catalog-star indices that project within
    // match_radius_pix of *some* top-detected source. A "findable
    // quad" needs all four star_ids in this set, with all four
    // projections inside the image bounds.
    let mut covered: HashSet<usize> = HashSet::new();
    for (i, proj) in projections.iter().enumerate() {
        let (px, py) = match proj {
            Some(p) => *p,
            None => continue,
        };
        if !(px >= 0.0 && px < w && py >= 0.0 && py < h) {
            continue;
        }
        // Inside the image — check against the top-N detections.
        for src in &top {
            let dx = px - src.x;
            let dy = py - src.y;
            if dx * dx + dy * dy <= cfg.match_radius_pix * cfg.match_radius_pix {
                covered.insert(i);
                break;
            }
        }
    }

    // Walk every per-band fragment. For each quad, its 4 star_ids
    // index into `multi.gaia_records` (per the bundle reader's
    // remapping contract). A quad is "findable" iff all 4 are in
    // `covered`. Separately count quads whose 4 stars all project
    // on-image (regardless of detection match) — useful sanity check.
    let mut n_quads_in_region = 0usize;
    let mut n_quads_on_image = 0usize;
    let mut n_findable_quads = 0usize;
    let mut per_band: Vec<usize> = Vec::with_capacity(multi.fragments.len());
    for frag in &multi.fragments {
        let mut band_findable = 0usize;
        for q in &frag.quads {
            n_quads_in_region += 1;
            let star_ids = &q.star_ids;
            let mut on_image = true;
            for &sid in star_ids {
                let proj = projections[sid];
                let (px, py) = match proj {
                    Some(p) => p,
                    None => {
                        on_image = false;
                        break;
                    }
                };
                if !(px >= 0.0 && px < w && py >= 0.0 && py < h) {
                    on_image = false;
                    break;
                }
            }
            if !on_image {
                continue;
            }
            n_quads_on_image += 1;
            if star_ids.iter().all(|&sid| covered.contains(&sid)) {
                n_findable_quads += 1;
                band_findable += 1;
            }
        }
        per_band.push(band_findable);
    }

    let findable_per_band = per_band
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>()
        .join("|");

    Ok(TriageRow {
        case: case_id.to_string(),
        n_detected: tc.sources.len(),
        n_top_field,
        n_catalog_in_region,
        n_catalog_on_image,
        n_top_with_match_3px: n_match_3,
        n_top_with_match_10px: n_match_10,
        median_top_residual_px,
        p95_top_residual_px,
        n_quads_in_region,
        n_quads_on_image,
        n_findable_quads,
        findable_per_band,
    })
}

/// Sweep the 8 standard CD orientations against each case and print
/// how many of the brightest 50 detected sources match a catalog
/// projection within 3 px (and 10 px). The winner is the renderer's
/// WCS convention.
fn probe_cd_orientations(
    bundle: &ZdclBundle,
    case_ids: &[String],
    cfg: &BenchTriageConfig,
) -> io::Result<()> {
    println!("case,variant,cd00,cd01,cd10,cd11,n_match_3px,n_match_10px,median_resid_px");
    for id in case_ids {
        let path = cfg.test_cases_dir.join(format!("{id}.json"));
        let raw = fs::read_to_string(&path)?;
        let tc: TestCase = serde_json::from_str(&raw).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("{}: {e}", path.display()),
            )
        })?;
        let region = SkyRegion::from_degrees(
            Equatorial::new(tc.ra_deg.to_radians(), tc.dec_deg.to_radians()),
            cfg.radius_deg,
        );
        let multi = bundle.load_region(&region)?;
        let mut top: Vec<TestCaseSource> = tc.sources.clone();
        top.sort_by(|a, b| {
            b.flux
                .partial_cmp(&a.flux)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        top.truncate(cfg.max_field_stars);

        let s = tc.plate_scale_arcsec * std::f64::consts::PI / (180.0 * 3600.0);
        // 8 standard orientations: 4 diagonal sign combos × 2 axis swaps.
        let variants: [(&str, [[f64; 2]; 2]); 8] = [
            ("++_diag", [[s, 0.0], [0.0, s]]),
            ("+-_diag", [[s, 0.0], [0.0, -s]]),
            ("-+_diag", [[-s, 0.0], [0.0, s]]),
            ("--_diag", [[-s, 0.0], [0.0, -s]]),
            ("++_swap", [[0.0, s], [s, 0.0]]),
            ("+-_swap", [[0.0, s], [-s, 0.0]]),
            ("-+_swap", [[0.0, -s], [s, 0.0]]),
            ("--_swap", [[0.0, -s], [-s, 0.0]]),
        ];

        for (name, cd) in variants {
            let wcs = truth_wcs_with_cd(&tc, cd);
            let projections: Vec<Option<(f64, f64)>> = multi
                .gaia_records
                .iter()
                .map(|g| wcs.radec_to_pixel(g.ra.to_radians(), g.dec.to_radians()))
                .collect();
            let mut residuals: Vec<f64> = Vec::with_capacity(top.len());
            let (mut n3, mut n10) = (0usize, 0usize);
            for src in &top {
                let mut best = f64::INFINITY;
                for &(px, py) in projections.iter().flatten() {
                    let dx = px - src.x;
                    let dy = py - src.y;
                    let d2 = dx * dx + dy * dy;
                    if d2 < best {
                        best = d2;
                    }
                }
                let d = best.sqrt();
                residuals.push(d);
                if d < 3.0 {
                    n3 += 1;
                }
                if d < 10.0 {
                    n10 += 1;
                }
            }
            let med = percentile(&mut residuals.clone(), 0.5);
            println!(
                "{},{},{:.3e},{:.3e},{:.3e},{:.3e},{},{},{:.2}",
                id, name, cd[0][0], cd[0][1], cd[1][0], cd[1][1], n3, n10, med
            );
        }
    }
    Ok(())
}

fn percentile(v: &mut [f64], p: f64) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let n = v.len();
    let idx = ((p * (n as f64 - 1.0)).round() as usize).min(n - 1);
    v[idx]
}
