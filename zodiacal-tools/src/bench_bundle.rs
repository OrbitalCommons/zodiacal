//! `bench-bundle` — sweep the 1000-case test corpus against a bundle.
//!
//! For each `NNNN.json` test case, opens a region of the bundle around
//! the truth center, builds one `Index` per band, and runs `solve()`.
//! Records solve success, wall-clock load + solve time, and angular
//! error vs. truth. Emits per-case CSV on stdout and a summary on
//! stderr.

use std::fs;
use std::io;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};
use starfield::Equatorial;
use starfield::time::{Time, Timescale};

use zodiacal::bundle::reader::ZdclBundle;
use zodiacal::extraction::DetectedSource;
use zodiacal::geom::sphere::{angular_distance, radec_to_xyz};
use zodiacal::index::Index;
use zodiacal::solver::{SkyRegion, Solution, SolverConfig, solve};
use zodiacal::verify::VerifyConfig;

#[derive(Debug, Deserialize)]
struct TestCase {
    image_width: f64,
    image_height: f64,
    ra_deg: f64,
    dec_deg: f64,
    plate_scale_arcsec: f64,
    sources: Vec<TestCaseSource>,
    #[serde(default)]
    hst: Option<HstBlock>,
}

#[derive(Debug, Deserialize)]
struct HstBlock {
    /// Exposure start in MJD (Modified Julian Date).
    #[serde(default)]
    t_min_mjd: Option<f64>,
    /// Exposure end in MJD.
    #[serde(default)]
    t_max_mjd: Option<f64>,
}

#[derive(Debug, Deserialize)]
struct TestCaseSource {
    x: f64,
    y: f64,
    flux: f64,
}

/// If the test case carries an `hst` block with MJD timestamps, return
/// the exposure midpoint as a `Time` suitable for PM propagation.
/// Returns `None` for synthetic test corpora.
///
/// MJD → JD: `jd = mjd + 2400000.5`. We then hand that to
/// [`Timescale::tt_jd`] which carries the TT semantics through. The
/// `Time` is only ever consumed by [`zodiacal::geom::sphere::propagate_pm`]
/// via `Time::j()` (TT Julian decimal year) — pure arithmetic on the
/// JD, no leap-second / delta-T table lookup — so an empty default
/// `Timescale` is sufficient.
fn obs_epoch_from_test_case(tc: &TestCase, timescale: &Timescale) -> Option<Time> {
    let hst = tc.hst.as_ref()?;
    let mid_mjd = match (hst.t_min_mjd, hst.t_max_mjd) {
        (Some(a), Some(b)) => 0.5 * (a + b),
        (Some(a), None) | (None, Some(a)) => a,
        (None, None) => return None,
    };
    Some(timescale.tt_jd(mid_mjd + 2400000.5, None))
}

pub struct BenchBundleConfig {
    pub bundle_path: PathBuf,
    pub test_cases_dir: PathBuf,
    /// Region radius in degrees. For a 2°×2° square, pass `√2 ≈ 1.4142`.
    pub radius_deg: f64,
    /// If set, stop after this many cases.
    pub limit: Option<usize>,
    /// If true, hint the solver with truth pixel scale ±5%.
    pub scale_hint: bool,
    /// Per-case solve timeout (seconds). 0 disables.
    pub timeout_secs: u64,
    /// Optional directory for per-case `<case>.trace.json` sidecars.
    /// When set and a case solves, the full solver trace is dumped:
    /// solved WCS, the matched quad's 4 (field, catalog) pairs, and the
    /// verification matched-pair list. Useful for visualising hits/
    /// misses against the source image.
    pub trace_out: Option<PathBuf>,
    /// Observation epoch override. When `None`, the harness auto-fills
    /// from the test case's `hst.t_min_mjd`/`t_max_mjd` midpoint (HST
    /// cases) and leaves it as `None` for synthetic corpora.
    pub obs_epoch: Option<Time>,
}

pub fn run(cfg: &BenchBundleConfig) -> io::Result<()> {
    let bundle = ZdclBundle::open(&cfg.bundle_path)?;
    let n_bands = bundle.bands().len();
    eprintln!(
        "Opened bundle: cell_depth={} bands={} populated={}",
        bundle.cell_depth(),
        n_bands,
        bundle.manifest().gaia.populated_cells,
    );

    let mut paths: Vec<PathBuf> = fs::read_dir(&cfg.test_cases_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| p.extension().and_then(|s| s.to_str()) == Some("json"))
        .collect();
    paths.sort();
    if let Some(n) = cfg.limit {
        paths.truncate(n);
    }
    // A single Timescale used to materialise every per-case `Time` —
    // leap-second / delta-T tables are not needed (see
    // `zodiacal::index::default_timescale`).
    let timescale = Timescale::default();

    eprintln!(
        "Running {} test case(s) at radius {:.3}°, scale_hint={}",
        paths.len(),
        cfg.radius_deg,
        cfg.scale_hint,
    );

    println!(
        "case,solved,solve_ms,load_ms,error_arcsec,solved_ra_deg,solved_dec_deg,n_total_quads,n_total_stars,n_verified,best_rejected_log_odds,best_rejected_n_matched,best_rejected_error_arcsec,timed_out"
    );

    let mut n_solved = 0usize;
    let mut total_load_ms = 0.0f64;
    let mut total_solve_ms = 0.0f64;
    let bench_start = Instant::now();

    for (i, p) in paths.iter().enumerate() {
        let raw = fs::read_to_string(p)?;
        let tc: TestCase = serde_json::from_str(&raw).map_err(|e| {
            io::Error::new(io::ErrorKind::InvalidData, format!("{}: {e}", p.display()))
        })?;

        let region = SkyRegion::from_degrees(
            Equatorial::new(tc.ra_deg.to_radians(), tc.dec_deg.to_radians()),
            cfg.radius_deg,
        );

        let load_start = Instant::now();
        let multi = bundle.load_region(&region)?;
        let indexes: Vec<Index> = multi.fragments.into_iter().map(Index::from).collect();
        let index_refs: Vec<&Index> = indexes.iter().collect();
        let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;

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
        // PM propagation: explicit --obs-epoch wins, otherwise auto-fill
        // from the HST exposure midpoint when the case carries one.
        // Synthetic cases at Gaia epoch leave obs_epoch = None (identity).
        solver_cfg.obs_epoch = cfg
            .obs_epoch
            .clone()
            .or_else(|| obs_epoch_from_test_case(&tc, &timescale));
        if cfg.scale_hint {
            // ±25% tolerance — wide enough to absorb sub-pixel residuals
            // and the scale-vs-band-edge interaction without destroying
            // the filter's discriminating power.
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

        let (solved, error_arcsec, solved_ra_deg, solved_dec_deg) = match &solution {
            Some(sol) => {
                let (got_ra, got_dec) = sol.wcs.field_center();
                (
                    true,
                    wcs_error_arcsec(&sol.wcs),
                    got_ra.to_degrees(),
                    got_dec.to_degrees(),
                )
            }
            None => (false, f64::NAN, f64::NAN, f64::NAN),
        };

        let (best_rej_log_odds, best_rej_n_matched, best_rej_error) =
            match (stats.best_rejected, stats.best_rejected_wcs.as_ref()) {
                (Some((lo, nm)), Some(wcs)) => (lo, nm as i64, wcs_error_arcsec(wcs)),
                _ => (f64::NAN, -1i64, f64::NAN),
            };

        let total_quads: usize = indexes.iter().map(|i| i.quads.len()).sum();
        let total_stars: usize = indexes.iter().map(|i| i.stars.len()).sum();

        if solved {
            n_solved += 1;
        }
        total_load_ms += load_ms;
        total_solve_ms += solve_ms;

        let case_name = p.file_stem().unwrap().to_string_lossy();

        // Optional per-case trace JSON sidecar. Dumped only on a
        // successful solve; the renderer uses it to draw the matched
        // quad and verification hits/misses.
        if let (Some(out_dir), Some(sol)) = (cfg.trace_out.as_ref(), solution.as_ref()) {
            std::fs::create_dir_all(out_dir)?;
            let trace_path = out_dir.join(format!("{case_name}.trace.json"));
            write_trace(&trace_path, &case_name, &tc, sol, &sources, &indexes)?;
        }
        let fmt_f = |v: f64| -> String {
            if v.is_nan() {
                String::new()
            } else {
                format!("{:.3}", v)
            }
        };
        let fmt_i = |v: i64| -> String { if v < 0 { String::new() } else { v.to_string() } };
        let fmt_deg = |v: f64| -> String {
            if v.is_nan() {
                String::new()
            } else {
                format!("{:.6}", v)
            }
        };
        println!(
            "{},{},{:.1},{:.1},{},{},{},{},{},{},{},{},{},{}",
            case_name,
            solved as u8,
            solve_ms,
            load_ms,
            fmt_f(error_arcsec),
            fmt_deg(solved_ra_deg),
            fmt_deg(solved_dec_deg),
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
                "  [{:>4}/{}] solved={} ({:.1}%) load_avg={:.0}ms solve_avg={:.0}ms",
                i + 1,
                paths.len(),
                n_solved,
                100.0 * n_solved as f64 / (i + 1) as f64,
                total_load_ms / (i + 1) as f64,
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
    eprintln!("Avg load:     {:.1} ms", total_load_ms / n);
    eprintln!("Avg solve:    {:.1} ms", total_solve_ms / n);
    eprintln!("Wall-clock:   {:.1} s", total_elapsed);

    Ok(())
}

/// Per-case solver trace dumped to `<trace_out>/<case>.trace.json`
/// when `--trace-out` is passed and the case solves. The schema is
/// stable enough for downstream renderers to consume directly.
#[derive(Serialize)]
struct TraceFile<'a> {
    case: &'a str,
    image_width: f64,
    image_height: f64,
    truth: TraceCenter,
    solved: TraceWcs,
    quad: TraceQuad,
    verification: TraceVerification,
}

#[derive(Serialize)]
struct TraceCenter {
    ra_deg: f64,
    dec_deg: f64,
    plate_scale_arcsec: f64,
}

#[derive(Serialize)]
struct TraceWcs {
    ra_deg: f64,
    dec_deg: f64,
    crpix: [f64; 2],
    cd: [[f64; 2]; 2],
    image_size: [f64; 2],
}

#[derive(Serialize)]
struct TraceQuad {
    /// Position of the matched index in the band slice (= band index
    /// for a multi-band bundle).
    band_idx: usize,
    /// 4 detected-source indices (into the test-case `sources` list,
    /// 0-based, in the brightness order the bench passes to `solve`).
    field_indices: [usize; 4],
    /// 4 catalog-star pixel positions (post-fit) and sky positions.
    catalog: [TraceCatalogStar; 4],
}

#[derive(Serialize)]
struct TraceCatalogStar {
    /// Pixel position of the catalog star projected through the
    /// solver's WCS — what the matched quad's vertex looks like on
    /// the image after the fit.
    px: f64,
    py: f64,
    ra_deg: f64,
    dec_deg: f64,
}

#[derive(Serialize)]
struct TraceVerification {
    log_odds: f64,
    n_matched: usize,
    n_distractor: usize,
    /// One entry per accepted match during verification: the field
    /// source index, the catalog star's projected pixel position, and
    /// its sky position.
    matches: Vec<TraceMatch>,
}

#[derive(Serialize)]
struct TraceMatch {
    field_idx: usize,
    px: f64,
    py: f64,
    ra_deg: f64,
    dec_deg: f64,
}

fn write_trace(
    path: &Path,
    case: &str,
    tc: &TestCase,
    sol: &Solution,
    _sources: &[DetectedSource],
    indexes: &[Index],
) -> io::Result<()> {
    let (got_ra, got_dec) = sol.wcs.field_center();
    let band = &indexes[sol.quad_match.index_id];
    let project = |ra: f64, dec: f64| -> (f64, f64) {
        sol.wcs
            .radec_to_pixel(ra, dec)
            .unwrap_or((f64::NAN, f64::NAN))
    };

    let catalog: [TraceCatalogStar; 4] = std::array::from_fn(|i| {
        let s = &band.stars[sol.quad_match.index_indices[i]];
        let (px, py) = project(s.position.ra, s.position.dec);
        TraceCatalogStar {
            px,
            py,
            ra_deg: s.position.ra.to_degrees(),
            dec_deg: s.position.dec.to_degrees(),
        }
    });

    let matches: Vec<TraceMatch> = sol
        .verify_result
        .matched_pairs
        .iter()
        .map(|&(field_idx, ref_idx)| {
            let s = &band.stars[ref_idx];
            let (px, py) = project(s.position.ra, s.position.dec);
            TraceMatch {
                field_idx,
                px,
                py,
                ra_deg: s.position.ra.to_degrees(),
                dec_deg: s.position.dec.to_degrees(),
            }
        })
        .collect();

    let trace = TraceFile {
        case,
        image_width: tc.image_width,
        image_height: tc.image_height,
        truth: TraceCenter {
            ra_deg: tc.ra_deg,
            dec_deg: tc.dec_deg,
            plate_scale_arcsec: tc.plate_scale_arcsec,
        },
        solved: TraceWcs {
            ra_deg: got_ra.to_degrees(),
            dec_deg: got_dec.to_degrees(),
            crpix: sol.wcs.crpix,
            cd: sol.wcs.cd,
            image_size: sol.wcs.image_size,
        },
        quad: TraceQuad {
            band_idx: sol.quad_match.index_id,
            field_indices: sol.quad_match.field_indices,
            catalog,
        },
        verification: TraceVerification {
            log_odds: sol.verify_result.log_odds,
            n_matched: sol.verify_result.n_matched,
            n_distractor: sol.verify_result.n_distractor,
            matches,
        },
    };

    let json = serde_json::to_string_pretty(&trace).map_err(io::Error::other)?;
    std::fs::write(path, json)?;
    Ok(())
}
