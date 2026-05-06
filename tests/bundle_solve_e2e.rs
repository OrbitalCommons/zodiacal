//! End-to-end integration test for the bundle pipeline.
//!
//! Builds a tiny synthetic bundle from in-memory stars, packages it via
//! the tidy phase, opens it through `ZdclBundle::open`, and runs
//! `solve()` against synthesized field detections at the bundle's
//! known star positions. This exercises the full
//! build → tidy → reader → solver path that unit tests cover only in
//! isolation, and would have caught (and now regression-guards) the
//! quad-vs-gaia ordering bug fixed alongside this test.

use std::f64::consts::PI;
use std::path::Path;

use chrono::DateTime;
use starfield::Equatorial;
use tempfile::TempDir;

use zodiacal::bundle::gaia_shard::GaiaRecord;
use zodiacal::bundle::manifest::{BuildMetadata, BuildSource};
use zodiacal::bundle::reader::ZdclBundle;
use zodiacal::bundle::tidy::{BandMetadata, GaiaMetadata, TidyMetadata, tidy_to_folder};
use zodiacal::extraction::DetectedSource;
use zodiacal::geom::sphere::{angular_distance, radec_to_xyz};
use zodiacal::geom::tan::TanWcs;
use zodiacal::index::Index;
use zodiacal::index::cell_builder::{CellStar, CellStarSource};
use zodiacal::index::multiband_cell_builder::{
    BundleWorkDirPaths, MultiBandCellBuildConfig, ScaleBand, build_bundle_work_dir,
};
use zodiacal::refinement::SidecarRecord;
use zodiacal::solver::{SkyRegion, SolverConfig, solve};
use zodiacal::verify::VerifyConfig;

const TEST_DEPTH: u8 = 5;
const N_POINTINGS: usize = 8;
const STARS_PER_CELL: usize = 200;
const PIXEL_SCALE_ARCSEC: f64 = 4.0; // 4″/px → 2048″ FOV at 512px
const IMAGE_SIZE: (f64, f64) = (512.0, 512.0);

/// xorshift PRNG — deterministic, no-deps.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() as f64) / (u64::MAX as f64)
    }
}

fn make_gaia(source_id: u64, ra_deg: f64, dec_deg: f64, mag: f64) -> GaiaRecord {
    GaiaRecord {
        source_id,
        ref_epoch: 2016.0,
        ra: ra_deg,
        dec: dec_deg,
        pmra: 0.0,
        pmdec: 0.0,
        parallax: 0.0,
        radial_velocity: f64::NAN,
        phot_g_mean_mag: mag,
        sigma_ra: 0.1,
        sigma_dec: 0.1,
        sigma_pmra: 0.0,
        sigma_pmdec: 0.0,
        sigma_parallax: 0.0,
        ra_dec_corr: f32::NAN,
        ruwe: f32::NAN,
        flags: 0,
    }
}

fn make_cell_star(source_id: u64, ra_rad: f64, dec_rad: f64, mag: f64) -> CellStar {
    CellStar {
        catalog_id: source_id,
        ra_rad,
        dec_rad,
        mag,
        sidecar: SidecarRecord {
            source_id,
            ref_epoch: 2016.0,
            ra: ra_rad.to_degrees(),
            dec: dec_rad.to_degrees(),
            pmra: 0.0,
            pmdec: 0.0,
            parallax: 0.0,
            radial_velocity: f64::NAN,
            sigma_ra: 0.1,
            sigma_dec: 0.1,
            sigma_pmra: 0.0,
            sigma_pmdec: 0.0,
            sigma_parallax: 0.0,
            flags: 0,
        },
        gaia: make_gaia(source_id, ra_rad.to_degrees(), dec_rad.to_degrees(), mag),
    }
}

/// Synthetic source: each cell in `cells` is populated with a dense
/// patch of `stars_per` stars uniformly distributed within ~0.5° of the
/// cell center. Stars near cell boundaries may legitimately spill into
/// neighbors — the test only requires the listed cells to have stars.
struct ClusteredSource {
    cells: Vec<u32>,
    stars_per_cell: Vec<Vec<CellStar>>,
}

impl ClusteredSource {
    fn new(rng: &mut Rng, cells: Vec<u32>, stars_per: usize) -> Self {
        let mut payload: Vec<Vec<CellStar>> = Vec::with_capacity(cells.len());
        for &cell_id in &cells {
            let (ra_c, dec_c) = cdshealpix::nested::center(TEST_DEPTH, cell_id as u64);
            let mut stars = Vec::with_capacity(stars_per);
            for _ in 0..stars_per {
                // Disk of 0.7° radius on the tangent plane around the
                // cell center, then back to spherical. 0.7° is just
                // under a depth-5 cell's radius (cell area = 3.4 sq.deg
                // ≈ 1.04° equivalent disk).
                let r = rng.next_f64().sqrt() * 0.7_f64.to_radians();
                let theta = rng.next_f64() * 2.0 * PI;
                let dra_tan = r * theta.cos();
                let ddec_tan = r * theta.sin();
                let dec = (dec_c + ddec_tan).clamp(-PI / 2.0, PI / 2.0);
                let cos_dec = dec_c.cos().max(0.05);
                let ra = (ra_c + dra_tan / cos_dec).rem_euclid(2.0 * PI);
                let mag = 8.0 + rng.next_f64() * 4.0;
                // Fully-random source_ids so the on-disk source_id-sort
                // permutation is non-trivial. Sequential ids would let
                // a quad-vs-gaia ordering bug masquerade as correct
                // (the identity permutation hides it). Real Gaia ids
                // are hashed in their high bits and arrive in
                // pseudo-random order, so this matches production.
                let source_id = rng.next_u64();
                stars.push(make_cell_star(source_id, ra, dec, mag));
            }
            payload.push(stars);
        }
        Self {
            cells,
            stars_per_cell: payload,
        }
    }

    fn stars_at_cell(&self, cell_id: u32) -> Option<&[CellStar]> {
        self.cells
            .iter()
            .position(|&c| c == cell_id)
            .map(|idx| self.stars_per_cell[idx].as_slice())
    }
}

impl CellStarSource for ClusteredSource {
    fn cell_count(&self) -> u32 {
        // Iterate `0..max_populated + 1` only — the orchestrator
        // fsyncs the manifest per empty cell, so keep this tight.
        self.cells.iter().copied().max().unwrap_or(0) + 1
    }
    fn stars_in_cell(&self, cell_id: u32) -> std::io::Result<Vec<CellStar>> {
        Ok(self
            .stars_at_cell(cell_id)
            .map(|s| s.to_vec())
            .unwrap_or_default())
    }
}

fn make_test_wcs(crval: [f64; 2], rotation_rad: f64) -> TanWcs {
    let scale = (PIXEL_SCALE_ARCSEC / 3600.0).to_radians();
    let c = rotation_rad.cos() * scale;
    let s = rotation_rad.sin() * scale;
    TanWcs {
        crval,
        crpix: [IMAGE_SIZE.0 / 2.0, IMAGE_SIZE.1 / 2.0],
        cd: [[c, -s], [s, c]],
        image_size: [IMAGE_SIZE.0, IMAGE_SIZE.1],
    }
}

fn synth_pointing(
    source: &ClusteredSource,
    cell_id: u32,
    rotation_rad: f64,
) -> (TanWcs, Vec<DetectedSource>) {
    // Pointing center is the cell's center.
    let (ra_c, dec_c) = cdshealpix::nested::center(TEST_DEPTH, cell_id as u64);
    let wcs = make_test_wcs([ra_c, dec_c], rotation_rad);

    // Project every star known to live in this cell. Keep only those
    // that land inside the image. Synthesize flux from magnitude
    // (brighter mag → higher flux).
    let mut sources = Vec::new();
    let stars = source.stars_at_cell(cell_id).expect("populated cell");
    for s in stars {
        if let Some((px, py)) = wcs.radec_to_pixel(s.ra_rad, s.dec_rad)
            && px >= 0.0
            && px < IMAGE_SIZE.0
            && py >= 0.0
            && py < IMAGE_SIZE.1
        {
            let flux = 10f64.powf(-0.4 * (s.mag - 12.0)); // 1.0 at mag 12
            sources.push(DetectedSource { x: px, y: py, flux });
        }
    }
    (wcs, sources)
}

fn band_set() -> Vec<ScaleBand> {
    // Bands sized to the synthetic FOV (2048″). Smallest band fits
    // tight clusters; largest covers nearly the whole image diagonal.
    [
        ("band_00", 50.0, 150.0, 400, 16),
        ("band_01", 150.0, 400.0, 400, 16),
        ("band_02", 400.0, 1000.0, 400, 16),
        ("band_03", 1000.0, 2000.0, 400, 16),
    ]
    .iter()
    .enumerate()
    .map(|(i, (label, lo, hi, qpc, mr))| ScaleBand {
        label: (*label).to_string(),
        band_idx: i as u32,
        scale_lower_arcsec: *lo,
        scale_upper_arcsec: *hi,
        quads_per_cell: *qpc,
        max_reuse: *mr,
    })
    .collect()
}

fn build_test_bundle(work_dir: &Path, output_dir: &Path, source: &ClusteredSource) {
    let bands = band_set();
    let cfg = MultiBandCellBuildConfig {
        bands: bands.clone(),
        max_stars_per_cell: 10_000,
        mag_limit: 16.0,
        cell_depth: TEST_DEPTH,
    };
    let paths = BundleWorkDirPaths {
        work_dir: work_dir.to_path_buf(),
    };
    build_bundle_work_dir(source, &cfg, &paths).expect("build_bundle_work_dir");

    let metadata = TidyMetadata {
        cell_depth: TEST_DEPTH,
        experiment: "bundle_solve_e2e".into(),
        build_metadata: BuildMetadata {
            tool: "bundle_solve_e2e test".into(),
            build_started_utc: "2026-05-06T00:00:00Z"
                .parse::<DateTime<chrono::Utc>>()
                .unwrap(),
            build_finished_utc: "2026-05-06T00:00:01Z"
                .parse::<DateTime<chrono::Utc>>()
                .unwrap(),
            source: BuildSource {
                kind: "synthetic".into(),
                release: "test".into(),
                path: "/dev/null".into(),
            },
        },
        gaia: GaiaMetadata {
            max_stars_per_cell: 10_000,
            mag_limit: 16.0,
            schema_version: 1,
        },
        bands: bands
            .iter()
            .map(|b| BandMetadata {
                label: b.label.clone(),
                scale_lower_arcsec: b.scale_lower_arcsec,
                scale_upper_arcsec: b.scale_upper_arcsec,
                quads_per_cell: b.quads_per_cell as u32,
                max_reuse: b.max_reuse as u32,
            })
            .collect(),
    };
    tidy_to_folder(work_dir, output_dir, &metadata).expect("tidy_to_folder");
}

#[test]
fn bundle_solves_synthetic_pointings() {
    let mut rng = Rng::new(0xCAFE_F00D);

    // Pick N_POINTINGS distinct, low-numbered cells so the build's
    // per-empty-cell manifest fsync stays cheap.
    let cells: Vec<u32> = (0..N_POINTINGS as u32).collect();
    let source = ClusteredSource::new(&mut rng, cells.clone(), STARS_PER_CELL);

    let work_tmp = TempDir::new().expect("work tmpdir");
    let out_tmp = TempDir::new().expect("out tmpdir");
    let bundle_path = out_tmp.path().join("test.zdcl.bundle");

    build_test_bundle(work_tmp.path(), &bundle_path, &source);

    let bundle = ZdclBundle::open(&bundle_path).expect("open bundle");
    assert_eq!(bundle.cell_depth(), TEST_DEPTH);
    assert_eq!(bundle.bands().len(), 4);

    let mut n_solved = 0usize;
    let mut per_case: Vec<(u32, usize, Option<f64>)> = Vec::new();

    for (i, &cell_id) in cells.iter().enumerate() {
        // Offset by half-step so we never hit rotation == 0 (axes
        // perfectly aligned with ICRS), which can produce degenerate
        // quad-code geometry that the canonical-code disambiguator
        // doesn't recover.
        let rotation = (i as f64 + 0.5) * (PI / 7.0);
        let (truth_wcs, sources) = synth_pointing(&source, cell_id, rotation);
        let n_in_fov = sources.len();

        let region =
            SkyRegion::from_degrees(Equatorial::new(truth_wcs.crval[0], truth_wcs.crval[1]), 1.5);
        let multi = bundle.load_region(&region).expect("load_region");
        let indexes: Vec<Index> = multi.fragments.into_iter().map(Index::from).collect();
        let index_refs: Vec<&Index> = indexes.iter().collect();

        let solver_cfg = SolverConfig {
            max_field_stars: 30,
            code_tolerance: 0.002,
            verify: VerifyConfig {
                match_radius_pix: 2.0,
                log_odds_accept: 25.0,
                min_matches: 8,
                ..VerifyConfig::default()
            },
            ..SolverConfig::default()
        };
        let (solution, stats) = solve(&sources, &index_refs, IMAGE_SIZE, &solver_cfg);

        let err = solution.map(|sol| {
            let (got_ra, got_dec) = sol.wcs.field_center();
            let truth_xyz = radec_to_xyz(truth_wcs.crval[0], truth_wcs.crval[1]);
            let got_xyz = radec_to_xyz(got_ra, got_dec);
            angular_distance(truth_xyz, got_xyz).to_degrees() * 3600.0
        });
        if matches!(err, Some(e) if e < 30.0) {
            n_solved += 1;
        }
        eprintln!(
            "  cell {:>5}: n_fov={} n_verified={} best_rej_log_odds={:?} best_rej_n_matched={:?} err={:?}",
            cell_id,
            n_in_fov,
            stats.n_verified,
            stats.best_rejected.map(|(lo, _)| lo),
            stats.best_rejected.map(|(_, n)| n),
            err,
        );
        per_case.push((cell_id, n_in_fov, err));
    }

    let n_attempted = cells.len();
    eprintln!(
        "bundle_solves_synthetic_pointings: {}/{}",
        n_solved, n_attempted
    );
    for (cell_id, n_in_fov, err) in &per_case {
        eprintln!(
            "  cell {:>5}: {} stars in FOV, err={}",
            cell_id,
            n_in_fov,
            match err {
                Some(e) => format!("{:.4} arcsec", e),
                None => "no solve".into(),
            },
        );
    }
    assert_eq!(
        n_solved, n_attempted,
        "expected all {} pointings to solve, got {}",
        n_attempted, n_solved,
    );
}
