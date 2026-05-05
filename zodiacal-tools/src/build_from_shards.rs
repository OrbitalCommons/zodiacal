//! `build-from-shards`: build a `.zdcl` index plus a refinement
//! `.zdcl.gaia` sidecar from a directory of Gaia DR3 CSV shards in
//! one pass.
//!
//! Reads every shard once, applies a magnitude limit, then builds the
//! quad index (v3 layout, HEALPix-grouped) and writes the sidecar
//! (v1 flat sorted-by-source_id binary).
//!
//! # Memory budget
//!
//! Each accepted star occupies ~120 bytes in memory (32 B index tuple +
//! 88 B sidecar record). At G ≤ 16 (~83 M stars) that's ≈ 10 GB and fits
//! in 64 GB hosts; at G ≤ 19 (~565 M stars) it's ≈ 70 GB and won't fit
//! on any sensible workstation. The tool refuses `--mag-limit > 17.0`
//! up front to avoid blowing the box.

use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;
use starfield_gaia::download::Downloader;
use starfield_gaia::{Dr3, Dr3Catalog, Dr3Entry};

use zodiacal::index::builder::{IndexBuilderConfig, build_index};
use zodiacal::refinement::{DEFAULT_PIVOT_STRIDE, SidecarRecord, write_sidecar};

/// Hard cap on the magnitude limit; deeper than this won't fit in
/// reasonable host RAM with the in-memory v1 sidecar contract.
pub const MAX_MAG_LIMIT: f64 = 17.0;

/// `(catalog_id, ra_radians, dec_radians, magnitude)` — the tuple shape
/// `build_index` expects.
pub type IndexStarTuple = (u64, f64, f64, f64);

/// Pair emitted per accepted Gaia row.
pub type ShardRow = (IndexStarTuple, SidecarRecord);

/// Find every `*.csv` or `*.csv.gz` in `dir`. Returns sorted paths.
///
/// Accepts both starfield-gaia's "excerpt" layout
/// (`shard_NNNN.csv.gz` under `~/.cache/starfield/gaia-excerpts/<name>/`)
/// and the raw ESA `GaiaSource_NNN-NNN-NNN.csv.gz` layout — the column
/// schema is identical in both cases, so `Dr3Catalog::from_csv_file`
/// reads either uniformly.
fn find_shard_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = Vec::new();
    for entry in std::fs::read_dir(dir)? {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let name = match path.file_name().and_then(|n| n.to_str()) {
            Some(n) => n,
            None => continue,
        };
        if name.ends_with(".csv.gz") || name.ends_with(".csv") {
            files.push(path);
        }
    }
    files.sort();
    Ok(files)
}

/// Convert a single `Dr3Entry` to (index tuple, sidecar record).
///
/// `f64::NAN` is used for unpublished optional astrometry, matching the
/// `SidecarRecord` schema. Sigmas (which `GaiaCore` stores as `f32`) are
/// passed through; `NaN` for unpublished optional sigmas.
pub fn entry_to_records(entry: &Dr3Entry) -> ShardRow {
    let core = &entry.core;
    let ra_rad = core.ra.to_radians();
    let dec_rad = core.dec.to_radians();
    let mag = core.phot_g_mean_mag;

    let radial_velocity = entry
        .radial_velocity
        .as_ref()
        .and_then(|rv| rv.radial_velocity)
        .unwrap_or(f64::NAN);

    let record = SidecarRecord {
        source_id: core.source_id,
        ref_epoch: core.ref_epoch,
        ra: core.ra,
        dec: core.dec,
        pmra: core.pmra.unwrap_or(f64::NAN),
        pmdec: core.pmdec.unwrap_or(f64::NAN),
        parallax: core.parallax.unwrap_or(f64::NAN),
        radial_velocity,
        sigma_ra: core.ra_error,
        sigma_dec: core.dec_error,
        sigma_pmra: core.pmra_error.unwrap_or(f32::NAN),
        sigma_pmdec: core.pmdec_error.unwrap_or(f32::NAN),
        sigma_parallax: core.parallax_error.unwrap_or(f32::NAN),
        flags: 0,
    };

    ((core.source_id, ra_rad, dec_rad, mag), record)
}

/// Read one shard via `Dr3Catalog::from_csv_file` (the canonical
/// high-level reader for Gaia DR3 CSV shards) and return all kept
/// stars as (tuple, record) pairs.
fn read_shard(path: &Path, mag_limit: f64) -> Result<Vec<ShardRow>, String> {
    let catalog = Dr3Catalog::from_csv_file(path, mag_limit)
        .map_err(|e| format!("{}: load failed: {e}", path.display()))?;

    // `brighter_than_ref(f64::INFINITY)` is the only inherent (non-trait)
    // accessor on `GaiaCatalogBase` that yields `&R::Entry` for every
    // loaded row — using `StarCatalog::stars()` would require both
    // crates to agree on a single `starfield` instance, which they
    // don't (starfield-gaia is git-pinned, zodiacal pulls from
    // crates.io).
    let entries = catalog.0.brighter_than_ref(f64::INFINITY);
    let mut out = Vec::with_capacity(entries.len());
    for entry in entries {
        // mag_limit is already applied at load time, but defensively
        // skip NaN mags — a NaN here would corrupt the brightness sort.
        if !entry.core.phot_g_mean_mag.is_finite() {
            continue;
        }
        out.push(entry_to_records(entry));
    }
    Ok(out)
}

/// Validated configuration for [`run`].
///
/// Construct via [`BuildFromShardsConfig::builder`]. Fields are
/// private so the only way to obtain a value is through the builder,
/// which validates on `build()` — `run` then trusts every field.
///
/// `shards_dir = None` tells the tool to use the shard set cached by
/// `starfield-gaia` (i.e. files under `~/.cache/starfield/gaia/dr3/`).
#[derive(Debug)]
pub struct BuildFromShardsConfig {
    shards_dir: Option<PathBuf>,
    output_prefix: PathBuf,
    mag_limit: f64,
    scale_lower_arcsec: f64,
    scale_upper_arcsec: f64,
    max_quads: usize,
    cell_depth: u8,
    threads: Option<usize>,
}

impl BuildFromShardsConfig {
    /// Begin assembling a config; call [`BuildFromShardsConfigBuilder::build`]
    /// to validate and finalise.
    pub fn builder() -> BuildFromShardsConfigBuilder {
        BuildFromShardsConfigBuilder::default()
    }
}

/// Fluent builder for [`BuildFromShardsConfig`]. Required fields
/// (`output_prefix`, `mag_limit`, `scale_lower_arcsec`,
/// `scale_upper_arcsec`, `max_quads`, `cell_depth`) error out at
/// `build()` if not set; optional fields (`shards_dir`, `threads`)
/// default to `None`.
#[derive(Default)]
pub struct BuildFromShardsConfigBuilder {
    shards_dir: Option<PathBuf>,
    output_prefix: Option<PathBuf>,
    mag_limit: Option<f64>,
    scale_lower_arcsec: Option<f64>,
    scale_upper_arcsec: Option<f64>,
    max_quads: Option<usize>,
    cell_depth: Option<u8>,
    threads: Option<usize>,
}

impl BuildFromShardsConfigBuilder {
    pub fn shards_dir(mut self, v: Option<PathBuf>) -> Self {
        self.shards_dir = v;
        self
    }
    pub fn output_prefix(mut self, v: PathBuf) -> Self {
        self.output_prefix = Some(v);
        self
    }
    pub fn mag_limit(mut self, v: f64) -> Self {
        self.mag_limit = Some(v);
        self
    }
    pub fn scale_lower_arcsec(mut self, v: f64) -> Self {
        self.scale_lower_arcsec = Some(v);
        self
    }
    pub fn scale_upper_arcsec(mut self, v: f64) -> Self {
        self.scale_upper_arcsec = Some(v);
        self
    }
    pub fn max_quads(mut self, v: usize) -> Self {
        self.max_quads = Some(v);
        self
    }
    pub fn cell_depth(mut self, v: u8) -> Self {
        self.cell_depth = Some(v);
        self
    }
    pub fn threads(mut self, v: Option<usize>) -> Self {
        self.threads = v;
        self
    }

    /// Validate fields and produce a [`BuildFromShardsConfig`]. The
    /// checks here used to live at the top of `run`; lifting them
    /// into the builder means a constructed config is always usable.
    pub fn build(self) -> Result<BuildFromShardsConfig, String> {
        let output_prefix = self
            .output_prefix
            .ok_or_else(|| "output_prefix is required".to_string())?;
        let mag_limit = self
            .mag_limit
            .ok_or_else(|| "mag_limit is required".to_string())?;
        let scale_lower_arcsec = self
            .scale_lower_arcsec
            .ok_or_else(|| "scale_lower_arcsec is required".to_string())?;
        let scale_upper_arcsec = self
            .scale_upper_arcsec
            .ok_or_else(|| "scale_upper_arcsec is required".to_string())?;
        let max_quads = self
            .max_quads
            .ok_or_else(|| "max_quads is required".to_string())?;
        let cell_depth = self
            .cell_depth
            .ok_or_else(|| "cell_depth is required".to_string())?;

        if !mag_limit.is_finite() || mag_limit > MAX_MAG_LIMIT {
            return Err(format!(
                "--mag-limit {mag_limit} exceeds hard cap {MAX_MAG_LIMIT}; v1 sidecar writer holds \
                 all records in RAM, and going deeper than ~17 won't fit on a 64 GB box. Either \
                 tighten the limit or wait for the streaming sidecar writer.",
            ));
        }
        if scale_lower_arcsec <= 0.0 || scale_upper_arcsec <= scale_lower_arcsec {
            return Err(format!(
                "invalid scale range: lower={scale_lower_arcsec}\" upper={scale_upper_arcsec}\" \
                 (must satisfy 0 < lower < upper)",
            ));
        }
        if max_quads == 0 {
            return Err("max_quads must be positive".into());
        }

        Ok(BuildFromShardsConfig {
            shards_dir: self.shards_dir,
            output_prefix,
            mag_limit,
            scale_lower_arcsec,
            scale_upper_arcsec,
            max_quads,
            cell_depth,
            threads: self.threads,
        })
    }
}

/// Top-level entry point. Trusts the config — validation happened at
/// `BuildFromShardsConfigBuilder::build` time.
pub fn run(cfg: &BuildFromShardsConfig) -> Result<(), String> {
    let t0 = Instant::now();

    // Resolve shards. If --shards-dir is provided, scan that directory.
    // Otherwise use the shard set cached by `starfield-gaia` (typically
    // `~/.cache/starfield/gaia/dr3/`), which is the canonical location
    // for already-downloaded GaiaSource files.
    let (shards, source_label) = match cfg.shards_dir.as_ref() {
        Some(dir) => {
            let files = find_shard_files(dir)
                .map_err(|e| format!("failed to scan shards directory {}: {e}", dir.display()))?;
            (files, format!("directory {}", dir.display()))
        }
        None => {
            let cached = Downloader::<Dr3>::list_cached().map_err(|e| {
                format!(
                    "failed to list starfield-gaia cached shards: {e}; \
                     pass --shards-dir to override"
                )
            })?;
            (cached, "starfield-gaia cache".to_string())
        }
    };
    if shards.is_empty() {
        return Err(match cfg.shards_dir.as_ref() {
            Some(dir) => format!("no .csv(.gz) shards found in {}", dir.display()),
            None => "starfield-gaia cache is empty (no Gaia DR3 shards downloaded yet); \
                     run starfield-gaia downloader or pass --shards-dir"
                .to_string(),
        });
    }
    eprintln!("Found {} shard(s) from {}", shards.len(), source_label);

    if let Some(n) = cfg.threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }

    let star_tuples: Mutex<Vec<IndexStarTuple>> = Mutex::new(Vec::new());
    let sidecar_records: Mutex<Vec<SidecarRecord>> = Mutex::new(Vec::new());
    let progress = std::sync::atomic::AtomicUsize::new(0);
    let n_shards = shards.len();
    let any_error: Mutex<Option<String>> = Mutex::new(None);

    shards.par_iter().for_each(|path| {
        if any_error.lock().unwrap().is_some() {
            return;
        }
        match read_shard(path, cfg.mag_limit) {
            Ok(rows) => {
                let n = rows.len();
                let mut tuples = star_tuples.lock().unwrap();
                let mut records = sidecar_records.lock().unwrap();
                tuples.reserve(n);
                records.reserve(n);
                for (t, r) in rows {
                    tuples.push(t);
                    records.push(r);
                }
                drop(tuples);
                drop(records);
                let done = progress.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
                if done.is_multiple_of(50) || done == n_shards {
                    let stars_so_far = star_tuples.lock().unwrap().len();
                    eprintln!("  shard {done}/{n_shards} done ({stars_so_far} stars accumulated)");
                }
            }
            Err(e) => {
                let mut slot = any_error.lock().unwrap();
                if slot.is_none() {
                    *slot = Some(e);
                }
            }
        }
    });

    if let Some(e) = any_error.into_inner().unwrap() {
        return Err(e);
    }

    let star_tuples = star_tuples.into_inner().unwrap();
    let sidecar_records = sidecar_records.into_inner().unwrap();
    let n_stars = star_tuples.len();
    let read_elapsed = t0.elapsed();
    eprintln!(
        "Read {} stars across {} shards in {:.1}s ({:.0} stars/s)",
        n_stars,
        n_shards,
        read_elapsed.as_secs_f64(),
        n_stars as f64 / read_elapsed.as_secs_f64().max(1e-9),
    );

    if n_stars == 0 {
        return Err("no stars survived the magnitude cut; nothing to index".into());
    }

    let scale_lower_rad = (cfg.scale_lower_arcsec / 3600.0).to_radians();
    let scale_upper_rad = (cfg.scale_upper_arcsec / 3600.0).to_radians();
    let max_quads = cfg.max_quads;
    let max_stars = n_stars;
    let builder_cfg = IndexBuilderConfig {
        scale_lower: scale_lower_rad,
        scale_upper: scale_upper_rad,
        max_stars,
        max_quads,
    };

    eprintln!(
        "Building index: scale=[{:.1}\",{:.1}\"], max_quads={}",
        cfg.scale_lower_arcsec, cfg.scale_upper_arcsec, max_quads,
    );

    let index = build_index(&star_tuples, &builder_cfg);
    drop(star_tuples);
    let build_elapsed = t0.elapsed() - read_elapsed;
    eprintln!(
        "Index built: {} stars, {} quads in {:.1}s",
        index.stars.len(),
        index.quads.len(),
        build_elapsed.as_secs_f64()
    );

    let zdcl_path = with_suffix(&cfg.output_prefix, "zdcl");
    let sidecar_path = with_suffix(&cfg.output_prefix, "zdcl.gaia");

    index
        .save_v3(&zdcl_path, cfg.cell_depth)
        .map_err(|e| format!("failed to write {}: {e}", zdcl_path.display()))?;
    let zdcl_size = std::fs::metadata(&zdcl_path).map(|m| m.len()).unwrap_or(0);

    write_sidecar(&sidecar_path, sidecar_records, DEFAULT_PIVOT_STRIDE)
        .map_err(|e| format!("failed to write {}: {e}", sidecar_path.display()))?;
    let sidecar_size = std::fs::metadata(&sidecar_path)
        .map(|m| m.len())
        .unwrap_or(0);

    let total = t0.elapsed();
    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("Stars read:   {n_stars}");
    eprintln!(
        "Index file:   {} ({:.1} MB)",
        zdcl_path.display(),
        zdcl_size as f64 / 1024.0 / 1024.0
    );
    eprintln!(
        "Sidecar file: {} ({:.1} MB)",
        sidecar_path.display(),
        sidecar_size as f64 / 1024.0 / 1024.0
    );
    eprintln!("Elapsed:      {:.1}s", total.as_secs_f64());

    Ok(())
}

/// Append `.<suffix>` to `prefix`. Used because `Path::set_extension`
/// would replace any trailing extension on the prefix; we want
/// `out/foo` → `out/foo.zdcl`, and `out/foo.bar` → `out/foo.bar.zdcl`.
fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(suffix);
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;
    use starfield_gaia::common::core::{GaiaCore, VarFlag};
    use starfield_gaia::dr3::entry::Dr3Entry;

    fn make_entry(source_id: u64, ra: f64, dec: f64, mag: f64) -> Dr3Entry {
        Dr3Entry {
            core: GaiaCore {
                source_id,
                solution_id: 1,
                ref_epoch: 2016.0,
                random_index: None,
                ra,
                ra_error: 0.1,
                dec,
                dec_error: 0.1,
                ra_dec_corr: None,
                parallax: Some(5.5),
                parallax_error: Some(0.2),
                pmra: Some(-12.3),
                pmra_error: Some(0.05),
                pmdec: Some(7.1),
                pmdec_error: Some(0.04),
                l: 0.0,
                b: 0.0,
                ecl_lon: 0.0,
                ecl_lat: 0.0,
                phot_g_mean_mag: mag,
                phot_g_mean_flux: None,
                phot_g_mean_flux_error: None,
                phot_g_n_obs: None,
                phot_variable_flag: VarFlag::NotAvailable,
                astrometric_n_obs_al: None,
                astrometric_excess_noise: None,
                astrometric_excess_noise_sig: None,
                astrometric_primary_flag: None,
                duplicated_source: None,
                matched_observations: None,
            },
            designation: None,
            pm: None,
            parallax_over_error: None,
            astrometric_extra: Default::default(),
            ipd: Default::default(),
            bp_rp: None,
            radial_velocity: None,
            gspphot: None,
            data_links: Default::default(),
            classifications: Default::default(),
        }
    }

    #[test]
    fn entry_to_records_converts_units_and_passes_through_sigmas() {
        let entry = make_entry(42, 90.0, -45.0, 12.5);
        let ((source_id, ra_rad, dec_rad, mag), record) = entry_to_records(&entry);
        assert_eq!(source_id, 42);
        assert!((ra_rad - std::f64::consts::FRAC_PI_2).abs() < 1e-12);
        assert!((dec_rad - (-std::f64::consts::FRAC_PI_4)).abs() < 1e-12);
        assert_eq!(mag, 12.5);

        // Sidecar record stores RA/Dec in degrees (matching the on-disk
        // schema documented in zodiacal::refinement::sidecar).
        assert_eq!(record.ra, 90.0);
        assert_eq!(record.dec, -45.0);
        assert_eq!(record.pmra, -12.3);
        assert_eq!(record.pmdec, 7.1);
        assert_eq!(record.parallax, 5.5);
        assert!(record.radial_velocity.is_nan()); // unpublished
        assert_eq!(record.sigma_ra, 0.1);
        assert_eq!(record.sigma_pmra, 0.05);
        assert_eq!(record.sigma_pmdec, 0.04);
        assert_eq!(record.sigma_parallax, 0.2);
    }

    #[test]
    fn entry_to_records_uses_nan_for_missing_optional_astrometry() {
        let mut entry = make_entry(7, 1.0, 2.0, 11.0);
        entry.core.pmra = None;
        entry.core.pmdec = None;
        entry.core.parallax = None;
        entry.core.parallax_error = None;
        entry.core.pmra_error = None;
        entry.core.pmdec_error = None;
        let (_, record) = entry_to_records(&entry);
        assert!(record.pmra.is_nan());
        assert!(record.pmdec.is_nan());
        assert!(record.parallax.is_nan());
        assert!(record.sigma_pmra.is_nan());
        assert!(record.sigma_pmdec.is_nan());
        assert!(record.sigma_parallax.is_nan());
    }

    fn valid_builder() -> BuildFromShardsConfigBuilder {
        BuildFromShardsConfig::builder()
            .shards_dir(Some(PathBuf::from("/nonexistent")))
            .output_prefix(PathBuf::from("/tmp/_zt_test"))
            .mag_limit(14.0)
            .scale_lower_arcsec(60.0)
            .scale_upper_arcsec(1800.0)
            .max_quads(1000)
            .cell_depth(5)
            .threads(Some(1))
    }

    #[test]
    fn build_rejects_excessive_mag_limit() {
        let err = valid_builder().mag_limit(19.5).build().unwrap_err();
        assert!(err.contains("exceeds hard cap"), "got: {err}");
    }

    #[test]
    fn build_rejects_inverted_scale_range() {
        let err = valid_builder()
            .scale_lower_arcsec(1800.0)
            .scale_upper_arcsec(60.0)
            .build()
            .unwrap_err();
        assert!(err.contains("invalid scale range"), "got: {err}");
    }

    #[test]
    fn build_rejects_zero_max_quads() {
        let err = valid_builder().max_quads(0).build().unwrap_err();
        assert!(err.contains("max_quads"), "got: {err}");
    }

    #[test]
    fn build_rejects_missing_required_fields() {
        let err = BuildFromShardsConfig::builder().build().unwrap_err();
        // Required fields fail in declaration order.
        assert!(err.contains("output_prefix"), "got: {err}");
    }

    #[test]
    fn build_succeeds_on_valid_inputs() {
        valid_builder().build().expect("valid builder must succeed");
    }
}
