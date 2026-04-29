//! `build-from-shards` CLI subcommand: build a `.zdcl` index plus a
//! refinement `.zdcl.gaia` sidecar from a directory of Gaia DR3 CSV
//! shards in one pass.
//!
//! Gated behind the `gaia-shards` feature so the public crate doesn't
//! need to pull `starfield-gaia` (which is a path dep — not yet on
//! crates.io).
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
use starfield_gaia::common::reader::CsvSourceReader;
use starfield_gaia::download::Downloader;
use starfield_gaia::{Dr3, Dr3Entry};

use crate::index::builder::{IndexBuilderConfig, build_index};
use crate::refinement::{DEFAULT_PIVOT_STRIDE, SidecarRecord, write_sidecar};

/// Hard cap on the magnitude limit; deeper than this won't fit in
/// reasonable host RAM with the in-memory v1 sidecar contract.
pub const MAX_MAG_LIMIT: f64 = 17.0;

/// `(catalog_id, ra_radians, dec_radians, magnitude)` — the tuple shape
/// `build_index` expects.
pub type IndexStarTuple = (u64, f64, f64, f64);

/// Pair emitted per accepted Gaia row.
pub type ShardRow = (IndexStarTuple, SidecarRecord);

/// Find every `GaiaSource_*.csv.gz` in `dir`. Returns sorted paths.
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
        if name.starts_with("GaiaSource_") && (name.ends_with(".csv.gz") || name.ends_with(".csv"))
        {
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

/// Read one shard file and return all kept stars as (tuple, record) pairs.
///
/// Uses the streaming `CsvSourceReader` to avoid materializing the whole
/// file into a `HashMap` (which would otherwise double our memory budget
/// before we ever start building).
fn read_shard(path: &Path, mag_limit: f64) -> Result<Vec<ShardRow>, String> {
    let reader = CsvSourceReader::<Dr3>::open(path, mag_limit)
        .map_err(|e| format!("{}: open failed: {e}", path.display()))?;

    let mut out = Vec::new();
    for entry_result in reader {
        let entry = entry_result.map_err(|e| format!("{}: parse failed: {e}", path.display()))?;
        // mag_limit is already applied inside CsvSourceReader, but skip NaN mags
        // defensively — a NaN here would corrupt the brightness sort.
        if !entry.core.phot_g_mean_mag.is_finite() {
            continue;
        }
        out.push(entry_to_records(&entry));
    }
    Ok(out)
}

/// Configuration for `run`.
///
/// `shards_dir = None` tells the tool to use the shard set cached by
/// `starfield-gaia` (i.e. files under `~/.cache/starfield/gaia/`).
pub struct BuildFromShardsConfig {
    pub shards_dir: Option<PathBuf>,
    pub output_prefix: PathBuf,
    pub mag_limit: f64,
    /// Lower scale bound in arcseconds.
    pub scale_lower_arcsec: f64,
    /// Upper scale bound in arcseconds.
    pub scale_upper_arcsec: f64,
    pub max_stars_per_cell: usize,
    pub max_quads: Option<usize>,
    pub cell_depth: u8,
    /// Rayon thread pool size; pass `None` to use rayon's default.
    pub threads: Option<usize>,
}

/// Top-level entry point.
pub fn run(cfg: &BuildFromShardsConfig) -> Result<(), String> {
    if !cfg.mag_limit.is_finite() || cfg.mag_limit > MAX_MAG_LIMIT {
        return Err(format!(
            "--mag-limit {} exceeds hard cap {}; v1 sidecar writer holds all records in RAM, \
             and going deeper than ~17 won't fit on a 64 GB box. Either tighten the limit or \
             wait for the streaming sidecar writer.",
            cfg.mag_limit, MAX_MAG_LIMIT,
        ));
    }
    if cfg.scale_lower_arcsec <= 0.0 || cfg.scale_upper_arcsec <= cfg.scale_lower_arcsec {
        return Err(format!(
            "invalid scale range: lower={}\" upper={}\" (must satisfy 0 < lower < upper)",
            cfg.scale_lower_arcsec, cfg.scale_upper_arcsec
        ));
    }

    let t0 = Instant::now();

    // Resolve shards. If --shards-dir is provided, scan that directory.
    // Otherwise use the shard set cached by `starfield-gaia` (typically at
    // `~/.cache/starfield/gaia/`), which is the canonical location for
    // already-downloaded GaiaSource files.
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
            Some(dir) => format!("no GaiaSource_*.csv(.gz) shards found in {}", dir.display()),
            None => "starfield-gaia cache is empty (no GaiaSource_*.csv.gz files \
                     downloaded yet); run starfield-gaia downloader or pass \
                     --shards-dir"
                .to_string(),
        });
    }
    eprintln!("Found {} shard(s) from {}", shards.len(), source_label);

    // Set up rayon (best-effort: ignore the error if the global pool is
    // already configured by another caller).
    if let Some(n) = cfg.threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }

    // Read shards in parallel. Each shard accumulates into a thread-local
    // Vec; we then merge under a single mutex to bound peak memory.
    let star_tuples: Mutex<Vec<IndexStarTuple>> = Mutex::new(Vec::new());
    let sidecar_records: Mutex<Vec<SidecarRecord>> = Mutex::new(Vec::new());
    let progress = std::sync::atomic::AtomicUsize::new(0);
    let n_shards = shards.len();
    let any_error: Mutex<Option<String>> = Mutex::new(None);

    shards.par_iter().for_each(|path| {
        // Skip remaining shards if a sibling already failed.
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

    // Build the quad index.
    let scale_lower_rad = (cfg.scale_lower_arcsec / 3600.0).to_radians();
    let scale_upper_rad = (cfg.scale_upper_arcsec / 3600.0).to_radians();
    let max_quads = cfg.max_quads.unwrap_or(n_stars * 8);
    let max_stars = n_stars; // build_index truncates after sorting; we want all
    let builder_cfg = IndexBuilderConfig {
        scale_lower: scale_lower_rad,
        scale_upper: scale_upper_rad,
        max_stars,
        max_quads,
    };

    eprintln!(
        "Building index: scale=[{:.1}\",{:.1}\"], max_stars_per_cell={} (advisory), max_quads={}",
        cfg.scale_lower_arcsec, cfg.scale_upper_arcsec, cfg.max_stars_per_cell, max_quads,
    );

    // Note: `max_stars_per_cell` is a `CatalogBuilderConfig` knob (HEALPix-
    // uniformized builder); the direct `build_index(&[(u64,f64,f64,f64)])`
    // path doesn't take it. We accept it on the CLI for forward-compat with
    // a future uniformization step but currently pass everything through.
    let _ = cfg.max_stars_per_cell;

    let index = build_index(&star_tuples, &builder_cfg);
    drop(star_tuples);
    let build_elapsed = t0.elapsed() - read_elapsed;
    eprintln!(
        "Index built: {} stars, {} quads in {:.1}s",
        index.stars.len(),
        index.quads.len(),
        build_elapsed.as_secs_f64()
    );

    // Persist outputs.
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

/// Append (or replace) the trailing extension on `prefix`. We append rather
/// than `set_extension` because callers commonly pass a prefix like
/// `out/gaia_g16` which has no extension; if they pass `out.foo` we still
/// want `out.foo.zdcl`, not `out.zdcl`.
fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(suffix);
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `Dr3Entry` with all the optional astrometry fields populated; useful
    /// for asserting we don't drop any fields on the way through.
    fn make_filled_entry(source_id: u64, ra_deg: f64, dec_deg: f64, mag: f64) -> Dr3Entry {
        use starfield_gaia::dr3::entry::{
            AstrometricExtra, BpRpPhotometry, Classifications, DataLinks, GspPhot, IpdQuality,
            RadialVelocityDr3,
        };
        use starfield_gaia::{GaiaCore, VarFlag};
        let core = GaiaCore {
            source_id,
            solution_id: 0,
            ref_epoch: 2016.0,
            random_index: None,
            ra: ra_deg,
            ra_error: 0.25,
            dec: dec_deg,
            dec_error: 0.18,
            ra_dec_corr: None,
            parallax: Some(2.5),
            parallax_error: Some(0.04),
            pmra: Some(-3.7),
            pmra_error: Some(0.05),
            pmdec: Some(1.2),
            pmdec_error: Some(0.06),
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
        };
        Dr3Entry {
            core,
            designation: None,
            pm: None,
            parallax_over_error: None,
            astrometric_extra: AstrometricExtra::default(),
            ipd: IpdQuality::default(),
            bp_rp: Some(BpRpPhotometry::default()),
            radial_velocity: Some(RadialVelocityDr3 {
                radial_velocity: Some(15.0),
                ..RadialVelocityDr3::default()
            }),
            gspphot: Some(GspPhot::default()),
            data_links: DataLinks::default(),
            classifications: Classifications::default(),
        }
    }

    /// A `Dr3Entry` with everything optional left as `None` — a faint
    /// 5-parameter or even 2-parameter source.
    fn make_minimal_entry(source_id: u64, ra_deg: f64, dec_deg: f64, mag: f64) -> Dr3Entry {
        use starfield_gaia::dr3::entry::{
            AstrometricExtra, Classifications, DataLinks, IpdQuality,
        };
        use starfield_gaia::{GaiaCore, VarFlag};
        let core = GaiaCore {
            source_id,
            solution_id: 0,
            ref_epoch: 2016.0,
            random_index: None,
            ra: ra_deg,
            ra_error: 1.5,
            dec: dec_deg,
            dec_error: 1.7,
            ra_dec_corr: None,
            parallax: None,
            parallax_error: None,
            pmra: None,
            pmra_error: None,
            pmdec: None,
            pmdec_error: None,
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
        };
        Dr3Entry {
            core,
            designation: None,
            pm: None,
            parallax_over_error: None,
            astrometric_extra: AstrometricExtra::default(),
            ipd: IpdQuality::default(),
            bp_rp: None,
            radial_velocity: None,
            gspphot: None,
            data_links: DataLinks::default(),
            classifications: Classifications::default(),
        }
    }

    #[test]
    fn filled_entry_round_trips_units() {
        let entry = make_filled_entry(1234, 45.0, 30.0, 14.5);
        let ((sid, ra_rad, dec_rad, mag), record) = entry_to_records(&entry);

        assert_eq!(sid, 1234);
        assert!((ra_rad - 45.0_f64.to_radians()).abs() < 1e-12);
        assert!((dec_rad - 30.0_f64.to_radians()).abs() < 1e-12);
        assert_eq!(mag, 14.5);

        // Sidecar stores degrees, not radians.
        assert_eq!(record.source_id, 1234);
        assert_eq!(record.ra, 45.0);
        assert_eq!(record.dec, 30.0);
        assert_eq!(record.ref_epoch, 2016.0);
        assert_eq!(record.pmra, -3.7);
        assert_eq!(record.pmdec, 1.2);
        assert_eq!(record.parallax, 2.5);
        assert_eq!(record.radial_velocity, 15.0);
        assert_eq!(record.sigma_ra, 0.25);
        assert_eq!(record.sigma_dec, 0.18);
        assert_eq!(record.sigma_pmra, 0.05);
        assert_eq!(record.sigma_pmdec, 0.06);
        assert_eq!(record.sigma_parallax, 0.04);
        assert_eq!(record.flags, 0);
    }

    #[test]
    fn minimal_entry_uses_nan_for_missing_fields() {
        let entry = make_minimal_entry(7, 0.0, 0.0, 19.5);
        let (_tuple, record) = entry_to_records(&entry);
        assert!(record.pmra.is_nan());
        assert!(record.pmdec.is_nan());
        assert!(record.parallax.is_nan());
        assert!(record.radial_velocity.is_nan());
        assert!(record.sigma_pmra.is_nan());
        assert!(record.sigma_pmdec.is_nan());
        assert!(record.sigma_parallax.is_nan());
        // ra_error / dec_error are non-optional in GaiaCore, so they pass through.
        assert_eq!(record.sigma_ra, 1.5);
        assert_eq!(record.sigma_dec, 1.7);
    }

    #[test]
    fn rv_present_but_value_missing_yields_nan() {
        use starfield_gaia::dr3::entry::RadialVelocityDr3;
        let mut entry = make_minimal_entry(99, 10.0, -5.0, 13.0);
        entry.radial_velocity = Some(RadialVelocityDr3 {
            radial_velocity: None,
            ..RadialVelocityDr3::default()
        });
        let (_tuple, record) = entry_to_records(&entry);
        assert!(record.radial_velocity.is_nan());
    }

    #[test]
    fn with_suffix_appends_not_replaces() {
        assert_eq!(
            with_suffix(Path::new("foo"), "zdcl"),
            PathBuf::from("foo.zdcl")
        );
        assert_eq!(
            with_suffix(Path::new("foo.bar"), "zdcl"),
            PathBuf::from("foo.bar.zdcl")
        );
        assert_eq!(
            with_suffix(Path::new("/tmp/out"), "zdcl.gaia"),
            PathBuf::from("/tmp/out.zdcl.gaia")
        );
    }
}
