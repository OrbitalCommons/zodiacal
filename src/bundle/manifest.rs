//! `BundleManifest` — the JSON `manifest.json` shape that ships inside a
//! finalized `.zdcl.bundle`.
//!
//! The manifest is the **commit point** of a bundle: a bundle is considered
//! finalized iff its `manifest.json` parses and passes [`BundleManifest::load`]'s
//! schema validation. See `docs/bundle-format.md` for the full specification.
//!
//! This file implements the typed schema, atomic write, and validating load.
//! It does **not** implement the build pipeline that writes one of these — the
//! cell-driven builder + tidy phase produce the manifest as their final step.
//!
//! # Schema
//!
//! ```text
//! BundleManifest
//! ├── format            "zdcl-bundle"     (validated on load)
//! ├── format_version    1                 (validated on load)
//! ├── cell_depth        u8
//! ├── n_cells           u32
//! ├── experiment        free-text human ops note
//! ├── build_metadata
//! │   ├── tool                            "zodiacal-tools 0.2.0"
//! │   ├── build_started_utc               RFC3339
//! │   ├── build_finished_utc              RFC3339
//! │   └── source { kind, release, path }
//! ├── gaia
//! │   ├── n_total                u64
//! │   ├── record_size            u32 = 104   (validated on load)
//! │   ├── schema_version         u32
//! │   ├── max_stars_per_cell     u32
//! │   ├── mag_limit              f64
//! │   └── populated_cells        u32
//! └── bands [BandInfo]
//!     ├── label                  "band_NN"
//!     ├── band_idx               u32         (array position == band_idx)
//!     ├── scale_lower_arcsec     f64
//!     ├── scale_upper_arcsec     f64
//!     ├── quads_per_cell         u32
//!     ├── max_reuse              u32
//!     ├── n_quads_total          u64
//!     └── populated_cells        u32
//! ```

use std::fs::File;
use std::io::{self, Read as _, Write as _};
use std::path::Path;

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// Magic string for the bundle format. A manifest with any other value is
/// rejected by [`BundleManifest::load`].
pub const FORMAT_MAGIC: &str = "zdcl-bundle";

/// Current on-disk schema version. `load` rejects manifests with a different
/// version (forward-compat is opt-in via a future match arm).
pub const FORMAT_VERSION: u32 = 1;

/// Required `gaia.record_size` for a v1 bundle. Mismatches are rejected by
/// `load` to prevent accidentally reading a 96 B-record bundle as 104 B (or
/// vice versa).
pub const GAIA_RECORD_SIZE: u32 = 104;

/// Top-level `manifest.json` schema for a finalized `.zdcl.bundle`.
///
/// Field order in the struct definition matches the JSON output order
/// produced by `serde_json::to_writer_pretty` — keep them aligned with the
/// spec example so a diff against `docs/bundle-format.md` stays cheap.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BundleManifest {
    /// Always `"zdcl-bundle"` for this format.
    pub format: String,
    /// On-disk schema version. `1` today.
    pub format_version: u32,
    /// HEALPix nested-scheme depth used for the bundle's cell sharding.
    pub cell_depth: u8,
    /// Total number of cells at `cell_depth` (`12 * 4^depth`). Stored
    /// redundantly so a reader doesn't have to derive it.
    pub n_cells: u32,
    /// Free-text human ops note. Tooling that needs structured params
    /// should look at `build_metadata` and the per-band entries.
    pub experiment: String,
    /// Structured build provenance.
    pub build_metadata: BuildMetadata,
    /// Per-cell Gaia-shard summary stats.
    pub gaia: GaiaBlock,
    /// One entry per scale band, in `band_idx` order. Array position must
    /// equal `band_idx` (validated by `load`).
    pub bands: Vec<BandInfo>,
}

/// Tooling-facing build provenance block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildMetadata {
    /// Builder identifier, typically `"<crate-name> <version>"` (e.g.
    /// `"zodiacal-tools 0.2.0"`).
    pub tool: String,
    /// UTC timestamp at which the build phase started. RFC3339 in JSON.
    pub build_started_utc: DateTime<Utc>,
    /// UTC timestamp at which the tidy phase finished and the manifest
    /// was written. RFC3339 in JSON.
    pub build_finished_utc: DateTime<Utc>,
    /// What the build was sourced from.
    pub source: BuildSource,
}

/// Description of the input data source the build consumed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BuildSource {
    /// Source-kind tag, e.g. `"starfield-datasources-bycell"`.
    pub kind: String,
    /// Catalog release identifier, e.g. `"Dr3"`.
    pub release: String,
    /// Resolved on-disk path the build read from. Stringly-typed for
    /// portability across platforms.
    pub path: String,
}

/// Gaia per-cell-shard summary block.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GaiaBlock {
    /// Total number of Gaia records across all populated cells.
    pub n_total: u64,
    /// Per-record byte size. Must equal [`GAIA_RECORD_SIZE`] for v1.
    pub record_size: u32,
    /// Per-record schema version (covaries with the bit layout described
    /// in `docs/bundle-format.md` § "Per-cell file formats").
    pub schema_version: u32,
    /// Brightness-truncation cap applied during the build.
    pub max_stars_per_cell: u32,
    /// Faintest G-magnitude included in the build.
    pub mag_limit: f64,
    /// How many `cell_NNNNN.zga` files the bundle actually carries
    /// (empty cells are absent on disk).
    pub populated_cells: u32,
}

/// One scale-band entry in `BundleManifest::bands`.
///
/// `band_idx` is stored explicitly even though it equals the array position;
/// keeping it in the manifest means consumers don't have to derive it from
/// position. The on-disk band table inside each `cell_NNNNN.zqd` carries the
/// same `band_idx`, so `manifest.bands[k].band_idx == on-disk band-table k`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct BandInfo {
    /// Human label, conventionally `"band_NN"` zero-padded.
    pub label: String,
    /// Zero-based band index. Array position == band_idx == on-disk
    /// band-table band_idx (validated by `load`).
    pub band_idx: u32,
    /// Lower scale bound for this band's quads, in arcseconds (between
    /// the closest two stars of the quad).
    pub scale_lower_arcsec: f64,
    /// Upper scale bound for this band's quads, in arcseconds.
    pub scale_upper_arcsec: f64,
    /// Per-cell quad count cap configured for this band.
    pub quads_per_cell: u32,
    /// Per-cell star-reuse cap configured for this band.
    pub max_reuse: u32,
    /// Total quads emitted across all populated cells in this band.
    pub n_quads_total: u64,
    /// Cells that committed at least one quad in this band.
    pub populated_cells: u32,
}

impl BundleManifest {
    /// Atomic write: serialize to `path.with_extension("json.partial")` (or
    /// a `.partial` sibling for non-`.json` paths), fsync, then rename
    /// onto `path`.
    ///
    /// On crash mid-write the only on-disk artifact is the orphaned
    /// `.partial` file; the canonical `path` is untouched. The next
    /// `save` (or the tidy-phase rerun) overwrites the partial
    /// idempotently.
    pub fn save(&self, path: &Path) -> io::Result<()> {
        if let Some(parent) = path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)?;
        }
        let tmp = partial_path_for(path);

        let json = serde_json::to_vec_pretty(self).map_err(io::Error::other)?;
        {
            let mut f = File::create(&tmp)?;
            f.write_all(&json)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, path)?;
        Ok(())
    }

    /// Load and validate a manifest from `path`.
    ///
    /// Validates:
    /// - `format == "zdcl-bundle"`
    /// - `format_version == 1`
    /// - `gaia.record_size == 104`
    /// - `bands` is sorted by `band_idx` ascending and dense from
    ///   `0..bands.len()` (no gaps, no duplicates, array position equals
    ///   `band_idx`)
    ///
    /// Other invariants (e.g. `n_cells == 12 * 4^cell_depth`,
    /// per-band `populated_cells <= n_cells`) are not checked here — they
    /// belong to the consumer that wires the manifest up to actual data.
    pub fn load(path: &Path) -> io::Result<Self> {
        let mut buf = String::new();
        File::open(path)?.read_to_string(&mut buf)?;
        let m: Self = serde_json::from_str(&buf).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse {}: {e}", path.display()),
            )
        })?;

        if m.format != FORMAT_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "unexpected bundle format string {:?} (expected {:?})",
                    m.format, FORMAT_MAGIC
                ),
            ));
        }

        match m.format_version {
            FORMAT_VERSION => {}
            other => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "unsupported bundle format_version {other} (this build supports {FORMAT_VERSION})",
                    ),
                ));
            }
        }

        if m.gaia.record_size != GAIA_RECORD_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "gaia.record_size = {} but this build expects {GAIA_RECORD_SIZE}",
                    m.gaia.record_size
                ),
            ));
        }

        for (pos, band) in m.bands.iter().enumerate() {
            if band.band_idx as usize != pos {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "bands[{pos}].band_idx = {} (expected dense 0..{} layout, position {pos})",
                        band.band_idx,
                        m.bands.len(),
                    ),
                ));
            }
        }

        Ok(m)
    }
}

/// Compute the `.partial` sibling path used for atomic writes.
fn partial_path_for(path: &Path) -> std::path::PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".partial");
    std::path::PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn band(idx: u32) -> BandInfo {
        // Use a non-SQRT_2 multiplier to avoid clippy::approx_constant on
        // 1.4142; the exact value doesn't matter for round-trip tests.
        let factor: f64 = 1.5_f64.powi(idx as i32);
        BandInfo {
            label: format!("band_{idx:02}"),
            band_idx: idx,
            scale_lower_arcsec: 10.0 * factor,
            scale_upper_arcsec: 15.0 * factor,
            quads_per_cell: 100,
            max_reuse: 8,
            n_quads_total: 1_228_800 - 100 * idx as u64,
            populated_cells: 12_288 - idx,
        }
    }

    fn sample_manifest(n_bands: u32) -> BundleManifest {
        // Use an explicit fixed timestamp so JSON output is deterministic
        // across test machines / clocks.
        let started: DateTime<Utc> = "2026-05-05T01:08:57Z".parse().unwrap();
        let finished: DateTime<Utc> = "2026-05-05T02:18:42Z".parse().unwrap();
        BundleManifest {
            format: FORMAT_MAGIC.to_string(),
            format_version: FORMAT_VERSION,
            cell_depth: 5,
            n_cells: 12_288,
            experiment: "test fixture".to_string(),
            build_metadata: BuildMetadata {
                tool: "zodiacal-tools 0.2.0".to_string(),
                build_started_utc: started,
                build_finished_utc: finished,
                source: BuildSource {
                    kind: "starfield-datasources-bycell".to_string(),
                    release: "Dr3".to_string(),
                    path: "/cache/dr3-mag20-bycell".to_string(),
                },
            },
            gaia: GaiaBlock {
                n_total: 112_080_742,
                record_size: GAIA_RECORD_SIZE,
                schema_version: 1,
                max_stars_per_cell: 10_000,
                mag_limit: 20.0,
                populated_cells: 12_288,
            },
            bands: (0..n_bands).map(band).collect(),
        }
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let dir = tempfile::tempdir().expect("tmpdir");
        let p = dir.path().join(name);
        // Leak the tempdir's drop guard for test simplicity — the OS
        // tmp dir will be cleaned by the system, and tests don't share
        // names. We keep the dir alive for the duration of the test
        // body by leaking the handle.
        std::mem::forget(dir);
        p
    }

    #[test]
    fn roundtrip_minimal() {
        let m = sample_manifest(1);
        let path = tmp_path("manifest_min.json");
        m.save(&path).unwrap();
        let loaded = BundleManifest::load(&path).unwrap();
        assert_eq!(loaded, m);
    }

    #[test]
    fn roundtrip_multi_band() {
        let m = sample_manifest(12);
        let path = tmp_path("manifest_multi.json");
        m.save(&path).unwrap();
        let loaded = BundleManifest::load(&path).unwrap();
        assert_eq!(loaded, m);
        assert_eq!(loaded.bands.len(), 12);
        for (i, b) in loaded.bands.iter().enumerate() {
            assert_eq!(b.band_idx as usize, i);
            assert_eq!(b.label, format!("band_{i:02}"));
        }
    }

    #[test]
    fn load_rejects_bad_format() {
        let mut m = sample_manifest(1);
        m.format = "wrong".to_string();
        let path = tmp_path("manifest_badfmt.json");
        m.save(&path).unwrap();
        let err = BundleManifest::load(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("format string"));
    }

    #[test]
    fn load_rejects_bad_format_version() {
        let mut m = sample_manifest(1);
        m.format_version = 99;
        let path = tmp_path("manifest_badver.json");
        m.save(&path).unwrap();
        let err = BundleManifest::load(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("format_version"));
    }

    #[test]
    fn load_rejects_bad_record_size() {
        let mut m = sample_manifest(1);
        m.gaia.record_size = 88;
        let path = tmp_path("manifest_badrec.json");
        m.save(&path).unwrap();
        let err = BundleManifest::load(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("record_size"));
    }

    #[test]
    fn load_rejects_band_idx_gap() {
        let mut m = sample_manifest(2);
        // Skip 1: bands[0] = 0, bands[1] = 2 (gap at 1).
        m.bands[1].band_idx = 2;
        m.bands[1].label = "band_02".to_string();
        let path = tmp_path("manifest_gap.json");
        m.save(&path).unwrap();
        let err = BundleManifest::load(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("band_idx"));
    }

    #[test]
    fn save_is_atomic() {
        let m = sample_manifest(3);
        let path = tmp_path("manifest_atomic.json");
        m.save(&path).unwrap();

        let partial = partial_path_for(&path);
        assert!(
            !partial.exists(),
            "leftover partial sibling: {}",
            partial.display()
        );
        assert!(path.exists());
        // And the file parses cleanly.
        let _loaded = BundleManifest::load(&path).unwrap();
    }

    #[test]
    fn save_overwrites_atomically() {
        let m1 = sample_manifest(1);
        let m2 = sample_manifest(4);
        let path = tmp_path("manifest_overwrite.json");

        m1.save(&path).unwrap();
        m2.save(&path).unwrap();

        // No partial residue; second save's content fully replaced first.
        let partial = partial_path_for(&path);
        assert!(!partial.exists());
        let loaded = BundleManifest::load(&path).unwrap();
        assert_eq!(loaded, m2);
        assert_eq!(loaded.bands.len(), 4);
    }
}
