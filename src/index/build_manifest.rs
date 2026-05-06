//! Crash-safe manifest for resumable index builds.
//!
//! A `BuildManifest` is the durable record of "which HEALPix cells have
//! been committed to the work directory". The cell-driven builder
//! consults it on resume to skip already-completed cells, and updates
//! it after every successful per-cell commit.
//!
//! Writes are atomic: the manifest is serialized to a sibling
//! `.partial` file, fsynced, and renamed into place. A crash during
//! the rename leaves either the old manifest or no manifest — never a
//! truncated one. The pattern mirrors `starfield-datasources`'s
//! `Manifest` writer (issue #65 refers).
//!
//! Layout: a single JSON file in the build's scratch directory. The
//! file grows with the cell count (a few KB at level 5, ~17 MB at
//! depth 8). [`BuildManifest::commit_cell`] only updates in-memory
//! state — callers explicitly drive [`BuildManifest::save`] on a
//! cadence that suits their throughput (single-cell save in the
//! single-threaded builder; periodic flush in the multi-band
//! actor).

use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

mod cell_stats_serde {
    use super::CellStats;
    use serde::{Deserialize, Deserializer, Serialize, Serializer};
    use std::collections::BTreeMap;

    pub fn serialize<S: Serializer>(
        map: &BTreeMap<u32, CellStats>,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        // BTreeMap iterates key-sorted, so the on-disk array is
        // already in the legacy order without any extra sort step.
        let pairs: Vec<(u32, &CellStats)> = map.iter().map(|(k, v)| (*k, v)).collect();
        pairs.serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(
        d: D,
    ) -> Result<BTreeMap<u32, CellStats>, D::Error> {
        let pairs: Vec<(u32, CellStats)> = Deserialize::deserialize(d)?;
        Ok(pairs.into_iter().collect())
    }
}

/// Filename the manifest lives under inside the scratch dir.
pub const MANIFEST_FILENAME: &str = "build-manifest.json";

/// On-disk schema version. Bump when adding non-trivially-decodable fields.
pub const MANIFEST_VERSION: u32 = 1;

/// Per-cell statistics committed alongside the cell's chunk file.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq)]
pub struct CellStats {
    pub n_stars: u64,
    pub n_quads: u64,
}

/// Crash-safe build manifest. Tracks which cells have been committed,
/// plus running totals so a resumed run can verify on-disk state
/// matches the manifest's claims.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct BuildManifest {
    pub version: u32,
    /// Cell IDs that have been fully committed to the work dir.
    /// `BTreeSet` for deterministic JSON output and O(log n) membership.
    pub completed_cells: BTreeSet<u32>,
    /// Per-cell statistics, keyed by cell_id. Wire format is a JSON
    /// array of `[cell_id, stats]` pairs (numeric keys, stable order)
    /// — see [`cell_stats_serde`]. In memory we keep a `BTreeMap` for
    /// O(log n) insert/lookup, which matters at depth 8 where the map
    /// grows to ~786k entries.
    #[serde(with = "cell_stats_serde")]
    pub cell_stats: BTreeMap<u32, CellStats>,
    /// Running totals; redundant with `cell_stats` but cheap to maintain
    /// and avoids a sum on resume.
    pub n_stars: u64,
    pub n_quads: u64,
    /// Per-scale-band per-cell completion tracking for the multi-band
    /// cell-driven builder.
    ///
    /// `completed_per_band[k]` is the set of cells that have committed
    /// quads for band `k` to the work-dir's per-cell `.zqd` file. A
    /// build that crashed after committing bands `[0..3]` for a cell
    /// but before band `4` resumes only band `4` for that cell.
    ///
    /// Empty when the build is single-band (legacy compat). The
    /// `#[serde(default)]` makes manifests written by the OLD code
    /// (without this field) still parse cleanly under the new code.
    /// Multi-band builders call [`BuildManifest::ensure_per_band`] at
    /// start to size this to `n_bands` empty sets.
    #[serde(default)]
    pub completed_per_band: Vec<BTreeSet<u32>>,
}

impl Default for BuildManifest {
    fn default() -> Self {
        Self {
            version: MANIFEST_VERSION,
            completed_cells: BTreeSet::new(),
            cell_stats: BTreeMap::new(),
            n_stars: 0,
            n_quads: 0,
            completed_per_band: Vec::new(),
        }
    }
}

impl BuildManifest {
    /// Path of the manifest file inside `scratch_dir`.
    pub fn path_in(scratch_dir: &Path) -> PathBuf {
        scratch_dir.join(MANIFEST_FILENAME)
    }

    /// Path of the `.partial` sibling used during atomic writes.
    pub fn tmp_path_in(scratch_dir: &Path) -> PathBuf {
        let mut s = Self::path_in(scratch_dir).into_os_string();
        s.push(".partial");
        PathBuf::from(s)
    }

    /// Load an existing manifest from `scratch_dir`. Returns
    /// `Ok(None)` if the file does not exist; an error on parse
    /// failure or version mismatch.
    pub fn load(scratch_dir: &Path) -> io::Result<Option<Self>> {
        let path = Self::path_in(scratch_dir);
        if !path.exists() {
            return Ok(None);
        }
        let mut buf = String::new();
        File::open(&path)?.read_to_string(&mut buf)?;
        let m: Self = serde_json::from_str(&buf).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse {}: {e}", path.display()),
            )
        })?;
        if m.version != MANIFEST_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "manifest at {} has version {} but expected {}",
                    path.display(),
                    m.version,
                    MANIFEST_VERSION,
                ),
            ));
        }
        Ok(Some(m))
    }

    /// Atomically persist the manifest to `scratch_dir`. Writes to a
    /// `.partial` sibling, fsyncs, then renames into place. Safe to
    /// call after every successful per-cell commit.
    pub fn save(&self, scratch_dir: &Path) -> io::Result<()> {
        std::fs::create_dir_all(scratch_dir)?;

        let tmp = Self::tmp_path_in(scratch_dir);
        let final_path = Self::path_in(scratch_dir);

        let json = serde_json::to_vec_pretty(self).map_err(io::Error::other)?;
        {
            let mut f = File::create(&tmp)?;
            f.write_all(&json)?;
            f.sync_all()?;
        }
        std::fs::rename(&tmp, &final_path)?;
        Ok(())
    }

    /// Mark `cell_id` complete with the given stats and update running
    /// totals. Pure in-memory; callers must call [`Self::save`]
    /// to persist.
    ///
    /// Idempotent: if the cell was already marked complete, the prior
    /// stats are replaced (running totals adjusted accordingly).
    pub fn commit_cell(&mut self, cell_id: u32, stats: CellStats) {
        if let Some(prev) = self.cell_stats.insert(cell_id, stats.clone()) {
            self.n_stars = self.n_stars.saturating_sub(prev.n_stars);
            self.n_quads = self.n_quads.saturating_sub(prev.n_quads);
        }
        self.completed_cells.insert(cell_id);
        self.n_stars += stats.n_stars;
        self.n_quads += stats.n_quads;
    }

    /// Quick membership check used by the cell-driven builder to skip
    /// already-committed cells on resume.
    pub fn is_complete(&self, cell_id: u32) -> bool {
        self.completed_cells.contains(&cell_id)
    }

    /// Resize `completed_per_band` to exactly `n_bands` entries, padding
    /// with empty `BTreeSet`s and discarding any extra entries beyond
    /// `n_bands`.
    ///
    /// Intended to be called once per build session before any
    /// [`BuildManifest::mark_band_complete`] calls, with the *current*
    /// band count from the bundle manifest (which is the source of
    /// truth for band layout). The `completed_per_band` field is a
    /// build-time scratchpad, so reconciling it to the current
    /// `n_bands` is always safe.
    ///
    /// If a manifest is loaded from disk with a stale
    /// `completed_per_band.len()` (because the band layout has changed
    /// since the previous run), calling `ensure_per_band` reconciles
    /// it: growing with empty sets when `n_bands` is larger, or
    /// truncating when `n_bands` is smaller. Stale completion records
    /// for removed bands are discarded by design — the bundle
    /// manifest's bands array is authoritative, and a band that no
    /// longer exists has no meaningful "completed" state.
    pub fn ensure_per_band(&mut self, n_bands: usize) {
        self.completed_per_band.resize_with(n_bands, BTreeSet::new);
    }

    /// True if cell `cell_id`'s band `band_idx` has been committed.
    ///
    /// Returns `false` for any `band_idx` outside the current
    /// `completed_per_band` length (including the legacy empty-vec
    /// case), so single-band call sites that never call
    /// `ensure_per_band` get a sensible default.
    pub fn is_band_complete(&self, cell_id: u32, band_idx: u32) -> bool {
        self.completed_per_band
            .get(band_idx as usize)
            .is_some_and(|set| set.contains(&cell_id))
    }

    /// Mark cell `cell_id`'s band `band_idx` as committed.
    ///
    /// Idempotent: marking the same `(cell, band)` twice is a no-op.
    /// If `completed_per_band` is shorter than `band_idx + 1` it is
    /// extended with empty sets to fit. This means callers that build
    /// without an explicit `ensure_per_band` still get correct
    /// per-band tracking, just with a `completed_per_band.len()` that
    /// reflects only the bands actually marked.
    pub fn mark_band_complete(&mut self, cell_id: u32, band_idx: u32) {
        let needed = band_idx as usize + 1;
        if self.completed_per_band.len() < needed {
            self.completed_per_band.resize_with(needed, BTreeSet::new);
        }
        self.completed_per_band[band_idx as usize].insert(cell_id);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "zodiacal-build-manifest-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    #[test]
    fn load_returns_none_when_missing() {
        let dir = tmp_dir("missing");
        std::fs::create_dir_all(&dir).unwrap();
        assert!(BuildManifest::load(&dir).unwrap().is_none());
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_then_load_roundtrip() {
        let dir = tmp_dir("rt");
        std::fs::create_dir_all(&dir).unwrap();

        let mut m = BuildManifest::default();
        m.commit_cell(
            42,
            CellStats {
                n_stars: 1234,
                n_quads: 567,
            },
        );
        m.commit_cell(
            7,
            CellStats {
                n_stars: 100,
                n_quads: 50,
            },
        );
        m.save(&dir).unwrap();

        let loaded = BuildManifest::load(&dir).unwrap().expect("manifest");
        assert_eq!(loaded, m);
        assert!(loaded.is_complete(42));
        assert!(loaded.is_complete(7));
        assert!(!loaded.is_complete(0));
        assert_eq!(loaded.n_stars, 1334);
        assert_eq!(loaded.n_quads, 617);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn commit_cell_is_idempotent_in_running_totals() {
        let mut m = BuildManifest::default();
        m.commit_cell(
            5,
            CellStats {
                n_stars: 10,
                n_quads: 4,
            },
        );
        // Re-commit the same cell with different stats — totals should
        // reflect the new value, not the sum.
        m.commit_cell(
            5,
            CellStats {
                n_stars: 20,
                n_quads: 8,
            },
        );
        assert_eq!(m.completed_cells.len(), 1);
        assert_eq!(m.n_stars, 20);
        assert_eq!(m.n_quads, 8);
    }

    #[test]
    fn version_mismatch_is_rejected() {
        let dir = tmp_dir("ver");
        std::fs::create_dir_all(&dir).unwrap();

        let path = BuildManifest::path_in(&dir);
        std::fs::write(
            &path,
            r#"{"version":99,"completed_cells":[],"cell_stats":[],"n_stars":0,"n_quads":0}"#,
        )
        .unwrap();

        let err = BuildManifest::load(&dir).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn save_is_atomic_no_partial_left_behind() {
        let dir = tmp_dir("atomic");
        std::fs::create_dir_all(&dir).unwrap();

        let m = BuildManifest::default();
        m.save(&dir).unwrap();

        // The .partial sibling must not survive a successful save.
        let partial = BuildManifest::tmp_path_in(&dir);
        assert!(!partial.exists(), "leftover partial: {}", partial.display());

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn per_band_legacy_compat() {
        // A manifest written by the OLD code (pre-`completed_per_band`)
        // must round-trip through the NEW deserializer with an empty
        // `completed_per_band`. The `#[serde(default)]` attribute is
        // what makes this work.
        let dir = tmp_dir("legacy");
        std::fs::create_dir_all(&dir).unwrap();

        let path = BuildManifest::path_in(&dir);
        // Hand-rolled JSON without the new field. Mirrors what the
        // pre-PR2 code would have written.
        std::fs::write(
            &path,
            r#"{"version":1,"completed_cells":[3,7],"cell_stats":[[3,{"n_stars":10,"n_quads":4}],[7,{"n_stars":5,"n_quads":2}]],"n_stars":15,"n_quads":6}"#,
        )
        .unwrap();

        let loaded = BuildManifest::load(&dir).unwrap().expect("manifest");
        assert!(loaded.completed_per_band.is_empty());
        // Existing fields still populate from the legacy JSON.
        assert_eq!(loaded.completed_cells.len(), 2);
        assert!(loaded.is_complete(3));
        assert!(loaded.is_complete(7));
        assert_eq!(loaded.n_stars, 15);
        assert_eq!(loaded.n_quads, 6);
        // And band queries on the legacy manifest answer false (no
        // band tracking yet).
        assert!(!loaded.is_band_complete(3, 0));

        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mark_band_complete_grows_set() {
        let mut m = BuildManifest::default();
        m.ensure_per_band(3);
        assert_eq!(m.completed_per_band.len(), 3);
        for set in &m.completed_per_band {
            assert!(set.is_empty());
        }

        m.mark_band_complete(5, 1);
        assert!(m.is_band_complete(5, 1));
        assert!(!m.is_band_complete(5, 0));
        assert!(!m.is_band_complete(5, 2));
        // Out-of-range bands answer false rather than panicking.
        assert!(!m.is_band_complete(5, 99));
        // Other cells unaffected.
        assert!(!m.is_band_complete(4, 1));
    }

    #[test]
    fn mark_band_complete_idempotent() {
        let mut m = BuildManifest::default();
        m.ensure_per_band(2);

        m.mark_band_complete(7, 0);
        m.mark_band_complete(7, 0);
        m.mark_band_complete(7, 0);

        assert!(m.is_band_complete(7, 0));
        // BTreeSet membership is set semantics — len stays 1.
        assert_eq!(m.completed_per_band[0].len(), 1);
    }

    #[test]
    fn ensure_per_band_is_idempotent() {
        // Two calls in a row must not clobber per-band state. This is
        // the "resume of a multi-band build" path.
        let mut m = BuildManifest::default();
        m.ensure_per_band(4);
        m.mark_band_complete(11, 2);

        m.ensure_per_band(4);
        assert_eq!(m.completed_per_band.len(), 4);
        assert!(m.is_band_complete(11, 2));
    }

    #[test]
    fn ensure_per_band_from_empty_produces_n_empty_sets() {
        let mut m = BuildManifest::default();
        assert!(m.completed_per_band.is_empty());

        m.ensure_per_band(5);
        assert_eq!(m.completed_per_band.len(), 5);
        for set in &m.completed_per_band {
            assert!(set.is_empty());
        }
    }

    #[test]
    fn ensure_per_band_same_size_preserves_contents() {
        let mut m = BuildManifest::default();
        m.ensure_per_band(3);
        m.mark_band_complete(1, 0);
        m.mark_band_complete(2, 1);
        m.mark_band_complete(3, 2);

        m.ensure_per_band(3);
        assert_eq!(m.completed_per_band.len(), 3);
        assert!(m.is_band_complete(1, 0));
        assert!(m.is_band_complete(2, 1));
        assert!(m.is_band_complete(3, 2));
    }

    #[test]
    fn ensure_per_band_grows_preserving_existing_entries() {
        let mut m = BuildManifest::default();
        m.ensure_per_band(3);
        m.mark_band_complete(1, 0);
        m.mark_band_complete(2, 1);
        m.mark_band_complete(3, 2);

        m.ensure_per_band(5);
        assert_eq!(m.completed_per_band.len(), 5);
        // First N entries preserved.
        assert!(m.is_band_complete(1, 0));
        assert!(m.is_band_complete(2, 1));
        assert!(m.is_band_complete(3, 2));
        // New entries are empty.
        assert!(m.completed_per_band[3].is_empty());
        assert!(m.completed_per_band[4].is_empty());
    }

    #[test]
    fn ensure_per_band_truncates_discarding_extra_entries() {
        let mut m = BuildManifest::default();
        m.ensure_per_band(4);
        m.mark_band_complete(1, 0);
        m.mark_band_complete(2, 1);
        m.mark_band_complete(3, 2);
        m.mark_band_complete(99, 3); // last band — will be discarded.

        m.ensure_per_band(3);
        assert_eq!(m.completed_per_band.len(), 3);
        // Surviving bands keep their contents.
        assert!(m.is_band_complete(1, 0));
        assert!(m.is_band_complete(2, 1));
        assert!(m.is_band_complete(3, 2));
        // The discarded band's entries are gone — the band index is
        // out of range now, so the query returns false.
        assert!(!m.is_band_complete(99, 3));
    }

    #[test]
    fn mark_band_complete_works_after_ensure_per_band() {
        let mut m = BuildManifest::default();
        m.ensure_per_band(4);

        // Marking any band index < n_bands must succeed without
        // panicking, for any cell_id.
        for band_idx in 0..4u32 {
            m.mark_band_complete(100 + band_idx, band_idx);
        }
        for band_idx in 0..4u32 {
            assert!(m.is_band_complete(100 + band_idx, band_idx));
        }
    }
}
