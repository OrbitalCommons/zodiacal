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
//! file is small (a few KB at level 5, a few hundred KB at level 12),
//! so re-serializing the whole document on every cell commit is fine.

use std::collections::BTreeSet;
use std::fs::File;
use std::io::{self, Read, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

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
    /// Optional per-cell statistics, keyed by cell_id.
    /// Stored as a `Vec<(cell_id, stats)>` so JSON keys stay numeric
    /// and reading order is stable.
    pub cell_stats: Vec<(u32, CellStats)>,
    /// Running totals; redundant with `cell_stats` but cheap to maintain
    /// and avoids a sum on resume.
    pub n_stars: u64,
    pub n_quads: u64,
}

impl Default for BuildManifest {
    fn default() -> Self {
        Self {
            version: MANIFEST_VERSION,
            completed_cells: BTreeSet::new(),
            cell_stats: Vec::new(),
            n_stars: 0,
            n_quads: 0,
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

    /// Mark `cell_id` complete with the given stats, update running
    /// totals, and persist atomically.
    ///
    /// Idempotent: if the cell was already marked complete, the prior
    /// stats are replaced (running totals adjusted accordingly). This
    /// matters for resume flows where a chunk file might be re-committed
    /// before the manifest update lands.
    pub fn commit_cell(
        &mut self,
        scratch_dir: &Path,
        cell_id: u32,
        stats: CellStats,
    ) -> io::Result<()> {
        if let Some(idx) = self.cell_stats.iter().position(|(c, _)| *c == cell_id) {
            let prev = self.cell_stats[idx].1.clone();
            self.n_stars = self.n_stars.saturating_sub(prev.n_stars);
            self.n_quads = self.n_quads.saturating_sub(prev.n_quads);
            self.cell_stats[idx].1 = stats.clone();
        } else {
            self.cell_stats.push((cell_id, stats.clone()));
            // Keep cell_stats stably sorted by cell_id for predictable
            // JSON output. Cheap given expected n ≤ 12,288 (level 5).
            self.cell_stats.sort_by_key(|(c, _)| *c);
        }
        self.completed_cells.insert(cell_id);
        self.n_stars += stats.n_stars;
        self.n_quads += stats.n_quads;
        self.save(scratch_dir)
    }

    /// Quick membership check used by the cell-driven builder to skip
    /// already-committed cells on resume.
    pub fn is_complete(&self, cell_id: u32) -> bool {
        self.completed_cells.contains(&cell_id)
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
            &dir,
            42,
            CellStats {
                n_stars: 1234,
                n_quads: 567,
            },
        )
        .unwrap();
        m.commit_cell(
            &dir,
            7,
            CellStats {
                n_stars: 100,
                n_quads: 50,
            },
        )
        .unwrap();

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
        let dir = tmp_dir("idem");
        std::fs::create_dir_all(&dir).unwrap();

        let mut m = BuildManifest::default();
        m.commit_cell(
            &dir,
            5,
            CellStats {
                n_stars: 10,
                n_quads: 4,
            },
        )
        .unwrap();
        // Re-commit the same cell with different stats — totals should
        // reflect the new value, not the sum.
        m.commit_cell(
            &dir,
            5,
            CellStats {
                n_stars: 20,
                n_quads: 8,
            },
        )
        .unwrap();
        assert_eq!(m.completed_cells.len(), 1);
        assert_eq!(m.n_stars, 20);
        assert_eq!(m.n_quads, 8);

        let _ = std::fs::remove_dir_all(&dir);
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
}
