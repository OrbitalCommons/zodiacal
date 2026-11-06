//! Checked regional bundle loading, index construction, and solving.
use std::io;
use std::time::Duration;
use web_time::Instant;

use super::ZdclBundle;
use crate::extraction::{DetectedSource, validate_sources};
use crate::index::Index;
use crate::input::SolveInputError;
use crate::solver::{SkyRegion, SolveReport, SolverConfig, solve_checked};

/// Stage timings and loaded-data counts for a regional solve.
#[derive(Debug, Clone)]
pub struct RegionalSolveProfile {
    /// Bundle region selection, I/O, decoding and fragment assembly.
    pub load_elapsed: Duration,
    /// Conversion of fragments to indexes, including KD-tree construction.
    pub index_elapsed: Duration,
    /// Checked solver call. The cooperative timeout applies to search only.
    pub solve_elapsed: Duration,
    pub total_elapsed: Duration,
    pub bands_loaded: usize,
    /// Number of Gaia records in the union of loaded cells, counted once across bands.
    pub stars_loaded: usize,
    /// Total quads across all loaded bands.
    pub quads_loaded: usize,
}

/// Regional solve result retaining indexes referenced by solution match indices.
pub struct RegionalSolveReport {
    pub solve: SolveReport,
    pub profile: RegionalSolveProfile,
    /// One index per manifest band in stable band order. Solution index_id and
    /// matched reference-star indices refer to these indexes. Drop the report
    /// (or just this field) when only the WCS is needed.
    pub indexes: Vec<Index>,
}

impl std::fmt::Debug for RegionalSolveReport {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RegionalSolveReport")
            .field("solve", &self.solve)
            .field("profile", &self.profile)
            .field("index_count", &self.indexes.len())
            .finish_non_exhaustive()
    }
}

/// Invalid caller input and bundle I/O/format failures remain distinguishable.
#[derive(Debug)]
pub enum RegionalSolveError {
    InvalidInput(SolveInputError),
    Load(io::Error),
}
impl std::fmt::Display for RegionalSolveError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InvalidInput(e) => e.fmt(f),
            Self::Load(e) => write!(f, "regional bundle load failed: {e}"),
        }
    }
}
impl std::error::Error for RegionalSolveError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::InvalidInput(e) => e,
            Self::Load(e) => e,
        })
    }
}
impl From<SolveInputError> for RegionalSolveError {
    fn from(e: SolveInputError) -> Self {
        Self::InvalidInput(e)
    }
}

impl ZdclBundle {
    /// Load every band over `region`, construct indexes, and run a checked solve.
    ///
    /// All input validation happens before loading. `region` selects overlapping
    /// HEALPix cells, not an acceptance constraint; use `config.within` to require
    /// the solved legacy field center to lie in a particular region. Configuration
    /// is used unchanged. The solver timeout excludes load/index construction.
    ///
    /// Reuse the bundle handle to reuse its shard cache. For repeated solves with
    /// unchanged indexes, retain `report.indexes` and call `solve_checked` directly.
    pub fn solve_region(
        &self,
        sources: &[DetectedSource],
        image_size: (f64, f64),
        region: &SkyRegion,
        config: &SolverConfig,
    ) -> Result<RegionalSolveReport, RegionalSolveError> {
        let total_start = Instant::now();
        validate_sources(sources, image_size).map_err(SolveInputError::from)?;
        region.validate().map_err(SolveInputError::from)?;
        config.validate().map_err(SolveInputError::from)?;

        let load_start = Instant::now();
        let multi = self.load_region(region).map_err(RegionalSolveError::Load)?;
        let load_elapsed = load_start.elapsed();
        let stars_loaded = multi.gaia_records.len();
        let quads_loaded = multi
            .fragments
            .iter()
            .map(|fragment| fragment.quads.len())
            .sum();
        let bands_loaded = multi.fragments.len();
        let index_start = Instant::now();
        let indexes: Vec<Index> = multi.fragments.into_iter().map(Index::from).collect();
        let index_elapsed = index_start.elapsed();
        let index_refs: Vec<&Index> = indexes.iter().collect();
        let solve_start = Instant::now();
        let solve = solve_checked(sources, &index_refs, image_size, config)?;
        let solve_elapsed = solve_start.elapsed();
        Ok(RegionalSolveReport {
            solve,
            indexes,
            profile: RegionalSolveProfile {
                load_elapsed,
                index_elapsed,
                solve_elapsed,
                total_elapsed: total_start.elapsed(),
                stars_loaded,
                quads_loaded,
                bands_loaded,
            },
        })
    }
}
