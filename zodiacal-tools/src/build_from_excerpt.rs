//! `build-from-excerpt`: cell-driven `.zdcl` + `.zdcl.gaia` build from
//! a HEALPix-sharded `starfield-gaia` excerpt directory.
//!
//! Drives [`zodiacal::index::cell_builder::build_index_cell_driven`]
//! against [`starfield_gaia::LazyLoadingCatalog`]'s per-cell reader
//! (`entries_in_cell`, upstream `starfield-datasources` PR #48).
//! Memory is bounded by one cell's worth of stars rather than the
//! whole catalog, so a 1.06 B-row G ≤ 20 build fits in <2 GB RAM.
//!
//! Each cell is brightness-truncated at the source via a max-heap
//! over `phot_g_mean_mag` so the cell-driven builder receives at most
//! `max_stars_per_cell` stars per cell — the same uniformization
//! pattern used by the in-process `CatalogBuilderConfig`, applied
//! per-cell at adapter time.

use std::collections::BinaryHeap;
use std::path::PathBuf;
use std::time::Instant;

use starfield_gaia::common::lazy::LazyLoadingCatalog;
use starfield_gaia::{Dr3, Dr3Entry};

use zodiacal::index::build_manifest::BuildManifest;
use zodiacal::index::cell_builder::{
    BuildPaths, BuildSummary, CellBuildConfig, CellStar, CellStarSource, build_index_cell_driven,
};
use zodiacal::refinement::SidecarRecord;

/// CLI configuration for `build-from-excerpt`.
pub struct BuildFromExcerptConfig {
    pub excerpt_dir: PathBuf,
    pub output_prefix: PathBuf,
    pub work_dir: PathBuf,
    pub mag_limit: f64,
    pub max_stars_per_cell: usize,
    pub quads_per_cell: usize,
    pub max_reuse: usize,
    pub scale_lower_arcsec: f64,
    pub scale_upper_arcsec: f64,
    pub final_cell_depth: u8,
    pub pivot_stride: u32,
    /// Rayon thread pool size; `None` uses rayon's default (all logical cores).
    pub threads: Option<usize>,
}

/// Top-level entry point.
pub fn run(cfg: &BuildFromExcerptConfig) -> Result<BuildSummary, String> {
    if !cfg.mag_limit.is_finite() || cfg.mag_limit <= 0.0 {
        return Err(format!("invalid --mag-limit: {}", cfg.mag_limit));
    }
    if cfg.max_stars_per_cell == 0 {
        return Err("--max-stars-per-cell must be positive".into());
    }
    if cfg.quads_per_cell == 0 {
        return Err("--quads-per-cell must be positive".into());
    }
    if cfg.scale_lower_arcsec <= 0.0 || cfg.scale_upper_arcsec <= cfg.scale_lower_arcsec {
        return Err(format!(
            "invalid scale range: lower={}\" upper={}\" (must satisfy 0 < lower < upper)",
            cfg.scale_lower_arcsec, cfg.scale_upper_arcsec
        ));
    }

    if let Some(n) = cfg.threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }

    let lazy = LazyLoadingCatalog::<Dr3>::open(&cfg.excerpt_dir)
        .map_err(|e| format!("LazyLoadingCatalog::open({}): {e}", cfg.excerpt_dir.display()))?;

    eprintln!(
        "Excerpt: {} ({} cells at HEALPix level {}, {} rows committed)",
        cfg.excerpt_dir.display(),
        lazy.cell_count(),
        lazy.healpix_level(),
        lazy.dir().manifest.kept_rows,
    );

    // Resume telemetry: print prior progress before kicking off so a
    // rerun makes its starting point obvious.
    if let Ok(Some(m)) = BuildManifest::load(&cfg.work_dir) {
        eprintln!(
            "Resuming: {} cells already committed, {} stars / {} quads in manifest",
            m.completed_cells.len(),
            m.n_stars,
            m.n_quads,
        );
    }

    let source = LazyExcerptSource {
        lazy,
        mag_limit: cfg.mag_limit,
        max_stars_per_cell: cfg.max_stars_per_cell,
    };

    let scale_lower_rad = (cfg.scale_lower_arcsec / 3600.0).to_radians();
    let scale_upper_rad = (cfg.scale_upper_arcsec / 3600.0).to_radians();

    let build_cfg = CellBuildConfig {
        scale_lower: scale_lower_rad,
        scale_upper: scale_upper_rad,
        quads_per_cell: cfg.quads_per_cell,
        max_reuse: cfg.max_reuse,
        final_cell_depth: cfg.final_cell_depth,
        pivot_stride: cfg.pivot_stride,
    };

    let paths = BuildPaths {
        work_dir: cfg.work_dir.clone(),
        final_index: with_suffix(&cfg.output_prefix, "zdcl"),
        final_sidecar: with_suffix(&cfg.output_prefix, "zdcl.gaia"),
    };

    eprintln!(
        "Building: scale=[{:.1}\",{:.1}\"], quads/cell={}, max_stars/cell={}, max_reuse={}",
        cfg.scale_lower_arcsec,
        cfg.scale_upper_arcsec,
        cfg.quads_per_cell,
        cfg.max_stars_per_cell,
        cfg.max_reuse,
    );

    let t0 = Instant::now();
    let summary = build_index_cell_driven(&source, &build_cfg, &paths)
        .map_err(|e| format!("build_index_cell_driven: {e}"))?;
    let elapsed = t0.elapsed();

    let zdcl_size = std::fs::metadata(&paths.final_index)
        .map(|m| m.len())
        .unwrap_or(0);
    let sidecar_size = std::fs::metadata(&paths.final_sidecar)
        .map(|m| m.len())
        .unwrap_or(0);

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!(
        "Cells:    processed = {}, resumed = {}, empty = {}",
        summary.n_cells_processed, summary.n_cells_resumed, summary.n_cells_empty,
    );
    eprintln!("Stars:    {}", summary.n_stars);
    eprintln!("Quads:    {}", summary.n_quads);
    eprintln!(
        "Index:    {} ({:.1} MB)",
        paths.final_index.display(),
        zdcl_size as f64 / 1024.0 / 1024.0
    );
    eprintln!(
        "Sidecar:  {} ({:.1} MB)",
        paths.final_sidecar.display(),
        sidecar_size as f64 / 1024.0 / 1024.0
    );
    eprintln!("Elapsed:  {:.1}s", elapsed.as_secs_f64());

    Ok(summary)
}

/// Adapter that turns a [`LazyLoadingCatalog<Dr3>`] into a
/// [`CellStarSource`]. Returns the brightest `max_stars_per_cell` rows
/// per cell after the source-side `mag_limit` gate.
struct LazyExcerptSource {
    lazy: LazyLoadingCatalog<Dr3>,
    mag_limit: f64,
    max_stars_per_cell: usize,
}

impl CellStarSource for LazyExcerptSource {
    fn cell_count(&self) -> u32 {
        self.lazy.cell_count()
    }

    fn stars_in_cell(&self, cell_id: u32) -> std::io::Result<Vec<CellStar>> {
        let it = self
            .lazy
            .entries_in_cell(cell_id, self.mag_limit)
            .map_err(|e| std::io::Error::other(format!("entries_in_cell({cell_id}): {e}")))?;

        // Max-heap on magnitude: faintest at top → ready to evict when
        // a brighter row arrives. Same shape as `HeapStar` in the
        // in-process uniformizer (`src/index/builder.rs`).
        let mut heap: BinaryHeap<MagOrdered> = BinaryHeap::with_capacity(self.max_stars_per_cell + 1);
        for r in it {
            let entry = r.map_err(|e| {
                std::io::Error::other(format!("entries_in_cell({cell_id}) row: {e}"))
            })?;
            // NaN mags would corrupt the heap ordering; skip defensively.
            if !entry.core.phot_g_mean_mag.is_finite() {
                continue;
            }
            let cs = entry_to_cell_star(&entry);
            if heap.len() < self.max_stars_per_cell {
                heap.push(MagOrdered(cs));
            } else if let Some(MagOrdered(faintest)) = heap.peek()
                && cs.mag < faintest.mag
            {
                heap.pop();
                heap.push(MagOrdered(cs));
            }
        }
        Ok(heap.into_iter().map(|MagOrdered(s)| s).collect())
    }
}

/// `CellStar` wrapped so a `BinaryHeap` orders by magnitude (faintest
/// at top). Ties are unordered — we just need a deterministic-enough
/// max-heap, not a strict total order on stars.
struct MagOrdered(CellStar);

impl PartialEq for MagOrdered {
    fn eq(&self, other: &Self) -> bool {
        self.0.mag == other.0.mag
    }
}
impl Eq for MagOrdered {}
impl PartialOrd for MagOrdered {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MagOrdered {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0
            .mag
            .partial_cmp(&other.0.mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Convert one `Dr3Entry` to a `CellStar`. Mirrors `entry_to_records`
/// in `build_from_shards.rs` — we don't reuse it directly because that
/// returns a tuple shape tied to the legacy in-RAM builder; the cell
/// builder expects the merged `CellStar` form.
fn entry_to_cell_star(entry: &Dr3Entry) -> CellStar {
    let core = &entry.core;
    let ra_rad = core.ra.to_radians();
    let dec_rad = core.dec.to_radians();
    let mag = core.phot_g_mean_mag;

    let radial_velocity = entry
        .radial_velocity
        .as_ref()
        .and_then(|rv| rv.radial_velocity)
        .unwrap_or(f64::NAN);

    let sidecar = SidecarRecord {
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

    CellStar {
        catalog_id: core.source_id,
        ra_rad,
        dec_rad,
        mag,
        sidecar,
    }
}

fn with_suffix(prefix: &std::path::Path, suffix: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(suffix);
    PathBuf::from(s)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_mag_limit() {
        let cfg = base_cfg(f64::NAN);
        assert!(run(&cfg).unwrap_err().contains("--mag-limit"));
    }

    #[test]
    fn rejects_zero_max_stars_per_cell() {
        let mut cfg = base_cfg(14.0);
        cfg.max_stars_per_cell = 0;
        assert!(
            run(&cfg)
                .unwrap_err()
                .contains("max-stars-per-cell")
        );
    }

    #[test]
    fn rejects_zero_quads_per_cell() {
        let mut cfg = base_cfg(14.0);
        cfg.quads_per_cell = 0;
        assert!(run(&cfg).unwrap_err().contains("quads-per-cell"));
    }

    #[test]
    fn rejects_inverted_scale_range() {
        let mut cfg = base_cfg(14.0);
        cfg.scale_lower_arcsec = 1800.0;
        cfg.scale_upper_arcsec = 30.0;
        assert!(run(&cfg).unwrap_err().contains("scale range"));
    }

    fn base_cfg(mag_limit: f64) -> BuildFromExcerptConfig {
        BuildFromExcerptConfig {
            excerpt_dir: PathBuf::from("/nonexistent"),
            output_prefix: PathBuf::from("/tmp/_zte_test"),
            work_dir: PathBuf::from("/tmp/_zte_work"),
            mag_limit,
            max_stars_per_cell: 100,
            quads_per_cell: 10,
            max_reuse: 8,
            scale_lower_arcsec: 30.0,
            scale_upper_arcsec: 1800.0,
            final_cell_depth: 5,
            pivot_stride: 4096,
            threads: Some(1),
        }
    }
}
