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

/// Validated configuration for [`run`]. Construct via
/// [`BuildFromExcerptConfig::builder`] — fields are private so the
/// only path to a value goes through builder validation, and `run`
/// trusts what it gets.
#[derive(Debug)]
pub struct BuildFromExcerptConfig {
    excerpt_dir: PathBuf,
    output_prefix: PathBuf,
    work_dir: PathBuf,
    mag_limit: f64,
    max_stars_per_cell: usize,
    quads_per_cell: usize,
    max_reuse: usize,
    scale_lower_arcsec: f64,
    scale_upper_arcsec: f64,
    final_cell_depth: u8,
    pivot_stride: u32,
    threads: Option<usize>,
}

impl BuildFromExcerptConfig {
    pub fn builder() -> BuildFromExcerptConfigBuilder {
        BuildFromExcerptConfigBuilder::default()
    }
}

/// Fluent builder for [`BuildFromExcerptConfig`]. Required fields
/// (`excerpt_dir`, `output_prefix`, `work_dir`, `mag_limit`,
/// `max_stars_per_cell`, `quads_per_cell`, `max_reuse`,
/// `scale_lower_arcsec`, `scale_upper_arcsec`, `final_cell_depth`,
/// `pivot_stride`) error out at `build()` if not set. Defaults for
/// optional knobs live in the clap layer (`main.rs`), not here, so
/// the API boundary stays declarative.
#[derive(Default)]
pub struct BuildFromExcerptConfigBuilder {
    excerpt_dir: Option<PathBuf>,
    output_prefix: Option<PathBuf>,
    work_dir: Option<PathBuf>,
    mag_limit: Option<f64>,
    max_stars_per_cell: Option<usize>,
    quads_per_cell: Option<usize>,
    max_reuse: Option<usize>,
    scale_lower_arcsec: Option<f64>,
    scale_upper_arcsec: Option<f64>,
    final_cell_depth: Option<u8>,
    pivot_stride: Option<u32>,
    threads: Option<usize>,
}

impl BuildFromExcerptConfigBuilder {
    pub fn excerpt_dir(mut self, v: PathBuf) -> Self {
        self.excerpt_dir = Some(v);
        self
    }
    pub fn output_prefix(mut self, v: PathBuf) -> Self {
        self.output_prefix = Some(v);
        self
    }
    pub fn work_dir(mut self, v: PathBuf) -> Self {
        self.work_dir = Some(v);
        self
    }
    pub fn mag_limit(mut self, v: f64) -> Self {
        self.mag_limit = Some(v);
        self
    }
    pub fn max_stars_per_cell(mut self, v: usize) -> Self {
        self.max_stars_per_cell = Some(v);
        self
    }
    pub fn quads_per_cell(mut self, v: usize) -> Self {
        self.quads_per_cell = Some(v);
        self
    }
    pub fn max_reuse(mut self, v: usize) -> Self {
        self.max_reuse = Some(v);
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
    pub fn final_cell_depth(mut self, v: u8) -> Self {
        self.final_cell_depth = Some(v);
        self
    }
    pub fn pivot_stride(mut self, v: u32) -> Self {
        self.pivot_stride = Some(v);
        self
    }
    pub fn threads(mut self, v: Option<usize>) -> Self {
        self.threads = v;
        self
    }

    /// Validate fields and finalise the config. Same checks the
    /// `run` function used to do up front, hoisted here so a
    /// constructed `BuildFromExcerptConfig` is always usable.
    pub fn build(self) -> Result<BuildFromExcerptConfig, String> {
        let excerpt_dir = self
            .excerpt_dir
            .ok_or_else(|| "excerpt_dir is required".to_string())?;
        let output_prefix = self
            .output_prefix
            .ok_or_else(|| "output_prefix is required".to_string())?;
        let work_dir = self
            .work_dir
            .ok_or_else(|| "work_dir is required".to_string())?;
        let mag_limit = self
            .mag_limit
            .ok_or_else(|| "mag_limit is required".to_string())?;
        let max_stars_per_cell = self
            .max_stars_per_cell
            .ok_or_else(|| "max_stars_per_cell is required".to_string())?;
        let quads_per_cell = self
            .quads_per_cell
            .ok_or_else(|| "quads_per_cell is required".to_string())?;
        let max_reuse = self
            .max_reuse
            .ok_or_else(|| "max_reuse is required".to_string())?;
        let scale_lower_arcsec = self
            .scale_lower_arcsec
            .ok_or_else(|| "scale_lower_arcsec is required".to_string())?;
        let scale_upper_arcsec = self
            .scale_upper_arcsec
            .ok_or_else(|| "scale_upper_arcsec is required".to_string())?;
        let final_cell_depth = self
            .final_cell_depth
            .ok_or_else(|| "final_cell_depth is required".to_string())?;
        let pivot_stride = self
            .pivot_stride
            .ok_or_else(|| "pivot_stride is required".to_string())?;

        if !mag_limit.is_finite() || mag_limit <= 0.0 {
            return Err(format!("invalid mag_limit: {mag_limit}"));
        }
        if max_stars_per_cell == 0 {
            return Err("max_stars_per_cell must be positive".into());
        }
        if quads_per_cell == 0 {
            return Err("quads_per_cell must be positive".into());
        }
        if max_reuse == 0 {
            return Err("max_reuse must be positive".into());
        }
        if pivot_stride == 0 {
            return Err("pivot_stride must be positive".into());
        }
        if scale_lower_arcsec <= 0.0 || scale_upper_arcsec <= scale_lower_arcsec {
            return Err(format!(
                "invalid scale range: lower={scale_lower_arcsec}\" upper={scale_upper_arcsec}\" \
                 (must satisfy 0 < lower < upper)",
            ));
        }

        Ok(BuildFromExcerptConfig {
            excerpt_dir,
            output_prefix,
            work_dir,
            mag_limit,
            max_stars_per_cell,
            quads_per_cell,
            max_reuse,
            scale_lower_arcsec,
            scale_upper_arcsec,
            final_cell_depth,
            pivot_stride,
            threads: self.threads,
        })
    }
}

/// Top-level entry point. Trusts the config — validation happened in
/// `BuildFromExcerptConfigBuilder::build`.
pub fn run(cfg: &BuildFromExcerptConfig) -> Result<BuildSummary, String> {
    if let Some(n) = cfg.threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }

    let lazy = LazyLoadingCatalog::<Dr3>::open(&cfg.excerpt_dir).map_err(|e| {
        format!(
            "LazyLoadingCatalog::open({}): {e}",
            cfg.excerpt_dir.display()
        )
    })?;

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
        let mut heap: BinaryHeap<MagOrdered> =
            BinaryHeap::with_capacity(self.max_stars_per_cell + 1);
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

    fn valid_builder() -> BuildFromExcerptConfigBuilder {
        BuildFromExcerptConfig::builder()
            .excerpt_dir(PathBuf::from("/nonexistent"))
            .output_prefix(PathBuf::from("/tmp/_zte_test"))
            .work_dir(PathBuf::from("/tmp/_zte_work"))
            .mag_limit(14.0)
            .max_stars_per_cell(100)
            .quads_per_cell(10)
            .max_reuse(8)
            .scale_lower_arcsec(30.0)
            .scale_upper_arcsec(1800.0)
            .final_cell_depth(5)
            .pivot_stride(4096)
            .threads(Some(1))
    }

    #[test]
    fn build_rejects_nan_mag_limit() {
        let err = valid_builder().mag_limit(f64::NAN).build().unwrap_err();
        assert!(err.contains("mag_limit"), "got: {err}");
    }

    #[test]
    fn build_rejects_negative_mag_limit() {
        let err = valid_builder().mag_limit(-1.0).build().unwrap_err();
        assert!(err.contains("mag_limit"), "got: {err}");
    }

    #[test]
    fn build_rejects_zero_max_stars_per_cell() {
        let err = valid_builder().max_stars_per_cell(0).build().unwrap_err();
        assert!(err.contains("max_stars_per_cell"), "got: {err}");
    }

    #[test]
    fn build_rejects_zero_quads_per_cell() {
        let err = valid_builder().quads_per_cell(0).build().unwrap_err();
        assert!(err.contains("quads_per_cell"), "got: {err}");
    }

    #[test]
    fn build_rejects_zero_max_reuse() {
        let err = valid_builder().max_reuse(0).build().unwrap_err();
        assert!(err.contains("max_reuse"), "got: {err}");
    }

    #[test]
    fn build_rejects_zero_pivot_stride() {
        let err = valid_builder().pivot_stride(0).build().unwrap_err();
        assert!(err.contains("pivot_stride"), "got: {err}");
    }

    #[test]
    fn build_rejects_inverted_scale_range() {
        let err = valid_builder()
            .scale_lower_arcsec(1800.0)
            .scale_upper_arcsec(30.0)
            .build()
            .unwrap_err();
        assert!(err.contains("scale range"), "got: {err}");
    }

    #[test]
    fn build_rejects_missing_required_fields() {
        let err = BuildFromExcerptConfig::builder().build().unwrap_err();
        assert!(err.contains("excerpt_dir"), "got: {err}");
    }

    #[test]
    fn build_succeeds_on_valid_inputs() {
        valid_builder().build().expect("valid builder must succeed");
    }
}
