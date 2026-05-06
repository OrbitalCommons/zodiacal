//! `build-from-excerpt-series`: drive the multi-band `.zdcl.bundle`
//! build pipeline from a `starfield-gaia` "bycell" excerpt directory.
//!
//! This is the user-facing CLI surface for PR6 of the bundle-format
//! roadmap. It glues PR3's `build_bundle_work_dir` (parallel cell-driven
//! shard build into a work_dir) and PR4's `tidy_to_folder` /
//! `tidy_to_zip` (single-threaded finalize into a folder or zip
//! artifact) behind a single subcommand that consumes a bycell excerpt
//! directory laid out as `shard_NNNN.csv.gz` files (one per HEALPix
//! cell at the excerpt's depth).
//!
//! ## Source-depth constraint (PR6 v1)
//!
//! The CLI only supports the **fast path** where the excerpt's HEALPix
//! depth equals the bundle's `--cell-depth`. Cross-depth resharding (a
//! bundle built deeper or shallower than its source excerpt) is a
//! follow-up and rejected with a clear error today.
//!
//! ## Scale-band derivation
//!
//! The user supplies `--scale-lower`, `--scale-upper`, `--bands`. The
//! per-band scale factor is derived as
//! `(scale_upper / scale_lower)^(1/bands)` so band edges are
//! geometrically spaced. There is intentionally no `--scale-factor`
//! flag — over-specifying invites inconsistent inputs.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use chrono::Utc;
use starfield_gaia::download::Downloader;
use starfield_gaia::{Dr3, Dr3Catalog, Dr3Entry};

use zodiacal::bundle::ZdclBundle;
use zodiacal::bundle::gaia_shard::GaiaRecord;
use zodiacal::bundle::manifest::{BuildMetadata, BuildSource};
use zodiacal::bundle::tidy::{
    BandMetadata, GaiaMetadata, TidyMetadata, prune_work_dir, tidy_to_folder, tidy_to_zip,
};
use zodiacal::index::cell_builder::{CellStar, CellStarSource};
use zodiacal::index::multiband_cell_builder::{
    BundleWorkDirPaths, MultiBandCellBuildConfig, ScaleBand, build_bundle_work_dir,
};
use zodiacal::refinement::SidecarRecord;

/// Hard cap on the magnitude limit. Bundles write per-cell `.zga`
/// shards rather than one global sidecar, so the original `build-from-shards`
/// 17.0 cap doesn't apply — individual cells are bounded by
/// `--max-stars-per-cell`. We keep a sanity ceiling at 22 (one mag past
/// what Gaia DR3 publishes for `phot_g_mean_mag`) just to catch typos.
pub const MAX_MAG_LIMIT: f64 = 22.0;

/// Output format selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputFormat {
    /// Folder bundle (`<prefix>.zdcl.bundle/`).
    Dir,
    /// Zip-archive bundle (`<prefix>.zdcl.bundle.zip`).
    Zip,
    /// Both forms: emit folder first, then zip.
    Both,
}

/// Parsed CLI configuration for the `build-from-excerpt-series`
/// subcommand. The CLI layer in `main.rs` parses these from clap and
/// invokes [`run`].
#[derive(Debug, Clone)]
pub struct BuildFromExcerptSeriesConfig {
    /// Bycell excerpt directory containing `shard_NNNN.csv.gz` files
    /// (one per HEALPix cell at the excerpt's depth). `None` falls back
    /// to the starfield-managed cache directory.
    pub excerpt_dir: Option<PathBuf>,
    /// Build scratch directory. Survives across runs to support
    /// resume-from-crash; only deleted if `--prune-work-dir` is set.
    pub work_dir: PathBuf,
    /// Output target prefix; final artifact is `<prefix>.zdcl.bundle/`
    /// or `<prefix>.zdcl.bundle.zip` (or both, per `output_format`).
    pub output_prefix: PathBuf,
    /// Which form(s) to emit.
    pub output_format: OutputFormat,
    /// G-band magnitude cutoff for the source excerpt. Stars fainter
    /// than this are discarded as the CSVs are read.
    pub mag_limit: f64,
    /// Per-cell brightness-truncation cap (applied after the per-cell
    /// stars are loaded but before quad emission).
    pub max_stars_per_cell: usize,
    /// HEALPix nested-scheme depth at which the bundle is sharded. Must
    /// match the source excerpt's depth (PR6 v1 constraint).
    pub cell_depth: u8,
    /// Number of scale bands to emit.
    pub bands: usize,
    /// Lower edge of band 0, in arcseconds.
    pub scale_lower_arcsec: f64,
    /// Upper edge of band `bands - 1`, in arcseconds.
    pub scale_upper_arcsec: f64,
    /// Per-band quad-count cap.
    pub quads_per_cell: usize,
    /// Per-cell, per-band cap on how many times a single star may be
    /// referenced across emitted quads.
    pub max_reuse: u32,
    /// Free-text experiment label embedded into the manifest.
    pub experiment: String,
    /// Rayon thread pool size; pass `None` to use rayon's default
    /// (one worker per logical core).
    pub threads: Option<usize>,
    /// If true, [`prune_work_dir`] is called after a successful tidy.
    pub prune_work_dir: bool,
}

/// Top-level entry point.
pub fn run(cfg: &BuildFromExcerptSeriesConfig) -> Result<(), String> {
    if !cfg.mag_limit.is_finite() || cfg.mag_limit > MAX_MAG_LIMIT {
        return Err(format!(
            "--mag-limit {} exceeds hard cap {}; tighten the limit or run a deeper build with a streaming sidecar.",
            cfg.mag_limit, MAX_MAG_LIMIT,
        ));
    }
    if cfg.scale_lower_arcsec <= 0.0 || cfg.scale_upper_arcsec <= cfg.scale_lower_arcsec {
        return Err(format!(
            "invalid scale range: lower={}\" upper={}\" (must satisfy 0 < lower < upper)",
            cfg.scale_lower_arcsec, cfg.scale_upper_arcsec,
        ));
    }
    if cfg.bands == 0 {
        return Err("--bands must be > 0".to_string());
    }
    if cfg.quads_per_cell == 0 {
        return Err("--quads-per-cell must be > 0".to_string());
    }
    if cfg.max_stars_per_cell == 0 {
        return Err("--max-stars-per-cell must be > 0".to_string());
    }

    let started = Utc::now();
    let t0 = Instant::now();

    if let Some(n) = cfg.threads {
        let _ = rayon::ThreadPoolBuilder::new()
            .num_threads(n)
            .build_global();
    }

    // Resolve excerpt source. `None` → starfield-gaia's cached dir.
    let (excerpt_dir, source_label) = match cfg.excerpt_dir.as_ref() {
        Some(d) => (d.clone(), format!("directory {}", d.display())),
        None => {
            let cached = Downloader::<Dr3>::list_cached().map_err(|e| {
                format!(
                    "failed to list starfield-gaia cached shards: {e}; \
                     pass --excerpt-dir to override"
                )
            })?;
            // The cached list returns CSV(.gz) file paths; their parent
            // dir is the excerpt root.
            let parent = cached
                .first()
                .and_then(|p| p.parent())
                .map(|p| p.to_path_buf())
                .ok_or_else(|| "starfield-gaia cache is empty; pass --excerpt-dir".to_string())?;
            (
                parent.clone(),
                format!("starfield-gaia cache {}", parent.display()),
            )
        }
    };
    eprintln!("Excerpt source: {source_label}");

    let source = BycellExcerptSource::open(&excerpt_dir, cfg.cell_depth, cfg.mag_limit)?;
    eprintln!(
        "Indexed {} populated shard(s) at source_depth={}; bundle_depth={} (cell_count={})",
        source.populated_count(),
        source.source_depth(),
        cfg.cell_depth,
        source.cell_count(),
    );

    let bands = derive_scale_bands(
        cfg.bands,
        cfg.scale_lower_arcsec,
        cfg.scale_upper_arcsec,
        cfg.quads_per_cell,
        cfg.max_reuse as usize,
    );
    eprintln!(
        "Building {} band(s): scale=[{:.2}\",{:.2}\"], quads/cell={}, max_reuse={}",
        bands.len(),
        cfg.scale_lower_arcsec,
        cfg.scale_upper_arcsec,
        cfg.quads_per_cell,
        cfg.max_reuse,
    );

    let multi_cfg = MultiBandCellBuildConfig {
        bands: bands.clone(),
        max_stars_per_cell: cfg.max_stars_per_cell,
        cell_depth: cfg.cell_depth,
    };
    let paths = BundleWorkDirPaths {
        work_dir: cfg.work_dir.clone(),
    };

    let summary = build_bundle_work_dir(&source, &multi_cfg, &paths)
        .map_err(|e| format!("build_bundle_work_dir failed: {e}"))?;
    let build_elapsed = t0.elapsed();
    eprintln!(
        "Build phase done in {:.1}s: processed={}, resumed={}, empty={}, n_stars={}",
        build_elapsed.as_secs_f64(),
        summary.n_cells_processed,
        summary.n_cells_resumed,
        summary.n_cells_empty,
        summary.n_stars,
    );

    // Tidy phase.
    let finished = Utc::now();
    let metadata = TidyMetadata {
        cell_depth: cfg.cell_depth,
        experiment: cfg.experiment.clone(),
        build_metadata: BuildMetadata {
            tool: format!("zodiacal-tools {}", env!("CARGO_PKG_VERSION")),
            build_started_utc: started,
            build_finished_utc: finished,
            source: BuildSource {
                kind: "starfield-datasources-bycell".to_string(),
                release: "Dr3".to_string(),
                path: excerpt_dir.display().to_string(),
            },
        },
        gaia: GaiaMetadata {
            max_stars_per_cell: cfg.max_stars_per_cell as u32,
            mag_limit: cfg.mag_limit,
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

    let folder_path = with_suffix(&cfg.output_prefix, "zdcl.bundle");
    let zip_path = with_suffix(&cfg.output_prefix, "zdcl.bundle.zip");

    match cfg.output_format {
        OutputFormat::Dir => {
            tidy_to_folder(&cfg.work_dir, &folder_path, &metadata)
                .map_err(|e| format!("tidy_to_folder failed: {e}"))?;
            eprintln!("Folder bundle written: {}", folder_path.display());
        }
        OutputFormat::Zip => {
            tidy_to_zip(&cfg.work_dir, &zip_path, &metadata)
                .map_err(|e| format!("tidy_to_zip failed: {e}"))?;
            eprintln!("Zip bundle written: {}", zip_path.display());
        }
        OutputFormat::Both => {
            tidy_to_folder(&cfg.work_dir, &folder_path, &metadata)
                .map_err(|e| format!("tidy_to_folder failed: {e}"))?;
            eprintln!("Folder bundle written: {}", folder_path.display());
            tidy_to_zip(&cfg.work_dir, &zip_path, &metadata)
                .map_err(|e| format!("tidy_to_zip failed: {e}"))?;
            eprintln!("Zip bundle written: {}", zip_path.display());
        }
    }

    if cfg.prune_work_dir {
        prune_work_dir(&cfg.work_dir).map_err(|e| format!("prune_work_dir failed: {e}"))?;
        eprintln!("Pruned work dir {}", cfg.work_dir.display());
    } else {
        eprintln!(
            "Work dir preserved at {} (pass --prune-work-dir to remove)",
            cfg.work_dir.display(),
        );
    }

    eprintln!();
    eprintln!("=== Summary ===");
    eprintln!("Bands:         {}", bands.len());
    eprintln!("Cells built:   {}", summary.n_cells_processed);
    eprintln!("Cells resumed: {}", summary.n_cells_resumed);
    eprintln!("Total stars:   {}", summary.n_stars);
    for (k, b) in bands.iter().enumerate() {
        let q = summary
            .per_band_quad_counts
            .get(k)
            .copied()
            .unwrap_or_default();
        let pop = summary
            .per_band_populated_cells
            .get(k)
            .copied()
            .unwrap_or_default();
        eprintln!(
            "  band {k:02} [{:.2}\",{:.2}\"]: n_quads={q}, populated_cells={pop}",
            b.scale_lower_arcsec, b.scale_upper_arcsec,
        );
    }
    eprintln!("Elapsed:       {:.1}s", t0.elapsed().as_secs_f64());

    // Final sanity check: confirm the produced bundle opens.
    let opened_path: &Path = match cfg.output_format {
        OutputFormat::Dir | OutputFormat::Both => &folder_path,
        OutputFormat::Zip => &zip_path,
    };
    let _b = ZdclBundle::open(opened_path).map_err(|e| {
        format!(
            "post-tidy ZdclBundle::open({}) failed: {e}",
            opened_path.display()
        )
    })?;

    Ok(())
}

// ---------------------------------------------------------------------------
//  Scale-band derivation
// ---------------------------------------------------------------------------

/// Build a `Vec<ScaleBand>` from `--scale-lower / --scale-upper / --bands`,
/// using `factor = (upper / lower)^(1/bands)` so band edges are
/// geometrically spaced.
fn derive_scale_bands(
    n_bands: usize,
    scale_lower_arcsec: f64,
    scale_upper_arcsec: f64,
    quads_per_cell: usize,
    max_reuse: usize,
) -> Vec<ScaleBand> {
    let factor = (scale_upper_arcsec / scale_lower_arcsec).powf(1.0 / n_bands as f64);
    (0..n_bands)
        .map(|i| {
            let lower = scale_lower_arcsec * factor.powi(i as i32);
            let upper = scale_lower_arcsec * factor.powi(i as i32 + 1);
            ScaleBand {
                label: format!("band_{i:02}"),
                band_idx: i as u32,
                scale_lower_arcsec: lower,
                scale_upper_arcsec: upper,
                quads_per_cell,
                max_reuse,
            }
        })
        .collect()
}

// ---------------------------------------------------------------------------
//  BycellExcerptSource: walk shard_NNNN.csv.gz files in a bycell excerpt
//  directory, expose them as a CellStarSource.
// ---------------------------------------------------------------------------

/// `CellStarSource` impl for a starfield-gaia bycell excerpt — a
/// directory of `shard_NNNN.csv.gz` files whose shard index equals the
/// HEALPix nested-scheme cell id at the excerpt's `source_depth`.
///
/// Supports two modes:
/// 1. **Identity passthrough** (`bundle_depth == source_depth`): each
///    bundle cell's stars come from the matching shard.
/// 2. **Finer bundle** (`bundle_depth > source_depth`): each bundle
///    cell's stars come from the parent source cell, partitioned by
///    HEALPix at `bundle_depth`. The parent's CSV is loaded and
///    re-hashed once, and all 4^(delta) child cells share the result
///    via a small LRU cache.
///
/// Coarser bundles (`bundle_depth < source_depth`) are rejected.
pub struct BycellExcerptSource {
    // (Debug impl below — Mutex doesn't auto-derive a useful Debug.)
    /// `source_cell_id (= shard index at source_depth) -> CSV path`.
    cell_files: HashMap<u32, PathBuf>,
    /// One past the largest populated bundle cell id. The orchestrator
    /// iterates `0..cell_count()`; we keep it tight so empty cells in
    /// the deeper bundle layout don't cost a manifest fsync each.
    cell_count: u32,
    /// HEALPix depth of the source excerpt (typically 5 for the bycell
    /// directory).
    source_depth: u8,
    /// HEALPix depth of the bundle being built.
    bundle_depth: u8,
    /// Magnitude cap applied to the loaded CSV rows.
    mag_limit: f64,
    /// Cached partitions of source-cell CSV → child-cell stars. Only
    /// populated when `bundle_depth > source_depth`. Bounded LRU; size
    /// is small (handful of recently-loaded source cells per worker).
    partition_cache: std::sync::Mutex<PartitionCache>,
}

impl std::fmt::Debug for BycellExcerptSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BycellExcerptSource")
            .field("source_depth", &self.source_depth)
            .field("bundle_depth", &self.bundle_depth)
            .field("populated", &self.cell_files.len())
            .field("mag_limit", &self.mag_limit)
            .finish()
    }
}

const PARTITION_CACHE_CAPACITY: usize = 32;

struct PartitionCache {
    /// Map: source_cell_id → Arc<HashMap<bundle_cell_id, stars>>.
    entries: HashMap<u32, std::sync::Arc<HashMap<u32, Vec<CellStar>>>>,
    /// Insertion order, oldest first.
    order: Vec<u32>,
}

impl PartitionCache {
    fn new() -> Self {
        Self {
            entries: HashMap::new(),
            order: Vec::with_capacity(PARTITION_CACHE_CAPACITY + 1),
        }
    }
    fn get(&self, parent: u32) -> Option<std::sync::Arc<HashMap<u32, Vec<CellStar>>>> {
        self.entries.get(&parent).cloned()
    }
    fn insert(&mut self, parent: u32, value: std::sync::Arc<HashMap<u32, Vec<CellStar>>>) {
        if self.entries.contains_key(&parent) {
            return;
        }
        if self.entries.len() >= PARTITION_CACHE_CAPACITY {
            // Drop the oldest entry. O(N) Vec::remove is fine at small N.
            if !self.order.is_empty() {
                let evict = self.order.remove(0);
                self.entries.remove(&evict);
            }
        }
        self.entries.insert(parent, value);
        self.order.push(parent);
    }
}

impl BycellExcerptSource {
    /// Scan `excerpt_dir` for `shard_NNNN.csv.gz` files. The shard
    /// index range determines `source_depth` (chosen as the smallest
    /// depth whose `12 * 4^depth` count exceeds the largest shard id).
    /// `bundle_depth` may equal or exceed `source_depth`.
    pub fn open(excerpt_dir: &Path, bundle_depth: u8, mag_limit: f64) -> Result<Self, String> {
        if !excerpt_dir.is_dir() {
            return Err(format!(
                "--excerpt-dir {} is not a directory",
                excerpt_dir.display(),
            ));
        }
        let mut cell_files = HashMap::new();
        for entry in std::fs::read_dir(excerpt_dir)
            .map_err(|e| format!("failed to read {}: {e}", excerpt_dir.display()))?
        {
            let entry = entry.map_err(|e| format!("read_dir entry failed: {e}"))?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            let stripped = match name
                .strip_prefix("shard_")
                .and_then(|s| s.strip_suffix(".csv.gz"))
                .or_else(|| {
                    name.strip_prefix("shard_")
                        .and_then(|s| s.strip_suffix(".csv"))
                }) {
                Some(s) => s,
                None => continue,
            };
            let idx: u32 = match stripped.parse() {
                Ok(v) => v,
                Err(_) => continue,
            };
            cell_files.insert(idx, path);
        }
        if cell_files.is_empty() {
            return Err(format!(
                "no shard_*.csv(.gz) files found in {}",
                excerpt_dir.display(),
            ));
        }
        // Infer source_depth from the largest shard index. The
        // excerpt's depth is the smallest D with `12 * 4^D > max_idx`.
        let max_idx = cell_files.keys().copied().max().unwrap_or(0) as u64;
        let mut source_depth: u8 = 0;
        while (12u64 << (2 * source_depth as u64)) <= max_idx {
            source_depth += 1;
            if source_depth > 29 {
                return Err(format!(
                    "shard index {max_idx} too large to derive a HEALPix depth (≤ 29 supported)",
                ));
            }
        }
        if bundle_depth < source_depth {
            return Err(format!(
                "--cell-depth {bundle_depth} is shallower than excerpt depth {source_depth}; \
                 coarser-bundle resharding is not supported. Pass --cell-depth >= {source_depth}.",
            ));
        }

        let bundle_cells = 12u64 << (2 * bundle_depth as u64);
        // For bundle_depth > source_depth, every parent's children
        // contribute to bundle cells. We only iterate up to one past
        // the largest *possible* child of the largest populated source.
        let cell_count: u32 = if bundle_depth == source_depth {
            cell_files.keys().copied().max().map(|m| m + 1).unwrap_or(0)
        } else {
            let delta = bundle_depth - source_depth;
            let max_source = cell_files.keys().copied().max().unwrap_or(0) as u64;
            // Children of parent P at delta deeper: P << (2*delta) .. (P+1) << (2*delta)
            let max_child = ((max_source + 1) << (2 * delta as u64)) - 1;
            (max_child + 1).min(bundle_cells) as u32
        };

        Ok(Self {
            cell_files,
            cell_count,
            source_depth,
            bundle_depth,
            mag_limit,
            partition_cache: std::sync::Mutex::new(PartitionCache::new()),
        })
    }

    /// Source-excerpt depth (inferred from the shard index range).
    pub fn source_depth(&self) -> u8 {
        self.source_depth
    }

    /// Number of source shards on disk.
    pub fn populated_count(&self) -> usize {
        self.cell_files.len()
    }

    /// Load + partition one source cell's CSV. Idempotent; the result
    /// is cached so subsequent `stars_in_cell` calls for any of this
    /// source cell's bundle-depth children get O(1) access.
    fn load_and_partition(
        &self,
        source_cell: u32,
    ) -> std::io::Result<std::sync::Arc<HashMap<u32, Vec<CellStar>>>> {
        if let Some(hit) = self
            .partition_cache
            .lock()
            .expect("partition cache poisoned")
            .get(source_cell)
        {
            return Ok(hit);
        }

        let path = match self.cell_files.get(&source_cell) {
            Some(p) => p,
            None => {
                let empty = std::sync::Arc::new(HashMap::new());
                self.partition_cache
                    .lock()
                    .expect("partition cache poisoned")
                    .insert(source_cell, empty.clone());
                return Ok(empty);
            }
        };
        let catalog = Dr3Catalog::from_csv_file(path, self.mag_limit)
            .map_err(|e| std::io::Error::other(format!("{}: load failed: {e}", path.display())))?;
        let entries = catalog.0.brighter_than_ref(f64::INFINITY);

        let mut partitions: HashMap<u32, Vec<CellStar>> = HashMap::new();
        if self.bundle_depth == self.source_depth {
            // Identity case: everything in one bucket keyed by source_cell.
            let mut bucket = Vec::with_capacity(entries.len());
            for entry in entries {
                if !entry.core.phot_g_mean_mag.is_finite() {
                    continue;
                }
                bucket.push(entry_to_cell_star(entry));
            }
            partitions.insert(source_cell, bucket);
        } else {
            // Re-hash each entry at bundle_depth and bin.
            for entry in entries {
                if !entry.core.phot_g_mean_mag.is_finite() {
                    continue;
                }
                let star = entry_to_cell_star(entry);
                let bundle_cell =
                    cdshealpix::nested::hash(self.bundle_depth, star.ra_rad, star.dec_rad) as u32;
                partitions.entry(bundle_cell).or_default().push(star);
            }
        }

        let arc = std::sync::Arc::new(partitions);
        self.partition_cache
            .lock()
            .expect("partition cache poisoned")
            .insert(source_cell, arc.clone());
        Ok(arc)
    }
}

impl CellStarSource for BycellExcerptSource {
    fn cell_count(&self) -> u32 {
        self.cell_count
    }

    fn stars_in_cell(&self, cell_id: u32) -> std::io::Result<Vec<CellStar>> {
        // Find the parent source cell for this bundle cell.
        let source_cell = if self.bundle_depth == self.source_depth {
            cell_id
        } else {
            let delta = self.bundle_depth - self.source_depth;
            cell_id >> (2 * delta as u32)
        };
        let partition = self.load_and_partition(source_cell)?;
        Ok(partition.get(&cell_id).cloned().unwrap_or_default())
    }
}

/// Convert one `Dr3Entry` row to the [`CellStar`] tuple consumed by the
/// cell-driven builder. `sidecar` and `gaia` are populated from the
/// same source columns; the multi-band builder reads `gaia`, the
/// legacy single-band path would read `sidecar` (unused here but we
/// populate both for forward compat with mixed pipelines).
pub fn entry_to_cell_star(entry: &Dr3Entry) -> CellStar {
    let core = &entry.core;
    let ra_rad = core.ra.to_radians();
    let dec_rad = core.dec.to_radians();
    let mag = core.phot_g_mean_mag;

    let radial_velocity = entry
        .radial_velocity
        .as_ref()
        .and_then(|rv| rv.radial_velocity)
        .unwrap_or(f64::NAN);

    let pmra = core.pmra.unwrap_or(f64::NAN);
    let pmdec = core.pmdec.unwrap_or(f64::NAN);
    let parallax = core.parallax.unwrap_or(f64::NAN);
    let pmra_err = core.pmra_error.unwrap_or(f32::NAN);
    let pmdec_err = core.pmdec_error.unwrap_or(f32::NAN);
    let parallax_err = core.parallax_error.unwrap_or(f32::NAN);
    let ra_dec_corr = core.ra_dec_corr.unwrap_or(f32::NAN);
    let ruwe = entry.ipd.ruwe.unwrap_or(f32::NAN);

    let mut flags: u32 = 0;
    if !pmra.is_nan() && !pmdec.is_nan() {
        flags |= GaiaRecord::FLAG_HAS_PM;
    }
    if !parallax.is_nan() {
        flags |= GaiaRecord::FLAG_HAS_PARALLAX;
    }
    if !radial_velocity.is_nan() {
        flags |= GaiaRecord::FLAG_HAS_RADIAL_VELOCITY;
    }
    if !ra_dec_corr.is_nan() {
        flags |= GaiaRecord::FLAG_HAS_RA_DEC_CORR;
    }
    if !ruwe.is_nan() {
        flags |= GaiaRecord::FLAG_HAS_RUWE;
    }

    let sidecar = SidecarRecord {
        source_id: core.source_id,
        ref_epoch: core.ref_epoch,
        ra: core.ra,
        dec: core.dec,
        pmra,
        pmdec,
        parallax,
        radial_velocity,
        sigma_ra: core.ra_error,
        sigma_dec: core.dec_error,
        sigma_pmra: pmra_err,
        sigma_pmdec: pmdec_err,
        sigma_parallax: parallax_err,
        flags: 0,
    };

    let gaia = GaiaRecord {
        source_id: core.source_id,
        ref_epoch: core.ref_epoch,
        ra: core.ra,
        dec: core.dec,
        pmra,
        pmdec,
        parallax,
        radial_velocity,
        phot_g_mean_mag: mag,
        sigma_ra: core.ra_error,
        sigma_dec: core.dec_error,
        sigma_pmra: pmra_err,
        sigma_pmdec: pmdec_err,
        sigma_parallax: parallax_err,
        ra_dec_corr,
        ruwe,
        flags,
    };

    CellStar {
        catalog_id: core.source_id,
        ra_rad,
        dec_rad,
        mag,
        sidecar,
        gaia,
    }
}

/// Append `.<suffix>` to `prefix`. Used for the same reason
/// `build-from-shards` does: `Path::set_extension` would clobber any
/// existing trailing extension on the prefix.
fn with_suffix(prefix: &Path, suffix: &str) -> PathBuf {
    let mut s = prefix.as_os_str().to_owned();
    s.push(".");
    s.push(suffix);
    PathBuf::from(s)
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Drop a zero-byte file with the expected name into `dir`. The
    /// excerpt-source open path only inspects filenames; CSV parsing is
    /// deferred to the per-cell `stars_in_cell` call, which the
    /// discovery-only tests don't exercise.
    fn touch_synthetic_shard(dir: &Path, shard_idx: u32) {
        let path = dir.join(format!("shard_{shard_idx:04}.csv.gz"));
        std::fs::write(&path, b"").unwrap();
    }

    #[test]
    fn derive_scale_bands_geometric() {
        let bands = derive_scale_bands(3, 10.0, 80.0, 100, 8);
        assert_eq!(bands.len(), 3);
        assert!((bands[0].scale_lower_arcsec - 10.0).abs() < 1e-9);
        assert!((bands[2].scale_upper_arcsec - 80.0).abs() < 1e-9);
        // Geometric: each upper == 2x lower.
        for b in &bands {
            let ratio = b.scale_upper_arcsec / b.scale_lower_arcsec;
            assert!((ratio - 2.0).abs() < 1e-9, "non-geometric band: {b:?}");
        }
        // band_idx is dense and labels are zero-padded.
        for (i, b) in bands.iter().enumerate() {
            assert_eq!(b.band_idx as usize, i);
            assert_eq!(b.label, format!("band_{i:02}"));
        }
    }

    #[test]
    fn excerpt_source_open_rejects_coarser_bundle_than_source() {
        let tmp = tempfile::tempdir().unwrap();
        // shard 99 → smallest source_depth with 12*4^D > 99 is D=2
        // (12*16=192). Asking for a bundle at depth 0 (coarser than 2)
        // is rejected — coarser-bundle resharding is unsupported.
        touch_synthetic_shard(tmp.path(), 99);
        let err = BycellExcerptSource::open(tmp.path(), 0, 16.0).unwrap_err();
        assert!(
            err.contains("shallower than excerpt depth"),
            "expected coarser-bundle rejection, got: {err}"
        );
    }

    #[test]
    fn excerpt_source_open_empty_dir_errors() {
        let tmp = tempfile::tempdir().unwrap();
        let err = BycellExcerptSource::open(tmp.path(), 5, 16.0).unwrap_err();
        assert!(
            err.contains("no shard_"),
            "expected empty-dir error, got: {err}"
        );
    }

    #[test]
    fn excerpt_source_identity_passthrough_at_inferred_depth() {
        let tmp = tempfile::tempdir().unwrap();
        // Shard ids ≤ 42 → smallest source_depth with 12*4^D > 42 is
        // D=1 (12*4=48 > 42). With bundle_depth == source_depth, the
        // identity passthrough applies; cell_count is one past the
        // largest populated.
        for &cid in &[0u32, 1, 7, 42] {
            touch_synthetic_shard(tmp.path(), cid);
        }
        let src = BycellExcerptSource::open(tmp.path(), 1, 16.0).unwrap();
        assert_eq!(src.populated_count(), 4);
        assert_eq!(src.source_depth(), 1);
        assert_eq!(src.cell_count(), 43);
    }

    #[test]
    fn excerpt_source_finer_bundle_expands_cell_count() {
        let tmp = tempfile::tempdir().unwrap();
        // Inferred source_depth=1; ask for bundle_depth=5 → each parent
        // has 4^4 = 256 children; cell_count = (max_parent+1) << 8.
        for &cid in &[0u32, 1, 7, 42] {
            touch_synthetic_shard(tmp.path(), cid);
        }
        let src = BycellExcerptSource::open(tmp.path(), 5, 16.0).unwrap();
        assert_eq!(src.source_depth(), 1);
        assert_eq!(src.cell_count(), 43 << 8);
    }

    /// End-to-end smoke: build a tiny bundle programmatically using a
    /// hand-rolled `CellStarSource` (which avoids the brittle CSV
    /// round-trip through `Dr3Catalog::from_csv_file`), then assert the
    /// produced bundle opens via `ZdclBundle::open` and has the
    /// expected band count.
    #[test]
    fn run_e2e_synthetic_two_band_bundle() {
        use chrono::{DateTime, Utc};
        use zodiacal::bundle::ZdclBundle;
        use zodiacal::bundle::gaia_shard::GaiaRecord;
        use zodiacal::bundle::manifest::{BuildMetadata, BuildSource};
        use zodiacal::bundle::tidy::{BandMetadata, GaiaMetadata, TidyMetadata, tidy_to_folder};
        use zodiacal::index::cell_builder::{CellStar, CellStarSource};
        use zodiacal::index::multiband_cell_builder::{
            BundleWorkDirPaths, MultiBandCellBuildConfig, build_bundle_work_dir,
        };
        use zodiacal::refinement::SidecarRecord;

        const TEST_DEPTH: u8 = 5;

        struct ClusteredSource {
            cells: Vec<u32>,
            stars_per_cell: usize,
        }
        impl CellStarSource for ClusteredSource {
            fn cell_count(&self) -> u32 {
                self.cells.iter().copied().max().unwrap_or(0) + 1
            }
            fn stars_in_cell(&self, cell_id: u32) -> std::io::Result<Vec<CellStar>> {
                if !self.cells.contains(&cell_id) {
                    return Ok(Vec::new());
                }
                // Place each cell's stars at distinct fixed offsets in
                // an arbitrary patch of sky. The bundle reader doesn't
                // re-validate that stars actually fall in their declared
                // cells — the source's cell-assignment is taken as
                // ground truth. This keeps the test independent of
                // cdshealpix's exact center coordinates.
                let base_ra = 0.5 + cell_id as f64 * 0.01;
                let base_dec = 0.1;
                let mut out = Vec::with_capacity(self.stars_per_cell);
                for i in 0..self.stars_per_cell {
                    let dra = (i as f64 - 4.0) * 1e-5;
                    let ddec = (i as f64 - 4.0) * 7e-6;
                    let ra = base_ra + dra;
                    let dec = base_dec + ddec;
                    let cell_id_12: u64 = (cell_id as u64) << (2 * (12 - TEST_DEPTH as u64));
                    let source_id: u64 = (cell_id_12 << 35) | (i as u64 + 1);
                    let mag = 12.0 + i as f64 * 0.01;
                    out.push(CellStar {
                        catalog_id: source_id,
                        ra_rad: ra,
                        dec_rad: dec,
                        mag,
                        sidecar: SidecarRecord {
                            source_id,
                            ref_epoch: 2016.0,
                            ra: ra.to_degrees(),
                            dec: dec.to_degrees(),
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
                        gaia: GaiaRecord {
                            source_id,
                            ref_epoch: 2016.0,
                            ra: ra.to_degrees(),
                            dec: dec.to_degrees(),
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
                        },
                    });
                }
                Ok(out)
            }
        }

        let work_tmp = tempfile::tempdir().unwrap();
        let out_tmp = tempfile::tempdir().unwrap();
        let cells = vec![0u32, 1, 7];
        let source = ClusteredSource {
            cells: cells.clone(),
            stars_per_cell: 8,
        };
        let bands = derive_scale_bands(2, 1.0, 50.0, 50, 8);
        let cfg = MultiBandCellBuildConfig {
            bands: bands.clone(),
            max_stars_per_cell: 1_000,
            cell_depth: TEST_DEPTH,
        };
        let paths = BundleWorkDirPaths {
            work_dir: work_tmp.path().to_path_buf(),
        };
        build_bundle_work_dir(&source, &cfg, &paths).unwrap();

        let bundle_path = out_tmp.path().join("smoke.zdcl.bundle");
        let metadata = TidyMetadata {
            cell_depth: TEST_DEPTH,
            experiment: "build-from-excerpt-series smoke".into(),
            build_metadata: BuildMetadata {
                tool: "zodiacal-tools test".into(),
                build_started_utc: "2026-05-05T00:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                build_finished_utc: "2026-05-05T01:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                source: BuildSource {
                    kind: "test-fixture".into(),
                    release: "test".into(),
                    path: "/tmp".into(),
                },
            },
            gaia: GaiaMetadata {
                max_stars_per_cell: 1_000,
                mag_limit: 20.0,
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
        tidy_to_folder(work_tmp.path(), &bundle_path, &metadata).unwrap();

        // Bundle opens, reports two bands.
        let bundle = ZdclBundle::open(&bundle_path).unwrap();
        assert_eq!(bundle.bands().len(), 2);
        assert!(bundle_path.join("manifest.json").is_file());

        // Loader returns one Index per band.
        let bands_loaded = zodiacal::index::store::load_bundle_bands(&bundle_path).unwrap();
        assert_eq!(bands_loaded.len(), 2);
    }
}
