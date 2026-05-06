//! Multi-band, cell-driven bundle work-dir builder.
//!
//! Phase 1 of the bundle build pipeline: workers iterate cells in
//! parallel, build every scale band's quads over each cell's star
//! buffer, and emit per-cell `.zqd` (multi-band) and `.zga` shards
//! into a work directory. The tidy phase finalizes the work_dir into
//! a folder/zip bundle.
//!
//! See `docs/bundle-format.md` § "Phase 1 — parallel shard build" for
//! the design.
//!
//! ## Two-phase build (PR8)
//!
//! - **Phase A** (gaia-only): walk cells in parallel, brightness-truncate,
//!   sort by `source_id`, and write `.zga` shards atomically. Each
//!   cell's gaia shard is the durable input for Phase B; once all cells
//!   are committed, every cross-cell quad emitter has the data it needs
//!   to build patch quads.
//! - **Phase B** (patch quads): walk cells in parallel again. For each
//!   cell C, parse its and its eight HEALPix edge-neighbors' `.zga`
//!   files into one combined patch buffer, build every band's quads over
//!   the patch, and write the resulting `.zqd` shard with a neighbor
//!   table referencing whichever neighbor cells the quads actually used.
//!
//! Phase B's per-cell builds are independent (they only read sibling
//! `.zga` files, never write to them), so the second pass parallelises
//! cleanly. Cells whose quads turn out to be entirely self-contained
//! emit a `.zqd` with an empty neighbor table — bit-identical (apart
//! from the `version = 2` field and the `n_neighbors = 0` slot) to the
//! pre-PR8 layout.
//!
//! ## Shape
//!
//! - `MultiBandCellBuildConfig` carries a `Vec<ScaleBand>` plus the
//!   global brightness-truncation knobs.
//! - `BundleWorkDirPaths` names the work directory.
//! - `build_bundle_work_dir(source, config, paths)` is the
//!   orchestrator. It returns a `BuildBundleSummary` and leaves
//!   `work_dir` populated with `quads/cell_NNNNN.zqd` +
//!   `gaia/cell_NNNNN.zga` files plus a `build-manifest.json`.
//!
//! ## Resume granularity
//!
//! A cell is "done" iff its `.zga` exists AND `is_complete(cell_id)`
//! AND every band is `is_band_complete(cell_id, band_idx)`. Phase A
//! commits a cell as `is_complete` (and as empty in `completed_per_band`
//! for empty cells) once its `.zga` is durable on disk. Phase B
//! flips each band complete after the `.zqd` rename succeeds. Resume
//! semantics: a cell whose `.zga` was committed but whose `.zqd` was
//! not skips Phase A (already done) and runs only Phase B.

use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use rayon::prelude::*;

use crate::bundle::gaia_shard::{GaiaRecord, GaiaShard, write_gaia_shard};
use crate::bundle::layout::cell_shard_path;
use crate::bundle::quad_shard::{
    BandEmit, MAX_NEIGHBORS, QuadShard, pack_star_id, write_quad_shard,
};
use crate::quads::{Code, DIMQUADS, Quad};

use super::build_manifest::{BuildManifest, CellStats};
use super::cell_builder::{CellStar, CellStarSource};
use super::quads::build_quads_for_cell_multiband;

/// One scale band in a multi-band bundle build.
#[derive(Debug, Clone)]
pub struct ScaleBand {
    /// Human-readable label written into the bundle manifest later
    /// (PR4+); not persisted by the work-dir builder. Carrying it on
    /// the in-memory config lets the same struct flow through the
    /// whole pipeline.
    pub label: String,
    /// Stable band index. Rendered into the `.zqd` band table; must be
    /// dense `[0, n_bands)` across the input slice.
    pub band_idx: u32,
    /// Lower bound on quad backbone length, arcseconds.
    pub scale_lower_arcsec: f64,
    /// Upper bound on quad backbone length, arcseconds. Must be
    /// strictly greater than `scale_lower_arcsec`.
    pub scale_upper_arcsec: f64,
    /// Quads to emit per HEALPix cell for this band.
    pub quads_per_cell: usize,
    /// Per-cell, per-band cap on how many times a single star may be
    /// referenced across the band's quads.
    pub max_reuse: usize,
}

/// Top-level multi-band cell-driven build configuration.
#[derive(Debug, Clone)]
pub struct MultiBandCellBuildConfig {
    /// One entry per scale band.
    pub bands: Vec<ScaleBand>,
    /// Brightness-truncation cap per cell. Sources may already filter
    /// upstream, but the orchestrator enforces this defensively.
    pub max_stars_per_cell: usize,
    /// Magnitude limit recorded as informational metadata. Not
    /// enforced by the orchestrator — the source filters upstream.
    pub mag_limit: f64,
    /// HEALPix depth at which cells are sharded.
    pub cell_depth: u8,
}

/// Filesystem layout for the build's work directory.
#[derive(Debug, Clone)]
pub struct BundleWorkDirPaths {
    /// Holds `quads/`, `gaia/`, and `build-manifest.json`.
    pub work_dir: PathBuf,
}

/// Summary returned by [`build_bundle_work_dir`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BuildBundleSummary {
    /// Cells processed in this invocation (excluding resumed-from-prior).
    pub n_cells_processed: u32,
    /// Cells already complete on entry, skipped via the manifest.
    pub n_cells_resumed: u32,
    /// Cells whose sources returned zero stars after filtering.
    pub n_cells_empty: u32,
    /// Total stars committed (including resumed).
    pub n_stars: u64,
    /// Per-band running quad totals.
    pub per_band_quad_counts: Vec<u64>,
    /// Number of populated cells per band.
    pub per_band_populated_cells: Vec<u32>,
}

/// Subdirectory names under work_dir.
pub const QUADS_SUBDIR: &str = "quads";
pub const GAIA_SUBDIR: &str = "gaia";

/// File extensions for per-cell shard files.
pub const QUAD_EXT: &str = "zqd";
pub const GAIA_EXT: &str = "zga";

// ---------------------------------------------------------------------------
//  Validation
// ---------------------------------------------------------------------------

fn validate_config(config: &MultiBandCellBuildConfig) -> io::Result<()> {
    if config.bands.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "MultiBandCellBuildConfig.bands must be non-empty",
        ));
    }
    let n = config.bands.len() as u32;
    let mut seen = vec![false; n as usize];
    for b in &config.bands {
        if b.scale_lower_arcsec <= 0.0 || b.scale_upper_arcsec <= b.scale_lower_arcsec {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "band {} has invalid scale range [{}, {}] arcsec",
                    b.band_idx, b.scale_lower_arcsec, b.scale_upper_arcsec
                ),
            ));
        }
        if b.band_idx >= n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "band_idx {} out of dense range [0, {n}); bands must be dense",
                    b.band_idx
                ),
            ));
        }
        if seen[b.band_idx as usize] {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("duplicate band_idx {}", b.band_idx),
            ));
        }
        seen[b.band_idx as usize] = true;
    }
    if !seen.iter().all(|x| *x) {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "band_idx values are not dense across [0, n_bands)",
        ));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
//  Cleanup of orphaned partials
// ---------------------------------------------------------------------------

/// Remove `*.part.*` files left behind by crashed worker writes.
///
/// Called at the start of every build session and exposed for tests.
pub fn cleanup_work_dir_partials(work_dir: &Path) -> io::Result<()> {
    for sub in [QUADS_SUBDIR, GAIA_SUBDIR] {
        let dir = work_dir.join(sub);
        if !dir.exists() {
            continue;
        }
        for entry in std::fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let name = name.to_string_lossy();
            if name.contains(".part.") {
                let _ = std::fs::remove_file(entry.path());
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
//  Per-cell write
// ---------------------------------------------------------------------------

/// Build an 8-hex-digit nonce derived from pid + thread + nanos +
/// cell_id + an optional caller seed. Stable enough across concurrent
/// workers that two competing partial filenames don't collide.
fn nonce_hex(cell_id: u32, seed: u64) -> String {
    let pid = std::process::id();
    let tid = format!("{:?}", std::thread::current().id());
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let mut h = DefaultHasher::new();
    pid.hash(&mut h);
    tid.hash(&mut h);
    nanos.hash(&mut h);
    cell_id.hash(&mut h);
    seed.hash(&mut h);
    let v = h.finish();
    format!("{:08x}", v as u32)
}

/// Phase A per-cell write: emit just the `.zga` shard atomically.
///
/// Returns the on-disk gaia records (sorted by source_id, mutated in
/// place by the writer). The caller may discard them — Phase B re-reads
/// every cell's `.zga` from disk so it always sees the durable bytes.
fn write_gaia_shard_only(
    work_dir: &Path,
    cell_depth: u8,
    cell_id: u32,
    mut gaia_records: Vec<GaiaRecord>,
    nonce_seed: u64,
) -> io::Result<()> {
    if gaia_records.is_empty() {
        return Ok(());
    }
    let nonce = nonce_hex(cell_id, nonce_seed);
    let zga_final = cell_shard_path(work_dir, GAIA_SUBDIR, GAIA_EXT, cell_depth, cell_id);
    let zga_partial: PathBuf = {
        let mut s = zga_final.as_os_str().to_owned();
        s.push(".part.");
        s.push(&nonce);
        PathBuf::from(s)
    };
    {
        let f = File::create(&zga_partial)?;
        let mut w = BufWriter::new(f);
        write_gaia_shard(&mut w, cell_id as u64, &mut gaia_records)?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    std::fs::rename(&zga_partial, &zga_final)?;
    Ok(())
}

/// Phase B per-cell write: emit the `.zqd` shard atomically, including
/// any neighbor-table entries referenced by patch quads.
fn write_quad_shard_only(
    work_dir: &Path,
    cell_depth: u8,
    cell_id: u32,
    neighbor_cells: &[u64],
    bands_emit: &[BandEmit<'_>],
    nonce_seed: u64,
) -> io::Result<()> {
    let all_quads_empty = bands_emit.iter().all(|b| b.quads.is_empty());
    // Even cells with no quads in any band still need a .zqd on disk so
    // that the reader's populated-cell enumeration (which lists `.zqd`
    // files) finds them. Skip only when there are no bands at all.
    if bands_emit.is_empty() && all_quads_empty {
        return Ok(());
    }

    let nonce = nonce_hex(cell_id, nonce_seed);
    let zqd_final = cell_shard_path(work_dir, QUADS_SUBDIR, QUAD_EXT, cell_depth, cell_id);
    let zqd_partial: PathBuf = {
        let mut s = zqd_final.as_os_str().to_owned();
        s.push(".part.");
        s.push(&nonce);
        PathBuf::from(s)
    };
    {
        let f = File::create(&zqd_partial)?;
        let mut w = BufWriter::new(f);
        write_quad_shard(&mut w, cell_id as u64, neighbor_cells, bands_emit)?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    std::fs::rename(&zqd_partial, &zqd_final)?;
    Ok(())
}

// ---------------------------------------------------------------------------
//  Orchestrator
// ---------------------------------------------------------------------------

/// Two-phase orchestrator: write every cell's `.zga` (Phase A), then
/// build cross-cell patch quads against each cell + its HEALPix
/// neighbors and write `.zqd` (Phase B).
///
/// Both phases iterate cells in parallel and update the build manifest
/// after every successful per-cell commit. Resume semantics are
/// per-phase: a cell whose `.zga` was committed but whose `.zqd` was
/// not skips Phase A and runs only Phase B on the next invocation.
pub fn build_bundle_work_dir<S: CellStarSource + ?Sized>(
    source: &S,
    config: &MultiBandCellBuildConfig,
    paths: &BundleWorkDirPaths,
) -> io::Result<BuildBundleSummary> {
    validate_config(config)?;

    let work_dir = paths.work_dir.as_path();
    std::fs::create_dir_all(work_dir.join(QUADS_SUBDIR))?;
    std::fs::create_dir_all(work_dir.join(GAIA_SUBDIR))?;

    cleanup_work_dir_partials(work_dir)?;

    let n_bands = config.bands.len();

    let mut manifest = BuildManifest::load(work_dir)?.unwrap_or_default();
    manifest.ensure_per_band(n_bands);
    manifest.save(work_dir)?;

    // Sort the bands once up front; pass them to both phases in a
    // stable band-idx order so the band table on disk matches the
    // configured layout.
    let mut sorted_bands: Vec<ScaleBand> = config.bands.clone();
    sorted_bands.sort_by_key(|b| b.band_idx);

    let cell_count = source.cell_count();
    let cell_depth = config.cell_depth;

    // Capture which cells were already fully done (both phases
    // committed) before this invocation so the summary's
    // `n_cells_resumed` only counts those.
    let initially_resumed: std::collections::HashSet<u32> = (0..cell_count)
        .filter(|&cell_id| {
            manifest.is_complete(cell_id)
                && (0..n_bands as u32).all(|b| manifest.is_band_complete(cell_id, b))
        })
        .collect();

    // -------- Phase A: walk cells, write .zga shards. ----------------
    //
    // A cell is in scope for Phase A iff its `.zga` is not yet on disk.
    // Resume picks up cleanly: previously committed `.zga`s skip the
    // (potentially expensive) per-cell pull, but their cell ids stay in
    // `manifest.completed_cells` so Phase B can iterate them.
    let mut phase_a_cells: Vec<u32> = Vec::new();
    for cell_id in 0..cell_count {
        let zga = cell_shard_path(work_dir, GAIA_SUBDIR, GAIA_EXT, cell_depth, cell_id);
        // Cell already had Phase A done iff:
        //   - it's marked complete in the manifest, AND
        //   - either its zga exists (had stars) OR commit_cell stats are zero (empty cell).
        // The simplest way to encode that is: skip iff manifest.is_complete(cell_id)
        // AND (zga exists OR the cell is recorded with zero n_stars).
        if manifest.is_complete(cell_id) {
            // Empty cells stay empty on resume; cells with stars must
            // have an existing zga (or we'd have crashed before the
            // commit_cell line).
            if zga.exists() || cell_recorded_empty(&manifest, cell_id) {
                continue;
            }
        }
        phase_a_cells.push(cell_id);
    }

    let manifest = Mutex::new(manifest);
    let n_cells_empty = std::sync::atomic::AtomicU32::new(0);
    let max_stars_per_cell = config.max_stars_per_cell;

    let phase_a: io::Result<()> = phase_a_cells.par_iter().try_for_each(|&cell_id| {
        let mut stars = source.stars_in_cell(cell_id)?;

        if stars.len() > max_stars_per_cell {
            stars.sort_by(|a, b| {
                a.mag
                    .partial_cmp(&b.mag)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
            stars.truncate(max_stars_per_cell);
        }

        if stars.is_empty() {
            let mut m = manifest.lock().unwrap();
            m.commit_cell(work_dir, cell_id, CellStats::default())?;
            // Empty cells are also "done" for every band — there are
            // no patch quads to build.
            for b in &sorted_bands {
                m.mark_band_complete(cell_id, b.band_idx);
            }
            m.save(work_dir)?;
            drop(m);
            n_cells_empty.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            return Ok::<(), io::Error>(());
        }

        let gaia_records: Vec<GaiaRecord> = stars
            .iter()
            .map(|s| {
                let mut g = s.gaia;
                if (s.gaia.source_id >> 63) & 1 == 1 {
                    g.flags |= GaiaRecord::FLAG_SOURCE_KIND_SUPPLEMENT;
                }
                g
            })
            .collect();

        let n_stars = stars.len() as u64;
        write_gaia_shard_only(work_dir, cell_depth, cell_id, gaia_records, cell_id as u64)?;

        let mut m = manifest.lock().unwrap();
        m.commit_cell(
            work_dir,
            cell_id,
            CellStats {
                n_stars,
                n_quads: 0, // Filled in by Phase B.
            },
        )?;
        m.save(work_dir)?;
        drop(m);

        Ok(())
    });
    phase_a?;

    // Best-effort fsync the gaia subdirectory so every committed `.zga`
    // is durably named before Phase B starts reading them.
    if let Ok(dir) = File::open(work_dir.join(GAIA_SUBDIR)) {
        let _ = dir.sync_all();
    }

    // -------- Phase B: walk cells, build patch quads, write .zqd. -----
    //
    // A cell is in scope for Phase B iff:
    //   - it has a `.zga` on disk (i.e. Phase A produced records), AND
    //   - at least one of its bands is not yet marked complete.
    let mut phase_b_cells: Vec<u32> = Vec::new();
    {
        let m = manifest.lock().unwrap();
        for cell_id in 0..cell_count {
            let zga = cell_shard_path(work_dir, GAIA_SUBDIR, GAIA_EXT, cell_depth, cell_id);
            if !zga.exists() {
                continue;
            }
            let all_bands_done = (0..n_bands as u32).all(|b| m.is_band_complete(cell_id, b));
            if !all_bands_done {
                phase_b_cells.push(cell_id);
            }
        }
    }
    let phase_b: io::Result<()> = phase_b_cells.par_iter().try_for_each(|&cell_id| {
        // Compute neighbor cell ids at this depth via cdshealpix. The
        // returned list is up to 8 ids; some of them may not have been
        // populated by Phase A (empty cells), so the patch loader filters
        // those out.
        let mut neighbors: Vec<u64> = Vec::with_capacity(8);
        cdshealpix::nested::append_bulk_neighbours(
            cell_depth,
            cell_id as u64,
            &mut neighbors,
        );
        // Self goes first so its records dominate the patch when ties
        // matter; neighbors follow in deterministic ascending order.
        // (cdshealpix returns them in a direction-enum order we don't
        // care about; sort for reproducibility regardless of caller.)
        neighbors.sort_unstable();
        let mut patch_cells: Vec<u64> = Vec::with_capacity(neighbors.len() + 1);
        patch_cells.push(cell_id as u64);
        for &nc in &neighbors {
            if nc == cell_id as u64 {
                continue; // defensive — shouldn't happen, but skip self repeats
            }
            patch_cells.push(nc);
        }

        // Read every patch cell's `.zga` once; assemble:
        //   combined_stars    Vec<CellStar>     for build_quads_for_cell_multiband
        //   per_star_origin   Vec<(cell_id, local_idx)>  parallel, one per star
        let mut combined_stars: Vec<CellStar> = Vec::new();
        let mut per_star_origin: Vec<(u64, u32)> = Vec::new();
        for &pc in &patch_cells {
            // Only depth-fitted u32 cells exist on disk in this build.
            if pc > u32::MAX as u64 {
                continue;
            }
            let zga_path =
                cell_shard_path(work_dir, GAIA_SUBDIR, GAIA_EXT, cell_depth, pc as u32);
            if !zga_path.exists() {
                continue;
            }
            let bytes = std::fs::read(&zga_path)?;
            let shard = GaiaShard::parse(&bytes).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "phase B: cell {cell_id} failed to parse neighbor {pc} .zga: {e}"
                    ),
                )
            })?;
            for (local_idx, rec) in shard.records().iter().enumerate() {
                combined_stars.push(gaia_record_to_cell_star(rec));
                per_star_origin.push((pc, local_idx as u32));
            }
        }

        if combined_stars.is_empty() {
            // Defensive: zga existed but was empty. Mark all bands
            // complete so we don't loop forever on resume.
            let mut m = manifest.lock().unwrap();
            for b in &sorted_bands {
                m.mark_band_complete(cell_id, b.band_idx);
            }
            m.save(work_dir)?;
            return Ok::<(), io::Error>(());
        }

        // Build all bands' quads over the combined patch buffer. Quad
        // star_ids are positions in `combined_stars`, which we then map
        // back to `(source_cell_id, local_idx)` via per_star_origin.
        //
        // Patch quads can span cell boundaries; the same physical
        // asterism would be discovered redundantly in every cell whose
        // patch encloses all 4 stars. To emit each quad exactly once
        // (and to keep per-cell `quads_per_cell` budgets honest), filter
        // each quad to the file of its centroid cell — the HEALPix cell
        // containing the unit-vector mean of the 4 member stars. The
        // builder runs with an inflated budget so the post-filter still
        // hits each cell's nominal quad count for the band.
        let inflated_bands: Vec<ScaleBand> = sorted_bands
            .iter()
            .map(|b| ScaleBand {
                quads_per_cell: b.quads_per_cell.saturating_mul(9),
                ..b.clone()
            })
            .collect();
        let raw_band_blocks =
            build_quads_for_cell_multiband(&combined_stars, &inflated_bands);

        let band_blocks: Vec<(u32, Vec<Quad>, Vec<Code>)> =
            raw_band_blocks
                .into_iter()
                .zip(sorted_bands.iter())
                .map(|((band_idx, raw_quads, raw_codes), band_cfg)| {
                    let mut keep_quads: Vec<Quad> =
                        Vec::with_capacity(band_cfg.quads_per_cell);
                    let mut keep_codes: Vec<Code> =
                        Vec::with_capacity(band_cfg.quads_per_cell);
                    for (q, c) in raw_quads.into_iter().zip(raw_codes) {
                        if keep_quads.len() >= band_cfg.quads_per_cell {
                            break;
                        }
                        let centroid = quad_centroid(&q, &combined_stars);
                        let centroid_cell = cdshealpix::nested::hash(
                            cell_depth,
                            centroid.0,
                            centroid.1,
                        );
                        if centroid_cell == cell_id as u64 {
                            keep_quads.push(q);
                            keep_codes.push(c);
                        }
                    }
                    (band_idx, keep_quads, keep_codes)
                })
                .collect();

        // Encode each quad's star_ids into the v2 packed format and
        // build the per-cell neighbor table from the unique non-self
        // source cell ids actually referenced.
        let self_cell = cell_id as u64;
        let mut neighbor_table: Vec<u64> = Vec::new();
        // Maps a referenced neighbor cell id → 1-based slot in neighbor_table.
        let mut neighbor_slot: std::collections::HashMap<u64, u32> =
            std::collections::HashMap::new();

        let mut packed_blocks: Vec<(u32, Vec<Quad>, Vec<Code>)> =
            Vec::with_capacity(band_blocks.len());
        for (band_idx, quads, codes) in band_blocks {
            let mut packed_quads: Vec<Quad> = Vec::with_capacity(quads.len());
            for q in quads {
                let mut new_ids = [0usize; DIMQUADS];
                for (i, &combined_idx) in q.star_ids.iter().enumerate() {
                    let (origin_cell, origin_local) = per_star_origin[combined_idx];
                    let nbr_idx = if origin_cell == self_cell {
                        0u32
                    } else {
                        match neighbor_slot.get(&origin_cell) {
                            Some(&slot) => slot,
                            None => {
                                let next_slot = neighbor_table.len() as u32 + 1;
                                if next_slot
                                    > MAX_NEIGHBORS as u32
                                {
                                    return Err(io::Error::new(
                                        io::ErrorKind::InvalidInput,
                                        format!(
                                            "phase B: cell {cell_id} would reference {} neighbor cells, exceeding the {}-cell on-disk cap",
                                            next_slot,
                                            MAX_NEIGHBORS,
                                        ),
                                    ));
                                }
                                neighbor_table.push(origin_cell);
                                neighbor_slot.insert(origin_cell, next_slot);
                                next_slot
                            }
                        }
                    };
                    let packed = pack_star_id(nbr_idx, origin_local)?;
                    new_ids[i] = packed as usize;
                }
                packed_quads.push(Quad { star_ids: new_ids });
            }
            packed_blocks.push((band_idx, packed_quads, codes));
        }

        let bands_emit: Vec<BandEmit<'_>> = packed_blocks
            .iter()
            .map(|(idx, qs, cs)| BandEmit {
                band_idx: *idx,
                quads: qs,
                codes: cs,
            })
            .collect();

        write_quad_shard_only(
            work_dir,
            cell_depth,
            cell_id,
            &neighbor_table,
            &bands_emit,
            cell_id as u64,
        )?;

        let n_quads_total: u64 = packed_blocks.iter().map(|(_, qs, _)| qs.len() as u64).sum();

        // Update the manifest: the cell is already in completed_cells
        // (Phase A added it); commit_cell with the same n_stars and the
        // new n_quads is idempotent in cell_stats but updates n_quads.
        let mut m = manifest.lock().unwrap();
        let existing_n_stars = m
            .cell_stats
            .iter()
            .find(|(c, _)| *c == cell_id)
            .map(|(_, s)| s.n_stars)
            .unwrap_or(0);
        m.commit_cell(
            work_dir,
            cell_id,
            CellStats {
                n_stars: existing_n_stars,
                n_quads: n_quads_total,
            },
        )?;
        for b in &sorted_bands {
            m.mark_band_complete(cell_id, b.band_idx);
        }
        m.save(work_dir)?;
        drop(m);

        Ok(())
    });
    phase_b?;

    let final_manifest = manifest.into_inner().unwrap();

    // n_cells_processed = unique cells that needed any work this run
    // (Phase A, Phase B, or both). Cells already done before the run
    // are tracked via `n_cells_resumed`. Empty cells are tracked
    // separately.
    let mut touched: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for &c in &phase_a_cells {
        touched.insert(c);
    }
    for &c in &phase_b_cells {
        touched.insert(c);
    }
    // Empty cells are part of phase_a_cells but tracked separately on
    // the summary. Subtract them from n_cells_processed so the field
    // reflects "cells that produced shards on disk" not "cells visited".
    let n_empty = n_cells_empty.into_inner();
    let n_cells_processed = (touched.len() as u32).saturating_sub(n_empty);
    let n_cells_resumed = initially_resumed.len() as u32;

    // ---------- per-band totals via mmapped re-scan ----------------------
    let mut per_band_quad_counts = vec![0u64; n_bands];
    let mut per_band_populated_cells = vec![0u32; n_bands];
    for &cell_id in &final_manifest.completed_cells {
        let zqd_path = cell_shard_path(work_dir, QUADS_SUBDIR, QUAD_EXT, cell_depth, cell_id);
        if !zqd_path.exists() {
            continue;
        }
        let bytes = std::fs::read(&zqd_path)?;
        let shard = QuadShard::parse(&bytes)?;
        for entry in shard.bands() {
            let k = entry.band_idx as usize;
            if k < n_bands {
                let n = entry.n_quads as u64;
                per_band_quad_counts[k] += n;
                if n > 0 {
                    per_band_populated_cells[k] += 1;
                }
            }
        }
    }

    Ok(BuildBundleSummary {
        n_cells_processed,
        n_cells_resumed,
        n_cells_empty: n_empty,
        n_stars: final_manifest.n_stars,
        per_band_quad_counts,
        per_band_populated_cells,
    })
}

/// Compute a quad's spherical centroid — the unit-vector mean of its
/// four member stars, projected back to (RA, Dec) in radians. Used to
/// decide which HEALPix cell's `.zqd` a patch quad belongs to: by
/// convention each quad is emitted once, in the file of the cell
/// containing its centroid.
fn quad_centroid(q: &Quad, stars: &[CellStar]) -> (f64, f64) {
    use crate::geom::sphere::radec_to_xyz;
    let mut sum = [0.0f64; 3];
    for &i in &q.star_ids {
        let s = &stars[i];
        let v = radec_to_xyz(s.ra_rad, s.dec_rad);
        for k in 0..3 {
            sum[k] += v[k];
        }
    }
    // Normalize back to a unit vector.
    let n = (sum[0] * sum[0] + sum[1] * sum[1] + sum[2] * sum[2]).sqrt();
    if n == 0.0 {
        return (0.0, 0.0);
    }
    let x = sum[0] / n;
    let y = sum[1] / n;
    let z = sum[2] / n;
    let dec = z.clamp(-1.0, 1.0).asin();
    let ra = y.atan2(x).rem_euclid(2.0 * std::f64::consts::PI);
    (ra, dec)
}

/// True iff the manifest records `cell_id` with zero stars (i.e. an
/// empty-cell commit). Used by Phase A's resume check to distinguish
/// "this cell was empty and we already noted that" from "this cell
/// crashed mid-write and needs a retry".
fn cell_recorded_empty(manifest: &BuildManifest, cell_id: u32) -> bool {
    manifest
        .cell_stats
        .iter()
        .any(|(c, s)| *c == cell_id && s.n_stars == 0)
}

/// Build a `CellStar` minimally from a `GaiaRecord` — just the four
/// fields `build_quads_for_cell_multiband` actually consumes (RA/Dec in
/// radians, magnitude, catalog_id). The unused sidecar/gaia fields are
/// populated from the same record for completeness, but the inner quad
/// builder never reads them.
fn gaia_record_to_cell_star(g: &GaiaRecord) -> CellStar {
    CellStar {
        catalog_id: g.source_id,
        ra_rad: g.ra.to_radians(),
        dec_rad: g.dec.to_radians(),
        mag: g.phot_g_mean_mag,
        // The patch builder never inspects these; populate them from
        // the same record so the struct is well-formed.
        sidecar: crate::refinement::SidecarRecord {
            source_id: g.source_id,
            ref_epoch: g.ref_epoch,
            ra: g.ra,
            dec: g.dec,
            pmra: g.pmra,
            pmdec: g.pmdec,
            parallax: g.parallax,
            radial_velocity: g.radial_velocity,
            sigma_ra: g.sigma_ra,
            sigma_dec: g.sigma_dec,
            sigma_pmra: g.sigma_pmra,
            sigma_pmdec: g.sigma_pmdec,
            sigma_parallax: g.sigma_parallax,
            flags: g.flags,
        },
        gaia: *g,
    }
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::gaia_shard::GaiaShard;
    use crate::index::cell_builder::CellStar;
    use crate::refinement::SidecarRecord;
    use std::cell::Cell;
    use std::sync::atomic::{AtomicU32, Ordering};

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "zodiacal-multiband-{}-{}-{}",
            name,
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
    }

    fn make_gaia(source_id: u64, mag: f64) -> GaiaRecord {
        GaiaRecord {
            source_id,
            ref_epoch: 2016.0,
            ra: 0.0,
            dec: 0.0,
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

    fn make_cell_star(catalog_id: u64, ra_rad: f64, dec_rad: f64, mag: f64) -> CellStar {
        CellStar {
            catalog_id,
            ra_rad,
            dec_rad,
            mag,
            sidecar: SidecarRecord {
                source_id: catalog_id,
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
            gaia: GaiaRecord {
                ra: ra_rad.to_degrees(),
                dec: dec_rad.to_degrees(),
                ..make_gaia(catalog_id, mag)
            },
        }
    }

    /// Synthetic source with `n_cells` cells, each populated with
    /// `stars_per_cell` deterministically-spaced stars. Gives quads
    /// for the small bands (it's not trying to fit large quads).
    struct SyntheticSource {
        n_cells: u32,
        stars_per_cell: usize,
        skip_cells: Vec<u32>,
    }

    impl CellStarSource for SyntheticSource {
        fn cell_count(&self) -> u32 {
            self.n_cells
        }
        fn stars_in_cell(&self, cell_id: u32) -> io::Result<Vec<CellStar>> {
            if self.skip_cells.contains(&cell_id) {
                return Ok(Vec::new());
            }
            let base_ra = 0.5 + cell_id as f64 * 0.01;
            let base_dec = 0.0;
            let mut out = Vec::with_capacity(self.stars_per_cell);
            for i in 0..self.stars_per_cell {
                let dra = (i as f64) * 0.0001;
                let ddec = (i as f64) * 0.00007;
                let catalog_id = (cell_id as u64) * 1000 + (i as u64) + 1;
                out.push(make_cell_star(
                    catalog_id,
                    base_ra + dra,
                    base_dec + ddec,
                    (i as f64) * 0.5 + 1.0,
                ));
            }
            Ok(out)
        }
    }

    fn three_bands() -> Vec<ScaleBand> {
        vec![
            ScaleBand {
                label: "band_00".into(),
                band_idx: 0,
                scale_lower_arcsec: 5.0,
                scale_upper_arcsec: 50.0,
                quads_per_cell: 50,
                max_reuse: 8,
            },
            ScaleBand {
                label: "band_01".into(),
                band_idx: 1,
                scale_lower_arcsec: 50.0,
                scale_upper_arcsec: 200.0,
                quads_per_cell: 50,
                max_reuse: 8,
            },
            ScaleBand {
                label: "band_02".into(),
                band_idx: 2,
                scale_lower_arcsec: 200.0,
                scale_upper_arcsec: 800.0,
                quads_per_cell: 50,
                max_reuse: 8,
            },
        ]
    }

    fn default_config() -> MultiBandCellBuildConfig {
        MultiBandCellBuildConfig {
            bands: three_bands(),
            max_stars_per_cell: 10_000,
            mag_limit: 20.0,
            cell_depth: 5,
        }
    }

    #[test]
    fn end_to_end_synthetic_4_cells_3_bands() {
        let work = tmp_dir("e2e");
        let paths = BundleWorkDirPaths {
            work_dir: work.clone(),
        };
        let source = SyntheticSource {
            n_cells: 4,
            stars_per_cell: 8,
            skip_cells: vec![],
        };
        let cfg = default_config();

        let summary = build_bundle_work_dir(&source, &cfg, &paths).unwrap();
        assert_eq!(summary.n_cells_processed, 4);
        assert_eq!(summary.n_cells_resumed, 0);
        assert_eq!(summary.n_cells_empty, 0);
        assert_eq!(summary.n_stars, 32);
        assert_eq!(summary.per_band_quad_counts.len(), 3);

        for cell_id in 0..4u32 {
            let zqd = cell_shard_path(&work, QUADS_SUBDIR, QUAD_EXT, cfg.cell_depth, cell_id);
            let zga = cell_shard_path(&work, GAIA_SUBDIR, GAIA_EXT, cfg.cell_depth, cell_id);
            assert!(zqd.exists(), ".zqd missing for cell {cell_id}");
            assert!(zga.exists(), ".zga missing for cell {cell_id}");

            let bytes = std::fs::read(&zqd).unwrap();
            let shard = QuadShard::parse(&bytes).unwrap();
            assert_eq!(shard.cell_id(), cell_id as u64);
            assert_eq!(shard.n_bands(), 3);

            let g_bytes = std::fs::read(&zga).unwrap();
            let g = GaiaShard::parse(&g_bytes).unwrap();
            assert_eq!(g.cell_id(), cell_id as u64);
            assert_eq!(g.len(), 8);
            // Sorted by source_id?
            let ids: Vec<u64> = g.records().iter().map(|r| r.source_id).collect();
            let mut expected = ids.clone();
            expected.sort();
            assert_eq!(ids, expected, "gaia not sorted by source_id");
        }

        // Manifest sanity.
        let m = BuildManifest::load(&work).unwrap().unwrap();
        for cell_id in 0..4 {
            assert!(m.is_complete(cell_id));
            for b in 0..3u32 {
                assert!(m.is_band_complete(cell_id, b));
            }
        }

        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn empty_cells_omitted_from_disk() {
        let work = tmp_dir("empty");
        let paths = BundleWorkDirPaths {
            work_dir: work.clone(),
        };
        let source = SyntheticSource {
            n_cells: 4,
            stars_per_cell: 8,
            skip_cells: vec![1],
        };
        let cfg = default_config();

        let summary = build_bundle_work_dir(&source, &cfg, &paths).unwrap();
        assert_eq!(summary.n_cells_empty, 1);

        let zqd1 = cell_shard_path(&work, QUADS_SUBDIR, QUAD_EXT, cfg.cell_depth, 1);
        let zga1 = cell_shard_path(&work, GAIA_SUBDIR, GAIA_EXT, cfg.cell_depth, 1);
        assert!(!zqd1.exists());
        assert!(!zga1.exists());

        let m = BuildManifest::load(&work).unwrap().unwrap();
        assert!(m.is_complete(1));

        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn crash_resume_band_aware() {
        // First run: source returns IO error on cell 2; expect cells
        // 0+1 to commit, then the build to error out.
        let work = tmp_dir("resume");
        let paths = BundleWorkDirPaths {
            work_dir: work.clone(),
        };

        // We need to control parallelism so the crash test is
        // deterministic. Run it under a 1-thread rayon pool.
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();

        struct CrashOnce {
            n_cells: u32,
            stars_per_cell: usize,
            crashed: Cell<bool>,
            crash_at: u32,
            pull_count: AtomicU32,
        }
        // SAFETY: tests are single-threaded under the rayon pool above; Cell suffices.
        unsafe impl Sync for CrashOnce {}

        impl CellStarSource for CrashOnce {
            fn cell_count(&self) -> u32 {
                self.n_cells
            }
            fn stars_in_cell(&self, cell_id: u32) -> io::Result<Vec<CellStar>> {
                self.pull_count.fetch_add(1, Ordering::Relaxed);
                if cell_id == self.crash_at && !self.crashed.get() {
                    self.crashed.set(true);
                    return Err(io::Error::other("simulated crash"));
                }
                let mut out = Vec::with_capacity(self.stars_per_cell);
                for i in 0..self.stars_per_cell {
                    let cid = (cell_id as u64) * 1000 + (i as u64) + 1;
                    out.push(make_cell_star(
                        cid,
                        0.5 + (cell_id as f64) * 0.01 + (i as f64) * 0.0001,
                        0.0 + (i as f64) * 0.00007,
                        (i as f64) * 0.5 + 1.0,
                    ));
                }
                Ok(out)
            }
        }

        let cfg = default_config();
        let source = CrashOnce {
            n_cells: 4,
            stars_per_cell: 8,
            crashed: Cell::new(false),
            crash_at: 2,
            pull_count: AtomicU32::new(0),
        };

        let err = pool
            .install(|| build_bundle_work_dir(&source, &cfg, &paths))
            .unwrap_err();
        assert!(err.to_string().contains("simulated crash"));

        // Manifest must record cells 0+1 complete.
        let m = BuildManifest::load(&work).unwrap().unwrap();
        assert!(m.is_complete(0));
        assert!(m.is_complete(1));
        assert!(!m.is_complete(2));

        // Phase 2: resume with non-erroring source.
        let source2 = CrashOnce {
            n_cells: 4,
            stars_per_cell: 8,
            crashed: Cell::new(true),
            crash_at: u32::MAX,
            pull_count: AtomicU32::new(0),
        };
        let summary = build_bundle_work_dir(&source2, &cfg, &paths).unwrap();
        // After a crash mid-Phase-A, cells 0+1 had Phase A done but
        // not Phase B; on resume they need Phase B work, so they're
        // counted in `n_cells_processed`. No cell was fully done before
        // the resume started (a fully-done cell needs both phases), so
        // `n_cells_resumed` is zero. All four cells go through Phase B.
        assert_eq!(summary.n_cells_resumed, 0);
        assert_eq!(summary.n_cells_processed, 4);
        // Cells 0+1 must NOT be re-pulled (their .zga is durable from
        // the previous run, so Phase A skips them).
        assert_eq!(source2.pull_count.load(Ordering::Relaxed), 2);

        // Compare to a fresh clean build.
        let work_clean = tmp_dir("resume-clean");
        let paths_clean = BundleWorkDirPaths {
            work_dir: work_clean.clone(),
        };
        let source_clean = CrashOnce {
            n_cells: 4,
            stars_per_cell: 8,
            crashed: Cell::new(true),
            crash_at: u32::MAX,
            pull_count: AtomicU32::new(0),
        };
        build_bundle_work_dir(&source_clean, &cfg, &paths_clean).unwrap();

        for cell_id in 0..4u32 {
            let a = cell_shard_path(&work, QUADS_SUBDIR, QUAD_EXT, cfg.cell_depth, cell_id);
            let b = cell_shard_path(&work_clean, QUADS_SUBDIR, QUAD_EXT, cfg.cell_depth, cell_id);
            let g_a = cell_shard_path(&work, GAIA_SUBDIR, GAIA_EXT, cfg.cell_depth, cell_id);
            let g_b = cell_shard_path(&work_clean, GAIA_SUBDIR, GAIA_EXT, cfg.cell_depth, cell_id);
            assert_eq!(std::fs::read(&a).unwrap(), std::fs::read(&b).unwrap());
            assert_eq!(std::fs::read(&g_a).unwrap(), std::fs::read(&g_b).unwrap());
        }

        let _ = std::fs::remove_dir_all(&work);
        let _ = std::fs::remove_dir_all(&work_clean);
    }

    #[test]
    fn output_byte_identical_1_thread_vs_8_thread() {
        let cfg = default_config();
        let source_1 = SyntheticSource {
            n_cells: 4,
            stars_per_cell: 8,
            skip_cells: vec![],
        };
        let source_8 = SyntheticSource {
            n_cells: 4,
            stars_per_cell: 8,
            skip_cells: vec![],
        };

        let work_1 = tmp_dir("threads-1");
        let work_8 = tmp_dir("threads-8");
        let paths_1 = BundleWorkDirPaths {
            work_dir: work_1.clone(),
        };
        let paths_8 = BundleWorkDirPaths {
            work_dir: work_8.clone(),
        };

        let pool_1 = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        let pool_8 = rayon::ThreadPoolBuilder::new()
            .num_threads(8)
            .build()
            .unwrap();

        pool_1
            .install(|| build_bundle_work_dir(&source_1, &cfg, &paths_1))
            .unwrap();
        pool_8
            .install(|| build_bundle_work_dir(&source_8, &cfg, &paths_8))
            .unwrap();

        for cell_id in 0..4u32 {
            let a = cell_shard_path(&work_1, QUADS_SUBDIR, QUAD_EXT, cfg.cell_depth, cell_id);
            let b = cell_shard_path(&work_8, QUADS_SUBDIR, QUAD_EXT, cfg.cell_depth, cell_id);
            let g_a = cell_shard_path(&work_1, GAIA_SUBDIR, GAIA_EXT, cfg.cell_depth, cell_id);
            let g_b = cell_shard_path(&work_8, GAIA_SUBDIR, GAIA_EXT, cfg.cell_depth, cell_id);
            assert_eq!(
                std::fs::read(&a).unwrap(),
                std::fs::read(&b).unwrap(),
                "zqd differs on cell {cell_id}"
            );
            assert_eq!(
                std::fs::read(&g_a).unwrap(),
                std::fs::read(&g_b).unwrap(),
                "zga differs on cell {cell_id}"
            );
        }

        let _ = std::fs::remove_dir_all(&work_1);
        let _ = std::fs::remove_dir_all(&work_8);
    }

    #[test]
    fn empty_band_in_middle_emits_zero_quads_block() {
        // Band 1 has a scale range that fits no quad in the synthetic
        // tight cluster. Bands 0 and 2 still produce quads.
        let work = tmp_dir("empty-band");
        let paths = BundleWorkDirPaths {
            work_dir: work.clone(),
        };

        let cfg = MultiBandCellBuildConfig {
            bands: vec![
                ScaleBand {
                    label: "band_00".into(),
                    band_idx: 0,
                    scale_lower_arcsec: 5.0,
                    scale_upper_arcsec: 50.0,
                    quads_per_cell: 50,
                    max_reuse: 8,
                },
                ScaleBand {
                    // Far too wide for the synthetic cluster (~tens of arcsec).
                    label: "band_01".into(),
                    band_idx: 1,
                    scale_lower_arcsec: 18000.0,
                    scale_upper_arcsec: 36000.0,
                    quads_per_cell: 50,
                    max_reuse: 8,
                },
                ScaleBand {
                    label: "band_02".into(),
                    band_idx: 2,
                    scale_lower_arcsec: 5.0,
                    scale_upper_arcsec: 200.0,
                    quads_per_cell: 50,
                    max_reuse: 8,
                },
            ],
            max_stars_per_cell: 10_000,
            mag_limit: 20.0,
            cell_depth: 5,
        };

        let source = SyntheticSource {
            n_cells: 1,
            stars_per_cell: 8,
            skip_cells: vec![],
        };

        let summary = build_bundle_work_dir(&source, &cfg, &paths).unwrap();
        assert_eq!(summary.per_band_quad_counts.len(), 3);
        assert_eq!(summary.per_band_quad_counts[1], 0);
        assert_eq!(summary.per_band_populated_cells[1], 0);

        let zqd = cell_shard_path(&work, QUADS_SUBDIR, QUAD_EXT, cfg.cell_depth, 0);
        let bytes = std::fs::read(&zqd).unwrap();
        let shard = QuadShard::parse(&bytes).unwrap();
        assert_eq!(shard.n_bands(), 3);
        let middle = shard.band(1).unwrap();
        assert_eq!(middle.n_quads(), 0);

        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn per_band_quad_counts_match_band_table() {
        let work = tmp_dir("counts-match");
        let paths = BundleWorkDirPaths {
            work_dir: work.clone(),
        };
        let cfg = default_config();
        let source = SyntheticSource {
            n_cells: 1,
            stars_per_cell: 12,
            skip_cells: vec![],
        };

        let summary = build_bundle_work_dir(&source, &cfg, &paths).unwrap();

        let zqd = cell_shard_path(&work, QUADS_SUBDIR, QUAD_EXT, cfg.cell_depth, 0);
        let bytes = std::fs::read(&zqd).unwrap();
        let shard = QuadShard::parse(&bytes).unwrap();
        for entry in shard.bands() {
            let k = entry.band_idx as usize;
            assert_eq!(
                entry.n_quads as u64, summary.per_band_quad_counts[k],
                "band {k} mismatch"
            );
        }

        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn mag_truncation_applied() {
        struct LotsOfStars;
        impl CellStarSource for LotsOfStars {
            fn cell_count(&self) -> u32 {
                1
            }
            fn stars_in_cell(&self, _cell_id: u32) -> io::Result<Vec<CellStar>> {
                let mut out = Vec::with_capacity(50);
                for i in 0..50u64 {
                    let mag = (50 - i) as f64 * 0.1; // descending mag → mostly fainter first
                    out.push(make_cell_star(
                        i + 1,
                        0.5 + (i as f64) * 0.0001,
                        0.0 + (i as f64) * 0.00007,
                        mag,
                    ));
                }
                Ok(out)
            }
        }

        let work = tmp_dir("mag-trunc");
        let paths = BundleWorkDirPaths {
            work_dir: work.clone(),
        };
        let cfg = MultiBandCellBuildConfig {
            bands: three_bands(),
            max_stars_per_cell: 10,
            mag_limit: 20.0,
            cell_depth: 5,
        };
        build_bundle_work_dir(&LotsOfStars, &cfg, &paths).unwrap();

        let zga = cell_shard_path(&work, GAIA_SUBDIR, GAIA_EXT, cfg.cell_depth, 0);
        let bytes = std::fs::read(&zga).unwrap();
        let shard = GaiaShard::parse(&bytes).unwrap();
        assert_eq!(shard.len(), 10);
        // The 10 brightest are the highest source_ids in this synthetic
        // (mag = (50-i)*0.1, so largest i wins). Pick a min-mag check.
        let mags: Vec<f64> = shard.records().iter().map(|r| r.phot_g_mean_mag).collect();
        for m in &mags {
            assert!(*m <= 1.0, "found a too-faint mag {m}, expected ≤ 1.0");
        }

        let _ = std::fs::remove_dir_all(&work);
    }

    #[test]
    fn invalid_band_config_rejected() {
        let bad_cases = vec![
            // duplicate band_idx
            vec![
                ScaleBand {
                    label: "0".into(),
                    band_idx: 0,
                    scale_lower_arcsec: 1.0,
                    scale_upper_arcsec: 2.0,
                    quads_per_cell: 1,
                    max_reuse: 1,
                },
                ScaleBand {
                    label: "0b".into(),
                    band_idx: 0,
                    scale_lower_arcsec: 1.0,
                    scale_upper_arcsec: 2.0,
                    quads_per_cell: 1,
                    max_reuse: 1,
                },
            ],
            // non-dense band_idx (skips 0)
            vec![
                ScaleBand {
                    label: "1".into(),
                    band_idx: 1,
                    scale_lower_arcsec: 1.0,
                    scale_upper_arcsec: 2.0,
                    quads_per_cell: 1,
                    max_reuse: 1,
                },
                ScaleBand {
                    label: "2".into(),
                    band_idx: 2,
                    scale_lower_arcsec: 1.0,
                    scale_upper_arcsec: 2.0,
                    quads_per_cell: 1,
                    max_reuse: 1,
                },
            ],
            // lower >= upper
            vec![ScaleBand {
                label: "0".into(),
                band_idx: 0,
                scale_lower_arcsec: 5.0,
                scale_upper_arcsec: 5.0,
                quads_per_cell: 1,
                max_reuse: 1,
            }],
            // empty bands list
            vec![],
        ];

        for bands in bad_cases {
            let work = tmp_dir("bad-cfg");
            let paths = BundleWorkDirPaths {
                work_dir: work.clone(),
            };
            let cfg = MultiBandCellBuildConfig {
                bands,
                max_stars_per_cell: 100,
                mag_limit: 20.0,
                cell_depth: 5,
            };
            let source = SyntheticSource {
                n_cells: 1,
                stars_per_cell: 4,
                skip_cells: vec![],
            };
            let err =
                build_bundle_work_dir(&source, &cfg, &paths).expect_err("invalid config rejected");
            assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
            let _ = std::fs::remove_dir_all(&work);
        }
    }
}
