//! Multi-band, cell-driven bundle work-dir builder (PR3 of the
//! `.zdcl.bundle` roadmap).
//!
//! Phase 1 of the bundle build pipeline: workers iterate cells in
//! parallel, build every scale band's quads over each cell's star
//! buffer, and emit per-cell `.zqd` (multi-band) and `.zga` shards
//! into a work directory. The tidy phase (PR4), reader (PR5), and
//! CLI (PR6) live in follow-up PRs; this module ships only the
//! work-dir-producing primitive.
//!
//! See `docs/bundle-format.md` § "Phase 1 — parallel shard build" for
//! the design.
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
//! A cell is "done" iff `completed_cells.contains(cell_id)` AND every
//! band is marked complete for it. The orchestrator marks all bands
//! complete in one critical section after the `.zqd` rename succeeds,
//! so `(cell, band)` granularity is effectively per-cell at this
//! revision — the per-band manifest field is recorded for forward
//! compatibility with PR4+ if a future builder ever races a single
//! cell's bands across workers.

use std::collections::hash_map::DefaultHasher;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc;
use std::thread;
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use rayon::prelude::*;

use crate::bundle::gaia_shard::{GaiaRecord, write_gaia_shard};
use crate::bundle::layout::cell_shard_path;
use crate::bundle::quad_shard::{BandEmit, QuadShard, write_quad_shard};

use super::build_manifest::{BuildManifest, CellStats};
use super::cell_builder::CellStarSource;
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
    /// HEALPix depth at which cells are sharded.
    pub cell_depth: u8,
    /// Wall-clock interval (seconds) between BuildManifest snapshots
    /// to disk. Defaults to 30.
    ///
    /// The orchestrator runs a dedicated actor thread that owns the
    /// manifest; workers send completion events to it via a channel
    /// and never block on a manifest mutex. The actor saves on this
    /// cadence regardless of throughput, so resume granularity is
    /// "at most this many seconds of progress redone after a crash".
    /// `0` means "save only at end of build".
    pub manifest_save_interval_secs: u64,
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
//  Build-manifest state (orchestrator-local)
// ---------------------------------------------------------------------------

/// One worker → actor message: "I committed cell X, please record
/// it." The actor thread (see `build_bundle_work_dir`) owns the
/// `BuildManifest` and processes these in arrival order. Workers
/// don't block on a mutex — they fire-and-forget into an unbounded
/// `mpsc::channel`, so the only synchronization cost is the channel's
/// internal lock-free queue.
struct CellCompletion {
    cell_id: u32,
    stats: CellStats,
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

/// Per-cell write entry-point: emit the `.zqd` and `.zga` shards
/// atomically into `work_dir/{quads,gaia}/`.
///
/// Skips writing entirely if both `bands_emit` is all-empty AND
/// `gaia_records` is empty. Otherwise each file is written to
/// `.part.HHHHHHHH`, fsynced, and atomically renamed onto the
/// canonical name. The fsync also acts as backpressure for the
/// kernel's writeback — without it, the page cache fills under
/// large builds (depth 8+) and every subsequent write blocks
/// erratically.
///
/// Returns one quad count per band in `bands_emit`, in input order.
pub fn write_cell_shards(
    work_dir: &Path,
    cell_depth: u8,
    cell_id: u32,
    bands_emit: &[BandEmit<'_>],
    mut gaia_records: Vec<GaiaRecord>,
    nonce_seed: u64,
) -> io::Result<Vec<u64>> {
    let counts: Vec<u64> = bands_emit.iter().map(|b| b.quads.len() as u64).collect();
    let all_quads_empty = bands_emit.iter().all(|b| b.quads.is_empty());
    if all_quads_empty && gaia_records.is_empty() {
        return Ok(counts);
    }

    let nonce = nonce_hex(cell_id, nonce_seed);

    let zqd_final = cell_shard_path(work_dir, QUADS_SUBDIR, QUAD_EXT, cell_depth, cell_id);
    atomic_write_shard(&zqd_final, &nonce, |w| {
        write_quad_shard(w, cell_id as u64, bands_emit)
    })?;

    let zga_final = cell_shard_path(work_dir, GAIA_SUBDIR, GAIA_EXT, cell_depth, cell_id);
    atomic_write_shard(&zga_final, &nonce, |w| {
        write_gaia_shard(w, cell_id as u64, &mut gaia_records)
    })?;

    Ok(counts)
}

/// Write `final_path` via a `final_path.part.<nonce>` sibling +
/// in-process buffer flush + `fsync` + atomic rename.
fn atomic_write_shard<F>(final_path: &Path, nonce: &str, write: F) -> io::Result<()>
where
    F: FnOnce(&mut BufWriter<File>) -> io::Result<()>,
{
    let partial: PathBuf = {
        let mut s = final_path.as_os_str().to_owned();
        s.push(".part.");
        s.push(nonce);
        PathBuf::from(s)
    };
    {
        let f = File::create(&partial)?;
        let mut w = BufWriter::new(f);
        write(&mut w)?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    std::fs::rename(&partial, final_path)
}

// ---------------------------------------------------------------------------
//  Orchestrator
// ---------------------------------------------------------------------------

/// Phase-1 orchestrator: build every cell's per-cell shards in
/// parallel, atomic-renaming each into place and updating the build
/// manifest after every successful commit.
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

    // Collect cells to process. Skip cells that are completed in
    // `completed_cells` *and* in every per-band set.
    let cell_count = source.cell_count();
    let mut to_process: Vec<u32> = Vec::with_capacity(cell_count as usize);
    let mut n_cells_resumed = 0u32;
    for cell_id in 0..cell_count {
        let cell_complete = manifest.is_complete(cell_id);
        let bands_complete = (0..n_bands as u32).all(|b| manifest.is_band_complete(cell_id, b));
        if cell_complete && bands_complete {
            n_cells_resumed += 1;
        } else {
            to_process.push(cell_id);
        }
    }

    // Group bundle cells by their source partition so a rayon worker
    // processes all siblings of one source partition before moving on.
    // This keeps the partition cache small: with random access, every
    // worker can hold a different partition concurrently (peak memory
    // ≈ N_workers × partition_size); with grouped access, each worker
    // touches one partition at a time and the cache only needs a few
    // slots to absorb cross-worker overlap.
    to_process.sort_by_key(|&c| source.source_partition_key(c));
    let mut partition_groups: Vec<Vec<u32>> = Vec::new();
    {
        let mut current_key: Option<u32> = None;
        let mut current: Vec<u32> = Vec::new();
        for cell_id in to_process {
            let key = source.source_partition_key(cell_id);
            if Some(key) != current_key {
                if !current.is_empty() {
                    partition_groups.push(std::mem::take(&mut current));
                }
                current_key = Some(key);
            }
            current.push(cell_id);
        }
        if !current.is_empty() {
            partition_groups.push(current);
        }
    }

    // Sort the bands once up front; pass them to the per-cell loop in
    // a stable band-idx order so the band table on disk matches the
    // configured layout.
    let mut sorted_bands: Vec<ScaleBand> = config.bands.clone();
    sorted_bands.sort_by_key(|b| b.band_idx);

    let save_interval_secs = config.manifest_save_interval_secs;
    let n_cells_empty = AtomicU32::new(0);
    let n_cells_processed = AtomicU32::new(0);
    let max_stars_per_cell = config.max_stars_per_cell;
    let cell_depth = config.cell_depth;

    // Run the rayon work plus the manifest-actor inside a scoped
    // thread context. The actor owns the `BuildManifest` exclusively
    // and saves to disk on a wall-clock cadence; workers send
    // `CellCompletion` messages via a channel and never block on
    // mutex contention.
    let final_manifest: BuildManifest = thread::scope(|s| -> io::Result<BuildManifest> {
        let (commit_tx, commit_rx) = mpsc::channel::<CellCompletion>();

        // Manifest actor. Receives `CellCompletion` events, applies
        // them in-memory, and saves to disk on the configured
        // wall-clock cadence. Returns the final manifest on shutdown.
        let actor_bands = sorted_bands.clone();
        let actor: thread::ScopedJoinHandle<'_, io::Result<BuildManifest>> = s.spawn(move || {
            let mut manifest = manifest;
            let mut dirty = false;
            let mut last_save = Instant::now();
            let interval = if save_interval_secs == 0 {
                Duration::MAX
            } else {
                Duration::from_secs(save_interval_secs)
            };
            // Use a recv timeout so an idle channel still triggers a
            // periodic flush. A `save_interval_secs == 0` config
            // means "save only on shutdown" — recv blocks indefinitely
            // and only flush in the disconnect arm.
            let recv_timeout = if save_interval_secs == 0 {
                Duration::from_secs(3600 * 24)
            } else {
                interval
            };
            let maybe_save = |m: &mut BuildManifest, d: &mut bool| -> io::Result<()> {
                if *d {
                    m.save(work_dir)?;
                    *d = false;
                }
                Ok(())
            };
            loop {
                match commit_rx.recv_timeout(recv_timeout) {
                    Ok(c) => {
                        manifest.commit_cell(c.cell_id, c.stats);
                        for b in &actor_bands {
                            manifest.mark_band_complete(c.cell_id, b.band_idx);
                        }
                        dirty = true;
                        if last_save.elapsed() >= interval {
                            maybe_save(&mut manifest, &mut dirty)?;
                            last_save = Instant::now();
                        }
                    }
                    Err(mpsc::RecvTimeoutError::Timeout) => {
                        maybe_save(&mut manifest, &mut dirty)?;
                        last_save = Instant::now();
                    }
                    Err(mpsc::RecvTimeoutError::Disconnected) => break,
                }
            }
            maybe_save(&mut manifest, &mut dirty)?;
            Ok(manifest)
        });

        // Worker chokepoint for "this cell is done"; both the empty-cell
        // and populated-cell paths funnel through here. A worker is
        // never blocked by another worker — channel send is lock-free
        // for unbounded mpsc.
        let commit_cell_progress = |cell_id: u32, stats: CellStats| -> io::Result<()> {
            commit_tx
                .send(CellCompletion { cell_id, stats })
                .map_err(|_| io::Error::other("manifest actor disconnected"))?;
            Ok(())
        };

        // Per-cell body, hoisted into a closure so the par-over-groups
        // loop below can call it in sequence within each group.
        let process_cell = |cell_id: u32| -> io::Result<()> {
            let mut stars = source.stars_in_cell(cell_id)?;

            // Brightness-truncate: sort by mag ascending, take the first N.
            if stars.len() > max_stars_per_cell {
                stars.sort_by(|a, b| {
                    a.mag
                        .partial_cmp(&b.mag)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
                stars.truncate(max_stars_per_cell);
            }

            if stars.is_empty() {
                // Empty cell: still mark complete so a resume run doesn't
                // re-pull it.
                commit_cell_progress(cell_id, CellStats::default())?;
                n_cells_empty.fetch_add(1, Ordering::Relaxed);
                return Ok(());
            }

            // Build all bands' quads over this cell's stars (sorted-by-mag
            // already; build_quads_for_cell re-sorts internally, that's
            // fine — same result).
            let mut band_blocks = build_quads_for_cell_multiband(&stars, &sorted_bands);

            // Build the gaia record vector for this cell, preserving
            // `stars` order so quad star_ids are still valid.
            let mut gaia_records: Vec<GaiaRecord> = stars
                .iter()
                .map(|s| {
                    let mut g = s.gaia;
                    // Set the supplement bit if the source_id's high bit
                    // is set; mirrors starfield-gaia's encoding.
                    if (s.gaia.source_id >> 63) & 1 == 1 {
                        g.flags |= GaiaRecord::FLAG_SOURCE_KIND_SUPPLEMENT;
                    }
                    g
                })
                .collect();

            // The on-disk `.zga` is sorted by source_id (write_gaia_shard
            // sorts internally). Quad star_ids reference cell-local
            // indices, so we must remap them through the same permutation
            // used to sort the gaia records — otherwise the reader walks
            // a quad's star_ids into the SORTED vec but the original
            // computation referenced UNSORTED positions, producing
            // garbage WCS hypotheses on lookup.
            //
            // Build the source_id-sort permutation, apply it to
            // `gaia_records`, then rewrite every quad's star_ids through
            // the inverse permutation `old_idx -> new_idx`.
            let n_records = gaia_records.len();
            let mut order: Vec<usize> = (0..n_records).collect();
            order.sort_by_key(|&i| gaia_records[i].source_id);
            let mut old_to_new = vec![0usize; n_records];
            for (new_idx, &old_idx) in order.iter().enumerate() {
                old_to_new[old_idx] = new_idx;
            }
            let sorted_gaia: Vec<GaiaRecord> = order.iter().map(|&i| gaia_records[i]).collect();
            gaia_records = sorted_gaia;
            for (_band_idx, quads, _codes) in band_blocks.iter_mut() {
                for q in quads.iter_mut() {
                    for slot in q.star_ids.iter_mut() {
                        *slot = old_to_new[*slot];
                    }
                }
            }

            // Materialize band emit slices for the writer (must be after
            // the remap above).
            let bands_emit: Vec<BandEmit<'_>> = band_blocks
                .iter()
                .map(|(idx, qs, cs)| BandEmit {
                    band_idx: *idx,
                    quads: qs,
                    codes: cs,
                })
                .collect();

            let _ = write_cell_shards(
                work_dir,
                cell_depth,
                cell_id,
                &bands_emit,
                gaia_records,
                cell_id as u64,
            )?;

            let n_stars = stars.len() as u64;
            let n_quads_total: u64 = band_blocks.iter().map(|(_, qs, _)| qs.len() as u64).sum();

            commit_cell_progress(
                cell_id,
                CellStats {
                    n_stars,
                    n_quads: n_quads_total,
                },
            )?;
            n_cells_processed.fetch_add(1, Ordering::Relaxed);
            Ok(())
        };

        // Iterate over partition groups (one group = all bundle siblings
        // of a source partition). Each rayon worker handles a whole
        // group in sequence so the partition cache only needs to absorb
        // cross-worker overlap, not random per-cell access.
        let result: io::Result<()> = partition_groups.par_iter().try_for_each(|group| {
            for &cell_id in group {
                process_cell(cell_id)?;
            }
            Ok(())
        });
        // Closing the channel signals the actor to drain remaining
        // messages, do a final flush, and exit. Always wait for the
        // actor before returning a result — both the rayon and actor
        // results may carry an error, and the actor's is usually the
        // root cause if both fired.
        drop(commit_tx);
        let actor_result = actor.join().expect("manifest actor panicked");
        match (result, actor_result) {
            (Ok(()), Ok(m)) => Ok(m),
            (_, Err(e)) => Err(e),
            (Err(e), Ok(_)) => Err(e),
        }
    })?;

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
        n_cells_processed: n_cells_processed.into_inner(),
        n_cells_resumed,
        n_cells_empty: n_cells_empty.into_inner(),
        n_stars: final_manifest.n_stars,
        per_band_quad_counts,
        per_band_populated_cells,
    })
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
            cell_depth: 5,
            manifest_save_interval_secs: 0,
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
        assert_eq!(summary.n_cells_resumed, 2);
        assert_eq!(summary.n_cells_processed, 2);
        // Cells 0+1 must NOT be re-pulled.
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
            cell_depth: 5,
            manifest_save_interval_secs: 0,
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
            cell_depth: 5,
            manifest_save_interval_secs: 0,
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
                cell_depth: 5,
                manifest_save_interval_secs: 0,
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
