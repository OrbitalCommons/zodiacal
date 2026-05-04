//! Cell-driven, resumable index builder.
//!
//! Replaces the "load every shard into RAM, build all quads, write at
//! the end" model of [`build_index`] with a cell-by-cell pipeline that
//! commits artifacts to a working directory after every cell. Memory is
//! bounded by the largest single cell. Crash-safe: a build interrupted
//! mid-flight resumes from the next uncommitted cell on the next run.
//!
//! The orchestrator is parameterized over a [`CellStarSource`] trait —
//! the concrete adapter for `LazyLoadingCatalog<Dr3>::entries_in_cell`
//! (issue #65, upstream-in-flight) lives in `zodiacal-tools`.
//!
//! [`build_index`]: super::builder::build_index
//!
//! ## Per-cell artifacts
//!
//! For each cell, three durable artifacts are written under `work_dir`:
//!
//! - `cell_artifacts/cell_NNNNN.idx-tmp` — the cell's stars, quads, and
//!   codes. Quads reference stars by `catalog_id` so the artifact is
//!   self-contained; index remapping happens at finalize time.
//! - `sidecar_chunks/chunk_NNNN.sidecar-tmp` — the cell's sidecar
//!   records, sorted by `source_id` (managed by [`SidecarStreamWriter`]).
//! - `build-manifest.json` — committed cell IDs + running totals
//!   (managed by [`BuildManifest`]). Updated atomically after each
//!   per-cell commit.
//!
//! At finalize, the orchestrator concatenates per-cell artifacts into
//! the final `.zdcl` (via the existing v3 writer) and invokes
//! `SidecarStreamWriter::finalize` to produce the `.zdcl.gaia`.
//!
//! ## What this builder does *not* do (yet)
//!
//! Per-cell quad construction here uses *only* the focus cell's stars,
//! mirroring issue #65's pseudocode. Quads whose backbones straddle a
//! HEALPix-cell boundary at the source's depth are dropped. At level 5
//! (cells ~1.8°) this loss is small for typical quad scales (30″–30′)
//! but non-zero. Adding neighbor context is a follow-up.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use crate::geom::sphere::{angular_distance, radec_to_xyz, star_midpoint};
use crate::quads::{Code, DIMCODES, DIMQUADS, Quad, compute_canonical_code};
use crate::refinement::{SidecarRecord, SidecarStreamWriter};

use super::{IndexStar};
use super::build_manifest::{BuildManifest, CellStats};

/// One star produced by a [`CellStarSource`]. Carries everything the
/// builder needs (index tuple + sidecar payload) so the caller doesn't
/// need to thread two parallel collections.
#[derive(Debug, Clone)]
pub struct CellStar {
    pub catalog_id: u64,
    /// Right ascension, radians. Used for the index file.
    pub ra_rad: f64,
    /// Declination, radians.
    pub dec_rad: f64,
    pub mag: f64,
    /// Full astrometric record for the refinement sidecar. Note: the
    /// sidecar stores RA/Dec in degrees per the existing on-disk
    /// schema; the source is expected to populate `sidecar.ra/dec` in
    /// degrees.
    pub sidecar: SidecarRecord,
}

/// Per-cell read interface. Implementations must be `Sync` because the
/// orchestrator may dispatch parallel cell loads in a future revision.
pub trait CellStarSource: Sync {
    /// Total cell count. Cell IDs are `0..cell_count`.
    fn cell_count(&self) -> u32;

    /// Load every star whose HEALPix cell at the source's depth equals
    /// `cell_id`, after applying any source-side magnitude/quality
    /// filters. Order is irrelevant — the builder sorts.
    fn stars_in_cell(&self, cell_id: u32) -> io::Result<Vec<CellStar>>;
}

/// Configuration for [`build_index_cell_driven`].
#[derive(Debug, Clone)]
pub struct CellBuildConfig {
    /// Lower angular size of quad backbones, radians.
    pub scale_lower: f64,
    /// Upper angular size of quad backbones, radians.
    pub scale_upper: f64,
    /// Quads to emit per HEALPix cell at the source's depth.
    pub quads_per_cell: usize,
    /// Max times any one star may be referenced across the cell's
    /// emitted quads. Tracked locally per cell.
    pub max_reuse: usize,
    /// HEALPix `cell_depth` for the *final* `.zdcl` v3 cell table.
    /// Independent of the source's read-time depth.
    pub final_cell_depth: u8,
    /// Stride for the sidecar pivot table. Use
    /// [`crate::refinement::DEFAULT_PIVOT_STRIDE`] for the standard
    /// 4096.
    pub pivot_stride: u32,
}

/// Filesystem layout for the build.
#[derive(Debug, Clone)]
pub struct BuildPaths {
    /// Holds the manifest, chunk temp files, and cell artifact temp
    /// files. Created if missing. Cleaned up on successful finalize.
    pub work_dir: PathBuf,
    /// Final `.zdcl` destination. Written atomically.
    pub final_index: PathBuf,
    /// Final `.zdcl.gaia` destination. Written atomically.
    pub final_sidecar: PathBuf,
}

/// Summary returned by [`build_index_cell_driven`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BuildSummary {
    /// Cells processed in this invocation (excludes those skipped via
    /// resume).
    pub n_cells_processed: u32,
    /// Cells already complete on entry, skipped via the manifest.
    pub n_cells_resumed: u32,
    /// Cells whose sources returned zero stars after filtering.
    pub n_cells_empty: u32,
    /// Final star count (sum across all cells, including resumed ones).
    pub n_stars: u64,
    /// Final quad count.
    pub n_quads: u64,
}

const CELL_ARTIFACTS_SUBDIR: &str = "cell_artifacts";
const SIDECAR_CHUNKS_SUBDIR: &str = "sidecar_chunks";
const CELL_ARTIFACT_MAGIC: &[u8; 8] = b"ZDCLCELL";

fn cell_artifact_path(work_dir: &Path, cell_id: u32) -> PathBuf {
    work_dir
        .join(CELL_ARTIFACTS_SUBDIR)
        .join(format!("cell_{cell_id:05}.idx-tmp"))
}

/// Top-level orchestrator. Iterates `0..source.cell_count()`, builds
/// per-cell artifacts, and finalizes both index and sidecar.
///
/// Resume semantics: any cells already listed in the manifest are
/// skipped. Their per-cell artifacts and sidecar chunks must still
/// exist on disk in `work_dir` (the manifest + on-disk artifacts are
/// the durable state).
///
/// Cells are processed sequentially. Parallelism is a follow-up — the
/// per-cell artifact writer + manifest are designed to allow it.
pub fn build_index_cell_driven<S: CellStarSource + ?Sized>(
    source: &S,
    config: &CellBuildConfig,
    paths: &BuildPaths,
) -> io::Result<BuildSummary> {
    if config.scale_lower <= 0.0 || config.scale_upper <= config.scale_lower {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "invalid scale range: [{}, {}] rad",
                config.scale_lower, config.scale_upper
            ),
        ));
    }
    if config.pivot_stride == 0 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "pivot_stride must be positive",
        ));
    }

    let work_dir = &paths.work_dir;
    let cell_artifacts_dir = work_dir.join(CELL_ARTIFACTS_SUBDIR);
    let sidecar_chunks_dir = work_dir.join(SIDECAR_CHUNKS_SUBDIR);
    std::fs::create_dir_all(&cell_artifacts_dir)?;
    std::fs::create_dir_all(&sidecar_chunks_dir)?;

    let mut manifest = BuildManifest::load(work_dir)?.unwrap_or_default();
    let mut sidecar_writer = if manifest.completed_cells.is_empty() {
        SidecarStreamWriter::new(&sidecar_chunks_dir)?
    } else {
        SidecarStreamWriter::resume(&sidecar_chunks_dir)?
    };

    let cell_count = source.cell_count();
    let mut summary = BuildSummary::default();

    for cell_id in 0..cell_count {
        if manifest.is_complete(cell_id) {
            summary.n_cells_resumed += 1;
            continue;
        }

        let stars = source.stars_in_cell(cell_id)?;
        if stars.is_empty() {
            // Mark complete (with zero stats) so a subsequent rerun
            // doesn't re-query an empty cell.
            manifest.commit_cell(work_dir, cell_id, CellStats::default())?;
            summary.n_cells_empty += 1;
            continue;
        }

        let (cell_quads, cell_codes) = build_quads_for_cell(&stars, config);

        // Persist per-cell index artifact.
        let artifact_path = cell_artifact_path(work_dir, cell_id);
        write_cell_artifact(&artifact_path, &stars, &cell_quads, &cell_codes)?;

        // Append sidecar chunk (records sorted internally by the writer).
        let sidecar_records: Vec<SidecarRecord> = stars.iter().map(|s| s.sidecar).collect();
        sidecar_writer.append_chunk(sidecar_records)?;

        // Atomically commit the cell to the manifest. From here, a
        // crash can resume cleanly because the artifact + chunk + this
        // manifest update are all durable.
        let stats = CellStats {
            n_stars: stars.len() as u64,
            n_quads: cell_quads.len() as u64,
        };
        manifest.commit_cell(work_dir, cell_id, stats)?;

        summary.n_cells_processed += 1;
    }

    // Finalize: assemble the .zdcl from per-cell artifacts.
    summary.n_stars = manifest.n_stars;
    summary.n_quads = manifest.n_quads;
    finalize_index(&manifest, work_dir, paths, config)?;

    // Finalize the sidecar.
    sidecar_writer.finalize(&paths.final_sidecar, config.pivot_stride)?;

    // Clean up: remove manifest + artifact dir + chunk dir on success.
    // Do *not* remove until both finals are in place — a crash after
    // the index finalize but before the sidecar finalize must still be
    // resumable, and a crash here is harmless.
    let _ = std::fs::remove_file(BuildManifest::path_in(work_dir));
    let _ = std::fs::remove_dir_all(&cell_artifacts_dir);
    // SidecarStreamWriter::finalize removes its own scratch contents
    // and tries to rmdir; if the dir is already gone, that's fine.
    let _ = std::fs::remove_dir(&sidecar_chunks_dir);
    let _ = std::fs::remove_dir(work_dir);

    Ok(summary)
}

/// Build quads for one cell using only the cell's own stars. Stars are
/// sorted by magnitude (brightest first); quad indices in the returned
/// `Quad` values are positions in the *sorted* per-cell list.
fn build_quads_for_cell(stars: &[CellStar], config: &CellBuildConfig) -> (Vec<Quad>, Vec<Code>) {
    if stars.len() < DIMQUADS {
        return (Vec::new(), Vec::new());
    }

    // Sort indices by magnitude (brightest first) so use_count limits
    // bias toward the brightest stars.
    let mut order: Vec<usize> = (0..stars.len()).collect();
    order.sort_by(|&a, &b| {
        stars[a]
            .mag
            .partial_cmp(&stars[b].mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let xyzs: Vec<[f64; 3]> = order
        .iter()
        .map(|&i| radec_to_xyz(stars[i].ra_rad, stars[i].dec_rad))
        .collect();

    let mut quads: Vec<Quad> = Vec::new();
    let mut codes: Vec<Code> = Vec::new();
    let mut seen: HashSet<[usize; DIMQUADS]> = HashSet::new();
    let mut use_count: Vec<usize> = vec![0; xyzs.len()];

    let chord_sq_upper = 2.0 * (1.0 - config.scale_upper.cos());

    'outer: for a_idx in 0..xyzs.len() {
        if quads.len() >= config.quads_per_cell {
            break;
        }
        if use_count[a_idx] >= config.max_reuse {
            continue;
        }

        let a_xyz = xyzs[a_idx];
        for b_idx in (a_idx + 1)..xyzs.len() {
            if use_count[b_idx] >= config.max_reuse {
                continue;
            }
            let b_xyz = xyzs[b_idx];
            let ab_dist = angular_distance(a_xyz, b_xyz);
            if ab_dist < config.scale_lower || ab_dist > config.scale_upper {
                continue;
            }

            let mid = star_midpoint(a_xyz, b_xyz);
            let cd_radius_sq = 2.0 * (1.0 - ab_dist.cos());

            // Linear scan for candidates near the midpoint. At
            // per-cell granularity n_stars is small (tens to a few
            // hundred typically), so an O(n) scan beats a KD-tree
            // construction. The chord_sq_upper outer guard avoids
            // computing midpoints on too-distant pairs.
            let _ = chord_sq_upper;

            let mut candidates: Vec<usize> = Vec::new();
            for (c_idx, c) in xyzs.iter().enumerate() {
                if c_idx == a_idx || c_idx == b_idx {
                    continue;
                }
                let dx = c[0] - mid[0];
                let dy = c[1] - mid[1];
                let dz = c[2] - mid[2];
                if dx * dx + dy * dy + dz * dz < cd_radius_sq {
                    candidates.push(c_idx);
                }
            }

            for ci in 0..candidates.len() {
                if quads.len() >= config.quads_per_cell {
                    break 'outer;
                }
                for di in (ci + 1)..candidates.len() {
                    let c_idx = candidates[ci];
                    let d_idx = candidates[di];

                    let mut key = [a_idx, b_idx, c_idx, d_idx];
                    key.sort();
                    if !seen.insert(key) {
                        continue;
                    }

                    if use_count[c_idx] >= config.max_reuse
                        || use_count[d_idx] >= config.max_reuse
                    {
                        continue;
                    }

                    let raw_xyz = [a_xyz, b_xyz, xyzs[c_idx], xyzs[d_idx]];
                    let raw_ids = [a_idx, b_idx, c_idx, d_idx];
                    let (ordered_xyz, ordered_ids) = canonical_quad_order(&raw_xyz, raw_ids);
                    let (code, canonical_ids, _) =
                        compute_canonical_code(&ordered_xyz, ordered_ids);

                    for &idx in &canonical_ids {
                        use_count[idx] += 1;
                    }

                    // Translate canonical_ids (positions in the sorted
                    // local list) back to positions in the *input*
                    // `stars` slice so the caller can look up
                    // `catalog_id` directly.
                    let star_ids_input: [usize; DIMQUADS] =
                        std::array::from_fn(|i| order[canonical_ids[i]]);
                    quads.push(Quad {
                        star_ids: star_ids_input,
                    });
                    codes.push(code);

                    if quads.len() >= config.quads_per_cell {
                        break 'outer;
                    }
                }
            }
        }
    }

    (quads, codes)
}

/// Local copy of `builder::canonical_quad_order` — exposing the original
/// would widen its visibility. Identical behaviour: order quad members
/// so the longest backbone is at indices `[0]` and `[1]`.
fn canonical_quad_order(
    star_xyz: &[[f64; 3]; DIMQUADS],
    star_ids: [usize; DIMQUADS],
) -> ([[f64; 3]; DIMQUADS], [usize; DIMQUADS]) {
    let mut best_pair = (0, 1);
    let mut best_dist = 0.0f64;
    for i in 0..DIMQUADS {
        for j in (i + 1)..DIMQUADS {
            let d = angular_distance(star_xyz[i], star_xyz[j]);
            if d > best_dist {
                best_dist = d;
                best_pair = (i, j);
            }
        }
    }
    let (ai, bi) = best_pair;
    let mut others: Vec<usize> = (0..DIMQUADS).filter(|&i| i != ai && i != bi).collect();
    others.sort_by_key(|&i| star_ids[i]);
    let order = [ai, bi, others[0], others[1]];
    let new_xyz: [[f64; 3]; DIMQUADS] = std::array::from_fn(|i| star_xyz[order[i]]);
    let new_ids: [usize; DIMQUADS] = std::array::from_fn(|i| star_ids[order[i]]);
    (new_xyz, new_ids)
}

// --- Per-cell artifact format -------------------------------------------
//
//   8B   magic       "ZDCLCELL"
//   4B   version     u32 = 1
//   4B   reserved
//   8B   n_stars     u64
//   8B   n_quads     u64
//   stars × n_stars   (40 B each: u64 catalog_id, 3 × f64 ra/dec/mag)
//   quads × n_quads   (32 B each: 4 × u64 catalog_id)
//   codes × n_quads   (32 B each: 4 × f64)
//
// Star records here use catalog_id rather than a local index so the
// artifact is self-contained for the finalize-time remap.

const CELL_ARTIFACT_VERSION: u32 = 1;

fn write_cell_artifact(
    path: &Path,
    stars: &[CellStar],
    quads: &[Quad],
    codes: &[Code],
) -> io::Result<()> {
    assert_eq!(quads.len(), codes.len());

    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".partial");
    let tmp_path = PathBuf::from(tmp);

    {
        let f = File::create(&tmp_path)?;
        let mut w = BufWriter::new(f);
        w.write_all(CELL_ARTIFACT_MAGIC)?;
        w.write_all(&CELL_ARTIFACT_VERSION.to_le_bytes())?;
        w.write_all(&[0u8; 4])?; // reserved
        w.write_all(&(stars.len() as u64).to_le_bytes())?;
        w.write_all(&(quads.len() as u64).to_le_bytes())?;

        for s in stars {
            w.write_all(&s.catalog_id.to_le_bytes())?;
            w.write_all(&s.ra_rad.to_le_bytes())?;
            w.write_all(&s.dec_rad.to_le_bytes())?;
            w.write_all(&s.mag.to_le_bytes())?;
        }
        for q in quads {
            for &local_idx in &q.star_ids {
                let cid = stars
                    .get(local_idx)
                    .map(|s| s.catalog_id)
                    .unwrap_or_else(|| panic!("quad references out-of-range star {local_idx}"));
                w.write_all(&cid.to_le_bytes())?;
            }
        }
        for c in codes {
            for &v in c {
                w.write_all(&v.to_le_bytes())?;
            }
        }
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    std::fs::rename(&tmp_path, path)
}

struct LoadedCellArtifact {
    stars: Vec<IndexStar>,
    /// Quads with members carrying catalog_id; remap to global indices
    /// at finalize time.
    quads_by_cid: Vec<[u64; DIMQUADS]>,
    codes: Vec<Code>,
}

fn read_cell_artifact(path: &Path) -> io::Result<LoadedCellArtifact> {
    let mut f = BufReader::new(OpenOptions::new().read(true).open(path)?);

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != CELL_ARTIFACT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("bad cell artifact magic at {}", path.display()),
        ));
    }
    let version = read_u32(&mut f)?;
    if version != CELL_ARTIFACT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported cell artifact version {version}"),
        ));
    }
    let _reserved = read_u32(&mut f)?;
    let n_stars = read_u64(&mut f)? as usize;
    let n_quads = read_u64(&mut f)? as usize;

    let mut stars: Vec<IndexStar> = Vec::with_capacity(n_stars);
    for _ in 0..n_stars {
        let catalog_id = read_u64(&mut f)?;
        let ra = read_f64(&mut f)?;
        let dec = read_f64(&mut f)?;
        let mag = read_f64(&mut f)?;
        stars.push(IndexStar {
            catalog_id,
            ra,
            dec,
            mag,
        });
    }
    let mut quads_by_cid: Vec<[u64; DIMQUADS]> = Vec::with_capacity(n_quads);
    for _ in 0..n_quads {
        let mut q = [0u64; DIMQUADS];
        for slot in &mut q {
            *slot = read_u64(&mut f)?;
        }
        quads_by_cid.push(q);
    }
    let mut codes: Vec<Code> = Vec::with_capacity(n_quads);
    for _ in 0..n_quads {
        let mut c = [0.0f64; DIMCODES];
        for slot in &mut c {
            *slot = read_f64(&mut f)?;
        }
        codes.push(c);
    }
    Ok(LoadedCellArtifact {
        stars,
        quads_by_cid,
        codes,
    })
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut b = [0u8; 4];
    r.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}
fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(u64::from_le_bytes(b))
}
fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut b = [0u8; 8];
    r.read_exact(&mut b)?;
    Ok(f64::from_le_bytes(b))
}

/// Read every per-cell artifact, remap catalog_id → global star index,
/// and write the final `.zdcl` v3 file.
fn finalize_index(
    manifest: &BuildManifest,
    work_dir: &Path,
    paths: &BuildPaths,
    config: &CellBuildConfig,
) -> io::Result<()> {
    let mut all_stars: Vec<IndexStar> = Vec::with_capacity(manifest.n_stars as usize);
    let mut quads: Vec<Quad> = Vec::with_capacity(manifest.n_quads as usize);
    let mut codes: Vec<Code> = Vec::with_capacity(manifest.n_quads as usize);
    let mut cid_to_global: HashMap<u64, usize> = HashMap::with_capacity(manifest.n_stars as usize);

    for &cell_id in &manifest.completed_cells {
        let stats = manifest
            .cell_stats
            .iter()
            .find(|(c, _)| *c == cell_id)
            .map(|(_, s)| s.clone())
            .unwrap_or_default();
        if stats.n_stars == 0 {
            continue;
        }
        let path = cell_artifact_path(work_dir, cell_id);
        let loaded = read_cell_artifact(&path)?;

        let base = all_stars.len();
        for (i, s) in loaded.stars.iter().enumerate() {
            let prior = cid_to_global.insert(s.catalog_id, base + i);
            if prior.is_some() {
                // Same catalog_id appearing in two cells means the
                // upstream source's cell partitioning is broken — this
                // builder assumes each star lives in exactly one cell.
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "catalog_id {} appears in multiple cells; source's cell partitioning is broken",
                        s.catalog_id
                    ),
                ));
            }
        }
        all_stars.extend(loaded.stars);

        for (q_cids, code) in loaded.quads_by_cid.into_iter().zip(loaded.codes) {
            let mut star_ids = [0usize; DIMQUADS];
            let mut all_resolved = true;
            for (i, cid) in q_cids.iter().enumerate() {
                match cid_to_global.get(cid) {
                    Some(&global) => star_ids[i] = global,
                    None => {
                        all_resolved = false;
                        break;
                    }
                }
            }
            if all_resolved {
                quads.push(Quad { star_ids });
                codes.push(code);
            }
            // Quads with unresolved members are dropped. This shouldn't
            // happen for the per-cell-only quad builder (every member
            // is from the same cell, so it's been added above), but we
            // leave the guard for future neighbor-aware builders.
        }
    }

    let _ = codes; // codes are recomputed by write_v3 from star positions

    write_index_to_path(
        &paths.final_index,
        &all_stars,
        &quads,
        config.scale_lower,
        config.scale_upper,
        config.final_cell_depth,
    )
}

fn write_index_to_path(
    path: &Path,
    stars: &[IndexStar],
    quads: &[Quad],
    scale_lower: f64,
    scale_upper: f64,
    cell_depth: u8,
) -> io::Result<()> {
    let mut tmp = path.as_os_str().to_owned();
    tmp.push(".partial");
    let tmp_path = PathBuf::from(tmp);

    {
        let f = File::create(&tmp_path)?;
        let mut w = BufWriter::new(f);
        super::source::write_v3(
            &mut w,
            None,
            stars,
            quads,
            scale_lower,
            scale_upper,
            cell_depth,
        )?;
        w.flush()?;
        w.get_ref().sync_all()?;
    }
    std::fs::rename(&tmp_path, path)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Index, IndexFragment, IndexSource};
    use crate::refinement::{SidecarReader, SidecarRecord};

    fn tmp_dir(name: &str) -> PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "zodiacal-cell-builder-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        std::fs::create_dir_all(&p).unwrap();
        p
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
                sigma_pmra: 0.01,
                sigma_pmdec: 0.01,
                sigma_parallax: 0.02,
                flags: 0,
            },
        }
    }

    /// Synthetic cell source: places `stars_per_cell` deterministically
    /// inside a small patch around each cell's nominal center. Cell IDs
    /// here are abstract — we don't bother computing real HEALPix
    /// neighbours for the test, just exercise the orchestrator's
    /// per-cell flow.
    struct SyntheticSource {
        n_cells: u32,
        stars_per_cell: usize,
    }

    impl CellStarSource for SyntheticSource {
        fn cell_count(&self) -> u32 {
            self.n_cells
        }

        fn stars_in_cell(&self, cell_id: u32) -> io::Result<Vec<CellStar>> {
            // Stars laid out in a small patch unique to each cell.
            // RA/Dec are *real* radians so the v3 cell-table writer
            // can hash them without complaint.
            let base_ra = 0.5 + (cell_id as f64) * 0.1; // radians; spread across the sky
            let base_dec = 0.0;
            let mut out = Vec::with_capacity(self.stars_per_cell);
            for i in 0..self.stars_per_cell {
                let dra = (i as f64) * 0.001;
                let ddec = (i as f64) * 0.0007;
                // Catalog IDs are unique across cells (encoded as
                // cell_id * 1000 + i) so the cross-cell-uniqueness
                // assertion in finalize_index passes.
                let catalog_id = (cell_id as u64) * 1000 + (i as u64) + 1;
                out.push(make_cell_star(
                    catalog_id,
                    base_ra + dra,
                    base_dec + ddec,
                    (i as f64) * 0.5,
                ));
            }
            Ok(out)
        }
    }

    fn make_paths(name: &str) -> BuildPaths {
        let work = tmp_dir(name);
        let final_index = work.parent().unwrap().join(format!("{name}-final.zdcl"));
        let final_sidecar = work.parent().unwrap().join(format!("{name}-final.zdcl.gaia"));
        BuildPaths {
            work_dir: work,
            final_index,
            final_sidecar,
        }
    }

    fn default_config() -> CellBuildConfig {
        CellBuildConfig {
            scale_lower: 0.0001,
            scale_upper: 0.05,
            quads_per_cell: 200,
            max_reuse: 8,
            final_cell_depth: 5,
            pivot_stride: 16,
        }
    }

    #[test]
    fn end_to_end_synthetic_build() {
        let paths = make_paths("e2e");
        let source = SyntheticSource {
            n_cells: 4,
            stars_per_cell: 8,
        };
        let cfg = default_config();

        let summary = build_index_cell_driven(&source, &cfg, &paths).unwrap();
        assert_eq!(summary.n_cells_processed, 4);
        assert_eq!(summary.n_cells_resumed, 0);
        assert_eq!(summary.n_stars, 32);
        assert!(summary.n_quads > 0, "expected some quads");

        // Final files must exist and be readable.
        assert!(paths.final_index.exists());
        assert!(paths.final_sidecar.exists());

        let zdcl = crate::index::ZdclFile::open(&paths.final_index).unwrap();
        assert_eq!(zdcl.star_count(), 32);
        assert_eq!(zdcl.quad_count() as u64, summary.n_quads);

        let sidecar = SidecarReader::open(&paths.final_sidecar).unwrap();
        assert_eq!(sidecar.len(), 32);
        // Spot-check a known catalog_id (cell 2, star 3 → id = 2*1000+3+1 = 2004).
        let r = sidecar.get(2004).expect("missing record");
        assert_eq!(r.source_id, 2004);

        // Work dir should be fully cleaned up on success.
        assert!(
            !paths.work_dir.exists()
                || std::fs::read_dir(&paths.work_dir)
                    .map(|d| d.count() == 0)
                    .unwrap_or(true),
            "work_dir not cleaned: {:?}",
            paths.work_dir
        );

        std::fs::remove_file(&paths.final_index).ok();
        std::fs::remove_file(&paths.final_sidecar).ok();
    }

    #[test]
    fn empty_cells_are_recorded_and_finalize_succeeds() {
        struct Source;
        impl CellStarSource for Source {
            fn cell_count(&self) -> u32 {
                3
            }
            fn stars_in_cell(&self, cell_id: u32) -> io::Result<Vec<CellStar>> {
                if cell_id == 1 {
                    return Ok(Vec::new());
                }
                Ok(vec![make_cell_star(
                    (cell_id as u64) + 1,
                    0.5 + (cell_id as f64) * 0.1,
                    0.0,
                    1.0,
                )])
            }
        }

        let paths = make_paths("empty-cells");
        let cfg = default_config();
        let summary = build_index_cell_driven(&Source, &cfg, &paths).unwrap();
        assert_eq!(summary.n_cells_processed, 2);
        assert_eq!(summary.n_cells_empty, 1);
        assert_eq!(summary.n_stars, 2);

        let zdcl = crate::index::ZdclFile::open(&paths.final_index).unwrap();
        assert_eq!(zdcl.star_count(), 2);

        std::fs::remove_file(&paths.final_index).ok();
        std::fs::remove_file(&paths.final_sidecar).ok();
    }

    /// Crash-resume simulation: a `CellStarSource` that panics on the
    /// first attempt to load the second cell, then on retry succeeds
    /// for every cell. Verifies the manifest survived the crash and
    /// the resumed run produces the same final output as a clean run.
    #[test]
    fn resumes_after_simulated_crash() {
        use std::cell::Cell;

        struct CrashOnce {
            crashed: Cell<bool>,
            crash_at_cell: u32,
            n_cells: u32,
            stars_per_cell: usize,
        }

        // SAFETY: tests are single-threaded; Cell is sufficient. The
        // trait requires Sync, so wrap in a stub Sync impl.
        unsafe impl Sync for CrashOnce {}

        impl CellStarSource for CrashOnce {
            fn cell_count(&self) -> u32 {
                self.n_cells
            }
            fn stars_in_cell(&self, cell_id: u32) -> io::Result<Vec<CellStar>> {
                if cell_id == self.crash_at_cell && !self.crashed.get() {
                    self.crashed.set(true);
                    return Err(io::Error::other("simulated crash"));
                }
                let mut out = Vec::with_capacity(self.stars_per_cell);
                for i in 0..self.stars_per_cell {
                    let catalog_id = (cell_id as u64) * 1000 + (i as u64) + 1;
                    out.push(make_cell_star(
                        catalog_id,
                        0.5 + (cell_id as f64) * 0.1 + (i as f64) * 0.001,
                        0.0 + (i as f64) * 0.0007,
                        (i as f64) * 0.5,
                    ));
                }
                Ok(out)
            }
        }

        let paths = make_paths("resume");
        let cfg = default_config();

        // Phase 1: crash partway. Source crashes on cell 2, so cells
        // 0 and 1 should already be committed when we abort.
        let source = CrashOnce {
            crashed: Cell::new(false),
            crash_at_cell: 2,
            n_cells: 4,
            stars_per_cell: 6,
        };
        let err = build_index_cell_driven(&source, &cfg, &paths).unwrap_err();
        assert!(err.to_string().contains("simulated crash"));

        // Manifest should record the committed cells.
        let manifest = BuildManifest::load(&paths.work_dir)
            .unwrap()
            .expect("manifest must persist after crash");
        assert_eq!(manifest.completed_cells.len(), 2, "{:?}", manifest);
        assert!(manifest.is_complete(0));
        assert!(manifest.is_complete(1));
        assert!(!manifest.is_complete(2));

        // Phase 2: resume — same paths, fresh `crashed` flag so the
        // source doesn't crash again.
        let source = CrashOnce {
            crashed: Cell::new(true), // already "crashed once"
            crash_at_cell: 2,
            n_cells: 4,
            stars_per_cell: 6,
        };
        let summary = build_index_cell_driven(&source, &cfg, &paths).unwrap();
        assert_eq!(summary.n_cells_resumed, 2);
        assert_eq!(summary.n_cells_processed, 2);
        assert_eq!(summary.n_stars, 24);

        // Verify resumed final output matches a clean rebuild byte-for-byte.
        let resumed_index_bytes = std::fs::read(&paths.final_index).unwrap();
        let resumed_sidecar_bytes = std::fs::read(&paths.final_sidecar).unwrap();
        std::fs::remove_file(&paths.final_index).ok();
        std::fs::remove_file(&paths.final_sidecar).ok();

        let clean_paths = make_paths("resume-clean");
        let clean_source = CrashOnce {
            crashed: Cell::new(true),
            crash_at_cell: u32::MAX,
            n_cells: 4,
            stars_per_cell: 6,
        };
        build_index_cell_driven(&clean_source, &cfg, &clean_paths).unwrap();
        let clean_index_bytes = std::fs::read(&clean_paths.final_index).unwrap();
        let clean_sidecar_bytes = std::fs::read(&clean_paths.final_sidecar).unwrap();
        std::fs::remove_file(&clean_paths.final_index).ok();
        std::fs::remove_file(&clean_paths.final_sidecar).ok();

        assert_eq!(resumed_sidecar_bytes, clean_sidecar_bytes);
        // Index bytes may differ if iteration order differed across
        // runs, but the cell-driven builder is deterministic, so
        // expect equality.
        assert_eq!(resumed_index_bytes, clean_index_bytes);
    }

    #[test]
    fn duplicate_catalog_id_across_cells_is_rejected() {
        struct DupSource;
        impl CellStarSource for DupSource {
            fn cell_count(&self) -> u32 {
                2
            }
            fn stars_in_cell(&self, cell_id: u32) -> io::Result<Vec<CellStar>> {
                // Both cells expose the same catalog_id 42 — the
                // finalize step must catch it.
                Ok(vec![make_cell_star(
                    42,
                    0.5 + (cell_id as f64) * 0.1,
                    0.0,
                    1.0,
                )])
            }
        }

        let paths = make_paths("dup");
        let cfg = default_config();
        let err = build_index_cell_driven(&DupSource, &cfg, &paths).unwrap_err();
        assert!(err.to_string().contains("multiple cells"), "got: {err}");

        // Cleanup left-over artifacts.
        let _ = std::fs::remove_dir_all(&paths.work_dir);
        let _ = std::fs::remove_file(&paths.final_index);
        let _ = std::fs::remove_file(&paths.final_sidecar);
    }

    #[test]
    fn invalid_scale_range_is_rejected() {
        let paths = make_paths("bad-scale");
        let mut cfg = default_config();
        cfg.scale_lower = 0.05;
        cfg.scale_upper = 0.01;
        let source = SyntheticSource {
            n_cells: 1,
            stars_per_cell: 4,
        };
        let err = build_index_cell_driven(&source, &cfg, &paths).unwrap_err();
        assert!(err.to_string().contains("scale range"));
        let _ = std::fs::remove_dir_all(&paths.work_dir);
    }

    /// Loading the final v3 file via `IndexSource::load_cells` over the
    /// full sky should return all stars and all quads — proves that the
    /// finalize step's catalog_id remap produced indices the v3 reader
    /// can resolve.
    #[test]
    fn final_v3_load_full_returns_everything() {
        let paths = make_paths("v3-full");
        let source = SyntheticSource {
            n_cells: 3,
            stars_per_cell: 6,
        };
        let cfg = default_config();
        build_index_cell_driven(&source, &cfg, &paths).unwrap();

        let zdcl = crate::index::ZdclFile::open(&paths.final_index).unwrap();
        let cells: Vec<_> = (0..(12u64 << (2 * cfg.final_cell_depth as u64)))
            .map(|id| crate::index::HealpixCell {
                depth: cfg.final_cell_depth,
                id,
            })
            .collect();
        // Touch only the cells that actually contain stars; load_cells
        // skips empty ones internally.
        let frag: IndexFragment = zdcl.load_cells(&cells).unwrap();
        assert_eq!(frag.stars.len(), 18);
        // Every quad must reference a valid star index in the fragment.
        for q in &frag.quads {
            for &sid in &q.star_ids {
                assert!(sid < frag.stars.len(), "quad sid {sid} out of range");
            }
        }
        // Reconstruct an Index to make sure tree-build succeeds.
        let _idx: Index = frag.into();

        std::fs::remove_file(&paths.final_index).ok();
        std::fs::remove_file(&paths.final_sidecar).ok();
    }
}
