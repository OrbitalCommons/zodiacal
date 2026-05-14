//! Tidy phase: package a complete `work_dir` into a final
//! `.zdcl.bundle/` directory or `.zdcl.bundle.zip` archive.
//!
//! The tidy phase is the **commit edge** of the bundle build pipeline:
//! a bundle is "real" iff its `manifest.json` is present and parseable.
//! This module promotes a populated work directory (as produced by
//! [`crate::index::multiband_cell_builder::build_bundle_work_dir`])
//! into one of two equally first-class output forms.
//!
//! See `docs/bundle-format.md` § "Phase 2 — tidy sweep into final
//! artifact" for the broader design.
//!
//! ## Two entry points
//!
//! - [`tidy_to_folder`] writes a directory bundle by **renaming**
//!   `work_dir` onto `<output_path>`. The work dir's `quads/`/`gaia/`
//!   layout is already byte-identical to the final bundle's, so we
//!   just write `manifest.json` into it and call `rename(2)`. Atomic
//!   on the same filesystem, instantaneous, no data movement, no
//!   per-file `fsync`. **Consumes the work directory.**
//! - [`tidy_to_zip`] writes a zip archive: per-cell entries are
//!   stream-zipped (Stored, no compression — binary shards are not
//!   very compressible) into `<output>.partial`, then `manifest.json`
//!   (Deflated) as the final entry, then the file is atomically
//!   renamed onto `<output>`. **Preserves the work directory** —
//!   reads from it without modifying it.
//!
//! To emit both forms from one build, run the zip tidy first (reads
//! from work_dir) then the folder tidy (consumes work_dir).
//!
//! ## Atomicity
//!
//! - The folder commit is one `rename(work_dir, output_path)`. POSIX
//!   `rename(2)` on the same filesystem is atomic against concurrent
//!   namespace observers, so a reader sees either the pre-tidy state
//!   (`work_dir` exists, `output_path` does not) or the post-tidy
//!   state (`output_path` exists, `work_dir` does not).
//! - The zip commit is `rename(<output>.partial, <output>)` after
//!   the full archive is stream-written. Pre-existing `.partial`
//!   from a crashed previous run is removed at entry.
//! - The manifest writes use their own atomic `.partial.json` +
//!   rename pattern, so a crash mid-manifest-write leaves the
//!   work_dir consistent (no `manifest.json` present, the next tidy
//!   tries again).
//!
//! We deliberately do **not** call `fsync` anywhere. Kernel page-cache
//! writeback auto-commits on its own (~5 s default `commit` interval
//! on ext4), and within seconds of `tidy_to_folder` returning all of
//! the work is durable. The crash window — power loss between
//! "tidy returned" and "writeback completed" — is at most that few
//! seconds wide, which is negligible relative to a multi-hour build.

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::{Path, PathBuf};

use zip::CompressionMethod;
use zip::write::FileOptions;

use super::gaia_shard::GaiaShard;
use super::layout::{cell_filename_width, cell_shard_path};
use super::manifest::{
    BandInfo, BuildMetadata, BundleManifest, FORMAT_MAGIC, FORMAT_VERSION, GAIA_RECORD_SIZE,
    GaiaBlock,
};
use super::quad_shard::QuadShard;
use crate::index::build_manifest::BuildManifest;
use crate::index::multiband_cell_builder::{GAIA_EXT, GAIA_SUBDIR, QUAD_EXT, QUADS_SUBDIR};

// ---------------------------------------------------------------------------
//  Public input types
// ---------------------------------------------------------------------------

/// User-supplied metadata required to build the final `BundleManifest`.
///
/// The on-disk-derived totals — `gaia.n_total`, `gaia.populated_cells`,
/// each band's `n_quads_total` and `populated_cells` — are computed by
/// the tidy phase from the work_dir and merged with the fields in this
/// struct. Per-band `band_idx` values are implicit from the position in
/// `bands` (`bands[k].band_idx == k`).
#[derive(Debug, Clone, PartialEq)]
pub struct TidyMetadata {
    /// HEALPix nested-scheme depth used for the bundle's cell sharding.
    pub cell_depth: u8,
    /// Free-text human ops note carried verbatim into the bundle
    /// manifest's `experiment` field.
    pub experiment: String,
    /// Structured build provenance. Emitted as-is into
    /// `manifest.build_metadata`.
    pub build_metadata: BuildMetadata,
    /// User-supplied Gaia-block fields. The on-disk-derived totals
    /// (`n_total`, `populated_cells`) are filled in from the work_dir.
    pub gaia: GaiaMetadata,
    /// One entry per scale band, in `band_idx` ascending order
    /// (position == band_idx).
    pub bands: Vec<BandMetadata>,
}

/// User-supplied subset of the bundle manifest's `gaia` block.
#[derive(Debug, Clone, PartialEq)]
pub struct GaiaMetadata {
    /// Brightness-truncation cap applied during the build.
    pub max_stars_per_cell: u32,
    /// Faintest G-magnitude included in the build.
    pub mag_limit: f64,
    /// Per-record schema version (covaries with the bit layout).
    pub schema_version: u32,
}

/// User-supplied subset of one band's manifest entry.
#[derive(Debug, Clone, PartialEq)]
pub struct BandMetadata {
    /// Human label, conventionally `"band_NN"` zero-padded.
    pub label: String,
    /// Lower scale bound for this band's quads, in arcseconds.
    pub scale_lower_arcsec: f64,
    /// Upper scale bound for this band's quads, in arcseconds.
    pub scale_upper_arcsec: f64,
    /// Per-cell quad count cap configured for this band.
    pub quads_per_cell: u32,
    /// Per-cell star-reuse cap configured for this band.
    pub max_reuse: u32,
}

// ---------------------------------------------------------------------------
//  Helpers: enumeration + per-shard re-scan
// ---------------------------------------------------------------------------

/// Enumerate cell ids that have a `.zqd` file under
/// `work_dir/quads/`. Returns the ids ascending.
///
/// Filenames are parsed using [`cell_filename_width`] for `depth`; any
/// file whose name doesn't match `cell_NNNN…N.zqd` at the canonical
/// width is skipped (defensive: e.g. dot-files or unrelated junk).
pub fn enumerate_populated_cells(work_dir: &Path, depth: u8) -> io::Result<Vec<u32>> {
    let quads_dir = work_dir.join(QUADS_SUBDIR);
    if !quads_dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("quads subdir missing under {}", work_dir.display()),
        ));
    }
    let width = cell_filename_width(depth);
    let mut cells: Vec<u32> = Vec::new();
    for entry in std::fs::read_dir(&quads_dir)? {
        let entry = entry?;
        let name = entry.file_name();
        let name = name.to_string_lossy();
        // Expected: cell_<N digits>.zqd
        let stripped = match name
            .strip_prefix("cell_")
            .and_then(|s| s.strip_suffix(".zqd"))
        {
            Some(s) => s,
            None => continue,
        };
        if stripped.len() != width {
            continue;
        }
        let cell_id: u32 = match stripped.parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        cells.push(cell_id);
    }
    cells.sort_unstable();
    Ok(cells)
}

/// Re-scan every populated cell's `.zqd` to compute per-band quad
/// totals and populated-cell counts.
///
/// Returns `(per_band_quad_counts, per_band_populated_cells)` indexed
/// by band position, both length `n_bands`. Validates that each cell's
/// matching `.zga` file exists, and that every populated band's
/// `band_idx` lies in `[0, n_bands)`.
pub fn compute_per_band_totals(
    work_dir: &Path,
    depth: u8,
    populated_cells: &[u32],
    n_bands: usize,
) -> io::Result<(Vec<u64>, Vec<u32>)> {
    let mut quad_counts = vec![0u64; n_bands];
    let mut pop_cells = vec![0u32; n_bands];

    for &cell_id in populated_cells {
        let zqd = cell_shard_path(work_dir, QUADS_SUBDIR, QUAD_EXT, depth, cell_id);
        let zga = cell_shard_path(work_dir, GAIA_SUBDIR, GAIA_EXT, depth, cell_id);
        if !zga.is_file() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "cell {cell_id} has {} but no matching {}",
                    zqd.display(),
                    zga.display()
                ),
            ));
        }
        let bytes = std::fs::read(&zqd)?;
        let shard = QuadShard::parse(&bytes).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse {}: {e}", zqd.display()),
            )
        })?;
        if shard.cell_id() != cell_id as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{}: filename cell_id={cell_id} but header cell_id={}",
                    zqd.display(),
                    shard.cell_id()
                ),
            ));
        }
        for entry in shard.bands() {
            let k = entry.band_idx as usize;
            if k >= n_bands {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "{}: band_idx {} >= configured n_bands {n_bands}",
                        zqd.display(),
                        entry.band_idx
                    ),
                ));
            }
            let n = entry.n_quads as u64;
            quad_counts[k] += n;
            if n > 0 {
                pop_cells[k] += 1;
            }
        }
    }
    Ok((quad_counts, pop_cells))
}

/// Sum `n_records` across every populated cell's `.zga` header.
pub fn compute_gaia_total(work_dir: &Path, depth: u8, populated_cells: &[u32]) -> io::Result<u64> {
    let mut total: u64 = 0;
    for &cell_id in populated_cells {
        let zga = cell_shard_path(work_dir, GAIA_SUBDIR, GAIA_EXT, depth, cell_id);
        let bytes = std::fs::read(&zga)?;
        let shard = GaiaShard::parse(&bytes).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("failed to parse {}: {e}", zga.display()),
            )
        })?;
        if shard.cell_id() != cell_id as u64 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "{}: filename cell_id={cell_id} but header cell_id={}",
                    zga.display(),
                    shard.cell_id()
                ),
            ));
        }
        total += shard.len() as u64;
    }
    Ok(total)
}

/// Build the final `BundleManifest` from the work_dir's on-disk state
/// plus the user-supplied [`TidyMetadata`].
pub fn build_final_manifest(
    work_dir: &Path,
    metadata: &TidyMetadata,
    populated_cells: &[u32],
) -> io::Result<BundleManifest> {
    let n_bands = metadata.bands.len();
    let (per_band_quads, per_band_pop) =
        compute_per_band_totals(work_dir, metadata.cell_depth, populated_cells, n_bands)?;
    let gaia_total = compute_gaia_total(work_dir, metadata.cell_depth, populated_cells)?;

    let n_cells = (12u64 << (2 * metadata.cell_depth as u64)) as u32;

    let bands: Vec<BandInfo> = metadata
        .bands
        .iter()
        .enumerate()
        .map(|(k, b)| BandInfo {
            label: b.label.clone(),
            band_idx: k as u32,
            scale_lower_arcsec: b.scale_lower_arcsec,
            scale_upper_arcsec: b.scale_upper_arcsec,
            quads_per_cell: b.quads_per_cell,
            max_reuse: b.max_reuse,
            n_quads_total: per_band_quads[k],
            populated_cells: per_band_pop[k],
        })
        .collect();

    Ok(BundleManifest {
        format: FORMAT_MAGIC.to_string(),
        format_version: FORMAT_VERSION,
        cell_depth: metadata.cell_depth,
        n_cells,
        experiment: metadata.experiment.clone(),
        build_metadata: metadata.build_metadata.clone(),
        gaia: GaiaBlock {
            n_total: gaia_total,
            record_size: GAIA_RECORD_SIZE,
            schema_version: metadata.gaia.schema_version,
            max_stars_per_cell: metadata.gaia.max_stars_per_cell,
            mag_limit: metadata.gaia.mag_limit,
            populated_cells: populated_cells.len() as u32,
        },
        bands,
    })
}

// ---------------------------------------------------------------------------
//  Shared validation
// ---------------------------------------------------------------------------

fn validate_work_dir(work_dir: &Path) -> io::Result<()> {
    if !work_dir.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("work_dir {} is not a directory", work_dir.display()),
        ));
    }
    let bm_path = work_dir.join(crate::index::build_manifest::MANIFEST_FILENAME);
    if !bm_path.is_file() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!(
                "build manifest missing at {}; cannot tidy an unfinalized work_dir",
                bm_path.display()
            ),
        ));
    }
    // Load it to confirm it parses; we don't currently consult its
    // contents during tidy (every populated cell is enumerated from
    // disk), but a missing/corrupt build-manifest is the canonical
    // signal that this work_dir is not ready to commit.
    let _ = BuildManifest::load(work_dir)?.ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::NotFound,
            format!("build manifest at {} did not load", bm_path.display()),
        )
    })?;
    if !work_dir.join(QUADS_SUBDIR).is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("missing {}/{QUADS_SUBDIR}", work_dir.display()),
        ));
    }
    if !work_dir.join(GAIA_SUBDIR).is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("missing {}/{GAIA_SUBDIR}", work_dir.display()),
        ));
    }
    Ok(())
}

fn partial_path_for(output: &Path) -> PathBuf {
    let mut s = output.as_os_str().to_owned();
    s.push(".partial");
    PathBuf::from(s)
}

/// Best-effort removal of a stale `.partial` artifact left by a
/// previous crashed tidy. NotFound is a no-op.
fn remove_partial(partial: &Path) -> io::Result<()> {
    let meta = match std::fs::symlink_metadata(partial) {
        Ok(m) => m,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    if meta.is_dir() {
        std::fs::remove_dir_all(partial)
    } else {
        std::fs::remove_file(partial)
    }
}

// ---------------------------------------------------------------------------
//  Folder output
// ---------------------------------------------------------------------------

/// Tidy a populated `work_dir` into a folder bundle at `output_path`.
///
/// The work directory's `quads/` and `gaia/` layout is already
/// identical to the final bundle's, so we promote it in place: write
/// `manifest.json` into `work_dir`, then `rename(work_dir, output_path)`.
/// POSIX `rename(2)` on the same filesystem is atomic against
/// concurrent namespace observers, so a reader never sees a half-built
/// bundle.
///
/// Consequences:
///
/// - The work directory is **consumed** by tidy. After a successful
///   commit, `work_dir` no longer exists — its contents are now under
///   `output_path`. `--prune-work-dir` becomes a no-op (it's implicit).
/// - `build-manifest.json` (the build phase's resume-state file) is
///   carried into the bundle alongside the proper `manifest.json`.
///   Readers ignore unknown top-level files, so this is harmless; it
///   also leaves the work-resume artifact in place for forensics if a
///   later step ever needs it.
/// - To emit both a folder and a zip from a single build, run the zip
///   tidy first (it reads from `work_dir` without modifying it), then
///   the folder tidy (which moves it).
pub fn tidy_to_folder(
    work_dir: &Path,
    output_path: &Path,
    metadata: &TidyMetadata,
) -> io::Result<()> {
    validate_work_dir(work_dir)?;
    let parent = output_path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "output_path {} has no parent directory",
                output_path.display()
            ),
        )
    })?;
    if !parent.as_os_str().is_empty() && !parent.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("output parent {} is not a directory", parent.display()),
        ));
    }

    let populated = enumerate_populated_cells(work_dir, metadata.cell_depth)?;
    let manifest = build_final_manifest(work_dir, metadata, &populated)?;

    // Write the final manifest into work_dir alongside the existing
    // `quads/`, `gaia/`, and `build-manifest.json`. `manifest.save`
    // itself uses an atomic `.partial.json` + rename pattern, so a
    // crash here leaves work_dir consistent with no `manifest.json`.
    let manifest_path = work_dir.join("manifest.json");
    manifest.save(&manifest_path)?;

    // Pre-existing canonical output is replaced atomically by `rename`
    // when the target is a directory; on Linux this requires the target
    // to be empty, so delegate to remove_dir_all if a stale committed
    // bundle is present.
    if let Ok(meta) = std::fs::symlink_metadata(output_path) {
        if meta.is_dir() {
            std::fs::remove_dir_all(output_path)?;
        } else {
            std::fs::remove_file(output_path)?;
        }
    }

    // The commit: rename work_dir onto output_path. Atomic on the
    // same filesystem, instantaneous, no data movement.
    std::fs::rename(work_dir, output_path)?;

    Ok(())
}

// ---------------------------------------------------------------------------
//  Zip output
// ---------------------------------------------------------------------------

/// Tidy a populated `work_dir` into a zip-archive bundle at
/// `output_zip_path`.
///
/// Per-cell entries are written `Stored` (no compression — the binary
/// shards aren't compressible enough to be worth the CPU cost, and
/// `Stored` lets a future reader decompress in O(1) and `mmap` the
/// extracted bytes if it wants). `manifest.json` is written `Deflated`
/// as the last entry, since it's small JSON that compresses well.
pub fn tidy_to_zip(
    work_dir: &Path,
    output_zip_path: &Path,
    metadata: &TidyMetadata,
) -> io::Result<()> {
    validate_work_dir(work_dir)?;
    let parent = output_zip_path.parent().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            format!(
                "output_zip_path {} has no parent directory",
                output_zip_path.display()
            ),
        )
    })?;
    if !parent.as_os_str().is_empty() && !parent.is_dir() {
        return Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("output parent {} is not a directory", parent.display()),
        ));
    }

    let populated = enumerate_populated_cells(work_dir, metadata.cell_depth)?;
    let manifest = build_final_manifest(work_dir, metadata, &populated)?;

    let partial = partial_path_for(output_zip_path);
    remove_partial(&partial)?;

    {
        let f = File::create(&partial)?;
        let mut buf = BufWriter::new(f);
        {
            let mut zip = zip::ZipWriter::new(&mut buf);
            let stored_opts = FileOptions::default().compression_method(CompressionMethod::Stored);
            let deflated_opts =
                FileOptions::default().compression_method(CompressionMethod::Deflated);

            let width = cell_filename_width(metadata.cell_depth);
            for &cell_id in &populated {
                let q_name = format!("{QUADS_SUBDIR}/cell_{cell_id:0width$}.{QUAD_EXT}");
                let q_src = cell_shard_path(
                    work_dir,
                    QUADS_SUBDIR,
                    QUAD_EXT,
                    metadata.cell_depth,
                    cell_id,
                );
                zip_add_file(&mut zip, &q_name, &q_src, stored_opts)?;

                let g_name = format!("{GAIA_SUBDIR}/cell_{cell_id:0width$}.{GAIA_EXT}");
                let g_src = cell_shard_path(
                    work_dir,
                    GAIA_SUBDIR,
                    GAIA_EXT,
                    metadata.cell_depth,
                    cell_id,
                );
                zip_add_file(&mut zip, &g_name, &g_src, stored_opts)?;
            }

            zip.start_file("manifest.json", deflated_opts)
                .map_err(io::Error::other)?;
            let json = serde_json::to_vec_pretty(&manifest).map_err(io::Error::other)?;
            zip.write_all(&json)?;

            zip.finish().map_err(io::Error::other)?;
        }
        buf.flush()?;
    }

    if let Ok(meta) = std::fs::symlink_metadata(output_zip_path) {
        if meta.is_dir() {
            std::fs::remove_dir_all(output_zip_path)?;
        } else {
            std::fs::remove_file(output_zip_path)?;
        }
    }

    std::fs::rename(&partial, output_zip_path)?;

    Ok(())
}

fn zip_add_file<W: Write + std::io::Seek>(
    zip: &mut zip::ZipWriter<W>,
    name: &str,
    src: &Path,
    opts: FileOptions,
) -> io::Result<()> {
    zip.start_file(name, opts).map_err(io::Error::other)?;
    let bytes = std::fs::read(src)?;
    zip.write_all(&bytes)?;
    Ok(())
}

// ---------------------------------------------------------------------------
//  Work-dir cleanup
// ---------------------------------------------------------------------------

/// Recursively delete `work_dir`. **Destructive** — only the operator
/// or CLI knows whether a successful tidy means the work_dir is no
/// longer needed (a second tidy producing the alternate output form
/// reads from the same work_dir, so default policy is to keep it).
///
/// A second call on a non-existent path returns `NotFound`; callers
/// that want idempotent cleanup should match on that error kind.
pub fn prune_work_dir(work_dir: &Path) -> io::Result<()> {
    std::fs::remove_dir_all(work_dir)
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bundle::gaia_shard::{GaiaRecord, GaiaShard, write_gaia_shard};
    use crate::bundle::quad_shard::{BandEmit, QuadShard, write_quad_shard};
    use crate::index::build_manifest::{BuildManifest, CellStats};
    use crate::quads::{Code, Quad};
    use chrono::{DateTime, Utc};
    use std::io::{Cursor, Read};

    const TEST_DEPTH: u8 = 5;

    fn make_quad(a: usize, b: usize, c: usize, d: usize) -> Quad {
        Quad {
            star_ids: [a, b, c, d],
        }
    }

    fn make_code(seed: f64) -> Code {
        [seed, seed + 1.0, seed + 2.0, seed + 3.0]
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

    fn write_zqd(work_dir: &Path, cell_id: u32, bands: &[BandEmit<'_>]) {
        let path = cell_shard_path(work_dir, QUADS_SUBDIR, QUAD_EXT, TEST_DEPTH, cell_id);
        let f = File::create(&path).unwrap();
        let mut buf = BufWriter::new(f);
        write_quad_shard(&mut buf, cell_id as u64, bands).unwrap();
        buf.flush().unwrap();
    }

    fn write_zga(work_dir: &Path, cell_id: u32, mut records: Vec<GaiaRecord>) {
        let path = cell_shard_path(work_dir, GAIA_SUBDIR, GAIA_EXT, TEST_DEPTH, cell_id);
        let f = File::create(&path).unwrap();
        let mut buf = BufWriter::new(f);
        write_gaia_shard(&mut buf, cell_id as u64, &mut records).unwrap();
        buf.flush().unwrap();
    }

    /// Per-cell test fixture data: hand-rolled quads + codes for two
    /// bands (band 0 has more quads than band 1) and a few gaia
    /// records.
    struct CellFixture {
        cell_id: u32,
        band0_quads: Vec<Quad>,
        band0_codes: Vec<Code>,
        band1_quads: Vec<Quad>,
        band1_codes: Vec<Code>,
        gaia: Vec<GaiaRecord>,
    }

    fn fixture_cells() -> Vec<CellFixture> {
        vec![
            CellFixture {
                cell_id: 0,
                band0_quads: vec![
                    make_quad(0, 1, 2, 3),
                    make_quad(0, 1, 2, 4),
                    make_quad(0, 1, 3, 4),
                ],
                band0_codes: vec![make_code(0.0), make_code(0.1), make_code(0.2)],
                band1_quads: vec![make_quad(0, 1, 2, 3)],
                band1_codes: vec![make_code(1.0)],
                gaia: vec![
                    make_gaia(1000, 10.0),
                    make_gaia(1001, 11.0),
                    make_gaia(1002, 12.0),
                    make_gaia(1003, 13.0),
                    make_gaia(1004, 14.0),
                ],
            },
            CellFixture {
                cell_id: 7,
                band0_quads: vec![make_quad(0, 1, 2, 3), make_quad(0, 1, 2, 4)],
                band0_codes: vec![make_code(2.0), make_code(2.1)],
                band1_quads: vec![make_quad(0, 1, 2, 3), make_quad(0, 1, 2, 4)],
                band1_codes: vec![make_code(3.0), make_code(3.1)],
                gaia: vec![
                    make_gaia(7000, 10.5),
                    make_gaia(7001, 11.5),
                    make_gaia(7002, 12.5),
                    make_gaia(7003, 13.5),
                ],
            },
            CellFixture {
                cell_id: 42,
                band0_quads: vec![
                    make_quad(0, 1, 2, 3),
                    make_quad(0, 1, 2, 4),
                    make_quad(0, 1, 3, 4),
                    make_quad(0, 2, 3, 4),
                ],
                band0_codes: vec![
                    make_code(4.0),
                    make_code(4.1),
                    make_code(4.2),
                    make_code(4.3),
                ],
                band1_quads: vec![],
                band1_codes: vec![],
                gaia: vec![
                    make_gaia(42_000, 10.0),
                    make_gaia(42_001, 11.0),
                    make_gaia(42_002, 12.0),
                    make_gaia(42_003, 13.0),
                    make_gaia(42_004, 14.0),
                    make_gaia(42_005, 15.0),
                ],
            },
        ]
    }

    /// Build a minimal `work_dir` populated by the supplied cell
    /// fixtures. Returns the work_dir path (an opaque tempdir) and the
    /// matching `TidyMetadata` for the fixture.
    fn build_work_dir() -> (tempfile::TempDir, Vec<CellFixture>, TidyMetadata) {
        let tmp = tempfile::tempdir().unwrap();
        let work_dir = tmp.path();
        std::fs::create_dir_all(work_dir.join(QUADS_SUBDIR)).unwrap();
        std::fs::create_dir_all(work_dir.join(GAIA_SUBDIR)).unwrap();

        let cells = fixture_cells();
        let mut bm = BuildManifest::default();
        bm.ensure_per_band(2);
        for cell in &cells {
            let bands = vec![
                BandEmit {
                    band_idx: 0,
                    quads: &cell.band0_quads,
                    codes: &cell.band0_codes,
                },
                BandEmit {
                    band_idx: 1,
                    quads: &cell.band1_quads,
                    codes: &cell.band1_codes,
                },
            ];
            write_zqd(work_dir, cell.cell_id, &bands);
            write_zga(work_dir, cell.cell_id, cell.gaia.clone());

            bm.commit_cell(
                cell.cell_id,
                CellStats {
                    n_stars: cell.gaia.len() as u64,
                    n_quads: (cell.band0_quads.len() + cell.band1_quads.len()) as u64,
                },
            );
            bm.mark_band_complete(cell.cell_id, 0);
            bm.mark_band_complete(cell.cell_id, 1);
        }
        bm.save(work_dir).unwrap();

        let metadata = TidyMetadata {
            cell_depth: TEST_DEPTH,
            experiment: "tidy fixture".to_string(),
            build_metadata: BuildMetadata {
                tool: "zodiacal-tools test".to_string(),
                build_started_utc: "2026-05-05T00:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                build_finished_utc: "2026-05-05T01:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                source: super::super::manifest::BuildSource {
                    kind: "test-fixture".to_string(),
                    release: "test".to_string(),
                    path: "/tmp/fixture".to_string(),
                },
            },
            gaia: GaiaMetadata {
                max_stars_per_cell: 10_000,
                mag_limit: 20.0,
                schema_version: 1,
            },
            bands: vec![
                BandMetadata {
                    label: "band_00".to_string(),
                    scale_lower_arcsec: 10.0,
                    scale_upper_arcsec: 14.142,
                    quads_per_cell: 100,
                    max_reuse: 8,
                },
                BandMetadata {
                    label: "band_01".to_string(),
                    scale_lower_arcsec: 14.142,
                    scale_upper_arcsec: 20.0,
                    quads_per_cell: 100,
                    max_reuse: 8,
                },
            ],
        };
        (tmp, cells, metadata)
    }

    fn expected_gaia_total(cells: &[CellFixture]) -> u64 {
        cells.iter().map(|c| c.gaia.len() as u64).sum()
    }

    fn expected_band0_quads(cells: &[CellFixture]) -> u64 {
        cells.iter().map(|c| c.band0_quads.len() as u64).sum()
    }

    fn expected_band1_quads(cells: &[CellFixture]) -> u64 {
        cells.iter().map(|c| c.band1_quads.len() as u64).sum()
    }

    fn expected_band1_pop(cells: &[CellFixture]) -> u32 {
        cells.iter().filter(|c| !c.band1_quads.is_empty()).count() as u32
    }

    #[test]
    fn folder_tidy_produces_expected_layout() {
        let (work_tmp, cells, metadata) = build_work_dir();
        let out_tmp = tempfile::tempdir().unwrap();
        let out = out_tmp.path().join("index.zdcl.bundle");

        tidy_to_folder(work_tmp.path(), &out, &metadata).unwrap();

        assert!(out.is_dir(), "output is not a directory");
        assert!(out.join("manifest.json").is_file());
        for cell in &cells {
            let q = cell_shard_path(&out, QUADS_SUBDIR, QUAD_EXT, TEST_DEPTH, cell.cell_id);
            let g = cell_shard_path(&out, GAIA_SUBDIR, GAIA_EXT, TEST_DEPTH, cell.cell_id);
            assert!(q.is_file(), "missing {}", q.display());
            assert!(g.is_file(), "missing {}", g.display());
        }

        let m = BundleManifest::load(&out.join("manifest.json")).unwrap();
        assert_eq!(m.cell_depth, TEST_DEPTH);
        assert_eq!(m.n_cells, 12_288);
        assert_eq!(m.gaia.n_total, expected_gaia_total(&cells));
        assert_eq!(m.gaia.populated_cells, cells.len() as u32);
        assert_eq!(m.bands.len(), 2);
        assert_eq!(m.bands[0].n_quads_total, expected_band0_quads(&cells));
        assert_eq!(m.bands[1].n_quads_total, expected_band1_quads(&cells));
        assert_eq!(m.bands[1].populated_cells, expected_band1_pop(&cells));
    }

    #[test]
    fn zip_tidy_produces_readable_archive() {
        let (work_tmp, cells, metadata) = build_work_dir();
        let out_tmp = tempfile::tempdir().unwrap();
        let zip_path = out_tmp.path().join("index.zdcl.bundle.zip");

        tidy_to_zip(work_tmp.path(), &zip_path, &metadata).unwrap();
        assert!(zip_path.is_file());

        let f = File::open(&zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(f).unwrap();
        // Walk by index so we see the *stored* order, not the
        // hash-map-iteration order that `file_names()` would return.
        let names: Vec<String> = (0..archive.len())
            .map(|i| archive.by_index(i).unwrap().name().to_string())
            .collect();

        // Cells appear in ascending id order; manifest is the last entry.
        let width = cell_filename_width(TEST_DEPTH);
        let mut expected: Vec<String> = Vec::new();
        for cell in &cells {
            let cid = cell.cell_id;
            expected.push(format!("{QUADS_SUBDIR}/cell_{cid:0width$}.{QUAD_EXT}"));
            expected.push(format!("{GAIA_SUBDIR}/cell_{cid:0width$}.{GAIA_EXT}"));
        }
        expected.push("manifest.json".to_string());
        assert_eq!(names, expected, "zip entry order mismatch");

        // Per-cell entries are Stored, manifest is Deflated.
        for name in &expected {
            let entry = archive.by_name(name).unwrap();
            if name == "manifest.json" {
                assert_eq!(entry.compression(), CompressionMethod::Deflated);
            } else {
                assert_eq!(entry.compression(), CompressionMethod::Stored);
            }
        }

        let mut mf_bytes = Vec::new();
        archive
            .by_name("manifest.json")
            .unwrap()
            .read_to_end(&mut mf_bytes)
            .unwrap();
        let m: BundleManifest = serde_json::from_slice(&mf_bytes).unwrap();
        assert_eq!(m.gaia.n_total, expected_gaia_total(&cells));
        assert_eq!(m.bands.len(), 2);
    }

    #[test]
    fn folder_tidy_consumes_work_dir() {
        let (work_tmp, _cells, metadata) = build_work_dir();
        let work_path = work_tmp.path().to_path_buf();
        let out_tmp = tempfile::tempdir().unwrap();
        let out = out_tmp.path().join("index.zdcl.bundle");

        tidy_to_folder(&work_path, &out, &metadata).unwrap();

        // work_dir was renamed onto output_path, so it should no
        // longer exist at its original location.
        assert!(
            !work_path.exists(),
            "work_dir survived tidy_to_folder; expected it to be moved"
        );
        // The output bundle is well-formed.
        assert!(out.join("manifest.json").is_file());
        assert!(out.join(QUADS_SUBDIR).is_dir());
        assert!(out.join(GAIA_SUBDIR).is_dir());

        // Defuse the TempDir Drop: the directory has been moved to a
        // sibling location, so the drop's cleanup is a no-op and the
        // OS will reclaim the temp parent normally.
        std::mem::forget(work_tmp);
    }

    #[test]
    fn folder_tidy_replaces_existing_output() {
        let (work_tmp, _cells, metadata) = build_work_dir();
        let out_tmp = tempfile::tempdir().unwrap();
        let out = out_tmp.path().join("index.zdcl.bundle");

        // Pre-create a stale committed bundle at the output path.
        std::fs::create_dir_all(out.join("quads")).unwrap();
        std::fs::write(out.join("garbage.bin"), b"stale committed bundle").unwrap();
        std::fs::write(out.join("manifest.json"), b"stale-manifest-bytes").unwrap();

        tidy_to_folder(work_tmp.path(), &out, &metadata).unwrap();

        // The stale output got replaced wholesale; the new manifest is parseable.
        assert!(out.join("manifest.json").is_file());
        assert!(!out.join("garbage.bin").exists(), "stale file survived");
        let _ = BundleManifest::load(&out.join("manifest.json")).unwrap();
        std::mem::forget(work_tmp);
    }

    #[test]
    fn tidy_zip_then_folder_byte_equivalent_logical() {
        // Order matters with the new tidy_to_folder semantics: zip
        // reads from work_dir without modifying it, folder consumes
        // it. So zip must come first if we want both outputs from the
        // same build.
        let (work_tmp, cells, metadata) = build_work_dir();
        let out_tmp = tempfile::tempdir().unwrap();
        let folder_out = out_tmp.path().join("index.zdcl.bundle");
        let zip_out = out_tmp.path().join("index.zdcl.bundle.zip");

        tidy_to_zip(work_tmp.path(), &zip_out, &metadata).unwrap();
        tidy_to_folder(work_tmp.path(), &folder_out, &metadata).unwrap();
        std::mem::forget(work_tmp);

        let f = File::open(&zip_out).unwrap();
        let mut archive = zip::ZipArchive::new(f).unwrap();

        let width = cell_filename_width(TEST_DEPTH);
        for cell in &cells {
            let cid = cell.cell_id;
            let q_name = format!("{QUADS_SUBDIR}/cell_{cid:0width$}.{QUAD_EXT}");
            let g_name = format!("{GAIA_SUBDIR}/cell_{cid:0width$}.{GAIA_EXT}");

            let folder_q = std::fs::read(cell_shard_path(
                &folder_out,
                QUADS_SUBDIR,
                QUAD_EXT,
                TEST_DEPTH,
                cid,
            ))
            .unwrap();
            let folder_g = std::fs::read(cell_shard_path(
                &folder_out,
                GAIA_SUBDIR,
                GAIA_EXT,
                TEST_DEPTH,
                cid,
            ))
            .unwrap();

            let mut zip_q = Vec::new();
            archive
                .by_name(&q_name)
                .unwrap()
                .read_to_end(&mut zip_q)
                .unwrap();
            let mut zip_g = Vec::new();
            archive
                .by_name(&g_name)
                .unwrap()
                .read_to_end(&mut zip_g)
                .unwrap();

            assert_eq!(folder_q, zip_q, "quads bytes diverge for cell {cid}");
            assert_eq!(folder_g, zip_g, "gaia bytes diverge for cell {cid}");

            // Sanity check: the bytes round-trip through both shard parsers.
            QuadShard::parse(&zip_q).unwrap();
            GaiaShard::parse(&zip_g).unwrap();
        }

        // Manifests parse to equal structs (both use deterministic field order).
        let folder_manifest = BundleManifest::load(&folder_out.join("manifest.json")).unwrap();
        let mut zip_manifest_bytes = Vec::new();
        archive
            .by_name("manifest.json")
            .unwrap()
            .read_to_end(&mut zip_manifest_bytes)
            .unwrap();
        let zip_manifest: BundleManifest = serde_json::from_slice(&zip_manifest_bytes).unwrap();
        assert_eq!(folder_manifest, zip_manifest);
    }

    #[test]
    fn per_band_totals_match_band_table() {
        let (work_tmp, cells, metadata) = build_work_dir();
        let out_tmp = tempfile::tempdir().unwrap();
        let out = out_tmp.path().join("index.zdcl.bundle");

        tidy_to_folder(work_tmp.path(), &out, &metadata).unwrap();

        let m = BundleManifest::load(&out.join("manifest.json")).unwrap();
        let expected_b0 = expected_band0_quads(&cells);
        let expected_b1 = expected_band1_quads(&cells);
        assert_eq!(m.bands[0].n_quads_total, expected_b0);
        assert_eq!(m.bands[1].n_quads_total, expected_b1);
        // Cross-check by re-parsing every cell's `.zqd` band table.
        let mut sum_b0 = 0u64;
        let mut sum_b1 = 0u64;
        let mut pop_b0 = 0u32;
        let mut pop_b1 = 0u32;
        for cell in &cells {
            let path = cell_shard_path(&out, QUADS_SUBDIR, QUAD_EXT, TEST_DEPTH, cell.cell_id);
            let bytes = std::fs::read(&path).unwrap();
            let shard = QuadShard::parse(&bytes).unwrap();
            for entry in shard.bands() {
                let n = entry.n_quads as u64;
                match entry.band_idx {
                    0 => {
                        sum_b0 += n;
                        if n > 0 {
                            pop_b0 += 1;
                        }
                    }
                    1 => {
                        sum_b1 += n;
                        if n > 0 {
                            pop_b1 += 1;
                        }
                    }
                    _ => panic!("unexpected band_idx {}", entry.band_idx),
                }
            }
        }
        assert_eq!(sum_b0, expected_b0);
        assert_eq!(sum_b1, expected_b1);
        assert_eq!(m.bands[0].populated_cells, pop_b0);
        assert_eq!(m.bands[1].populated_cells, pop_b1);
    }

    #[test]
    fn missing_zga_for_zqd_rejected() {
        let tmp = tempfile::tempdir().unwrap();
        let work_dir = tmp.path();
        std::fs::create_dir_all(work_dir.join(QUADS_SUBDIR)).unwrap();
        std::fs::create_dir_all(work_dir.join(GAIA_SUBDIR)).unwrap();

        // Write a `.zqd` for cell 1, but no `.zga`.
        let q = vec![make_quad(0, 1, 2, 3)];
        let c = vec![make_code(0.0)];
        write_zqd(
            work_dir,
            1,
            &[BandEmit {
                band_idx: 0,
                quads: &q,
                codes: &c,
            }],
        );

        // Build manifest exists but doesn't matter for this validation.
        let mut bm = BuildManifest::default();
        bm.ensure_per_band(1);
        bm.save(work_dir).unwrap();

        let metadata = TidyMetadata {
            cell_depth: TEST_DEPTH,
            experiment: "test".into(),
            build_metadata: BuildMetadata {
                tool: "tool".into(),
                build_started_utc: "2026-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                build_finished_utc: "2026-01-01T00:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                source: super::super::manifest::BuildSource {
                    kind: "k".into(),
                    release: "r".into(),
                    path: "p".into(),
                },
            },
            gaia: GaiaMetadata {
                max_stars_per_cell: 1,
                mag_limit: 20.0,
                schema_version: 1,
            },
            bands: vec![BandMetadata {
                label: "band_00".into(),
                scale_lower_arcsec: 10.0,
                scale_upper_arcsec: 20.0,
                quads_per_cell: 1,
                max_reuse: 1,
            }],
        };

        let out_tmp = tempfile::tempdir().unwrap();
        let out = out_tmp.path().join("index.zdcl.bundle");
        let err = tidy_to_folder(work_dir, &out, &metadata).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(
            err.to_string().contains("no matching"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn prune_work_dir_removes_work_dir() {
        let (work_tmp, _cells, _metadata) = build_work_dir();
        let path = work_tmp.path().to_path_buf();
        // Drop the TempDir's ownership so it doesn't try to clean up
        // an already-deleted path.
        let _ = work_tmp.keep();
        assert!(path.exists());

        prune_work_dir(&path).unwrap();
        assert!(!path.exists());

        // Second call returns NotFound.
        let err = prune_work_dir(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::NotFound);
    }

    #[test]
    fn zip_entry_bytes_round_trip_through_parsers() {
        // Independent sanity check: the zip output's per-cell entries
        // can be parsed by their respective shard parsers without
        // reference to the folder output. (Catches a regression where
        // we accidentally compress with Deflated and forget to
        // decompress on read.)
        let (work_tmp, cells, metadata) = build_work_dir();
        let out_tmp = tempfile::tempdir().unwrap();
        let zip_path = out_tmp.path().join("index.zdcl.bundle.zip");
        tidy_to_zip(work_tmp.path(), &zip_path, &metadata).unwrap();

        let f = File::open(&zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(f).unwrap();
        let width = cell_filename_width(TEST_DEPTH);
        for cell in &cells {
            let q_name = format!(
                "{QUADS_SUBDIR}/cell_{cid:0width$}.{QUAD_EXT}",
                cid = cell.cell_id
            );
            let mut q_bytes = Vec::new();
            archive
                .by_name(&q_name)
                .unwrap()
                .read_to_end(&mut q_bytes)
                .unwrap();
            let shard = QuadShard::parse(&q_bytes).unwrap();
            assert_eq!(shard.cell_id(), cell.cell_id as u64);
        }
    }

    /// Sanity check: `serde_json::from_reader` on a `Cursor<Vec<u8>>` of
    /// the zip-extracted manifest bytes parses identically to
    /// `serde_json::from_slice`. Exists so the test surface keeps the
    /// "manifest is plain JSON in the zip" property explicit even
    /// though no production code path hits it.
    #[test]
    fn manifest_json_in_zip_is_plain_json() {
        let (work_tmp, _cells, metadata) = build_work_dir();
        let out_tmp = tempfile::tempdir().unwrap();
        let zip_path = out_tmp.path().join("b.zip");
        tidy_to_zip(work_tmp.path(), &zip_path, &metadata).unwrap();

        let f = File::open(&zip_path).unwrap();
        let mut archive = zip::ZipArchive::new(f).unwrap();
        let mut bytes = Vec::new();
        archive
            .by_name("manifest.json")
            .unwrap()
            .read_to_end(&mut bytes)
            .unwrap();
        let from_slice: BundleManifest = serde_json::from_slice(&bytes).unwrap();
        let from_reader: BundleManifest = serde_json::from_reader(Cursor::new(&bytes)).unwrap();
        assert_eq!(from_slice, from_reader);
    }
}
