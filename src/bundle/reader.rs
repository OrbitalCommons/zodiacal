//! `ZdclBundle` — consumption-side reader for a `.zdcl.bundle`.
//!
//! The reader wraps a [`SubfileAccessor`] (either an [`FsAccessor`] over a
//! finalized bundle directory, or a [`ZipAccessor`] over the equivalent
//! `.zip` archive) and serves three kinds of query:
//!
//! - **Region load:** [`ZdclBundle::load_region`] returns a
//!   [`MultiBandFragment`] — the union of every cell's gaia records
//!   intersecting the [`SkyRegion`], plus one [`IndexFragment`] per scale
//!   band whose quads/codes have been concatenated and remapped to indices
//!   into the unioned gaia-record vector. [`ZdclBundle::load_region_band`]
//!   returns a single band's `IndexFragment`.
//! - **Per-source-id lookup:** [`ZdclBundle::gaia_get`] /
//!   [`ZdclBundle::gaia_get_many`] resolve a Gaia source_id to its 104-byte
//!   record. The cell can be supplied as a hint or derived from the
//!   source_id's HEALPix-12 prefix (Gaia DR3 convention).
//! - **Structural verification:** [`ZdclBundle::verify`] walks every
//!   populated cell and checks magic / version / cell_id / size invariants.
//!
//! Per-cell shard bytes are cached on first access in an internal
//! `Arc`-shared map so repeated region queries don't re-read the
//! accessor's underlying mmap (FS) or re-decompress (zip).
//!
//! ## Caveat: shared gaia records vs. per-band fragments
//!
//! The current revision duplicates the constructed `Vec<IndexStar>` into
//! every band's `IndexFragment.stars`. The format spec calls out a future
//! optimization that lets all bands share one star vector; that's deferred
//! to a later PR. The duplication only costs a few hundred kB per band
//! per region load and never changes solver semantics.

use std::collections::BTreeSet;
use std::io;
use std::num::NonZeroUsize;
use std::path::Path;
use std::sync::{Arc, Mutex};

use cdshealpix::nested;
use lru::LruCache;
use starfield::Equatorial;
use starfield::time::Timescale;

use crate::geom::ProperMotion;
use crate::index::RefEpoch;

/// Default eviction cap for [`ZdclBundle`]'s per-cell shard LRU. Tuned
/// so a 1000-case bench-bundle sweep at ~6,000 cells/case (5° hint on
/// a depth-9 bundle) stays within typical host RAM. Override via
/// [`ZdclBundle::open_with_capacity`].
pub const ZDCL_BUNDLE_DEFAULT_CACHE_CAP: usize = 100_000;

use crate::index::IndexStar;
use crate::index::source::IndexFragment;
use crate::quads::{Code, Quad};
use crate::solver::SkyRegion;

use super::accessor::{FsAccessor, SubfileAccessor, ZipAccessor};
use super::gaia_shard::{GaiaRecord, GaiaShard};
use super::layout::cell_filename_width;
use super::manifest::{BandInfo, BundleManifest, FORMAT_MAGIC, FORMAT_VERSION, GAIA_RECORD_SIZE};
use super::quad_shard::QuadShard;

/// Subdirectory holding `.zqd` quad shards inside a bundle.
const QUADS_SUBDIR: &str = "quads";
/// Subdirectory holding `.zga` gaia shards inside a bundle.
const GAIA_SUBDIR: &str = "gaia";
/// `.zqd` extension (without leading dot).
const QUAD_EXT: &str = "zqd";
/// `.zga` extension (without leading dot).
const GAIA_EXT: &str = "zga";
/// Manifest entry name inside a bundle.
const MANIFEST_NAME: &str = "manifest.json";

/// Multi-band slice of a bundle over a [`SkyRegion`].
///
/// `gaia_records` is the union of every overlapping cell's records, in
/// stable order (cells visited ascending, records within a cell preserved
/// in their on-disk order). `fragments[k]` corresponds to
/// `bands[k]` of the bundle's manifest, and its quads/codes have been
/// remapped from per-cell-local star indices to indices into
/// `gaia_records` (and equivalently into `fragments[k].stars`).
#[derive(Debug, Clone)]
pub struct MultiBandFragment {
    /// Concatenated 104-byte Gaia records from every cell that overlapped
    /// the region.
    pub gaia_records: Vec<GaiaRecord>,
    /// One per band, in `band_idx` order (which equals position).
    pub fragments: Vec<IndexFragment>,
}

/// Report returned by [`ZdclBundle::verify`].
#[derive(Debug, Clone, Default)]
pub struct VerifyReport {
    /// Number of populated cells walked.
    pub n_cells_checked: u32,
    /// Per-cell errors. Empty iff the bundle passed.
    pub errors: Vec<VerifyError>,
}

/// One structural problem found by [`ZdclBundle::verify`].
#[derive(Debug, Clone)]
pub struct VerifyError {
    /// Cell whose shard tripped the check. `MissingShard` errors carry
    /// the manifest-claimed cell id of the missing file.
    pub cell_id: u32,
    /// Categorized error kind.
    pub kind: VerifyErrorKind,
    /// Human-readable description (full underlying error message).
    pub detail: String,
}

/// Categorization of [`VerifyError`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyErrorKind {
    /// Magic bytes did not match `b"ZDCLQUAD"` or `b"ZDCLGAIA"`.
    BadMagic,
    /// On-disk format version was not the expected `1`.
    BadVersion,
    /// Embedded `cell_id` field disagreed with the filename's cell id.
    BadCellId,
    /// Slice length / record-size invariant was violated.
    BadSize,
    /// A `.zqd` or `.zga` was missing for a populated cell.
    MissingShard,
}

/// Cached per-cell shard bytes. `Arc<[u8]>` lets the cache hand out a
/// cheap clone of the underlying buffer without holding the cache mutex
/// across the parse.
#[derive(Debug)]
struct CellEntries {
    quads: Arc<[u8]>,
    gaia: Arc<[u8]>,
}

/// Bundle-format reader.
///
/// Construct via [`ZdclBundle::open`] (auto-detects directory vs zip).
pub struct ZdclBundle {
    accessor: Box<dyn SubfileAccessor>,
    manifest: BundleManifest,
    populated_cells: BTreeSet<u32>,
    cell_cache: Mutex<LruCache<u32, Arc<CellEntries>>>,
    /// Timescale used to materialise `Time` from Gaia `ref_epoch` values
    /// at load time. Held here so every `IndexStar` produced from this
    /// bundle shares the same `Arc<TimescaleInner>`.
    timescale: Timescale,
}

impl std::fmt::Debug for ZdclBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZdclBundle")
            .field("cell_depth", &self.manifest.cell_depth)
            .field("n_bands", &self.manifest.bands.len())
            .field("populated_cells", &self.populated_cells.len())
            .finish()
    }
}

impl ZdclBundle {
    /// Open a bundle at `path`.
    ///
    /// Auto-detects packaging:
    /// - directory → [`FsAccessor`]
    /// - file ending in `.zip` → [`ZipAccessor`]
    /// - anything else → `InvalidInput`
    ///
    /// Reads and validates `manifest.json`, then enumerates the populated
    /// cell ids by listing entries under `quads/`. The populated-cell set
    /// is cached for the bundle's lifetime.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        Self::open_with_capacity(path, ZDCL_BUNDLE_DEFAULT_CACHE_CAP)
    }

    /// Open a bundle with an explicit per-cell shard LRU cap. Use a
    /// larger cap for long-running query servers; smaller caps trade
    /// hit rate for tighter memory use.
    pub fn open_with_capacity(path: impl AsRef<Path>, cap: usize) -> io::Result<Self> {
        let cap = NonZeroUsize::new(cap.max(1)).expect("max(1) ensures non-zero");
        let path = path.as_ref();
        let meta = std::fs::metadata(path)?;
        let accessor: Box<dyn SubfileAccessor> = if meta.is_dir() {
            Box::new(FsAccessor::open(path)?)
        } else if meta.is_file() {
            // Match `.zip` (case-insensitively) on the file extension —
            // anything else is rejected so we don't try to interpret a
            // stray binary as a bundle.
            let is_zip = path
                .extension()
                .and_then(|e| e.to_str())
                .map(|e| e.eq_ignore_ascii_case("zip"))
                .unwrap_or(false);
            if !is_zip {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    format!("{} is a file but not a .zip bundle", path.display()),
                ));
            }
            Box::new(ZipAccessor::open(path)?)
        } else {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "{} is neither a directory nor a regular file",
                    path.display()
                ),
            ));
        };

        let manifest = load_manifest_from_accessor(accessor.as_ref())?;
        let populated_cells = enumerate_populated_cells(accessor.as_ref(), manifest.cell_depth)?;

        Ok(Self {
            accessor,
            manifest,
            populated_cells,
            cell_cache: Mutex::new(LruCache::new(cap)),
            timescale: Timescale::default(),
        })
    }

    /// Reference to the parsed `manifest.json`.
    #[inline]
    pub fn manifest(&self) -> &BundleManifest {
        &self.manifest
    }

    /// HEALPix nested-scheme depth used for cell sharding.
    #[inline]
    pub fn cell_depth(&self) -> u8 {
        self.manifest.cell_depth
    }

    /// Per-band manifest entries, in `band_idx` order.
    #[inline]
    pub fn bands(&self) -> &[BandInfo] {
        &self.manifest.bands
    }

    /// Sorted set of populated cell ids (cells with both a `.zqd` and a
    /// `.zga` on disk, per the manifest's enumeration).
    #[inline]
    pub fn populated_cells(&self) -> &BTreeSet<u32> {
        &self.populated_cells
    }

    /// Slice every band over `region`.
    pub fn load_region(&self, region: &SkyRegion) -> io::Result<MultiBandFragment> {
        let cells = self.cells_in_region(region);

        // Per-cell parsed shards plus the running base index of each
        // cell's gaia records inside the concatenated `gaia_records`.
        struct CellLoaded {
            entries: Arc<CellEntries>,
            base: usize,
            n_records: usize,
        }
        let mut loaded: Vec<CellLoaded> = Vec::with_capacity(cells.len());
        let mut gaia_records: Vec<GaiaRecord> = Vec::new();

        for cell_id in &cells {
            let entries = self.cell_bytes(*cell_id)?;
            let gaia = GaiaShard::parse(&entries.gaia).map_err(|e| {
                io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("cell {cell_id} gaia shard: {e}"),
                )
            })?;
            let base = gaia_records.len();
            let n = gaia.len();
            gaia_records.extend_from_slice(gaia.records());
            loaded.push(CellLoaded {
                entries,
                base,
                n_records: n,
            });
        }

        // Pre-build a per-band scale range pulled from the manifest.
        let n_bands = self.manifest.bands.len();
        let scale_ranges: Vec<(f64, f64)> = self
            .manifest
            .bands
            .iter()
            .map(|b| {
                (
                    arcsec_to_radians(b.scale_lower_arcsec),
                    arcsec_to_radians(b.scale_upper_arcsec),
                )
            })
            .collect();

        // Build the canonical IndexStar vector once from the unioned
        // gaia records. Each band's fragment shares this list (cloned).
        let stars: Vec<IndexStar> = gaia_records
            .iter()
            .map(|g| gaia_record_to_index_star(g, &self.timescale))
            .collect();

        // Concatenate per-band quads + codes across cells, remapping
        // local cell-relative star indices to global indices into
        // `gaia_records` via each cell's `base`.
        let mut per_band_quads: Vec<Vec<Quad>> = (0..n_bands).map(|_| Vec::new()).collect();
        let mut per_band_codes: Vec<Vec<Code>> = (0..n_bands).map(|_| Vec::new()).collect();

        for cell in &loaded {
            let qs = QuadShard::parse(&cell.entries.quads).map_err(|e| {
                io::Error::new(io::ErrorKind::InvalidData, format!("cell quad shard: {e}"))
            })?;
            for entry in qs.bands() {
                let band_idx = entry.band_idx as usize;
                if band_idx >= n_bands {
                    // Manifest says n_bands; on-disk band_idx out of
                    // range is malformed but we surface a clear error
                    // rather than panic.
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "cell {} carries band_idx {} but manifest has only {n_bands} bands",
                            qs.cell_id(),
                            entry.band_idx
                        ),
                    ));
                }
                let view = qs
                    .band(entry.band_idx)
                    .expect("band_idx came from this shard's table");
                let q_dst = &mut per_band_quads[band_idx];
                let c_dst = &mut per_band_codes[band_idx];
                q_dst.reserve(entry.n_quads as usize);
                c_dst.reserve(entry.n_quads as usize);
                let base = cell.base;
                let n_records = cell.n_records;
                for q in view.quads_iter() {
                    let mut remapped = q;
                    for slot in remapped.star_ids.iter_mut() {
                        if *slot >= n_records {
                            return Err(io::Error::new(
                                io::ErrorKind::InvalidData,
                                format!(
                                    "cell {} band {} quad references star idx {slot} but cell has only {n_records} records",
                                    qs.cell_id(),
                                    entry.band_idx
                                ),
                            ));
                        }
                        *slot += base;
                    }
                    q_dst.push(remapped);
                }
                c_dst.extend(view.codes_iter());
            }
        }

        let fragments: Vec<IndexFragment> = (0..n_bands)
            .map(|k| IndexFragment {
                stars: stars.clone(),
                quads: std::mem::take(&mut per_band_quads[k]),
                codes: std::mem::take(&mut per_band_codes[k]),
                scale_lower: scale_ranges[k].0,
                scale_upper: scale_ranges[k].1,
                metadata: None,
            })
            .collect();

        Ok(MultiBandFragment {
            gaia_records,
            fragments,
        })
    }

    /// Load a single band's quads/codes for `region`.
    ///
    /// Equivalent to `load_region(region).fragments[band_idx]` but with a
    /// bounds check on `band_idx` returning `InvalidInput` instead of
    /// panicking.
    pub fn load_region_band(
        &self,
        band_idx: usize,
        region: &SkyRegion,
    ) -> io::Result<IndexFragment> {
        let n_bands = self.manifest.bands.len();
        if band_idx >= n_bands {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("band_idx {band_idx} out of range; bundle has {n_bands} bands"),
            ));
        }
        let multi = self.load_region(region)?;
        let mut fragments = multi.fragments;
        Ok(fragments.swap_remove(band_idx))
    }

    /// Look up a single Gaia record by `source_id`.
    ///
    /// `cell_hint`:
    /// - `Some(cell_id)`: open that cell's `.zga` directly and binsearch.
    ///   Returns `Ok(None)` if `cell_id` is not populated in this bundle
    ///   or if the source_id is not present in that cell's shard.
    /// - `None`: derive the cell from the source_id's HEALPix-12 prefix
    ///   (Gaia DR3 convention). Returns `InvalidInput` if the bundle's
    ///   `cell_depth > 12` (the encoding can't disambiguate) or if
    ///   `source_id` has bit 63 set (supplement record — caller must pass
    ///   `cell_hint`).
    pub fn gaia_get(
        &self,
        source_id: u64,
        cell_hint: Option<u32>,
    ) -> io::Result<Option<GaiaRecord>> {
        let cell_id = match cell_hint {
            Some(c) => c,
            None => match self.derive_cell_from_source_id(source_id)? {
                Some(c) => c,
                None => return Ok(None),
            },
        };
        if !self.populated_cells.contains(&cell_id) {
            return Ok(None);
        }
        let entries = self.cell_bytes(cell_id)?;
        let shard = GaiaShard::parse(&entries.gaia).map_err(|e| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                format!("cell {cell_id} gaia shard: {e}"),
            )
        })?;
        Ok(shard.find(source_id).copied())
    }

    /// Look up many Gaia records.
    ///
    /// Returns a `Vec<Option<GaiaRecord>>` whose length matches `source_ids`.
    /// V1 implementation iterates [`Self::gaia_get`] per id without
    /// bulk optimization; a future revision could group ids by derived
    /// cell and amortize shard parsing across calls. The output ordering
    /// matches `source_ids` exactly.
    pub fn gaia_get_many(&self, source_ids: &[u64]) -> io::Result<Vec<Option<GaiaRecord>>> {
        // TODO(future): group by derived cell, parse each cell's shard
        // once, run a galloping search per group.
        let mut out = Vec::with_capacity(source_ids.len());
        for &id in source_ids {
            out.push(self.gaia_get(id, None)?);
        }
        Ok(out)
    }

    /// Walk every populated cell, checking magic, version, embedded
    /// cell_id, and basic size invariants.
    ///
    /// Errors are accumulated; the report carries every problem found.
    /// Returning `Ok` does **not** mean the bundle passed — check
    /// [`VerifyReport::errors`].
    pub fn verify(&self) -> io::Result<VerifyReport> {
        let mut report = VerifyReport::default();
        let depth = self.manifest.cell_depth;

        for &cell_id in &self.populated_cells {
            report.n_cells_checked += 1;

            let q_path = quad_entry_name(depth, cell_id);
            let g_path = gaia_entry_name(depth, cell_id);

            // Quads.
            match self.accessor.read_entry(&q_path) {
                Ok(bytes) => {
                    if let Err(err) = QuadShard::parse(&bytes) {
                        let kind = classify_shard_parse_error(&err);
                        report.errors.push(VerifyError {
                            cell_id,
                            kind,
                            detail: format!("{q_path}: {err}"),
                        });
                    } else {
                        // Parse succeeded — cross-check embedded cell_id.
                        let shard = QuadShard::parse(&bytes).expect("parsed above");
                        if shard.cell_id() != cell_id as u64 {
                            report.errors.push(VerifyError {
                                cell_id,
                                kind: VerifyErrorKind::BadCellId,
                                detail: format!(
                                    "{q_path}: header cell_id={} but filename cell_id={cell_id}",
                                    shard.cell_id()
                                ),
                            });
                        }
                    }
                }
                Err(err) => {
                    report.errors.push(VerifyError {
                        cell_id,
                        kind: VerifyErrorKind::MissingShard,
                        detail: format!("{q_path}: {err}"),
                    });
                }
            }

            // Gaia.
            match self.accessor.read_entry(&g_path) {
                Ok(bytes) => match GaiaShard::parse(&bytes) {
                    Ok(shard) => {
                        if shard.cell_id() != cell_id as u64 {
                            report.errors.push(VerifyError {
                                cell_id,
                                kind: VerifyErrorKind::BadCellId,
                                detail: format!(
                                    "{g_path}: header cell_id={} but filename cell_id={cell_id}",
                                    shard.cell_id()
                                ),
                            });
                        }
                    }
                    Err(err) => {
                        let kind = classify_shard_parse_error(&err);
                        report.errors.push(VerifyError {
                            cell_id,
                            kind,
                            detail: format!("{g_path}: {err}"),
                        });
                    }
                },
                Err(err) => {
                    report.errors.push(VerifyError {
                        cell_id,
                        kind: VerifyErrorKind::MissingShard,
                        detail: format!("{g_path}: {err}"),
                    });
                }
            }
        }

        Ok(report)
    }

    // -----------------------------------------------------------------
    //  Internal helpers
    // -----------------------------------------------------------------

    /// Resolve `region` to the populated cells inside the bundle that
    /// cover it. Cells the manifest doesn't list (because they were
    /// empty at build time) are filtered out.
    fn cells_in_region(&self, region: &SkyRegion) -> Vec<u32> {
        let depth = self.manifest.cell_depth;
        let bmoc = nested::cone_coverage_approx(
            depth,
            region.center.ra,
            region.center.dec,
            region.radius_rad,
        );
        let mut cells: Vec<u32> = bmoc
            .flat_iter()
            .filter_map(|id| {
                if id <= u32::MAX as u64 {
                    let id32 = id as u32;
                    if self.populated_cells.contains(&id32) {
                        Some(id32)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect();
        cells.sort_unstable();
        cells.dedup();
        cells
    }

    /// Load the per-cell `.zqd` + `.zga` byte buffers (cached).
    fn cell_bytes(&self, cell_id: u32) -> io::Result<Arc<CellEntries>> {
        // Fast path: LRU hit. `get` promotes the entry to most-recent.
        {
            let mut cache = self.cell_cache.lock().expect("cell cache poisoned");
            if let Some(entry) = cache.get(&cell_id) {
                return Ok(Arc::clone(entry));
            }
        }

        let depth = self.manifest.cell_depth;
        let q_name = quad_entry_name(depth, cell_id);
        let g_name = gaia_entry_name(depth, cell_id);

        let q_bytes = self.accessor.read_entry(&q_name)?;
        let q_arc: Arc<[u8]> = Arc::from(q_bytes.as_ref().to_vec().into_boxed_slice());
        let g_bytes = self.accessor.read_entry(&g_name)?;
        let g_arc: Arc<[u8]> = Arc::from(g_bytes.as_ref().to_vec().into_boxed_slice());

        let entries = Arc::new(CellEntries {
            quads: q_arc,
            gaia: g_arc,
        });

        let mut cache = self.cell_cache.lock().expect("cell cache poisoned");
        // Lost a race? Use whatever's already there so concurrent
        // callers converge on the same Arc when possible.
        if let Some(existing) = cache.get(&cell_id) {
            return Ok(Arc::clone(existing));
        }
        cache.put(cell_id, Arc::clone(&entries));
        Ok(entries)
    }

    /// Derive the bundle-depth cell id from a Gaia DR3 `source_id`.
    ///
    /// Returns:
    /// - `Ok(Some(cell_id))` for a well-formed native (bit-63 unset) id
    ///   when `cell_depth <= 12`.
    /// - `Ok(None)` if the derived cell is not populated (caller treats
    ///   as a miss — but this method itself doesn't peek at the
    ///   populated set; that's the caller's job; `Ok(None)` is reserved
    ///   for "the encoding doesn't apply" cases that shouldn't error).
    ///   In v1, this branch isn't taken — the populated check happens at
    ///   the call site.
    /// - `Err(InvalidInput)` for supplement source_ids (bit 63 set) or
    ///   if `cell_depth > 12`.
    fn derive_cell_from_source_id(&self, source_id: u64) -> io::Result<Option<u32>> {
        if (source_id >> 63) & 1 == 1 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "supplement source_id (bit 63 set) cannot be mapped to a cell without cell_hint",
            ));
        }
        let depth = self.manifest.cell_depth;
        if depth > 12 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "cell_depth {depth} > 12: HEALPix-12 source_id prefix can't disambiguate without cell_hint"
                ),
            ));
        }
        let cell_id_12: u64 = source_id >> 35;
        let cell_id: u64 = cell_id_12 >> (2 * (12 - depth as u64));
        Ok(Some(cell_id as u32))
    }
}

/// Convert arcseconds to radians.
#[inline]
fn arcsec_to_radians(arcsec: f64) -> f64 {
    // 1 arcsec = π / (180 * 3600) radians.
    arcsec.to_radians() / 3600.0
}

/// Construct an `IndexStar` from a 104-byte Gaia record.
///
/// The shard stores RA/Dec in degrees per the spec; the runtime
/// `Equatorial` uses radians. PM is present iff the record's
/// `FLAG_HAS_PM` bit is set; otherwise `None`. The Gaia `ref_epoch`
/// (e.g. 2016.0 for DR3) is materialised as a `Time` from the
/// caller-supplied `Timescale`; every star produced from one
/// `Timescale` shares its underlying `Arc<TimescaleInner>`.
#[inline]
fn gaia_record_to_index_star(g: &GaiaRecord, timescale: &Timescale) -> IndexStar {
    IndexStar {
        catalog_id: g.source_id,
        position: Equatorial::new(g.ra.to_radians(), g.dec.to_radians()),
        mag: g.phot_g_mean_mag,
        proper_motion: if g.has_pm() {
            Some(ProperMotion {
                pmra: g.pmra,
                pmdec: g.pmdec,
            })
        } else {
            None
        },
        ref_epoch: RefEpoch::new(timescale.j(g.ref_epoch)),
    }
}

/// Build the bundle-relative entry path for a cell's quad shard.
fn quad_entry_name(depth: u8, cell_id: u32) -> String {
    let width = cell_filename_width(depth);
    format!("{QUADS_SUBDIR}/cell_{cell_id:0width$}.{QUAD_EXT}")
}

/// Build the bundle-relative entry path for a cell's gaia shard.
fn gaia_entry_name(depth: u8, cell_id: u32) -> String {
    let width = cell_filename_width(depth);
    format!("{GAIA_SUBDIR}/cell_{cell_id:0width$}.{GAIA_EXT}")
}

/// Read + parse the manifest via an arbitrary [`SubfileAccessor`].
fn load_manifest_from_accessor(accessor: &dyn SubfileAccessor) -> io::Result<BundleManifest> {
    let bytes = accessor.read_entry(MANIFEST_NAME)?;
    let manifest: BundleManifest = serde_json::from_slice(&bytes).map_err(|e| {
        io::Error::new(
            io::ErrorKind::InvalidData,
            format!("failed to parse manifest.json: {e}"),
        )
    })?;
    if manifest.format != FORMAT_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unexpected bundle format string {:?} (expected {:?})",
                manifest.format, FORMAT_MAGIC
            ),
        ));
    }
    if manifest.format_version != FORMAT_VERSION {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "unsupported bundle format_version {} (this build supports {FORMAT_VERSION})",
                manifest.format_version
            ),
        ));
    }
    if manifest.gaia.record_size != GAIA_RECORD_SIZE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!(
                "gaia.record_size = {} but this build expects {GAIA_RECORD_SIZE}",
                manifest.gaia.record_size
            ),
        ));
    }
    for (pos, band) in manifest.bands.iter().enumerate() {
        if band.band_idx as usize != pos {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "bands[{pos}].band_idx = {} (expected dense 0..{} layout)",
                    band.band_idx,
                    manifest.bands.len()
                ),
            ));
        }
    }
    Ok(manifest)
}

/// Walk `quads/` via the accessor and collect cell ids whose filenames
/// match the canonical `cell_NNNN…N.zqd` pattern at this depth.
fn enumerate_populated_cells(
    accessor: &dyn SubfileAccessor,
    depth: u8,
) -> io::Result<BTreeSet<u32>> {
    let width = cell_filename_width(depth);
    let prefix = format!("{QUADS_SUBDIR}/");
    let entries = accessor.list_prefix(&prefix)?;
    let mut out = BTreeSet::new();
    for name in entries {
        // Trim "<QUADS_SUBDIR>/" and ".<QUAD_EXT>".
        let stripped = match name
            .strip_prefix(&prefix)
            .and_then(|s| s.strip_prefix("cell_"))
            .and_then(|s| s.strip_suffix(&format!(".{QUAD_EXT}")))
        {
            Some(s) => s,
            None => continue,
        };
        if stripped.len() != width {
            continue;
        }
        if let Ok(cell_id) = stripped.parse::<u32>() {
            out.insert(cell_id);
        }
    }
    Ok(out)
}

/// Best-effort categorization of a shard parse error into one of the
/// `VerifyErrorKind` buckets. The shard parsers wrap their errors in
/// `io::Error` with `ErrorKind::InvalidData` and a descriptive message,
/// so we string-match on the message.
fn classify_shard_parse_error(err: &io::Error) -> VerifyErrorKind {
    let msg = err.to_string();
    if msg.contains("magic") || msg.contains("Magic") {
        VerifyErrorKind::BadMagic
    } else if msg.contains("version") || msg.contains("Version") {
        VerifyErrorKind::BadVersion
    } else {
        VerifyErrorKind::BadSize
    }
}

// ---------------------------------------------------------------------------
//  Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    use std::path::PathBuf;

    use chrono::{DateTime, Utc};
    use starfield::Equatorial;
    use tempfile::TempDir;

    use crate::bundle::manifest::{BuildMetadata, BuildSource};
    use crate::bundle::tidy::{
        BandMetadata, GaiaMetadata, TidyMetadata, tidy_to_folder, tidy_to_zip,
    };
    use crate::index::cell_builder::{CellStar, CellStarSource};
    use crate::index::multiband_cell_builder::{
        BundleWorkDirPaths, MultiBandCellBuildConfig, ScaleBand, build_bundle_work_dir,
    };

    const TEST_DEPTH: u8 = 5;

    /// Synthetic source: a small set of cells, each containing a tight
    /// cluster of stars whose RA/Dec puts them inside the matching
    /// HEALPix cell at TEST_DEPTH=5. Cell-center coordinates are
    /// computed via cdshealpix so the synthetic clusters genuinely live
    /// inside their assigned cells.
    struct ClusteredSource {
        cells: Vec<u32>,
        stars_per_cell: usize,
    }

    impl ClusteredSource {
        fn new(cells: Vec<u32>, stars_per_cell: usize) -> Self {
            Self {
                cells,
                stars_per_cell,
            }
        }
    }

    /// Compute a cell's center (ra, dec) in radians via cdshealpix.
    fn cell_center(depth: u8, cell_id: u32) -> (f64, f64) {
        cdshealpix::nested::center(depth, cell_id as u64)
    }

    impl CellStarSource for ClusteredSource {
        fn cell_count(&self) -> u32 {
            // Only iterate up to one past the largest populated cell.
            // The orchestrator iterates `0..cell_count()` and fsyncs
            // the manifest for every empty cell — at depth 5 that's
            // 12 * 4^5 = 12,288 fsyncs per test, which dominates wall
            // time. The full-sky count is only needed at production
            // build time, not in tests.
            self.cells.iter().copied().max().unwrap_or(0) + 1
        }

        fn stars_in_cell(&self, cell_id: u32) -> io::Result<Vec<CellStar>> {
            if !self.cells.contains(&cell_id) {
                return Ok(Vec::new());
            }
            let (ra_c, dec_c) = cell_center(TEST_DEPTH, cell_id);
            let mut out = Vec::with_capacity(self.stars_per_cell);
            for i in 0..self.stars_per_cell {
                // Tight cluster: ~1 arcsec scatter.
                let dra = (i as f64 - 4.0) * 1e-5;
                let ddec = (i as f64 - 4.0) * 7e-6;
                let ra = ra_c + dra;
                let dec = dec_c + ddec;
                // Build a synthetic Gaia source_id whose HEALPix-12
                // prefix encodes cell_id at TEST_DEPTH. Going from
                // depth-5 cell to depth-12 cell requires multiplying by
                // 4^7 = 16_384 (one of the depth-12 cells inside this
                // depth-5 cell). Use the lowest depth-12 child.
                let cell_id_12: u64 = (cell_id as u64) << (2 * (12 - TEST_DEPTH as u64));
                let source_id: u64 = (cell_id_12 << 35) | (i as u64 + 1);
                out.push(make_cell_star(source_id, ra, dec, 12.0 + i as f64 * 0.01));
            }
            Ok(out)
        }
    }

    fn make_gaia(source_id: u64, ra_deg: f64, dec_deg: f64, mag: f64) -> GaiaRecord {
        GaiaRecord {
            source_id,
            ref_epoch: 2016.0,
            ra: ra_deg,
            dec: dec_deg,
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
            gaia: make_gaia(catalog_id, ra_rad.to_degrees(), dec_rad.to_degrees(), mag),
        }
    }

    fn two_bands() -> Vec<ScaleBand> {
        vec![
            ScaleBand {
                label: "band_00".into(),
                band_idx: 0,
                scale_lower_arcsec: 1.0,
                scale_upper_arcsec: 50.0,
                quads_per_cell: 50,
                max_reuse: 8,
            },
            ScaleBand {
                label: "band_01".into(),
                band_idx: 1,
                scale_lower_arcsec: 50.0,
                scale_upper_arcsec: 500.0,
                quads_per_cell: 50,
                max_reuse: 8,
            },
        ]
    }

    fn tidy_metadata() -> TidyMetadata {
        TidyMetadata {
            cell_depth: TEST_DEPTH,
            experiment: "reader test fixture".into(),
            build_metadata: BuildMetadata {
                tool: "zodiacal-tools test".into(),
                build_started_utc: "2026-05-05T00:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                build_finished_utc: "2026-05-05T01:00:00Z".parse::<DateTime<Utc>>().unwrap(),
                source: BuildSource {
                    kind: "test-fixture".into(),
                    release: "test".into(),
                    path: "/tmp/fixture".into(),
                },
            },
            gaia: GaiaMetadata {
                max_stars_per_cell: 10_000,
                mag_limit: 20.0,
                schema_version: 1,
            },
            bands: vec![
                BandMetadata {
                    label: "band_00".into(),
                    scale_lower_arcsec: 1.0,
                    scale_upper_arcsec: 50.0,
                    quads_per_cell: 50,
                    max_reuse: 8,
                },
                BandMetadata {
                    label: "band_01".into(),
                    scale_lower_arcsec: 50.0,
                    scale_upper_arcsec: 500.0,
                    quads_per_cell: 50,
                    max_reuse: 8,
                },
            ],
        }
    }

    /// Build a small bundle in `out_path`, returning the temp work dir
    /// (kept alive) and the cells that were populated.
    fn build_test_bundle_folder() -> (TempDir, TempDir, PathBuf, Vec<u32>) {
        let work_tmp = TempDir::new().expect("work tmpdir");
        let out_tmp = TempDir::new().expect("out tmpdir");
        let cells = vec![0u32, 1, 7, 42];
        let source = ClusteredSource::new(cells.clone(), 8);
        let cfg = MultiBandCellBuildConfig {
            bands: two_bands(),
            max_stars_per_cell: 1_000,
            cell_depth: TEST_DEPTH,
            manifest_save_interval_secs: 0,
        };
        let paths = BundleWorkDirPaths {
            work_dir: work_tmp.path().to_path_buf(),
        };
        build_bundle_work_dir(&source, &cfg, &paths).expect("build work dir");

        let bundle_path = out_tmp.path().join("index.zdcl.bundle");
        tidy_to_folder(work_tmp.path(), &bundle_path, &tidy_metadata()).expect("tidy_to_folder");
        (work_tmp, out_tmp, bundle_path, cells)
    }

    fn build_test_bundle_zip() -> (TempDir, TempDir, PathBuf, Vec<u32>) {
        let work_tmp = TempDir::new().expect("work tmpdir");
        let out_tmp = TempDir::new().expect("out tmpdir");
        let cells = vec![0u32, 1, 7, 42];
        let source = ClusteredSource::new(cells.clone(), 8);
        let cfg = MultiBandCellBuildConfig {
            bands: two_bands(),
            max_stars_per_cell: 1_000,
            cell_depth: TEST_DEPTH,
            manifest_save_interval_secs: 0,
        };
        let paths = BundleWorkDirPaths {
            work_dir: work_tmp.path().to_path_buf(),
        };
        build_bundle_work_dir(&source, &cfg, &paths).expect("build work dir");

        let zip_path = out_tmp.path().join("index.zdcl.bundle.zip");
        tidy_to_zip(work_tmp.path(), &zip_path, &tidy_metadata()).expect("tidy_to_zip");
        (work_tmp, out_tmp, zip_path, cells)
    }

    #[test]
    fn open_folder_bundle() {
        let (_w, _o, bundle, cells) = build_test_bundle_folder();
        let b = ZdclBundle::open(&bundle).expect("open folder bundle");
        assert_eq!(b.cell_depth(), TEST_DEPTH);
        assert_eq!(b.bands().len(), 2);
        assert_eq!(b.manifest().gaia.n_total, cells.len() as u64 * 8);
        // Populated cells exactly match what we built.
        let pop: Vec<u32> = b.populated_cells().iter().copied().collect();
        assert_eq!(pop, cells);
    }

    #[test]
    fn open_zip_bundle() {
        let (_w, _o, bundle, cells) = build_test_bundle_zip();
        let b = ZdclBundle::open(&bundle).expect("open zip bundle");
        assert_eq!(b.cell_depth(), TEST_DEPTH);
        assert_eq!(b.bands().len(), 2);
        assert_eq!(b.manifest().gaia.n_total, cells.len() as u64 * 8);
        let pop: Vec<u32> = b.populated_cells().iter().copied().collect();
        assert_eq!(pop, cells);
    }

    #[test]
    fn open_rejects_non_bundle_path() {
        let tmp = TempDir::new().unwrap();
        let nonsense = tmp.path().join("not-a-bundle.bin");
        std::fs::write(&nonsense, b"hello").unwrap();
        let err = ZdclBundle::open(&nonsense).expect_err("should reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn load_region_returns_only_overlapping_cells() {
        let (_w, _o, bundle, cells) = build_test_bundle_folder();
        let b = ZdclBundle::open(&bundle).unwrap();
        // Tight region around cell 0's center → should return only that
        // cell's stars (or a small handful of neighbours via the cone
        // approximation, but always a strict subset of all built cells).
        let (ra_c, dec_c) = cell_center(TEST_DEPTH, 0);
        let region = SkyRegion::from_radians(Equatorial::new(ra_c, dec_c), 0.001);
        let frag = b.load_region(&region).unwrap();
        assert!(
            frag.gaia_records.len() < cells.len() * 8,
            "tight region should be a subset; got {} of {} possible records",
            frag.gaia_records.len(),
            cells.len() * 8
        );
        // And it must include cell 0's records (8 of them).
        assert!(frag.gaia_records.len() >= 8);
    }

    #[test]
    fn load_region_multi_band_fragment_per_band() {
        let (_w, _o, bundle, _cells) = build_test_bundle_folder();
        let b = ZdclBundle::open(&bundle).unwrap();
        // Whole-sky region.
        let region = SkyRegion::from_radians(Equatorial::new(0.0, 0.0), std::f64::consts::PI);
        let frag = b.load_region(&region).unwrap();
        assert_eq!(frag.fragments.len(), 2);
        // Both fragments share the same star list (cloned).
        assert_eq!(frag.fragments[0].stars.len(), frag.fragments[1].stars.len());
        assert_eq!(frag.fragments[0].stars.len(), frag.gaia_records.len());
        // At least one band emitted some quads on this synthetic.
        let total_quads: usize = frag.fragments.iter().map(|f| f.quads.len()).sum();
        assert!(total_quads > 0);
        // Each fragment's quads.len() == codes.len().
        for f in &frag.fragments {
            assert_eq!(f.quads.len(), f.codes.len());
        }
    }

    #[test]
    fn load_region_band_single_band() {
        let (_w, _o, bundle, _cells) = build_test_bundle_folder();
        let b = ZdclBundle::open(&bundle).unwrap();
        let region = SkyRegion::from_radians(Equatorial::new(0.0, 0.0), std::f64::consts::PI);
        let multi = b.load_region(&region).unwrap();
        let band0 = b.load_region_band(0, &region).unwrap();
        assert_eq!(band0.quads.len(), multi.fragments[0].quads.len());
        assert_eq!(band0.codes.len(), multi.fragments[0].codes.len());
        assert_eq!(band0.stars.len(), multi.fragments[0].stars.len());

        // Out-of-range band_idx is rejected.
        let err = b.load_region_band(99, &region).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn gaia_get_with_hint() {
        let (_w, _o, bundle, _cells) = build_test_bundle_folder();
        let b = ZdclBundle::open(&bundle).unwrap();
        // The synthetic populates 8 stars in cell 7. Their source_ids
        // are derived from cell 7's HEALPix-12 prefix. The exact ids we
        // wrote come back via the gaia shard.
        let cell_id_12: u64 = 7u64 << (2 * (12 - TEST_DEPTH as u64));
        let want_id: u64 = (cell_id_12 << 35) | 1; // i = 0 -> +1
        let got = b
            .gaia_get(want_id, Some(7))
            .expect("io ok")
            .expect("record present");
        assert_eq!(got.source_id, want_id);
    }

    #[test]
    fn gaia_get_without_hint_for_normal_source_id() {
        let (_w, _o, bundle, _cells) = build_test_bundle_folder();
        let b = ZdclBundle::open(&bundle).unwrap();
        // Same id construction as in the with_hint test, but pass
        // cell_hint = None and let the bundle derive cell 42 from the
        // prefix.
        let cell_id_12: u64 = 42u64 << (2 * (12 - TEST_DEPTH as u64));
        let want_id: u64 = (cell_id_12 << 35) | 3;
        let got = b
            .gaia_get(want_id, None)
            .expect("io ok")
            .expect("record present");
        assert_eq!(got.source_id, want_id);
    }

    #[test]
    fn gaia_get_supplement_requires_hint() {
        let (_w, _o, bundle, _cells) = build_test_bundle_folder();
        let b = ZdclBundle::open(&bundle).unwrap();
        // A synthetic supplement-style id (bit 63 set). Without a hint
        // the bundle must error rather than silently miss.
        let supp_id: u64 = (1u64 << 63) | 12345;
        let err = b.gaia_get(supp_id, None).expect_err("must require hint");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        // With a (correct-but-irrelevant) hint, the bundle returns
        // None because no record with that id was written. The
        // important property is "it doesn't error".
        let got = b.gaia_get(supp_id, Some(0)).expect("io ok");
        assert!(got.is_none());
    }

    #[test]
    fn gaia_get_many_returns_correct_length() {
        let (_w, _o, bundle, _cells) = build_test_bundle_folder();
        let b = ZdclBundle::open(&bundle).unwrap();
        let cell_id_12: u64 = 0u64 << (2 * (12 - TEST_DEPTH as u64));
        let id1 = (cell_id_12 << 35) | 1;
        let id2 = 9_999_999_999u64; // unlikely to exist
        let cell_id_12_b: u64 = 7u64 << (2 * (12 - TEST_DEPTH as u64));
        let id3 = (cell_id_12_b << 35) | 2;
        let got = b.gaia_get_many(&[id1, id2, id3]).expect("io ok");
        assert_eq!(got.len(), 3);
        assert!(got[0].is_some());
        assert!(got[2].is_some());
    }

    #[test]
    fn folder_zip_parity() {
        let (_w1, _o1, folder, _cells) = build_test_bundle_folder();
        let (_w2, _o2, zip, _cells2) = build_test_bundle_zip();

        let bf = ZdclBundle::open(&folder).unwrap();
        let bz = ZdclBundle::open(&zip).unwrap();

        let region = SkyRegion::from_radians(Equatorial::new(0.0, 0.0), std::f64::consts::PI);
        let mf = bf.load_region(&region).unwrap();
        let mz = bz.load_region(&region).unwrap();

        // Bit-for-bit gaia records.
        assert_eq!(mf.gaia_records.len(), mz.gaia_records.len());
        for (a, b) in mf.gaia_records.iter().zip(mz.gaia_records.iter()) {
            let ab: &[u8] = bytemuck::bytes_of(a);
            let bb: &[u8] = bytemuck::bytes_of(b);
            assert_eq!(ab, bb);
        }
        // Per-band quads + codes.
        assert_eq!(mf.fragments.len(), mz.fragments.len());
        for (fa, fb) in mf.fragments.iter().zip(mz.fragments.iter()) {
            assert_eq!(fa.quads.len(), fb.quads.len());
            for (qa, qb) in fa.quads.iter().zip(fb.quads.iter()) {
                assert_eq!(qa.star_ids, qb.star_ids);
            }
            assert_eq!(fa.codes, fb.codes);
            assert_eq!(fa.scale_lower, fb.scale_lower);
            assert_eq!(fa.scale_upper, fb.scale_upper);
        }
    }

    #[test]
    fn verify_passes_clean_bundle() {
        let (_w, _o, bundle, cells) = build_test_bundle_folder();
        let b = ZdclBundle::open(&bundle).unwrap();
        let report = b.verify().expect("verify io ok");
        assert!(
            report.errors.is_empty(),
            "expected clean bundle, got errors: {:?}",
            report.errors
        );
        assert_eq!(report.n_cells_checked as usize, cells.len());
    }

    #[test]
    fn verify_catches_corrupted_zqd() {
        let (_w, _o, bundle, _cells) = build_test_bundle_folder();
        // Mutate the magic byte of one .zqd file. Pick cell 7's quad
        // shard.
        let width = cell_filename_width(TEST_DEPTH);
        let cid = 7u32;
        let zqd = bundle
            .join(QUADS_SUBDIR)
            .join(format!("cell_{cid:0width$}.{QUAD_EXT}"));
        let mut bytes = std::fs::read(&zqd).expect("read zqd");
        bytes[0] = b'X'; // breaks "ZDCLQUAD"
        std::fs::write(&zqd, &bytes).expect("write zqd");

        let b = ZdclBundle::open(&bundle).unwrap();
        let report = b.verify().expect("verify io ok");
        let bad: Vec<_> = report
            .errors
            .iter()
            .filter(|e| e.cell_id == cid && e.kind == VerifyErrorKind::BadMagic)
            .collect();
        assert!(
            !bad.is_empty(),
            "expected BadMagic for cell {cid}, got: {:?}",
            report.errors
        );
    }

    #[test]
    fn bands_returns_manifest_bands() {
        let (_w, _o, bundle, _cells) = build_test_bundle_folder();
        let b = ZdclBundle::open(&bundle).unwrap();
        let bands = b.bands();
        assert_eq!(bands.len(), 2);
        assert_eq!(bands[0].band_idx, 0);
        assert_eq!(bands[1].band_idx, 1);
        assert_eq!(bands[0].label, "band_00");
        assert_eq!(bands[1].label, "band_01");
    }
}
