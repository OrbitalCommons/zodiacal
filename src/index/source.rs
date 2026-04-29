//! `IndexSource` — sparse cell-based access to a `.zdcl` file.
//!
//! Plan 2 of the deployment-mode roadmap (`plans/02-healpix-format.md`):
//! supports loading just the HEALPix cells that intersect a given
//! [`SkyRegion`], so a 1° hint loads ~MB of stars off disk instead of GB.
//!
//! Backwards-compatible with the v2 format: v2 files appear as a single
//! virtual cell containing all stars, and `load_cells` returns the full set.

use std::fs::File;
use std::io;
use std::path::Path;

use cdshealpix::nested;
use memmap2::Mmap;

use crate::geom::sphere::radec_to_xyz;
use crate::quads::{Code, DIMCODES, DIMQUADS, Quad};
use crate::solver::SkyRegion;

use super::{IndexMetadata, IndexStar};

/// Sentinel cell-depth value used to signal "v2 file synthesized as a
/// single virtual cell". Real v3 cell depths are 0..=29.
pub(super) const SYNTHESIZED_CELL_DEPTH: u8 = 0xFF;

/// Default HEALPix depth used when grouping stars in a v3 file. Depth 5
/// gives 12 × 4^5 = 12,288 cells, ~5° per cell — small enough that
/// typical FOVs land in 1–10 cells, large enough that the cell table
/// stays well under 1 MB.
pub const DEFAULT_CELL_DEPTH: u8 = 5;

/// One HEALPix cell, identified by depth + nested-scheme cell id.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HealpixCell {
    pub depth: u8,
    pub id: u64,
}

/// A subset of an index loaded from a particular set of cells. Returned by
/// `IndexSource::load_cells`. Indices in `quads` are *into the
/// `stars` vector of this fragment*, not into the original file's full
/// star list.
#[derive(Debug, Clone)]
pub struct IndexFragment {
    pub stars: Vec<IndexStar>,
    pub quads: Vec<Quad>,
    pub codes: Vec<Code>,
    pub scale_lower: f64,
    pub scale_upper: f64,
    pub metadata: Option<IndexMetadata>,
}

/// Abstraction over "where does index data come from". Implemented by
/// [`ZdclFile`] for on-disk v3 files; future implementations could read
/// from an in-memory buffer or a remote service.
pub trait IndexSource: Send + Sync {
    /// All HEALPix cells that overlap `region`.
    fn cells_intersecting(&self, region: &SkyRegion) -> Vec<HealpixCell>;

    /// Load the contents of the given cells. Order of `cells` is
    /// preserved in the returned star order, but not required to be
    /// sorted by the caller.
    fn load_cells(&self, cells: &[HealpixCell]) -> io::Result<IndexFragment>;

    /// HEALPix depth at which this source's cell table is built. Returns
    /// `SYNTHESIZED_CELL_DEPTH` for v2 files exposed as a single virtual cell.
    fn cell_depth(&self) -> u8;

    /// Build metadata, if present.
    fn metadata(&self) -> Option<&IndexMetadata>;

    /// Total stars across all cells.
    fn star_count(&self) -> usize;

    /// Total quads.
    fn quad_count(&self) -> usize;

    /// Quad-scale band carried by this source: (lower, upper) in radians.
    /// Useful for the solver to skip whole sources whose scale range can't
    /// match the field's backbone.
    fn scale_range(&self) -> (f64, f64);
}

/// Per-cell offset entry in the v3 file header.
#[derive(Debug, Clone, Copy)]
struct CellEntry {
    cell_id: u64,
    /// Index of the first star of this cell in the global stars block
    /// (in star units, not bytes).
    star_offset: u64,
    star_count: u32,
}

/// On-disk index file (`.zdcl`) memory-mapped for sparse cell access.
pub struct ZdclFile {
    mmap: Mmap,
    cell_depth: u8,
    /// Sorted by `cell_id` ascending. Empty for v2 files (we synthesize a
    /// single virtual cell on the fly in `cells_intersecting`).
    cell_table: Vec<CellEntry>,
    metadata: Option<IndexMetadata>,
    /// Byte offset in mmap where the stars block begins.
    stars_offset: usize,
    n_stars: usize,
    n_quads: usize,
    scale_lower: f64,
    scale_upper: f64,
    quads_offset: usize,
    codes_offset: usize,
}

const STAR_RECORD_SIZE: usize = 32; // u64 + 3 × f64
const QUAD_RECORD_SIZE: usize = DIMQUADS * 4; // 4 × u32
const CODE_RECORD_SIZE: usize = DIMCODES * 8; // 4 × f64
const CELL_TABLE_ENTRY_SIZE: usize = 8 + 8 + 4; // u64 + u64 + u32 (no padding here)

const MAGIC: &[u8; 4] = b"ZDCL";

impl ZdclFile {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "file shorter than header magic+version",
            ));
        }
        if &mmap[0..4] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "invalid magic bytes",
            ));
        }
        let version = u32::from_le_bytes(mmap[4..8].try_into().unwrap());

        match version {
            1 | 2 => Self::open_v1_v2(mmap, version),
            3 => Self::open_v3(mmap),
            _ => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version: {version}"),
            )),
        }
    }

    fn open_v1_v2(mmap: Mmap, version: u32) -> io::Result<Self> {
        let mut cursor = 8usize;
        let metadata = if version >= 2 {
            let meta_len = read_u64_at(&mmap, cursor)? as usize;
            cursor += 8;
            if meta_len > mmap.len().saturating_sub(cursor) {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!("metadata_len {meta_len} exceeds remaining file size"),
                ));
            }
            if meta_len > 0 {
                let bytes = read_bytes_at(&mmap, cursor, meta_len)?;
                cursor += meta_len;
                let m: IndexMetadata = serde_json::from_slice(bytes)
                    .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
                Some(m)
            } else {
                None
            }
        } else {
            None
        };

        let n_stars = read_u64_at(&mmap, cursor)? as usize;
        cursor += 8;
        let n_quads = read_u64_at(&mmap, cursor)? as usize;
        cursor += 8;
        let scale_lower = read_f64_at(&mmap, cursor)?;
        cursor += 8;
        let scale_upper = read_f64_at(&mmap, cursor)?;
        cursor += 8;

        let stars_offset = cursor;
        let stars_size = n_stars
            .checked_mul(STAR_RECORD_SIZE)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "stars block overflow"))?;
        let quads_size = n_quads
            .checked_mul(QUAD_RECORD_SIZE)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "quads block overflow"))?;
        let codes_size = n_quads
            .checked_mul(CODE_RECORD_SIZE)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "codes block overflow"))?;
        let quads_offset = stars_offset + stars_size;
        let codes_offset = quads_offset + quads_size;
        let total_expected = codes_offset + codes_size;
        if mmap.len() < total_expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "file truncated: expected at least {total_expected} bytes, got {}",
                    mmap.len()
                ),
            ));
        }

        Ok(Self {
            mmap,
            cell_depth: SYNTHESIZED_CELL_DEPTH,
            cell_table: Vec::new(),
            metadata,
            stars_offset,
            n_stars,
            n_quads,
            scale_lower,
            scale_upper,
            quads_offset,
            codes_offset,
        })
    }

    fn open_v3(mmap: Mmap) -> io::Result<Self> {
        let mut cursor = 8usize;
        let meta_len = read_u64_at(&mmap, cursor)? as usize;
        cursor += 8;
        // Bound metadata length to remaining file size before slicing.
        if meta_len > mmap.len().saturating_sub(cursor) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("metadata_len {meta_len} exceeds remaining file size"),
            ));
        }
        let metadata = if meta_len > 0 {
            let bytes = read_bytes_at(&mmap, cursor, meta_len)?;
            cursor += meta_len;
            let m: IndexMetadata = serde_json::from_slice(bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Some(m)
        } else {
            None
        };

        let cell_depth = read_u8_at(&mmap, cursor)?;
        cursor += 1;
        // Validate cell_depth is in the valid HEALPix range. cdshealpix
        // panics on depths > 29; reject corrupt files cleanly here. The
        // SYNTHESIZED_CELL_DEPTH sentinel is for v2 backwards-compat only
        // and must never appear on disk in a v3 file.
        if cell_depth > 29 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("cell_depth {cell_depth} outside valid HEALPix range 0..=29"),
            ));
        }
        // 7 reserved bytes
        cursor += 7;
        let n_cells = read_u64_at(&mmap, cursor)? as usize;
        cursor += 8;

        // Bound n_cells against the maximum possible HEALPix cells at this
        // depth — and against the remaining file size — before allocating.
        let max_cells_at_depth = 12usize
            .checked_shl((cell_depth as u32) * 2)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "cell_depth overflow"))?;
        if n_cells > max_cells_at_depth {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("n_cells {n_cells} exceeds 12*4^cell_depth = {max_cells_at_depth}"),
            ));
        }
        let cell_table_bytes = n_cells.checked_mul(CELL_TABLE_ENTRY_SIZE).ok_or_else(|| {
            io::Error::new(io::ErrorKind::InvalidData, "cell_table size overflow")
        })?;
        if cell_table_bytes > mmap.len().saturating_sub(cursor) {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("cell_table_bytes {cell_table_bytes} exceeds remaining file"),
            ));
        }

        let mut cell_table: Vec<CellEntry> = Vec::with_capacity(n_cells);
        for _ in 0..n_cells {
            let cell_id = read_u64_at(&mmap, cursor)?;
            cursor += 8;
            let star_offset = read_u64_at(&mmap, cursor)?;
            cursor += 8;
            let star_count = read_u32_at(&mmap, cursor)?;
            cursor += 4;
            cell_table.push(CellEntry {
                cell_id,
                star_offset,
                star_count,
            });
        }

        // Align cursor to 8 bytes before reading n_stars, n_quads, scales.
        cursor = (cursor + 7) & !7;

        let n_stars = read_u64_at(&mmap, cursor)? as usize;
        cursor += 8;
        let n_quads = read_u64_at(&mmap, cursor)? as usize;
        cursor += 8;
        let scale_lower = read_f64_at(&mmap, cursor)?;
        cursor += 8;
        let scale_upper = read_f64_at(&mmap, cursor)?;
        cursor += 8;

        let stars_offset = cursor;
        let stars_size = n_stars
            .checked_mul(STAR_RECORD_SIZE)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "stars block overflow"))?;
        let quads_size = n_quads
            .checked_mul(QUAD_RECORD_SIZE)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "quads block overflow"))?;
        let codes_size = n_quads
            .checked_mul(CODE_RECORD_SIZE)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "codes block overflow"))?;
        let quads_offset = stars_offset + stars_size;
        let codes_offset = quads_offset + quads_size;
        let total_expected = codes_offset + codes_size;
        if mmap.len() < total_expected {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "file truncated: expected at least {total_expected} bytes, got {}",
                    mmap.len()
                ),
            ));
        }

        // Validate every cell entry's range is a subrange of [0, n_stars).
        for (i, entry) in cell_table.iter().enumerate() {
            let end = (entry.star_offset as usize)
                .checked_add(entry.star_count as usize)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("cell entry {i} range overflow"),
                    )
                })?;
            if end > n_stars {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "cell entry {i}: range [{}, {end}) exceeds n_stars {n_stars}",
                        entry.star_offset
                    ),
                ));
            }
        }

        Ok(Self {
            mmap,
            cell_depth,
            cell_table,
            metadata,
            stars_offset,
            n_stars,
            n_quads,
            scale_lower,
            scale_upper,
            quads_offset,
            codes_offset,
        })
    }

    /// Load every star and quad — equivalent to the legacy `Index::load`.
    pub fn load_full(&self) -> io::Result<IndexFragment> {
        let mut stars = Vec::with_capacity(self.n_stars);
        for i in 0..self.n_stars {
            stars.push(self.read_star(i)?);
        }
        let mut quads = Vec::with_capacity(self.n_quads);
        for i in 0..self.n_quads {
            quads.push(self.read_quad(i)?);
        }
        let mut codes = Vec::with_capacity(self.n_quads);
        for i in 0..self.n_quads {
            codes.push(self.read_code(i)?);
        }
        Ok(IndexFragment {
            stars,
            quads,
            codes,
            scale_lower: self.scale_lower,
            scale_upper: self.scale_upper,
            metadata: self.metadata.clone(),
        })
    }

    fn read_star(&self, idx: usize) -> io::Result<IndexStar> {
        let off = self.stars_offset + idx * STAR_RECORD_SIZE;
        if off + STAR_RECORD_SIZE > self.mmap.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "star OOB"));
        }
        let catalog_id = u64::from_le_bytes(self.mmap[off..off + 8].try_into().unwrap());
        let ra = f64::from_le_bytes(self.mmap[off + 8..off + 16].try_into().unwrap());
        let dec = f64::from_le_bytes(self.mmap[off + 16..off + 24].try_into().unwrap());
        let mag = f64::from_le_bytes(self.mmap[off + 24..off + 32].try_into().unwrap());
        Ok(IndexStar {
            catalog_id,
            ra,
            dec,
            mag,
        })
    }

    fn read_quad(&self, idx: usize) -> io::Result<Quad> {
        let off = self.quads_offset + idx * QUAD_RECORD_SIZE;
        if off + QUAD_RECORD_SIZE > self.mmap.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "quad OOB"));
        }
        let mut star_ids = [0usize; DIMQUADS];
        for (i, sid) in star_ids.iter_mut().enumerate() {
            let p = off + i * 4;
            *sid = u32::from_le_bytes(self.mmap[p..p + 4].try_into().unwrap()) as usize;
        }
        Ok(Quad { star_ids })
    }

    fn read_code(&self, idx: usize) -> io::Result<Code> {
        let off = self.codes_offset + idx * CODE_RECORD_SIZE;
        if off + CODE_RECORD_SIZE > self.mmap.len() {
            return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "code OOB"));
        }
        let mut code = [0.0f64; DIMCODES];
        for (i, v) in code.iter_mut().enumerate() {
            let p = off + i * 8;
            *v = f64::from_le_bytes(self.mmap[p..p + 8].try_into().unwrap());
        }
        Ok(code)
    }
}

impl IndexSource for ZdclFile {
    fn cells_intersecting(&self, region: &SkyRegion) -> Vec<HealpixCell> {
        if self.cell_depth == SYNTHESIZED_CELL_DEPTH {
            // v2 file: single virtual cell containing all stars.
            return vec![HealpixCell {
                depth: SYNTHESIZED_CELL_DEPTH,
                id: 0,
            }];
        }
        // For depth 0..=29, use cdshealpix's cone coverage. The "approx"
        // variant slightly over-includes (returns a superset of true
        // intersecting cells), which is exactly what we want — false
        // positives just cost a few extra MB of read; false negatives
        // would silently drop stars.
        let bmoc = nested::cone_coverage_approx(
            self.cell_depth,
            region.center.ra,
            region.center.dec,
            region.radius_rad,
        );
        let depth = self.cell_depth;
        bmoc.flat_iter()
            .map(|id| HealpixCell { depth, id })
            .collect()
    }

    fn load_cells(&self, cells: &[HealpixCell]) -> io::Result<IndexFragment> {
        if self.cell_depth == SYNTHESIZED_CELL_DEPTH {
            // v2 backwards-compat: any non-empty request returns the full
            // file (we don't track sub-cells in a v2 layout).
            return self.load_full();
        }

        // Resolve each requested cell to its CellEntry by binary search.
        // Cells not present in the table simply contribute zero stars.
        let mut star_ranges: Vec<(usize, usize)> = Vec::with_capacity(cells.len());
        for c in cells {
            if c.depth != self.cell_depth {
                continue;
            }
            if let Ok(idx) = self.cell_table.binary_search_by_key(&c.id, |e| e.cell_id) {
                let entry = &self.cell_table[idx];
                let start = entry.star_offset as usize;
                let count = entry.star_count as usize;
                star_ranges.push((start, count));
            }
        }

        // De-dup + sort ranges so we emit stars in deterministic order.
        star_ranges.sort_unstable();
        star_ranges.dedup();
        let total_stars: usize = star_ranges.iter().map(|(_, c)| *c).sum();

        // Build a kept-set of original star indices and a remap.
        let mut star_remap: Vec<Option<usize>> = vec![None; self.n_stars];
        let mut stars: Vec<IndexStar> = Vec::with_capacity(total_stars);
        for &(start, count) in &star_ranges {
            for (i, slot) in star_remap.iter_mut().enumerate().skip(start).take(count) {
                *slot = Some(stars.len());
                stars.push(self.read_star(i)?);
            }
        }

        // Quads: keep those whose stars are all in the kept set, remap
        // indices into the compact local stars[] vector.
        let mut quads: Vec<Quad> = Vec::with_capacity(self.n_quads / 4);
        let mut codes: Vec<Code> = Vec::with_capacity(self.n_quads / 4);
        for q_idx in 0..self.n_quads {
            let q = self.read_quad(q_idx)?;
            let mut new_ids = [0usize; DIMQUADS];
            let mut all_kept = true;
            for (i, &sid) in q.star_ids.iter().enumerate() {
                match star_remap.get(sid).copied().flatten() {
                    Some(new_idx) => new_ids[i] = new_idx,
                    None => {
                        all_kept = false;
                        break;
                    }
                }
            }
            if all_kept {
                quads.push(Quad { star_ids: new_ids });
                codes.push(self.read_code(q_idx)?);
            }
        }

        Ok(IndexFragment {
            stars,
            quads,
            codes,
            scale_lower: self.scale_lower,
            scale_upper: self.scale_upper,
            metadata: self.metadata.clone(),
        })
    }

    fn cell_depth(&self) -> u8 {
        self.cell_depth
    }

    fn metadata(&self) -> Option<&IndexMetadata> {
        self.metadata.as_ref()
    }

    fn star_count(&self) -> usize {
        self.n_stars
    }

    fn quad_count(&self) -> usize {
        self.n_quads
    }

    fn scale_range(&self) -> (f64, f64) {
        (self.scale_lower, self.scale_upper)
    }
}

// --- byte readers ---------------------------------------------------------

fn read_u8_at(buf: &[u8], off: usize) -> io::Result<u8> {
    buf.get(off)
        .copied()
        .ok_or_else(|| io::Error::new(io::ErrorKind::UnexpectedEof, "u8 OOB"))
}

fn read_u32_at(buf: &[u8], off: usize) -> io::Result<u32> {
    if off + 4 > buf.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "u32 OOB"));
    }
    Ok(u32::from_le_bytes(buf[off..off + 4].try_into().unwrap()))
}

fn read_u64_at(buf: &[u8], off: usize) -> io::Result<u64> {
    if off + 8 > buf.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "u64 OOB"));
    }
    Ok(u64::from_le_bytes(buf[off..off + 8].try_into().unwrap()))
}

fn read_f64_at(buf: &[u8], off: usize) -> io::Result<f64> {
    if off + 8 > buf.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "f64 OOB"));
    }
    Ok(f64::from_le_bytes(buf[off..off + 8].try_into().unwrap()))
}

fn read_bytes_at(buf: &[u8], off: usize, len: usize) -> io::Result<&[u8]> {
    if off + len > buf.len() {
        return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "slice OOB"));
    }
    Ok(&buf[off..off + len])
}

// --- v3 writer helpers ----------------------------------------------------

/// Emit a v3 file at `path` with stars sorted by HEALPix cell at
/// `cell_depth`. Indices in `quads_in` are remapped to point at the
/// post-sort star positions. Codes are recomputed from the sorted star
/// positions.
///
/// Used by `Index::save_v3` and the upgrade tooling.
pub(super) fn write_v3<W: io::Write>(
    w: &mut W,
    metadata: Option<&IndexMetadata>,
    stars: &[IndexStar],
    quads: &[Quad],
    scale_lower: f64,
    scale_upper: f64,
    cell_depth: u8,
) -> io::Result<()> {
    // Sort stars by (cell_id, original index) — preserving the existing
    // brightness order within a cell. Build a remap from old index → new.
    let mut indexed_with_cell: Vec<(usize, u64)> = stars
        .iter()
        .enumerate()
        .map(|(i, s)| (i, nested::hash(cell_depth, s.ra, s.dec)))
        .collect();
    indexed_with_cell.sort_by_key(|&(orig_idx, cell_id)| (cell_id, orig_idx));

    let n_stars = stars.len();
    let mut star_remap = vec![0usize; n_stars];
    let mut sorted_stars: Vec<IndexStar> = Vec::with_capacity(n_stars);
    for (new_idx, &(orig_idx, _)) in indexed_with_cell.iter().enumerate() {
        star_remap[orig_idx] = new_idx;
        sorted_stars.push(stars[orig_idx].clone());
    }

    // Build cell_table from the sorted star list.
    let mut cell_table: Vec<CellEntry> = Vec::new();
    let mut current_cell: Option<u64> = None;
    let mut run_start = 0usize;
    for (i, (_, cell_id)) in indexed_with_cell.iter().enumerate() {
        match current_cell {
            None => {
                current_cell = Some(*cell_id);
                run_start = i;
            }
            Some(prev) if prev != *cell_id => {
                cell_table.push(CellEntry {
                    cell_id: prev,
                    star_offset: run_start as u64,
                    star_count: (i - run_start) as u32,
                });
                current_cell = Some(*cell_id);
                run_start = i;
            }
            _ => {}
        }
    }
    if let Some(prev) = current_cell {
        cell_table.push(CellEntry {
            cell_id: prev,
            star_offset: run_start as u64,
            star_count: (n_stars - run_start) as u32,
        });
    }

    // Remap quad star_ids; codes get recomputed from sorted positions.
    let star_xyz: Vec<[f64; 3]> = sorted_stars
        .iter()
        .map(|s| radec_to_xyz(s.ra, s.dec))
        .collect();

    // Header.
    w.write_all(MAGIC)?;
    w.write_all(&3u32.to_le_bytes())?; // version

    let meta_bytes = match metadata {
        Some(m) => serde_json::to_vec(m).map_err(io::Error::other)?,
        None => Vec::new(),
    };
    w.write_all(&(meta_bytes.len() as u64).to_le_bytes())?;
    if !meta_bytes.is_empty() {
        w.write_all(&meta_bytes)?;
    }

    w.write_all(&[cell_depth])?;
    w.write_all(&[0u8; 7])?; // reserved
    w.write_all(&(cell_table.len() as u64).to_le_bytes())?;

    // Cell table. `bytes_so_far` tracks the *writer's* current absolute
    // file offset so we can pad to an 8-byte boundary before the
    // n_stars/n_quads/scales block. Header so far:
    //   magic (4) + version (4) + meta_len (8) + meta_bytes (meta_len)
    //   + cell_depth (1) + reserved (7) + n_cells (8)
    //   = 32 + meta_bytes.len()
    let mut bytes_so_far = 4 + 4 + 8 + meta_bytes.len() + 1 + 7 + 8;
    for entry in &cell_table {
        w.write_all(&entry.cell_id.to_le_bytes())?;
        w.write_all(&entry.star_offset.to_le_bytes())?;
        w.write_all(&entry.star_count.to_le_bytes())?;
        bytes_so_far += CELL_TABLE_ENTRY_SIZE;
    }

    // Pad to 8-byte alignment.
    let aligned = (bytes_so_far + 7) & !7;
    let pad = aligned - bytes_so_far;
    if pad > 0 {
        w.write_all(&vec![0u8; pad])?;
    }

    w.write_all(&(n_stars as u64).to_le_bytes())?;
    w.write_all(&(quads.len() as u64).to_le_bytes())?;
    w.write_all(&scale_lower.to_le_bytes())?;
    w.write_all(&scale_upper.to_le_bytes())?;

    for s in &sorted_stars {
        w.write_all(&s.catalog_id.to_le_bytes())?;
        w.write_all(&s.ra.to_le_bytes())?;
        w.write_all(&s.dec.to_le_bytes())?;
        w.write_all(&s.mag.to_le_bytes())?;
    }

    let mut remapped_quads: Vec<Quad> = Vec::with_capacity(quads.len());
    for q in quads {
        let mut new_ids = [0usize; DIMQUADS];
        for (i, &sid) in q.star_ids.iter().enumerate() {
            new_ids[i] = star_remap[sid];
        }
        remapped_quads.push(Quad { star_ids: new_ids });
    }
    for q in &remapped_quads {
        for &id in &q.star_ids {
            w.write_all(&(id as u32).to_le_bytes())?;
        }
    }

    for q in &remapped_quads {
        let xyz: [_; DIMQUADS] = std::array::from_fn(|i| star_xyz[q.star_ids[i]]);
        let (code, _, _) = crate::quads::compute_canonical_code(&xyz, q.star_ids);
        for &v in &code {
            w.write_all(&v.to_le_bytes())?;
        }
    }

    w.flush()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{Index, IndexMetadata};
    use crate::kdtree::KdTree;

    fn make_test_index(n: usize) -> Index {
        let mut stars = Vec::with_capacity(n);
        for i in 0..n {
            let frac = i as f64 / n.max(1) as f64;
            stars.push(IndexStar {
                catalog_id: 100 + i as u64,
                ra: 1.0 + frac * 0.5,
                dec: 0.3 + frac * 0.4,
                mag: 5.0 + frac * 5.0,
            });
        }
        let star_points: Vec<[f64; 3]> = stars.iter().map(|s| radec_to_xyz(s.ra, s.dec)).collect();
        let star_indices: Vec<usize> = (0..n).collect();
        let star_tree = KdTree::<3>::build(star_points.clone(), star_indices);
        // Build a few quads if we have enough stars.
        let quads = if n >= 4 {
            (0..n.saturating_sub(3))
                .step_by(2)
                .take(5)
                .map(|base| Quad {
                    star_ids: [base, base + 1, base + 2, base + 3],
                })
                .collect::<Vec<_>>()
        } else {
            Vec::new()
        };
        let mut codes: Vec<Code> = Vec::with_capacity(quads.len());
        for q in &quads {
            let xyz: [_; DIMQUADS] = std::array::from_fn(|i| star_points[q.star_ids[i]]);
            let (code, _, _) = crate::quads::compute_canonical_code(&xyz, q.star_ids);
            codes.push(code);
        }
        let code_indices: Vec<usize> = (0..quads.len()).collect();
        let code_tree = KdTree::<{ DIMCODES }>::build(codes, code_indices);

        Index {
            star_tree,
            stars,
            code_tree,
            quads,
            scale_lower: 0.001,
            scale_upper: 0.05,
            metadata: Some(IndexMetadata {
                scale_lower_arcsec: 60.0,
                scale_upper_arcsec: 600.0,
                n_stars: n,
                n_quads: 5,
                max_stars_per_cell: 10,
                uniformize_depth: 6,
                quad_depth: 6,
                passes: 16,
                max_reuse: 8,
                build_timestamp: 1_700_000_000,
                catalog_path: None,
                band_index: None,
                scale_factor: None,
                mag_range: None,
            }),
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "zodiacal_v3_test_{name}_{}.bin",
            std::process::id()
        ))
    }

    #[test]
    fn v3_roundtrip_preserves_content() {
        let idx = make_test_index(40);
        let path = temp_path("v3_roundtrip");
        idx.save_v3(&path, DEFAULT_CELL_DEPTH).unwrap();

        let zf = ZdclFile::open(&path).unwrap();
        assert_eq!(zf.cell_depth(), DEFAULT_CELL_DEPTH);
        assert_eq!(zf.star_count(), 40);
        assert_eq!(zf.quad_count(), 5);

        let frag = zf.load_full().unwrap();
        std::fs::remove_file(&path).ok();

        // Content is preserved as a *set* (order may differ — stars are sorted
        // by HEALPix cell + original index, not by the order they were
        // inserted into the catalog).
        let original_ids: std::collections::HashSet<u64> =
            idx.stars.iter().map(|s| s.catalog_id).collect();
        let loaded_ids: std::collections::HashSet<u64> =
            frag.stars.iter().map(|s| s.catalog_id).collect();
        assert_eq!(original_ids, loaded_ids);
        assert_eq!(frag.scale_lower, idx.scale_lower);
        assert_eq!(frag.scale_upper, idx.scale_upper);
        assert!(frag.metadata.is_some());
    }

    #[test]
    fn v3_loads_v2_files_via_synthesis() {
        let idx = make_test_index(30);
        let path = temp_path("v3_reads_v2");
        // Use the legacy v2 writer.
        idx.save(&path).unwrap();

        let zf = ZdclFile::open(&path).unwrap();
        assert_eq!(zf.cell_depth(), SYNTHESIZED_CELL_DEPTH);
        assert_eq!(zf.star_count(), 30);

        // cells_intersecting on the synthesized layout returns the single
        // virtual cell, which load_cells expands to the full content.
        let region = SkyRegion::from_radians(starfield::Equatorial::new(1.2, 0.5), 0.5);
        let cells = zf.cells_intersecting(&region);
        assert_eq!(cells.len(), 1);
        assert_eq!(cells[0].depth, SYNTHESIZED_CELL_DEPTH);

        let frag = zf.load_cells(&cells).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(frag.stars.len(), 30);
    }

    #[test]
    fn cells_intersecting_disjoint_region_returns_none() {
        let idx = make_test_index(60);
        let path = temp_path("v3_disjoint");
        idx.save_v3(&path, DEFAULT_CELL_DEPTH).unwrap();

        let zf = ZdclFile::open(&path).unwrap();
        // Region nowhere near the stars (which sit around RA=1.0..1.5, Dec=0.3..0.7).
        let region = SkyRegion::from_radians(starfield::Equatorial::new(4.0, -1.2), 0.05);
        let cells = zf.cells_intersecting(&region);
        let frag = zf.load_cells(&cells).unwrap();
        std::fs::remove_file(&path).ok();
        // Cells may exist (the star area might overlap if depth is coarse),
        // but for a tight region 4 rad away in RA and far in Dec, no stars
        // should be loaded.
        assert_eq!(frag.stars.len(), 0);
        assert_eq!(frag.quads.len(), 0);
    }

    #[test]
    fn cells_intersecting_full_sky_returns_everything() {
        let idx = make_test_index(50);
        let path = temp_path("v3_fullsky");
        idx.save_v3(&path, DEFAULT_CELL_DEPTH).unwrap();

        let zf = ZdclFile::open(&path).unwrap();
        let region =
            SkyRegion::from_radians(starfield::Equatorial::new(0.0, 0.0), std::f64::consts::PI);
        let cells = zf.cells_intersecting(&region);
        let frag = zf.load_cells(&cells).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(frag.stars.len(), 50);
    }

    #[test]
    fn load_cells_empty_returns_empty_fragment() {
        let idx = make_test_index(20);
        let path = temp_path("v3_empty_cells");
        idx.save_v3(&path, DEFAULT_CELL_DEPTH).unwrap();
        let zf = ZdclFile::open(&path).unwrap();
        let frag = zf.load_cells(&[]).unwrap();
        std::fs::remove_file(&path).ok();
        assert_eq!(frag.stars.len(), 0);
        assert_eq!(frag.quads.len(), 0);
    }

    #[test]
    fn v3_solve_smoke_via_index_from_fragment() {
        // End-to-end: build an index, save as v3, open via ZdclFile,
        // load full, convert IndexFragment → Index, run solve(), verify
        // the recovered WCS matches the truth.
        use crate::extraction::DetectedSource;
        use crate::geom::tan::TanWcs;
        use crate::index::builder::{IndexBuilderConfig, build_index};
        use crate::solver::{SolverConfig, solve};
        use crate::verify::VerifyConfig;
        use std::f64::consts::PI;

        let image_size = (512.0, 512.0);
        let pixel_scale_arcsec: f64 = 2.0;
        let scale_rad = (pixel_scale_arcsec / 3600.0).to_radians();
        let truth_wcs = TanWcs {
            crval: [1.0, 0.5],
            crpix: [256.0, 256.0],
            cd: [[scale_rad, 0.0], [0.0, scale_rad]],
            image_size: [image_size.0, image_size.1],
        };

        let mut state: u64 = 314_159_265;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        let mut catalog = Vec::new();
        let mut sources = Vec::new();
        for i in 0..25 {
            let px = 30.0 + rng() * 452.0;
            let py = 30.0 + rng() * 452.0;
            let (ra, dec) = truth_wcs.pixel_to_radec(px, py);
            catalog.push((i as u64, ra, dec, i as f64));
            sources.push(DetectedSource {
                x: px,
                y: py,
                flux: 1000.0 - i as f64 * 10.0,
            });
        }

        let field_diag = (image_size.0 * image_size.0 + image_size.1 * image_size.1).sqrt();
        let max_angle = field_diag * scale_rad;
        let cfg = IndexBuilderConfig {
            scale_lower: scale_rad * 10.0,
            scale_upper: max_angle,
            max_stars: 25,
            max_quads: 50_000,
        };
        let index = build_index(&catalog, &cfg);

        let path = temp_path("v3_solve_smoke");
        index.save_v3(&path, DEFAULT_CELL_DEPTH).unwrap();

        let zf = ZdclFile::open(&path).unwrap();
        let frag = zf.load_full().unwrap();
        std::fs::remove_file(&path).ok();
        let reloaded: Index = frag.into();
        assert_eq!(reloaded.stars.len(), index.stars.len());
        assert_eq!(reloaded.quads.len(), index.quads.len());
        assert_eq!(reloaded.star_tree.len(), index.stars.len());
        assert_eq!(reloaded.code_tree.len(), index.quads.len());

        let solver_cfg = SolverConfig {
            scale_range: None,
            max_field_stars: 25,
            code_tolerance: 0.002,
            verify: VerifyConfig {
                match_radius_pix: 3.0,
                log_odds_accept: 10.0,
                min_matches: 3,
                ..VerifyConfig::default()
            },
            ..SolverConfig::default()
        };
        let (solution, _stats) = solve(&sources, &[&reloaded], image_size, &solver_cfg);
        let solution = solution.expect("v3-loaded index should solve");

        let (solved_ra, solved_dec) = solution.wcs.field_center();
        let (truth_ra, truth_dec) = truth_wcs.field_center();
        let arcsec = PI / (180.0 * 3600.0);
        let dra = (solved_ra - truth_ra).abs() * truth_dec.cos();
        let ddec = (solved_dec - truth_dec).abs();
        let sep_arcsec = ((dra * dra + ddec * ddec).sqrt()) / arcsec;
        assert!(
            sep_arcsec < 30.0,
            "v3 sparse-load+solve image-center separation {sep_arcsec:.2} arcsec exceeds 30\""
        );
    }

    #[test]
    fn v3_sparse_load_matches_full_load_for_full_region() {
        let idx = make_test_index(60);
        let path = temp_path("v3_sparse_full");
        idx.save_v3(&path, DEFAULT_CELL_DEPTH).unwrap();

        let zf = ZdclFile::open(&path).unwrap();
        let region =
            SkyRegion::from_radians(starfield::Equatorial::new(0.0, 0.0), std::f64::consts::PI);
        let cells = zf.cells_intersecting(&region);
        let fragment_via_cells = zf.load_cells(&cells).unwrap();
        let fragment_full = zf.load_full().unwrap();
        std::fs::remove_file(&path).ok();

        // Same star set (order may differ between the two paths if cell
        // iteration order does not match the on-disk star order — our
        // load_cells walks star_ranges sorted ascending, which matches the
        // on-disk grouping, so they should be identical).
        assert_eq!(fragment_via_cells.stars.len(), fragment_full.stars.len());
        assert_eq!(fragment_via_cells.quads.len(), fragment_full.quads.len());
        let ids_full: std::collections::HashSet<u64> =
            fragment_full.stars.iter().map(|s| s.catalog_id).collect();
        let ids_cells: std::collections::HashSet<u64> = fragment_via_cells
            .stars
            .iter()
            .map(|s| s.catalog_id)
            .collect();
        assert_eq!(ids_full, ids_cells);
    }

    #[test]
    fn v3_rejects_corrupt_cell_depth() {
        // Hand-construct a v3 header with cell_depth = 200 (> 29), verify
        // ZdclFile::open returns InvalidData rather than panicking.
        use std::io::Write as _;
        let path = temp_path("v3_bad_depth");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"ZDCL").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap(); // meta_len
        f.write_all(&[200u8]).unwrap(); // bad cell_depth
        f.write_all(&[0u8; 7]).unwrap(); // reserved
        f.write_all(&0u64.to_le_bytes()).unwrap(); // n_cells = 0
        f.write_all(&0u64.to_le_bytes()).unwrap(); // n_stars = 0
        f.write_all(&0u64.to_le_bytes()).unwrap(); // n_quads = 0
        f.write_all(&0.0f64.to_le_bytes()).unwrap(); // scale_lower
        f.write_all(&0.0f64.to_le_bytes()).unwrap(); // scale_upper
        drop(f);
        let err = match ZdclFile::open(&path) {
            Ok(_) => panic!("should have rejected bad cell_depth"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn v3_rejects_oversized_n_cells() {
        // A v3 header that claims more cells than fit in the file (or even
        // could exist at the chosen depth) must error, not panic on
        // Vec::with_capacity.
        use std::io::Write as _;
        let path = temp_path("v3_bad_ncells");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"ZDCL").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&[5u8]).unwrap(); // valid cell_depth
        f.write_all(&[0u8; 7]).unwrap();
        // Claim u64::MAX cells — would OOM Vec::with_capacity.
        f.write_all(&u64::MAX.to_le_bytes()).unwrap();
        drop(f);
        let err = match ZdclFile::open(&path) {
            Ok(_) => panic!("should have rejected oversized n_cells"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn v3_rejects_truncated_file() {
        // Header claims n_stars > 0 but the stars block is missing.
        use std::io::Write as _;
        let path = temp_path("v3_truncated");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"ZDCL").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&[5u8]).unwrap();
        f.write_all(&[0u8; 7]).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap(); // n_cells = 0
        f.write_all(&100u64.to_le_bytes()).unwrap(); // n_stars = 100 (lying)
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0.0f64.to_le_bytes()).unwrap();
        f.write_all(&0.0f64.to_le_bytes()).unwrap();
        // Don't write the stars block.
        drop(f);
        let err = match ZdclFile::open(&path) {
            Ok(_) => panic!("should have rejected truncated file"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn v3_rejects_cell_entry_out_of_bounds() {
        // A cell entry whose star_offset+star_count exceeds n_stars must
        // error at open, not silently truncate at load time.
        use std::io::Write as _;
        let path = temp_path("v3_bad_cell");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"ZDCL").unwrap();
        f.write_all(&3u32.to_le_bytes()).unwrap();
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&[5u8]).unwrap();
        f.write_all(&[0u8; 7]).unwrap();
        f.write_all(&1u64.to_le_bytes()).unwrap(); // n_cells = 1
        // One bogus cell entry: cell_id = 0, star_offset = 50, count = 100
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&50u64.to_le_bytes()).unwrap();
        f.write_all(&100u32.to_le_bytes()).unwrap();
        // Padding to 8-byte alignment (cell entry is 20 bytes; cursor at
        // 32+20=52 needs 4 bytes pad)
        f.write_all(&[0u8; 4]).unwrap();
        f.write_all(&30u64.to_le_bytes()).unwrap(); // n_stars = 30 (cell range exceeds)
        f.write_all(&0u64.to_le_bytes()).unwrap();
        f.write_all(&0.0f64.to_le_bytes()).unwrap();
        f.write_all(&0.0f64.to_le_bytes()).unwrap();
        // 30 stars × 32 bytes
        f.write_all(&vec![0u8; 30 * 32]).unwrap();
        drop(f);
        let err = match ZdclFile::open(&path) {
            Ok(_) => panic!("should have rejected out-of-bounds cell entry"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn v3_load_cells_drops_quads_with_outside_member() {
        let idx = make_test_index(40);
        let path = temp_path("v3_drops_outside");
        idx.save_v3(&path, DEFAULT_CELL_DEPTH).unwrap();
        let zf = ZdclFile::open(&path).unwrap();

        // Pick a tight region that catches only some of the stars.
        let region = SkyRegion::from_radians(starfield::Equatorial::new(1.0, 0.3), 0.005);
        let cells = zf.cells_intersecting(&region);
        let frag = zf.load_cells(&cells).unwrap();
        std::fs::remove_file(&path).ok();

        // Whatever made it through, every quad's star_ids must be in-bounds
        // for the kept stars.
        for q in &frag.quads {
            for &sid in &q.star_ids {
                assert!(
                    sid < frag.stars.len(),
                    "quad references invalid index {sid} (only {} stars loaded)",
                    frag.stars.len()
                );
            }
        }
    }
}
