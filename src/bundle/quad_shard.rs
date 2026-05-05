//! Per-cell `.zqd` quad-shard format: quads + codes for a single HEALPix cell,
//! across every scale band, with a band table at the head pointing at each
//! band's quad/code blocks.
//!
//! See `docs/bundle-format.md` — section "`quads/cell_NNNNN.zqd` — quads +
//! codes for one cell, all bands" — for the canonical layout. This module
//! implements both the writer (used by the cell-driven multi-band builder)
//! and a borrowing zero-copy-ish reader suitable for mmap-backed use.
//!
//! ## Position-independent shards
//!
//! All offsets stored in the band table are **relative to the start of the
//! shard's bytes** — i.e., relative to the magic at byte 0 of the shard, not
//! to the start of whatever stream the shard happens to be embedded in. A
//! shard is therefore position-independent: it can be concatenated mid-stream
//! (into a tar, zip, or any other container) and `QuadShard::parse` will
//! still decode it correctly when handed the embedded byte slice. The writer
//! never consults its underlying stream's position when computing offsets;
//! readers always interpret offsets relative to the slice they're given.
//!
//! Byte layout (all little-endian):
//!
//! ```text
//! HEADER (32 B)
//!   magic         8 B   b"ZDCLQUAD"
//!   version       4 B   u32 LE = 1
//!   reserved      4 B   zero
//!   cell_id       8 B   u64 LE
//!   n_bands       4 B   u32 LE
//!   reserved      4 B   pad to 8
//!
//! BAND TABLE   n_bands × 24 B
//!   band_idx     4 B  u32 LE
//!   n_quads      4 B  u32 LE
//!   quads_offset 8 B  u64 LE   byte offset relative to shard start
//!   codes_offset 8 B  u64 LE   byte offset relative to shard start
//!
//! Per band, in band_idx order, no padding between blocks:
//!   quads_block   n_quads × 16 B   (4 × u32 LE local star indices)
//!   codes_block   n_quads × 32 B   (4 × f64 LE)
//! ```

use std::io::{self, Write};

use crate::quads::{Code, DIMCODES, DIMQUADS, Quad};

#[cfg(target_endian = "big")]
compile_error!(
    "Quad shard on-disk layout assumes little-endian; big-endian targets are not supported."
);

/// 8-byte file magic identifying a per-cell quad shard.
pub const QUAD_SHARD_MAGIC: &[u8; 8] = b"ZDCLQUAD";

/// On-disk format version this module reads and writes.
pub const QUAD_SHARD_VERSION: u32 = 1;

/// Size of the fixed file header (magic + version + reserved + cell_id +
/// n_bands + reserved).
pub const HEADER_SIZE: usize = 32;

/// Size of one band-table entry on disk.
pub const BAND_ENTRY_SIZE: usize = 24;

/// Encoded size of a single quad record on disk: `DIMQUADS × u32`.
pub const QUAD_RECORD_SIZE: usize = DIMQUADS * 4;

/// Encoded size of a single code record on disk: `DIMCODES × f64`.
pub const CODE_RECORD_SIZE: usize = DIMCODES * 8;

// Compile-time sanity: the spec pins the on-disk layout at 16 B per quad and
// 32 B per code, which assumes DIMQUADS = 4 and DIMCODES = 4. If those ever
// change, the on-disk format must be revisited.
const _: () = assert!(QUAD_RECORD_SIZE == 16);
const _: () = assert!(CODE_RECORD_SIZE == 32);

/// One band's quads + codes to be emitted into a `.zqd` file.
pub struct BandEmit<'a> {
    pub band_idx: u32,
    pub quads: &'a [Quad],
    pub codes: &'a [Code],
}

/// One entry in the on-disk band table.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BandEntry {
    pub band_idx: u32,
    pub n_quads: u32,
    pub quads_offset: u64,
    pub codes_offset: u64,
}

/// Borrowing reader over a mmapped (or otherwise loaded) `.zqd` byte slice.
///
/// The header + band-table is parsed eagerly on construction (it's tiny —
/// typically under 320 B for a dozen bands). Per-band quad / code blocks are
/// decoded on demand via [`BandView`].
#[derive(Debug)]
pub struct QuadShard<'a> {
    raw: &'a [u8],
    cell_id: u64,
    band_table: Vec<BandEntry>,
}

/// View into a single band's quads + codes within a `QuadShard`.
#[derive(Debug)]
pub struct BandView<'a> {
    raw: &'a [u8],
    n_quads: usize,
    quads_offset: usize,
    codes_offset: usize,
}

// --- writer ----------------------------------------------------------------

/// Write a `.zqd` shard for one cell containing every band's quads/codes.
///
/// The writer accepts the bands in any order; entries are sorted ascending
/// by `band_idx` before being written, so the on-disk band table is always
/// monotonic. `band_idx` values must be unique across the input slice.
///
/// Returns `InvalidInput` if any `Quad.star_ids` element does not fit in a
/// `u32`, or if duplicate `band_idx` values are present, or if a band's
/// `quads.len() != codes.len()`.
///
/// **Offsets are slice-relative.** All `quads_offset` / `codes_offset`
/// values stored in the band table are byte offsets from the start of the
/// shard's bytes (the `b"ZDCLQUAD"` magic), *not* from the start of the
/// underlying stream. The writer therefore never consults `w`'s stream
/// position; the resulting shard is position-independent and round-trips
/// when concatenated mid-stream and parsed via the embedded byte slice.
/// Do not reintroduce a `header_start` term — `QuadShard::parse` interprets
/// every offset relative to the slice it's given.
///
/// The bound is `Write` only (no `Seek`): the shard is built into an
/// in-memory buffer and emitted with one `write_all` + `flush`, so the
/// caller is guaranteed the bytes are in the underlying writer when this
/// function returns.
pub fn write_quad_shard<W: Write>(
    w: &mut W,
    cell_id: u64,
    bands: &[BandEmit<'_>],
) -> io::Result<()> {
    // Build a sorted view of the input without mutating the caller's slice.
    // We sort indirectly via a Vec of references so we don't have to clone
    // BandEmit (which would also force its lifetime to be 'static).
    let mut order: Vec<&BandEmit<'_>> = bands.iter().collect();
    order.sort_by_key(|b| b.band_idx);

    // Reject duplicate band_idx values up front — the spec wants one entry
    // per band, and a duplicate would silently fight for the same slot.
    for pair in order.windows(2) {
        if pair[0].band_idx == pair[1].band_idx {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!("duplicate band_idx {} in BandEmit slice", pair[0].band_idx),
            ));
        }
    }

    // Reject mismatched quads/codes lengths and overflow before writing
    // anything.
    for b in &order {
        if b.quads.len() != b.codes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "band {} has {} quads but {} codes",
                    b.band_idx,
                    b.quads.len(),
                    b.codes.len()
                ),
            ));
        }
        for q in b.quads {
            for &sid in &q.star_ids {
                if sid > u32::MAX as usize {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "quad star_id {sid} does not fit in u32; \
                             cannot encode in .zqd format"
                        ),
                    ));
                }
            }
        }
    }

    let n_bands = order.len() as u32;
    let band_table_size = order.len() * BAND_ENTRY_SIZE;

    // Compute total shard size, then allocate one buffer. Offsets baked
    // into the band table are slice-relative (start of `buf` == start of
    // magic), so the shard is position-independent regardless of where it
    // ends up in the underlying stream.
    let mut data_size: usize = 0;
    for b in &order {
        let n_quads = b.quads.len();
        data_size += n_quads * QUAD_RECORD_SIZE + n_quads * CODE_RECORD_SIZE;
    }
    let total_size = HEADER_SIZE + band_table_size + data_size;
    let mut buf: Vec<u8> = Vec::with_capacity(total_size);

    // --- header --- (32 B)
    buf.extend_from_slice(QUAD_SHARD_MAGIC);
    buf.extend_from_slice(&QUAD_SHARD_VERSION.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // reserved
    buf.extend_from_slice(&cell_id.to_le_bytes());
    buf.extend_from_slice(&n_bands.to_le_bytes());
    buf.extend_from_slice(&0u32.to_le_bytes()); // reserved (header pad to 8)
    debug_assert_eq!(buf.len(), HEADER_SIZE);

    // --- placeholder band table --- we'll fill the real entries in below
    // once we know each band's slice-relative quads/codes offsets.
    let band_table_start = buf.len();
    buf.resize(band_table_start + band_table_size, 0);

    // --- per-band data blocks ---
    let mut entries: Vec<BandEntry> = Vec::with_capacity(order.len());
    for b in &order {
        let n_quads = b.quads.len();
        // Slice-relative offset: simply the current length of `buf`.
        let quads_offset = buf.len() as u64;
        for q in b.quads {
            let mut record = [0u8; QUAD_RECORD_SIZE];
            for (i, &sid) in q.star_ids.iter().enumerate() {
                // Bounds-checked above; this cast is safe.
                let v = sid as u32;
                record[i * 4..i * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
            buf.extend_from_slice(&record);
        }
        let codes_offset = buf.len() as u64;
        debug_assert_eq!(
            codes_offset,
            quads_offset + (n_quads * QUAD_RECORD_SIZE) as u64
        );
        for c in b.codes {
            let mut record = [0u8; CODE_RECORD_SIZE];
            for (i, &v) in c.iter().enumerate() {
                record[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
            }
            buf.extend_from_slice(&record);
        }

        entries.push(BandEntry {
            band_idx: b.band_idx,
            n_quads: n_quads as u32,
            quads_offset,
            codes_offset,
        });
    }
    debug_assert_eq!(buf.len(), total_size);

    // --- fill the real band table at the head of the buffer ---
    for (k, e) in entries.iter().enumerate() {
        let off = band_table_start + k * BAND_ENTRY_SIZE;
        buf[off..off + 4].copy_from_slice(&e.band_idx.to_le_bytes());
        buf[off + 4..off + 8].copy_from_slice(&e.n_quads.to_le_bytes());
        buf[off + 8..off + 16].copy_from_slice(&e.quads_offset.to_le_bytes());
        buf[off + 16..off + 24].copy_from_slice(&e.codes_offset.to_le_bytes());
    }

    // Single emission, then flush — the shard is fully durable in the
    // underlying writer by the time we return.
    w.write_all(&buf)?;
    w.flush()?;
    Ok(())
}

// --- reader ----------------------------------------------------------------

impl<'a> QuadShard<'a> {
    /// Parse the header + band table out of `bytes`, validating magic,
    /// version, and that every (offset, length) pair lands inside the
    /// slice. Per-band quad / code blocks are decoded on demand via
    /// [`Self::band`].
    ///
    /// **Offsets are interpreted relative to the start of `bytes`.** The
    /// writer emits position-independent shards, so this works whether
    /// `bytes` is a standalone shard file, an mmap of one, or a sub-slice
    /// of a larger buffer (e.g., a shard concatenated mid-stream into a
    /// tar / zip / custom container) — as long as the slice starts at the
    /// shard's `b"ZDCLQUAD"` magic and ends at or past the last byte of
    /// the last band's codes block.
    pub fn parse(bytes: &'a [u8]) -> io::Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "quad shard truncated: {} bytes < header size {}",
                    bytes.len(),
                    HEADER_SIZE
                ),
            ));
        }

        if &bytes[0..8] != QUAD_SHARD_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "quad shard bad magic; expected b\"ZDCLQUAD\"",
            ));
        }

        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != QUAD_SHARD_VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "quad shard version {version} unsupported \
                     (this build understands version {QUAD_SHARD_VERSION})"
                ),
            ));
        }
        // bytes[12..16] reserved
        let cell_id = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let n_bands = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
        // bytes[28..32] reserved (pad)

        let band_table_start = HEADER_SIZE;
        let band_table_end = band_table_start
            .checked_add(n_bands.checked_mul(BAND_ENTRY_SIZE).ok_or_else(|| {
                io::Error::new(io::ErrorKind::InvalidData, "band-table size overflow")
            })?)
            .ok_or_else(|| io::Error::new(io::ErrorKind::InvalidData, "band-table end overflow"))?;
        if band_table_end > bytes.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "quad shard truncated: band-table ends at {band_table_end}, \
                     file is {} bytes",
                    bytes.len()
                ),
            ));
        }

        let mut band_table: Vec<BandEntry> = Vec::with_capacity(n_bands);
        for k in 0..n_bands {
            let off = band_table_start + k * BAND_ENTRY_SIZE;
            let band_idx = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap());
            let n_quads = u32::from_le_bytes(bytes[off + 4..off + 8].try_into().unwrap());
            let quads_offset = u64::from_le_bytes(bytes[off + 8..off + 16].try_into().unwrap());
            let codes_offset = u64::from_le_bytes(bytes[off + 16..off + 24].try_into().unwrap());

            // Validate this entry's blocks lie inside `bytes`.
            let nq = n_quads as usize;
            let quads_end = (quads_offset as usize)
                .checked_add(nq.checked_mul(QUAD_RECORD_SIZE).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "quads block size overflow")
                })?)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "quads block end overflow")
                })?;
            let codes_end = (codes_offset as usize)
                .checked_add(nq.checked_mul(CODE_RECORD_SIZE).ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "codes block size overflow")
                })?)
                .ok_or_else(|| {
                    io::Error::new(io::ErrorKind::InvalidData, "codes block end overflow")
                })?;
            if quads_end > bytes.len() || codes_end > bytes.len() {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "band {band_idx} blocks out of range: \
                         quads_end={quads_end}, codes_end={codes_end}, \
                         file_size={}",
                        bytes.len()
                    ),
                ));
            }
            if nq > 0 {
                // Populated bands' offsets must clear the header +
                // band-table region.
                if (quads_offset as usize) < band_table_end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "band {band_idx} quads_offset {quads_offset} overlaps \
                             header/band-table (ends at {band_table_end})"
                        ),
                    ));
                }
                if (codes_offset as usize) < band_table_end {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "band {band_idx} codes_offset {codes_offset} overlaps \
                             header/band-table (ends at {band_table_end})"
                        ),
                    ));
                }
            } else {
                // Invariant for empty bands (n_quads == 0): the writer
                // emits zero bytes between the quads_offset and
                // codes_offset bookmarks, so the two must compare equal
                // and must point at a valid in-slice byte (at or beyond
                // the band table, at or before the slice end).
                if quads_offset != codes_offset {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "empty band {band_idx} has quads_offset={quads_offset} != \
                             codes_offset={codes_offset}; expected equal offsets"
                        ),
                    ));
                }
                if (quads_offset as usize) < band_table_end || (quads_offset as usize) > bytes.len()
                {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!(
                            "empty band {band_idx} offset {quads_offset} outside data \
                             region [{band_table_end}, {})",
                            bytes.len()
                        ),
                    ));
                }
            }

            band_table.push(BandEntry {
                band_idx,
                n_quads,
                quads_offset,
                codes_offset,
            });
        }

        // The on-disk band table is required to be strictly monotonic in
        // band_idx (the writer sorts; readers rely on that for
        // binary-search lookup).
        for pair in band_table.windows(2) {
            if pair[0].band_idx >= pair[1].band_idx {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "band table not strictly ascending by band_idx: \
                         {} followed by {}",
                        pair[0].band_idx, pair[1].band_idx
                    ),
                ));
            }
        }

        // Validate that no two band data regions overlap. Each populated
        // band contributes two regions (quads block, codes block); empty
        // bands contribute none. We collect, sort by start, then walk
        // pairwise.
        let mut regions: Vec<(usize, usize, u32, &'static str)> =
            Vec::with_capacity(band_table.len() * 2);
        for entry in &band_table {
            let nq = entry.n_quads as usize;
            if nq == 0 {
                continue;
            }
            let q_start = entry.quads_offset as usize;
            let q_end = q_start + nq * QUAD_RECORD_SIZE;
            let c_start = entry.codes_offset as usize;
            let c_end = c_start + nq * CODE_RECORD_SIZE;
            regions.push((q_start, q_end, entry.band_idx, "quads"));
            regions.push((c_start, c_end, entry.band_idx, "codes"));
        }
        regions.sort_by_key(|r| r.0);
        for w in regions.windows(2) {
            let (a_start, a_end, a_band, a_kind) = w[0];
            let (b_start, b_end, b_band, b_kind) = w[1];
            if a_end > b_start {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "band data regions overlap: band {a_band} {a_kind} \
                         [{a_start}, {a_end}) vs band {b_band} {b_kind} \
                         [{b_start}, {b_end})"
                    ),
                ));
            }
        }

        Ok(QuadShard {
            raw: bytes,
            cell_id,
            band_table,
        })
    }

    pub fn cell_id(&self) -> u64 {
        self.cell_id
    }

    pub fn n_bands(&self) -> usize {
        self.band_table.len()
    }

    pub fn bands(&self) -> &[BandEntry] {
        &self.band_table
    }

    /// Look up the entry for `band_idx` via binary search (O(log n_bands)).
    pub fn band(&self, band_idx: u32) -> Option<BandView<'_>> {
        let pos = self
            .band_table
            .binary_search_by_key(&band_idx, |e| e.band_idx)
            .ok()?;
        let e = &self.band_table[pos];
        Some(BandView {
            raw: self.raw,
            n_quads: e.n_quads as usize,
            quads_offset: e.quads_offset as usize,
            codes_offset: e.codes_offset as usize,
        })
    }
}

impl<'a> BandView<'a> {
    pub fn n_quads(&self) -> usize {
        self.n_quads
    }

    /// Iterate this band's quads, decoded from 4 × u32 LE on disk into the
    /// in-memory `[usize; 4]` representation.
    pub fn quads_iter(&self) -> impl Iterator<Item = Quad> + '_ {
        let base = self.quads_offset;
        let raw = self.raw;
        (0..self.n_quads).map(move |i| {
            let off = base + i * QUAD_RECORD_SIZE;
            let mut star_ids = [0usize; DIMQUADS];
            for (k, slot) in star_ids.iter_mut().enumerate() {
                let p = off + k * 4;
                *slot = u32::from_le_bytes(raw[p..p + 4].try_into().unwrap()) as usize;
            }
            Quad { star_ids }
        })
    }

    /// Iterate this band's codes, decoded from 4 × f64 LE on disk.
    pub fn codes_iter(&self) -> impl Iterator<Item = Code> + '_ {
        let base = self.codes_offset;
        let raw = self.raw;
        (0..self.n_quads).map(move |i| {
            let off = base + i * CODE_RECORD_SIZE;
            let mut code = [0.0f64; DIMCODES];
            for (k, slot) in code.iter_mut().enumerate() {
                let p = off + k * 8;
                *slot = f64::from_le_bytes(raw[p..p + 8].try_into().unwrap());
            }
            code
        })
    }
}

// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    fn make_quad(a: usize, b: usize, c: usize, d: usize) -> Quad {
        Quad {
            star_ids: [a, b, c, d],
        }
    }

    fn make_code(seed: f64) -> Code {
        [seed, seed + 1.0, seed + 2.0, seed + 3.0]
    }

    fn write_to_vec(cell_id: u64, bands: &[BandEmit<'_>]) -> Vec<u8> {
        let mut buf = Vec::new();
        {
            let mut cur = Cursor::new(&mut buf);
            write_quad_shard(&mut cur, cell_id, bands).expect("write_quad_shard");
        }
        buf
    }

    #[test]
    fn roundtrip_single_band() {
        let quads = vec![
            make_quad(1, 2, 3, 4),
            make_quad(10, 20, 30, 40),
            make_quad(7, 8, 9, 11),
        ];
        let codes = vec![make_code(0.0), make_code(0.5), make_code(-1.25)];
        let bytes = write_to_vec(
            42,
            &[BandEmit {
                band_idx: 3,
                quads: &quads,
                codes: &codes,
            }],
        );

        let shard = QuadShard::parse(&bytes).expect("parse");
        assert_eq!(shard.cell_id(), 42);
        assert_eq!(shard.n_bands(), 1);
        assert_eq!(shard.bands()[0].band_idx, 3);
        assert_eq!(shard.bands()[0].n_quads, 3);

        let view = shard.band(3).expect("band 3 exists");
        assert_eq!(view.n_quads(), 3);
        let got_quads: Vec<Quad> = view.quads_iter().collect();
        let got_codes: Vec<Code> = view.codes_iter().collect();
        assert_eq!(got_quads.len(), 3);
        for (lhs, rhs) in got_quads.iter().zip(quads.iter()) {
            assert_eq!(lhs.star_ids, rhs.star_ids);
        }
        assert_eq!(got_codes, codes);

        // Missing band → None.
        assert!(shard.band(0).is_none());
        assert!(shard.band(99).is_none());
    }

    #[test]
    fn roundtrip_multi_band() {
        let q0: Vec<Quad> = (0..5).map(|i| make_quad(i, i + 1, i + 2, i + 3)).collect();
        let c0: Vec<Code> = (0..5).map(|i| make_code(i as f64 * 0.1)).collect();

        let q1: Vec<Quad> = vec![]; // empty middle band, intentionally
        let c1: Vec<Code> = vec![];

        let q2: Vec<Quad> = (0..12)
            .map(|i| make_quad(100 + i, 200 + i, 300 + i, 400 + i))
            .collect();
        let c2: Vec<Code> = (0..12).map(|i| make_code(i as f64 + 100.0)).collect();

        let bytes = write_to_vec(
            7,
            &[
                BandEmit {
                    band_idx: 0,
                    quads: &q0,
                    codes: &c0,
                },
                BandEmit {
                    band_idx: 1,
                    quads: &q1,
                    codes: &c1,
                },
                BandEmit {
                    band_idx: 2,
                    quads: &q2,
                    codes: &c2,
                },
            ],
        );

        let shard = QuadShard::parse(&bytes).expect("parse");
        assert_eq!(shard.cell_id(), 7);
        assert_eq!(shard.n_bands(), 3);

        let v0 = shard.band(0).unwrap();
        let v1 = shard.band(1).unwrap();
        let v2 = shard.band(2).unwrap();
        assert_eq!(v0.n_quads(), 5);
        assert_eq!(v1.n_quads(), 0);
        assert_eq!(v2.n_quads(), 12);

        let got_q0: Vec<Quad> = v0.quads_iter().collect();
        let got_q2: Vec<Quad> = v2.quads_iter().collect();
        let got_c0: Vec<Code> = v0.codes_iter().collect();
        let got_c2: Vec<Code> = v2.codes_iter().collect();
        assert_eq!(got_q0.len(), 5);
        for (got, want) in got_q0.iter().zip(q0.iter()) {
            assert_eq!(got.star_ids, want.star_ids);
        }
        for (got, want) in got_q2.iter().zip(q2.iter()) {
            assert_eq!(got.star_ids, want.star_ids);
        }
        assert_eq!(got_c0, c0);
        assert_eq!(got_c2, c2);

        // Empty band iterators yield nothing without panicking.
        assert_eq!(v1.quads_iter().count(), 0);
        assert_eq!(v1.codes_iter().count(), 0);

        // band(missing) returns None for an idx not in the table.
        assert!(shard.band(3).is_none());
        assert!(shard.band(u32::MAX).is_none());
    }

    #[test]
    fn empty_band_in_middle() {
        // Bands [0, 5, 11], with band 5 empty. Verify that band 5's table
        // entry has n_quads = 0 and the offsets nominally land inside the
        // file (right between band 0's data and band 11's).
        let q0: Vec<Quad> = (0..3).map(|i| make_quad(i, i + 1, i + 2, i + 3)).collect();
        let c0: Vec<Code> = (0..3).map(|i| make_code(i as f64)).collect();
        let q5: Vec<Quad> = vec![];
        let c5: Vec<Code> = vec![];
        let q11: Vec<Quad> = (0..2)
            .map(|i| make_quad(50 + i, 60 + i, 70 + i, 80 + i))
            .collect();
        let c11: Vec<Code> = (0..2).map(|i| make_code(50.0 + i as f64)).collect();

        let bytes = write_to_vec(
            123,
            &[
                BandEmit {
                    band_idx: 0,
                    quads: &q0,
                    codes: &c0,
                },
                BandEmit {
                    band_idx: 5,
                    quads: &q5,
                    codes: &c5,
                },
                BandEmit {
                    band_idx: 11,
                    quads: &q11,
                    codes: &c11,
                },
            ],
        );

        let shard = QuadShard::parse(&bytes).expect("parse");
        assert_eq!(shard.cell_id(), 123);
        assert_eq!(shard.n_bands(), 3);

        let entry5 = shard
            .bands()
            .iter()
            .find(|e| e.band_idx == 5)
            .expect("band 5 entry");
        assert_eq!(entry5.n_quads, 0);
        // Offsets must land within the file size.
        assert!((entry5.quads_offset as usize) <= bytes.len());
        assert!((entry5.codes_offset as usize) <= bytes.len());

        let v5 = shard.band(5).unwrap();
        assert_eq!(v5.n_quads(), 0);
        assert_eq!(v5.quads_iter().count(), 0);
        assert_eq!(v5.codes_iter().count(), 0);

        let got11_q: Vec<Quad> = shard.band(11).unwrap().quads_iter().collect();
        for (got, want) in got11_q.iter().zip(q11.iter()) {
            assert_eq!(got.star_ids, want.star_ids);
        }
    }

    #[test]
    fn bad_magic_rejected() {
        // 8 bytes "ZDCLQUAX" + enough garbage for a header-sized blob.
        let mut bytes = Vec::with_capacity(HEADER_SIZE);
        bytes.extend_from_slice(b"ZDCLQUAX");
        bytes.extend_from_slice(&[0u8; HEADER_SIZE - 8]);
        let err = QuadShard::parse(&bytes).expect_err("bad magic must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn bad_version_rejected() {
        let mut bytes = Vec::with_capacity(HEADER_SIZE);
        bytes.extend_from_slice(QUAD_SHARD_MAGIC);
        bytes.extend_from_slice(&99u32.to_le_bytes()); // bogus version
        bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved
        bytes.extend_from_slice(&0u64.to_le_bytes()); // cell_id
        bytes.extend_from_slice(&0u32.to_le_bytes()); // n_bands
        bytes.extend_from_slice(&0u32.to_le_bytes()); // reserved
        let err = QuadShard::parse(&bytes).expect_err("bad version must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn truncated_file_rejected() {
        // Build a valid-looking shard, then chop the tail off so the band
        // table's offsets/sizes overshoot the slice.
        let q: Vec<Quad> = (0..8).map(|i| make_quad(i, i + 1, i + 2, i + 3)).collect();
        let c: Vec<Code> = (0..8).map(|i| make_code(i as f64)).collect();
        let bytes = write_to_vec(
            1,
            &[BandEmit {
                band_idx: 0,
                quads: &q,
                codes: &c,
            }],
        );
        // Drop the last 16 bytes — that hacks off (at least) the last
        // code's 32 bytes worth of content if we drop more, but 16 alone
        // straddles a code record so the codes_end check trips.
        let truncated = &bytes[..bytes.len() - 16];
        let err = QuadShard::parse(truncated).expect_err("truncated file must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        // Header-only truncation (cut into the band table) also rejects.
        let head_only = &bytes[..HEADER_SIZE + 4];
        let err2 = QuadShard::parse(head_only).expect_err("band-table truncation must reject");
        assert_eq!(err2.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn band_idx_unsorted_input_sorted_internally() {
        // Writer is given bands in non-ascending order (5, 1, 9). We
        // chose to sort internally; verify the on-disk band table comes
        // out monotonic and parsing succeeds.
        let q1: Vec<Quad> = vec![make_quad(1, 2, 3, 4)];
        let c1: Vec<Code> = vec![make_code(1.0)];
        let q5: Vec<Quad> = vec![make_quad(5, 6, 7, 8), make_quad(9, 10, 11, 12)];
        let c5: Vec<Code> = vec![make_code(5.0), make_code(5.5)];
        let q9: Vec<Quad> = vec![make_quad(13, 14, 15, 16)];
        let c9: Vec<Code> = vec![make_code(9.0)];

        let bytes = write_to_vec(
            999,
            &[
                BandEmit {
                    band_idx: 5,
                    quads: &q5,
                    codes: &c5,
                },
                BandEmit {
                    band_idx: 1,
                    quads: &q1,
                    codes: &c1,
                },
                BandEmit {
                    band_idx: 9,
                    quads: &q9,
                    codes: &c9,
                },
            ],
        );

        let shard = QuadShard::parse(&bytes).expect("parse with sorted output");
        let idxs: Vec<u32> = shard.bands().iter().map(|e| e.band_idx).collect();
        assert_eq!(idxs, vec![1, 5, 9], "band table must be sorted ascending");

        // Each band's content survived the sort.
        let got1: Vec<Quad> = shard.band(1).unwrap().quads_iter().collect();
        let got5: Vec<Quad> = shard.band(5).unwrap().quads_iter().collect();
        let got9: Vec<Quad> = shard.band(9).unwrap().quads_iter().collect();
        assert_eq!(got1.len(), 1);
        assert_eq!(got1[0].star_ids, q1[0].star_ids);
        assert_eq!(got5.len(), 2);
        for (g, w) in got5.iter().zip(q5.iter()) {
            assert_eq!(g.star_ids, w.star_ids);
        }
        assert_eq!(got9.len(), 1);
        assert_eq!(got9[0].star_ids, q9[0].star_ids);
    }

    #[test]
    fn duplicate_band_idx_rejected() {
        // Two BandEmit entries with the same band_idx is a programming
        // error and must be rejected before write.
        let q: Vec<Quad> = vec![make_quad(1, 2, 3, 4)];
        let c: Vec<Code> = vec![make_code(0.0)];

        let mut buf = Vec::new();
        let mut cur = Cursor::new(&mut buf);
        let err = write_quad_shard(
            &mut cur,
            7,
            &[
                BandEmit {
                    band_idx: 2,
                    quads: &q,
                    codes: &c,
                },
                BandEmit {
                    band_idx: 2,
                    quads: &q,
                    codes: &c,
                },
            ],
        )
        .expect_err("duplicate band_idx must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn u32_overflow_in_star_ids() {
        // A star_id that doesn't fit in u32 must fail with a clear error
        // rather than silently truncate to a wrong index. This test only
        // makes sense on 64-bit platforms where usize > u32::MAX is
        // representable.
        if std::mem::size_of::<usize>() <= 4 {
            return;
        }
        let big = (u32::MAX as usize) + 1;
        let q = vec![Quad {
            star_ids: [big, 1, 2, 3],
        }];
        let c = vec![make_code(0.0)];

        let mut buf = Vec::new();
        let mut cur = Cursor::new(&mut buf);
        let err = write_quad_shard(
            &mut cur,
            0,
            &[BandEmit {
                band_idx: 0,
                quads: &q,
                codes: &c,
            }],
        )
        .expect_err("oversize star_id must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn quads_codes_length_mismatch_rejected() {
        let q: Vec<Quad> = vec![make_quad(1, 2, 3, 4), make_quad(5, 6, 7, 8)];
        let c: Vec<Code> = vec![make_code(0.0)]; // one fewer than quads
        let mut buf = Vec::new();
        let mut cur = Cursor::new(&mut buf);
        let err = write_quad_shard(
            &mut cur,
            0,
            &[BandEmit {
                band_idx: 0,
                quads: &q,
                codes: &c,
            }],
        )
        .expect_err("mismatch must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    #[test]
    fn header_and_first_band_byte_layout() {
        // Pin the exact 32-byte header + first 24-byte band-table entry.
        // Single band, 1 quad, so the resulting offsets are determinate.
        let q: Vec<Quad> = vec![make_quad(0xDEAD_BEEF, 1, 2, 3)];
        let c: Vec<Code> = vec![[1.0_f64, 2.0, 3.0, 4.0]];
        let bytes = write_to_vec(
            0x0123_4567_89AB_CDEF,
            &[BandEmit {
                band_idx: 7,
                quads: &q,
                codes: &c,
            }],
        );

        // Expected header (32 B):
        let expected_head: [u8; 32] = [
            // magic "ZDCLQUAD"
            b'Z', b'D', b'C', b'L', b'Q', b'U', b'A', b'D', // version = 1 (u32 LE)
            0x01, 0x00, 0x00, 0x00, // reserved
            0x00, 0x00, 0x00, 0x00, // cell_id (u64 LE)
            0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, // n_bands = 1
            0x01, 0x00, 0x00, 0x00, // reserved (pad)
            0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(&bytes[..HEADER_SIZE], &expected_head[..]);

        // First band entry: 24 B at byte 32.
        // band_idx = 7, n_quads = 1.
        // quads_offset = HEADER_SIZE + BAND_ENTRY_SIZE = 32 + 24 = 56
        // codes_offset = 56 + 16 = 72
        let q_off: u64 = (HEADER_SIZE + BAND_ENTRY_SIZE) as u64;
        let c_off: u64 = q_off + QUAD_RECORD_SIZE as u64;
        let mut expected_entry = [0u8; BAND_ENTRY_SIZE];
        expected_entry[0..4].copy_from_slice(&7u32.to_le_bytes());
        expected_entry[4..8].copy_from_slice(&1u32.to_le_bytes());
        expected_entry[8..16].copy_from_slice(&q_off.to_le_bytes());
        expected_entry[16..24].copy_from_slice(&c_off.to_le_bytes());
        assert_eq!(
            &bytes[HEADER_SIZE..HEADER_SIZE + BAND_ENTRY_SIZE],
            &expected_entry[..]
        );
    }

    #[test]
    fn roundtrip_many_bands() {
        // Twelve bands with varying quad counts, including a mid-table
        // empty band, to exercise band-table indexing under load.
        let mut bands_data: Vec<(u32, Vec<Quad>, Vec<Code>)> = Vec::new();
        for k in 0..12u32 {
            let n = if k == 5 { 0 } else { (k as usize) + 1 };
            let qs: Vec<Quad> = (0..n)
                .map(|i| make_quad(k as usize * 100 + i, i + 1, i + 2, i + 3))
                .collect();
            let cs: Vec<Code> = (0..n)
                .map(|i| make_code((k as f64) * 10.0 + i as f64))
                .collect();
            bands_data.push((k, qs, cs));
        }
        let bands: Vec<BandEmit<'_>> = bands_data
            .iter()
            .map(|(idx, qs, cs)| BandEmit {
                band_idx: *idx,
                quads: qs,
                codes: cs,
            })
            .collect();

        let bytes = write_to_vec(0xABCD, &bands);
        let shard = QuadShard::parse(&bytes).expect("parse 12-band shard");
        assert_eq!(shard.n_bands(), 12);

        for (idx, qs, cs) in &bands_data {
            let view = shard.band(*idx).expect("band exists");
            let got_q: Vec<Quad> = view.quads_iter().collect();
            let got_c: Vec<Code> = view.codes_iter().collect();
            assert_eq!(got_q.len(), qs.len(), "band {idx} quad count");
            for (g, w) in got_q.iter().zip(qs.iter()) {
                assert_eq!(g.star_ids, w.star_ids);
            }
            assert_eq!(got_c, *cs, "band {idx} codes");
        }
    }

    #[test]
    fn nan_and_infinity_codes_roundtrip_bitwise() {
        // Codes carry arbitrary f64s including NaN/Inf. The on-disk
        // format is "raw bits" so these must round-trip bit-for-bit
        // (PartialEq on NaN would lie).
        let q: Vec<Quad> = vec![make_quad(1, 2, 3, 4), make_quad(5, 6, 7, 8)];
        let c: Vec<Code> = vec![
            [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1.5],
            // A signaling-NaN-ish payload: pin it via from_bits.
            [
                f64::from_bits(0x7FF0_0000_0000_0001),
                f64::from_bits(0xFFF8_0000_DEAD_BEEF),
                0.0,
                -0.0,
            ],
        ];
        let bytes = write_to_vec(
            1,
            &[BandEmit {
                band_idx: 0,
                quads: &q,
                codes: &c,
            }],
        );
        let shard = QuadShard::parse(&bytes).expect("parse");
        let got: Vec<Code> = shard.band(0).unwrap().codes_iter().collect();
        assert_eq!(got.len(), c.len());
        for (g, w) in got.iter().zip(c.iter()) {
            for (gi, wi) in g.iter().zip(w.iter()) {
                assert_eq!(
                    gi.to_bits(),
                    wi.to_bits(),
                    "f64 bit pattern mismatch in code"
                );
            }
        }
    }

    #[test]
    fn overlapping_band_regions_rejected() {
        // Hand-craft a shard with two bands whose quad blocks overlap.
        // Both bands carry n_quads = 1 (16 B for quads, 32 B for codes),
        // but we point both at the same quads_offset. The codes_offset
        // values are non-overlapping so only the quads regions clash.
        let header_size = HEADER_SIZE;
        let n_bands: u32 = 2;
        let band_table_size = (n_bands as usize) * BAND_ENTRY_SIZE;
        let body_start = header_size + band_table_size;
        // Layout in the body:
        //   [body_start         .. body_start + 16)   shared quads block
        //   [body_start + 16    .. body_start + 48)   band 0 codes
        //   [body_start + 48    .. body_start + 80)   band 1 codes
        let total = body_start + 16 + 32 + 32;
        let mut bytes = vec![0u8; total];

        // Header.
        bytes[0..8].copy_from_slice(QUAD_SHARD_MAGIC);
        bytes[8..12].copy_from_slice(&QUAD_SHARD_VERSION.to_le_bytes());
        bytes[12..16].copy_from_slice(&0u32.to_le_bytes());
        bytes[16..24].copy_from_slice(&0u64.to_le_bytes()); // cell_id
        bytes[24..28].copy_from_slice(&n_bands.to_le_bytes());
        bytes[28..32].copy_from_slice(&0u32.to_le_bytes());

        let q_off = body_start as u64;
        let c0_off = (body_start + 16) as u64;
        let c1_off = (body_start + 48) as u64;

        // Band 0.
        let off0 = header_size;
        bytes[off0..off0 + 4].copy_from_slice(&0u32.to_le_bytes());
        bytes[off0 + 4..off0 + 8].copy_from_slice(&1u32.to_le_bytes());
        bytes[off0 + 8..off0 + 16].copy_from_slice(&q_off.to_le_bytes());
        bytes[off0 + 16..off0 + 24].copy_from_slice(&c0_off.to_le_bytes());

        // Band 1 — shares the same quads_offset, which must trigger the
        // overlap check.
        let off1 = header_size + BAND_ENTRY_SIZE;
        bytes[off1..off1 + 4].copy_from_slice(&1u32.to_le_bytes());
        bytes[off1 + 4..off1 + 8].copy_from_slice(&1u32.to_le_bytes());
        bytes[off1 + 8..off1 + 16].copy_from_slice(&q_off.to_le_bytes());
        bytes[off1 + 16..off1 + 24].copy_from_slice(&c1_off.to_le_bytes());

        let err = QuadShard::parse(&bytes).expect_err("overlapping regions must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("overlap"), "got: {err}");
    }

    #[test]
    fn out_of_order_band_idx_rejected() {
        // Hand-craft a band table whose band_idx values are not
        // strictly ascending. The writer always sorts, so the only way
        // to exercise this path is to mutate the on-disk band table by
        // hand.
        let q0: Vec<Quad> = vec![make_quad(1, 2, 3, 4)];
        let c0: Vec<Code> = vec![make_code(0.0)];
        let q1: Vec<Quad> = vec![make_quad(5, 6, 7, 8)];
        let c1: Vec<Code> = vec![make_code(1.0)];
        let mut bytes = write_to_vec(
            0,
            &[
                BandEmit {
                    band_idx: 1,
                    quads: &q0,
                    codes: &c0,
                },
                BandEmit {
                    band_idx: 2,
                    quads: &q1,
                    codes: &c1,
                },
            ],
        );

        // Swap the band_idx values in the on-disk band table so the
        // table reads [2, 1] — i.e., not strictly ascending.
        let off0 = HEADER_SIZE;
        let off1 = HEADER_SIZE + BAND_ENTRY_SIZE;
        bytes[off0..off0 + 4].copy_from_slice(&2u32.to_le_bytes());
        bytes[off1..off1 + 4].copy_from_slice(&1u32.to_le_bytes());

        let err = QuadShard::parse(&bytes).expect_err("out-of-order band_idx must reject");
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("ascending"), "got: {err}");
    }

    /// Regression for issue #84: shard offsets must be slice-relative, so
    /// a shard written mid-stream still round-trips when its embedded byte
    /// slice is handed to `QuadShard::parse`. Before the fix, the writer
    /// added the underlying stream's `header_start` to every offset in the
    /// band table; reads against the embedded slice would then point past
    /// the slice's end and the parser would reject the shard (or, worse,
    /// silently load the wrong bytes if absolute offsets happened to land
    /// inside the slice).
    #[test]
    fn shard_roundtrips_at_nonzero_stream_offset() {
        // Build the same body of bands as `roundtrip_multi_band`, then
        // stuff the shard between a prefix and a suffix in a single Vec
        // and parse it from the embedded slice.
        let q0: Vec<Quad> = (0..5).map(|i| make_quad(i, i + 1, i + 2, i + 3)).collect();
        let c0: Vec<Code> = (0..5).map(|i| make_code(i as f64 * 0.1)).collect();
        let q1: Vec<Quad> = vec![]; // empty middle band
        let c1: Vec<Code> = vec![];
        let q2: Vec<Quad> = (0..7)
            .map(|i| make_quad(100 + i, 200 + i, 300 + i, 400 + i))
            .collect();
        let c2: Vec<Code> = (0..7).map(|i| make_code(i as f64 + 100.0)).collect();

        let mut buf: Vec<u8> = Vec::new();
        // Prefix: bytes that misalign the shard so it does not start at 0.
        let prefix = b"PREFIX_PREFIX_";
        buf.extend_from_slice(prefix);
        let shard_start = buf.len();
        assert_ne!(shard_start, 0, "test would not catch the bug at offset 0");

        {
            let mut cur = Cursor::new(&mut buf);
            // Seek past the prefix so the cursor's stream_position is
            // shard_start before the writer runs. This is what would have
            // poisoned the absolute offsets in the pre-fix writer.
            cur.set_position(shard_start as u64);
            write_quad_shard(
                &mut cur,
                123,
                &[
                    BandEmit {
                        band_idx: 0,
                        quads: &q0,
                        codes: &c0,
                    },
                    BandEmit {
                        band_idx: 1,
                        quads: &q1,
                        codes: &c1,
                    },
                    BandEmit {
                        band_idx: 2,
                        quads: &q2,
                        codes: &c2,
                    },
                ],
            )
            .expect("write_quad_shard at nonzero offset");
        }
        let shard_end = buf.len();
        // Suffix: trailing bytes that must not be parsed as part of the
        // shard. If the parser reads past the band-data blocks the suffix
        // would be misinterpreted.
        buf.extend_from_slice(b"_SUFFIX_TRAILING_BYTES");

        // Parse the embedded slice — must succeed and round-trip every
        // band's contents identically.
        let embedded = &buf[shard_start..shard_end];
        let shard = QuadShard::parse(embedded).expect("parse embedded shard");
        assert_eq!(shard.cell_id(), 123);
        assert_eq!(shard.n_bands(), 3);

        let got_q0: Vec<Quad> = shard.band(0).unwrap().quads_iter().collect();
        let got_c0: Vec<Code> = shard.band(0).unwrap().codes_iter().collect();
        assert_eq!(got_q0.len(), 5);
        for (g, w) in got_q0.iter().zip(q0.iter()) {
            assert_eq!(g.star_ids, w.star_ids);
        }
        assert_eq!(got_c0, c0);

        // Empty middle band still parses cleanly.
        let v1 = shard.band(1).unwrap();
        assert_eq!(v1.n_quads(), 0);
        assert_eq!(v1.quads_iter().count(), 0);
        assert_eq!(v1.codes_iter().count(), 0);

        let got_q2: Vec<Quad> = shard.band(2).unwrap().quads_iter().collect();
        let got_c2: Vec<Code> = shard.band(2).unwrap().codes_iter().collect();
        assert_eq!(got_q2.len(), 7);
        for (g, w) in got_q2.iter().zip(q2.iter()) {
            assert_eq!(g.star_ids, w.star_ids);
        }
        assert_eq!(got_c2, c2);

        // The prefix and suffix must be preserved in the surrounding
        // buffer — the writer must not have stomped over them.
        assert_eq!(&buf[..shard_start], prefix);
        assert_eq!(&buf[shard_end..], b"_SUFFIX_TRAILING_BYTES");

        // Independent sanity check: the offsets in the band table are
        // slice-relative, so they must all be < shard_end - shard_start
        // (the embedded slice's length), and each populated band's offset
        // must be >= HEADER_SIZE + n_bands * BAND_ENTRY_SIZE.
        let shard_len = (shard_end - shard_start) as u64;
        let band_table_end = (HEADER_SIZE + 3 * BAND_ENTRY_SIZE) as u64;
        for entry in shard.bands() {
            if entry.n_quads > 0 {
                assert!(
                    entry.quads_offset >= band_table_end,
                    "quads_offset {} overlaps band table (ends at {})",
                    entry.quads_offset,
                    band_table_end
                );
                assert!(
                    entry.quads_offset < shard_len,
                    "quads_offset {} not slice-relative (shard_len = {})",
                    entry.quads_offset,
                    shard_len
                );
                assert!(
                    entry.codes_offset < shard_len,
                    "codes_offset {} not slice-relative (shard_len = {})",
                    entry.codes_offset,
                    shard_len
                );
            }
        }
    }
}
