//! Flat sorted-by-source_id binary sidecar for Gaia astrometry.
//!
//! The sidecar sits next to a `.zdcl` index file (`my_index.zdcl.gaia`) and
//! carries the refinement-minimal Gaia astrometry subset needed to compute
//! apparent-place positions. Access pattern is point lookups by Gaia
//! `source_id`, optimized for batch FOV queries via galloping search from
//! the previous hit.
//!
//! See issue #45 for the full rationale; briefly: Gaia's `source_id` has
//! HEALPix locality baked in, so sorting records by source_id gives free
//! spatial clustering on disk, and compression doesn't help enough to
//! justify parquet's decompression cost on the query path.

use std::collections::BinaryHeap;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

const MAGIC: &[u8; 8] = b"ZDCLGAIA";
const VERSION: u32 = 1;
const HEADER_SIZE: usize = 64;
pub const RECORD_SIZE: u32 = 88;
pub const DEFAULT_PIVOT_STRIDE: u32 = 4096;

/// One row of the refinement sidecar. `repr(C)` so the on-disk layout
/// matches the in-memory struct byte-for-byte; this lets us cast slices
/// of the mmap directly to `&[SidecarRecord]` without deserialization.
///
/// Unpublished fields (no parallax solution, no radial velocity) are
/// stored as `f64::NAN` so the reader doesn't need a validity bitmap.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SidecarRecord {
    pub source_id: u64,
    pub ref_epoch: f64,
    pub ra: f64,
    pub dec: f64,
    pub pmra: f64,
    pub pmdec: f64,
    pub parallax: f64,
    pub radial_velocity: f64,
    pub sigma_ra: f32,
    pub sigma_dec: f32,
    pub sigma_pmra: f32,
    pub sigma_pmdec: f32,
    pub sigma_parallax: f32,
    pub flags: u32,
}

// Compile-time assertion that the Rust struct is exactly 88 bytes.
const _: () = assert!(std::mem::size_of::<SidecarRecord>() == RECORD_SIZE as usize);

/// Write a sidecar file. Records are sorted by `source_id` before being
/// written, so the input iterator does not need to be pre-sorted.
///
/// `pivot_stride` controls the in-file pivot table density. `DEFAULT_PIVOT_STRIDE`
/// (4096) gives ~7.5k pivots for a 31M-record sidecar — about 60 KB of
/// in-memory index, which fits comfortably in L2.
pub fn write_sidecar<I>(path: &Path, records: I, pivot_stride: u32) -> io::Result<()>
where
    I: IntoIterator<Item = SidecarRecord>,
{
    assert!(pivot_stride > 0, "pivot_stride must be positive");

    let mut records: Vec<SidecarRecord> = records.into_iter().collect();
    records.sort_by_key(|r| r.source_id);

    let n_records = records.len() as u64;
    let pivot_count = if records.is_empty() {
        0
    } else {
        ((records.len() - 1) / pivot_stride as usize + 1) as u32
    };

    let file = File::create(path)?;
    let mut w = BufWriter::new(file);

    // Header.
    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&RECORD_SIZE.to_le_bytes())?;
    w.write_all(&n_records.to_le_bytes())?;
    w.write_all(&pivot_stride.to_le_bytes())?;
    w.write_all(&pivot_count.to_le_bytes())?;
    w.write_all(&[0u8; 32])?; // reserved

    // Pivot table: source_id at each k*pivot_stride.
    for k in 0..pivot_count as usize {
        let idx = k * pivot_stride as usize;
        w.write_all(&records[idx].source_id.to_le_bytes())?;
    }

    // Pad to the records-start offset, which must be 8-aligned.
    let written_so_far = HEADER_SIZE + (pivot_count as usize) * 8;
    let records_start = (written_so_far + 7) & !7;
    let pad = records_start - written_so_far;
    if pad > 0 {
        w.write_all(&vec![0u8; pad])?;
    }

    // Records. Safe to cast because SidecarRecord is repr(C) with no padding
    // gaps (verified by the size_of assertion above).
    let records_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            records.as_ptr() as *const u8,
            records.len() * RECORD_SIZE as usize,
        )
    };
    w.write_all(records_bytes)?;

    w.flush()?;
    Ok(())
}

/// Streaming sidecar writer with bounded peak memory.
///
/// Each call to [`SidecarStreamWriter::append_chunk`] sorts the chunk
/// internally by `source_id` and dumps it to a numbered temp file under
/// the scratch directory. [`SidecarStreamWriter::finalize`] then performs
/// an external k-way merge across the temp files, building the pivot
/// table on the fly, and writes the final sorted-and-pivoted sidecar.
///
/// Peak memory is bounded by the largest chunk (callers typically size
/// chunks to one cell's worth of records — tens of MB at HEALPix level
/// 5). Disk usage is bounded by the total record count: each record is
/// written once to a temp file and once to the final output.
///
/// Crash semantics: temp files live under `scratch_dir`. If the writer
/// is dropped before `finalize`, partial chunks remain on disk so a
/// caller-managed manifest can resume from the next chunk. `finalize`
/// removes all temp files only after the final atomic rename succeeds.
pub struct SidecarStreamWriter {
    scratch_dir: PathBuf,
    /// (chunk_path, n_records) — appended in the order chunks were
    /// committed. Order doesn't matter for the merge (records are sorted
    /// by source_id inside each file), but stable order helps debugging.
    chunks: Vec<(PathBuf, u64)>,
    next_chunk_idx: u64,
}

impl SidecarStreamWriter {
    /// Create a new streaming writer. `scratch_dir` will be created if
    /// missing; chunk temp files are written there as
    /// `chunk_NNNN.sidecar-tmp`.
    ///
    /// The writer is independent of the final output path until
    /// [`Self::finalize`] is called, which lets the caller pick a
    /// final destination only after all chunks have been committed
    /// successfully (a useful split for crash-resume flows).
    pub fn new(scratch_dir: impl Into<PathBuf>) -> io::Result<Self> {
        let scratch_dir = scratch_dir.into();
        std::fs::create_dir_all(&scratch_dir)?;
        Ok(Self {
            scratch_dir,
            chunks: Vec::new(),
            next_chunk_idx: 0,
        })
    }

    /// Resume a writer from a scratch directory previously populated by
    /// `append_chunk` calls. The discovered chunks contribute to the
    /// final merge but are not re-sorted.
    ///
    /// Resume scans the directory for `chunk_NNNN.sidecar-tmp` files,
    /// reads each one's record count from its header, and seeds the
    /// internal chunk list. Chunks the caller already committed in a
    /// prior run will round-trip into the final sidecar without rework.
    pub fn resume(scratch_dir: impl Into<PathBuf>) -> io::Result<Self> {
        let scratch_dir = scratch_dir.into();
        std::fs::create_dir_all(&scratch_dir)?;

        let mut found: Vec<(u64, PathBuf, u64)> = Vec::new();
        for entry in std::fs::read_dir(&scratch_dir)? {
            let entry = entry?;
            let path = entry.path();
            if !path.is_file() {
                continue;
            }
            let name = match path.file_name().and_then(|n| n.to_str()) {
                Some(n) => n,
                None => continue,
            };
            let idx = match name
                .strip_prefix("chunk_")
                .and_then(|s| s.strip_suffix(".sidecar-tmp"))
                .and_then(|s| s.parse::<u64>().ok())
            {
                Some(i) => i,
                None => continue,
            };
            let n_records = read_chunk_header(&path)?;
            found.push((idx, path, n_records));
        }
        found.sort_by_key(|(i, _, _)| *i);

        let next_chunk_idx = found.last().map(|(i, _, _)| i + 1).unwrap_or(0);
        let chunks: Vec<(PathBuf, u64)> = found.into_iter().map(|(_, p, n)| (p, n)).collect();

        Ok(Self {
            scratch_dir,
            chunks,
            next_chunk_idx,
        })
    }

    /// Write a chunk of records to a numbered temp file, sorted by
    /// `source_id`. Returns the chunk's index and on-disk path.
    ///
    /// Empty chunks are accepted but produce no temp file — the caller
    /// can use the returned index to drive a manifest without special-casing.
    pub fn append_chunk(&mut self, mut records: Vec<SidecarRecord>) -> io::Result<u64> {
        let idx = self.next_chunk_idx;
        self.next_chunk_idx += 1;

        if records.is_empty() {
            return Ok(idx);
        }

        records.sort_by_key(|r| r.source_id);

        let path = self.chunk_path(idx);
        let tmp_path = {
            let mut s = path.as_os_str().to_owned();
            s.push(".partial");
            PathBuf::from(s)
        };

        let n_records = records.len() as u64;
        {
            let file = File::create(&tmp_path)?;
            let mut w = BufWriter::new(file);
            w.write_all(&n_records.to_le_bytes())?;
            // Records bytes (88 B each, repr(C)).
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    records.as_ptr() as *const u8,
                    records.len() * RECORD_SIZE as usize,
                )
            };
            w.write_all(bytes)?;
            w.flush()?;
            w.get_ref().sync_all()?;
        }
        std::fs::rename(&tmp_path, &path)?;

        self.chunks.push((path, n_records));
        Ok(idx)
    }

    /// Number of chunks committed so far.
    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Total records committed across all chunks (sum of chunk sizes).
    pub fn total_records(&self) -> u64 {
        self.chunks.iter().map(|(_, n)| *n).sum()
    }

    /// Merge all committed chunks into `final_path`, build the pivot
    /// table, and remove the scratch directory's temp files. The final
    /// file is written atomically (tmp file + rename).
    ///
    /// Dropping the writer without calling `finalize` deliberately
    /// leaves the chunk temp files in place — the on-disk chunks plus
    /// a sibling [`crate::index::build_manifest::BuildManifest`] are
    /// the durable resume state.
    ///
    /// `pivot_stride` controls the in-file pivot density; pass
    /// [`DEFAULT_PIVOT_STRIDE`] for the standard 4096.
    pub fn finalize(self, final_path: &Path, pivot_stride: u32) -> io::Result<()> {
        assert!(pivot_stride > 0, "pivot_stride must be positive");

        let n_records: u64 = self.chunks.iter().map(|(_, n)| *n).sum();
        let pivot_count = if n_records == 0 {
            0u32
        } else {
            ((n_records - 1) / pivot_stride as u64 + 1) as u32
        };

        // Final file is written to a sibling .partial first so a crash
        // mid-write doesn't leave a half-formed `.zdcl.gaia` in place.
        let tmp_final = match final_path.extension() {
            Some(ext) => {
                let mut s = ext.to_owned();
                s.push(".partial");
                final_path.with_extension(s)
            }
            None => final_path.with_extension("partial"),
        };

        // Open chunk readers; collect first-record peeks for the heap.
        let mut readers: Vec<ChunkReader> = Vec::with_capacity(self.chunks.len());
        for (path, n) in &self.chunks {
            if *n == 0 {
                continue;
            }
            readers.push(ChunkReader::open(path)?);
        }

        // Min-heap keyed by current head record's source_id.
        let mut heap: BinaryHeap<HeapEntry> = BinaryHeap::with_capacity(readers.len());
        for (i, r) in readers.iter_mut().enumerate() {
            if let Some(rec) = r.next()? {
                heap.push(HeapEntry {
                    source_id: rec.source_id,
                    reader_idx: i,
                    record: rec,
                });
            }
        }

        let file = File::create(&tmp_final)?;
        let mut w = BufWriter::new(file);

        // --- Header ---
        w.write_all(MAGIC)?;
        w.write_all(&VERSION.to_le_bytes())?;
        w.write_all(&RECORD_SIZE.to_le_bytes())?;
        w.write_all(&n_records.to_le_bytes())?;
        w.write_all(&pivot_stride.to_le_bytes())?;
        w.write_all(&pivot_count.to_le_bytes())?;
        w.write_all(&[0u8; 32])?;

        // --- Reserve pivot table region; we'll overwrite it after the
        // merge pass once we know each pivot's source_id. We can't emit
        // pivots inline because the file layout puts the pivot table
        // BEFORE the records.
        let pivots_offset = HEADER_SIZE as u64;
        let pivots_size = (pivot_count as usize) * 8;
        if pivots_size > 0 {
            w.write_all(&vec![0u8; pivots_size])?;
        }

        let pre_padding = HEADER_SIZE + pivots_size;
        let records_offset = (pre_padding + 7) & !7;
        let pad = records_offset - pre_padding;
        if pad > 0 {
            w.write_all(&vec![0u8; pad])?;
        }

        // --- Merge pass ---
        let mut pivots: Vec<u64> = Vec::with_capacity(pivot_count as usize);
        let stride = pivot_stride as u64;
        let mut written: u64 = 0;
        let mut prev_source_id: Option<u64> = None;
        while let Some(top) = heap.pop() {
            // Validate sort: streaming merge requires monotonically
            // non-decreasing source_id across the merged stream.
            if let Some(prev) = prev_source_id
                && top.source_id < prev
            {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    format!(
                        "merge stream out of order: prev={prev} next={} (chunks must be sorted)",
                        top.source_id,
                    ),
                ));
            }

            if written.is_multiple_of(stride) && pivots.len() < pivot_count as usize {
                pivots.push(top.source_id);
            }

            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(
                    &top.record as *const SidecarRecord as *const u8,
                    RECORD_SIZE as usize,
                )
            };
            w.write_all(bytes)?;
            written += 1;
            prev_source_id = Some(top.source_id);

            // Refill from this reader.
            let r = &mut readers[top.reader_idx];
            if let Some(rec) = r.next()? {
                heap.push(HeapEntry {
                    source_id: rec.source_id,
                    reader_idx: top.reader_idx,
                    record: rec,
                });
            }
        }

        if written != n_records {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "merged {written} records but chunk headers reported {n_records}; \
                     a chunk file may be truncated"
                ),
            ));
        }

        w.flush()?;

        // Rewind, overwrite the pivot table region with real pivots, and
        // sync. `BufWriter::seek` flushes its buffer first.
        if pivot_count > 0 {
            assert_eq!(pivots.len(), pivot_count as usize);
            w.seek(SeekFrom::Start(pivots_offset))?;
            for p in &pivots {
                w.write_all(&p.to_le_bytes())?;
            }
            w.flush()?;
        }
        w.get_ref().sync_all()?;
        drop(w);

        std::fs::rename(&tmp_final, final_path)?;

        // Clean up chunk temp files; only after the rename succeeds so
        // a crashed finalize stays resumable.
        for (path, _) in &self.chunks {
            let _ = std::fs::remove_file(path);
        }
        // Best-effort: remove scratch dir if empty. Caller may share it
        // with a sibling manifest, in which case rmdir is harmless on
        // EEXIST/ENOTEMPTY.
        let _ = std::fs::remove_dir(&self.scratch_dir);

        Ok(())
    }

    fn chunk_path(&self, idx: u64) -> PathBuf {
        self.scratch_dir
            .join(format!("chunk_{idx:04}.sidecar-tmp"))
    }
}

/// Read the n_records header from a chunk temp file without loading
/// records.
fn read_chunk_header(path: &Path) -> io::Result<u64> {
    let mut f = File::open(path)?;
    let mut buf = [0u8; 8];
    f.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

/// Buffered reader over a single sidecar chunk temp file. Yields
/// records in the file's stored (already-sorted) order.
struct ChunkReader {
    inner: BufReader<File>,
    remaining: u64,
}

impl ChunkReader {
    fn open(path: &Path) -> io::Result<Self> {
        let mut f = OpenOptions::new().read(true).open(path)?;
        let mut hdr = [0u8; 8];
        f.read_exact(&mut hdr)?;
        let remaining = u64::from_le_bytes(hdr);
        Ok(Self {
            inner: BufReader::new(f),
            remaining,
        })
    }

    fn next(&mut self) -> io::Result<Option<SidecarRecord>> {
        if self.remaining == 0 {
            return Ok(None);
        }
        let mut buf = [0u8; RECORD_SIZE as usize];
        self.inner.read_exact(&mut buf)?;
        self.remaining -= 1;
        // Safe: SidecarRecord is repr(C), the buffer is RECORD_SIZE
        // bytes, and we copy out (no aliasing the buffer).
        let rec = unsafe { std::ptr::read_unaligned(buf.as_ptr() as *const SidecarRecord) };
        Ok(Some(rec))
    }
}

/// Heap entry for the k-way merge. `BinaryHeap` is a max-heap, so we
/// invert `cmp` to get min-on-source_id behavior.
struct HeapEntry {
    source_id: u64,
    reader_idx: usize,
    record: SidecarRecord,
}

impl PartialEq for HeapEntry {
    fn eq(&self, other: &Self) -> bool {
        self.source_id == other.source_id && self.reader_idx == other.reader_idx
    }
}
impl Eq for HeapEntry {}
impl PartialOrd for HeapEntry {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for HeapEntry {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Min-heap on source_id, ties broken by reader_idx for
        // deterministic output across runs.
        other
            .source_id
            .cmp(&self.source_id)
            .then(other.reader_idx.cmp(&self.reader_idx))
    }
}

/// Mmap-backed random-access reader over a sidecar file.
///
/// Thread-safe for concurrent reads; the mmap and pivot table are immutable
/// after `open`.
pub struct SidecarReader {
    mmap: Mmap,
    n_records: usize,
    pivot_stride: u32,
    /// Pivot table kept in memory (small — ~60 KB for a 31M-row sidecar).
    pivots: Vec<u64>,
    /// Byte offset in the mmap where records[0] begins.
    records_offset: usize,
}

impl SidecarReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };

        if mmap.len() < HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sidecar file smaller than header",
            ));
        }
        if &mmap[0..8] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sidecar magic mismatch",
            ));
        }
        let version = u32::from_le_bytes(mmap[8..12].try_into().unwrap());
        let record_size = u32::from_le_bytes(mmap[12..16].try_into().unwrap());
        if version != VERSION || record_size != RECORD_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported sidecar: version={version} record_size={record_size}"),
            ));
        }
        let n_records = u64::from_le_bytes(mmap[16..24].try_into().unwrap()) as usize;
        let pivot_stride = u32::from_le_bytes(mmap[24..28].try_into().unwrap());
        let pivot_count = u32::from_le_bytes(mmap[28..32].try_into().unwrap()) as usize;

        // Parse pivot table.
        let pivot_bytes_start = HEADER_SIZE;
        let pivot_bytes_end = pivot_bytes_start + pivot_count * 8;
        if mmap.len() < pivot_bytes_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sidecar truncated in pivot table",
            ));
        }
        let mut pivots = Vec::with_capacity(pivot_count);
        for k in 0..pivot_count {
            let off = pivot_bytes_start + k * 8;
            pivots.push(u64::from_le_bytes(mmap[off..off + 8].try_into().unwrap()));
        }

        // Records start at the next 8-aligned offset after the pivot table.
        let records_offset = (pivot_bytes_end + 7) & !7;
        let expected_end = records_offset + n_records * RECORD_SIZE as usize;
        if mmap.len() < expected_end {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "sidecar truncated in records",
            ));
        }

        Ok(Self {
            mmap,
            n_records,
            pivot_stride,
            pivots,
            records_offset,
        })
    }

    pub fn len(&self) -> usize {
        self.n_records
    }

    pub fn is_empty(&self) -> bool {
        self.n_records == 0
    }

    /// Byte slice that covers `n_records * RECORD_SIZE` bytes of records.
    fn records_bytes(&self) -> &[u8] {
        &self.mmap[self.records_offset..self.records_offset + self.n_records * RECORD_SIZE as usize]
    }

    /// Borrow a single record by its index in the sorted records array.
    fn record_at(&self, idx: usize) -> &SidecarRecord {
        debug_assert!(idx < self.n_records);
        let offset = idx * RECORD_SIZE as usize;
        let bytes = &self.records_bytes()[offset..offset + RECORD_SIZE as usize];
        // Safe: mmap is 8-aligned, records_offset is 8-aligned, and
        // SidecarRecord is repr(C) with native alignment.
        unsafe { &*(bytes.as_ptr() as *const SidecarRecord) }
    }

    /// Point lookup. O(log n).
    pub fn get(&self, source_id: u64) -> Option<&SidecarRecord> {
        if self.n_records == 0 {
            return None;
        }
        let (lo, hi) = self.pivot_window(source_id);
        self.binsearch_range(lo, hi, source_id)
    }

    /// Bulk lookup. Sorts the input internally (so callers may pass an
    /// arbitrary order) and uses galloping search from the prior hit for
    /// amortized linear-time scanning of spatially clustered queries.
    ///
    /// Returns one `Option<SidecarRecord>` per input source_id, in the
    /// same order as the input (not sorted order).
    pub fn get_many(&self, source_ids: &[u64]) -> Vec<Option<SidecarRecord>> {
        let mut indexed: Vec<(usize, u64)> = source_ids.iter().copied().enumerate().collect();
        indexed.sort_by_key(|&(_, id)| id);

        let mut out = vec![None; source_ids.len()];
        if self.n_records == 0 {
            return out;
        }

        // Galloping search from the previous hit's index.
        let mut last_idx: usize = 0;
        for &(orig_i, id) in &indexed {
            match self.gallop_find(last_idx, id) {
                Some(idx) => {
                    out[orig_i] = Some(*self.record_at(idx));
                    last_idx = idx;
                }
                None => {
                    // id not present; last_idx unchanged.
                }
            }
        }
        out
    }

    /// Locate the window [lo, hi) of record indices that could contain
    /// `source_id`, using the pivot table.
    fn pivot_window(&self, source_id: u64) -> (usize, usize) {
        if self.pivots.is_empty() {
            return (0, self.n_records);
        }
        // partition_point: first pivot whose id > source_id.
        let p = self.pivots.partition_point(|&pid| pid <= source_id);
        let lo = if p == 0 {
            0
        } else {
            (p - 1) * self.pivot_stride as usize
        };
        let hi = (p * self.pivot_stride as usize).min(self.n_records);
        (lo, hi)
    }

    /// Binsearch for `source_id` within [lo, hi). Returns the record index
    /// if found, else None.
    fn binsearch_range(
        &self,
        mut lo: usize,
        mut hi: usize,
        source_id: u64,
    ) -> Option<&SidecarRecord> {
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let mid_id = self.record_at(mid).source_id;
            match mid_id.cmp(&source_id) {
                std::cmp::Ordering::Equal => return Some(self.record_at(mid)),
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }

    /// Galloping search starting from `start` for a record with the given
    /// `source_id`. Returns the index if found. Backward galloping also
    /// supported for occasional unsorted input.
    fn gallop_find(&self, start: usize, source_id: u64) -> Option<usize> {
        if self.n_records == 0 {
            return None;
        }
        let start = start.min(self.n_records.saturating_sub(1));
        let start_id = self.record_at(start).source_id;

        if start_id == source_id {
            return Some(start);
        }

        if source_id > start_id {
            // Gallop forward.
            let mut step = 1usize;
            let mut prev = start;
            loop {
                let next = (start + step).min(self.n_records);
                if next >= self.n_records {
                    // Target is in [prev+1, n_records).
                    let found_idx = self.binsearch_idx(prev + 1, self.n_records, source_id);
                    return found_idx;
                }
                let id = self.record_at(next).source_id;
                if id == source_id {
                    return Some(next);
                }
                if id > source_id {
                    // Target, if present, is in [prev+1, next).
                    return self.binsearch_idx(prev + 1, next, source_id);
                }
                prev = next;
                step *= 2;
            }
        } else {
            // Gallop backward.
            let mut step = 1usize;
            let mut prev = start;
            loop {
                let next = start.saturating_sub(step);
                let id = self.record_at(next).source_id;
                if id == source_id {
                    return Some(next);
                }
                if id < source_id {
                    // Target, if present, is in [next+1, prev).
                    return self.binsearch_idx(next + 1, prev, source_id);
                }
                if next == 0 {
                    // Target, if present, is in [0, prev).
                    return self.binsearch_idx(0, prev, source_id);
                }
                prev = next;
                step *= 2;
            }
        }
    }

    /// Binsearch within [lo, hi) returning the matched index, else None.
    fn binsearch_idx(&self, mut lo: usize, mut hi: usize, source_id: u64) -> Option<usize> {
        while lo < hi {
            let mid = lo + (hi - lo) / 2;
            let mid_id = self.record_at(mid).source_id;
            match mid_id.cmp(&source_id) {
                std::cmp::Ordering::Equal => return Some(mid),
                std::cmp::Ordering::Less => lo = mid + 1,
                std::cmp::Ordering::Greater => hi = mid,
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    fn rec(source_id: u64, ra: f64, dec: f64) -> SidecarRecord {
        SidecarRecord {
            source_id,
            ref_epoch: 2016.0,
            ra,
            dec,
            pmra: 1.5,
            pmdec: -0.7,
            parallax: 5.25,
            radial_velocity: 12.0,
            sigma_ra: 0.1,
            sigma_dec: 0.1,
            sigma_pmra: 0.01,
            sigma_pmdec: 0.01,
            sigma_parallax: 0.02,
            flags: 0,
        }
    }

    fn tmp_path(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "zodiacal-sidecar-test-{}-{}.gaia",
            name,
            std::process::id()
        ));
        p
    }

    #[test]
    fn roundtrip_small() {
        let path = tmp_path("rt");
        let records: Vec<_> = (0..50u64)
            .map(|i| rec(1000 + i * 7, i as f64, -(i as f64)))
            .collect();
        // Pass in a shuffled order — writer should sort.
        let mut shuffled = records.clone();
        shuffled.reverse();
        write_sidecar(&path, shuffled, 8).unwrap();

        let reader = SidecarReader::open(&path).unwrap();
        assert_eq!(reader.len(), 50);

        for r in &records {
            let got = reader.get(r.source_id).expect("should be present");
            assert_eq!(got.source_id, r.source_id);
            assert_eq!(got.ra, r.ra);
            assert_eq!(got.dec, r.dec);
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn get_many_matches_get() {
        let path = tmp_path("gm");
        let records: Vec<_> = (0..200u64)
            .map(|i| rec(10_000 + i * 3, i as f64 * 0.1, i as f64 * -0.05))
            .collect();
        write_sidecar(&path, records.clone(), 16).unwrap();
        let reader = SidecarReader::open(&path).unwrap();

        // Shuffled subset of ids, some not present.
        let queries = vec![
            10_015, 10_303, 10_000, 99_999, 10_597, 10_006, 10_003, 12_345, 10_594,
        ];

        let bulk = reader.get_many(&queries);
        for (q, got) in queries.iter().zip(bulk.iter()) {
            let single = reader.get(*q).cloned();
            assert_eq!(single, *got, "mismatch at source_id {q}");
        }

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn nan_fields_preserved() {
        let path = tmp_path("nan");
        let mut r = rec(42, 1.0, 2.0);
        r.parallax = f64::NAN;
        r.radial_velocity = f64::NAN;
        write_sidecar(&path, vec![r], 1).unwrap();
        let reader = SidecarReader::open(&path).unwrap();
        let got = reader.get(42).unwrap();
        assert!(got.parallax.is_nan());
        assert!(got.radial_velocity.is_nan());

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn empty_sidecar_opens() {
        let path = tmp_path("empty");
        write_sidecar(&path, std::iter::empty::<SidecarRecord>(), 1).unwrap();
        let reader = SidecarReader::open(&path).unwrap();
        assert_eq!(reader.len(), 0);
        assert!(reader.get(42).is_none());
        assert!(reader.get_many(&[1, 2, 3]).iter().all(|x| x.is_none()));

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn bad_magic_rejected() {
        let path = tmp_path("bad");
        let mut f = std::fs::File::create(&path).unwrap();
        f.write_all(b"NOPE----").unwrap();
        f.write_all(&[0u8; 64]).unwrap();
        drop(f);
        let err = match SidecarReader::open(&path) {
            Ok(_) => panic!("bad-magic file should not open"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);

        let _ = std::fs::remove_file(&path);
    }

    #[test]
    fn clustered_queries_via_galloping() {
        // 10k records with clustered id queries — exercises galloping forward
        // across many matches without falling back to full-range binsearch.
        let path = tmp_path("gallop");
        let records: Vec<_> = (0..10_000u64)
            .map(|i| rec(100_000 + i * 2, 0.0, 0.0))
            .collect();
        write_sidecar(&path, records, 256).unwrap();
        let reader = SidecarReader::open(&path).unwrap();

        // Query a contiguous cluster of 500 ids, shuffled.
        let mut queries: Vec<u64> = (0..500u64).map(|i| 100_000 + (5000 + i) * 2).collect();
        queries.reverse();
        let results = reader.get_many(&queries);
        assert_eq!(results.len(), 500);
        assert!(results.iter().all(|r| r.is_some()));

        let _ = std::fs::remove_file(&path);
    }

    fn tmp_dir(name: &str) -> std::path::PathBuf {
        let mut p = std::env::temp_dir();
        p.push(format!(
            "zodiacal-sidecar-stream-{}-{}-{}",
            name,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0),
        ));
        p
    }

    fn collect_records(reader: &SidecarReader) -> Vec<SidecarRecord> {
        (0..reader.len()).map(|i| *reader.record_at(i)).collect()
    }

    #[test]
    fn stream_writer_roundtrip_matches_in_ram_writer() {
        // Same records via both writers should produce byte-identical
        // contents, since the on-disk format is fixed and both paths
        // sort by source_id deterministically.
        let scratch = tmp_dir("rt-scratch");
        let final_stream = tmp_dir("rt-final-stream");
        let final_inram = tmp_dir("rt-final-inram");

        let mut records: Vec<SidecarRecord> = (0..1000u64)
            .map(|i| rec(7919 * (i + 1), i as f64 * 0.001, -(i as f64) * 0.0005))
            .collect();
        // Shuffle to exercise the writer's sort.
        records.swap(0, 999);
        records.swap(123, 456);

        // In-RAM baseline.
        write_sidecar(&final_inram, records.clone(), 64).unwrap();

        // Streaming writer: chunk into 7 pieces of varying sizes, with
        // overlapping source_id ranges to mimic cell ordering.
        let mut writer = SidecarStreamWriter::new(&scratch).unwrap();
        let chunks: Vec<Vec<SidecarRecord>> = (0..7)
            .map(|c| {
                records
                    .iter()
                    .enumerate()
                    .filter(|(i, _)| i % 7 == c)
                    .map(|(_, r)| *r)
                    .collect()
            })
            .collect();
        for ch in chunks {
            writer.append_chunk(ch).unwrap();
        }
        assert_eq!(writer.total_records(), 1000);
        writer.finalize(&final_stream, 64).unwrap();

        // Compare byte streams.
        let a = std::fs::read(&final_inram).unwrap();
        let b = std::fs::read(&final_stream).unwrap();
        assert_eq!(a, b, "stream and in-RAM writers diverged");

        // Reader-level equivalence as a belt-and-suspenders check.
        let r_in = SidecarReader::open(&final_inram).unwrap();
        let r_st = SidecarReader::open(&final_stream).unwrap();
        assert_eq!(collect_records(&r_in), collect_records(&r_st));

        let _ = std::fs::remove_file(&final_inram);
        let _ = std::fs::remove_file(&final_stream);
    }

    #[test]
    fn stream_writer_empty_input() {
        let scratch = tmp_dir("empty-scratch");
        let final_path = tmp_dir("empty-final");

        let writer = SidecarStreamWriter::new(&scratch).unwrap();
        writer.finalize(&final_path, 64).unwrap();

        let reader = SidecarReader::open(&final_path).unwrap();
        assert_eq!(reader.len(), 0);

        let _ = std::fs::remove_file(&final_path);
    }

    #[test]
    fn stream_writer_resume_replays_committed_chunks() {
        // Simulate a crashed builder that committed some chunks but
        // never finalized; resume should pick them up and merge them
        // alongside new chunks.
        let scratch = tmp_dir("resume-scratch");
        let final_path = tmp_dir("resume-final");

        let all_records: Vec<SidecarRecord> = (0..500u64).map(|i| rec(i * 11 + 3, 0.1, 0.2)).collect();
        let split = 300;

        // Phase 1: writer crashes after committing the first 3 chunks.
        {
            let mut writer = SidecarStreamWriter::new(&scratch).unwrap();
            for c in 0..3 {
                let lo = c * 100;
                let hi = lo + 100;
                writer.append_chunk(all_records[lo..hi].to_vec()).unwrap();
            }
            assert_eq!(writer.chunk_count(), 3);
            // Drop without finalize — chunks remain on disk.
        }

        // Phase 2: resume, add the remaining chunks, finalize.
        {
            let mut writer = SidecarStreamWriter::resume(&scratch).unwrap();
            assert_eq!(writer.chunk_count(), 3);
            assert_eq!(writer.total_records(), split as u64);
            for c in 3..5 {
                let lo = c * 100;
                let hi = lo + 100;
                writer.append_chunk(all_records[lo..hi].to_vec()).unwrap();
            }
            writer.finalize(&final_path, 32).unwrap();
        }

        // Final file should contain every record exactly once.
        let reader = SidecarReader::open(&final_path).unwrap();
        assert_eq!(reader.len(), all_records.len());
        for r in &all_records {
            let got = reader.get(r.source_id).expect("missing record after resume");
            assert_eq!(got.source_id, r.source_id);
        }

        let _ = std::fs::remove_file(&final_path);
    }

    #[test]
    fn stream_writer_skips_empty_chunks() {
        // A cell with zero stars (e.g. a high-galactic-latitude void at
        // a very tight mag cap) should produce a noop append, not a
        // zero-byte temp file that confuses the merge.
        let scratch = tmp_dir("skip-scratch");
        let final_path = tmp_dir("skip-final");

        let mut writer = SidecarStreamWriter::new(&scratch).unwrap();
        writer.append_chunk(Vec::new()).unwrap();
        writer.append_chunk(vec![rec(42, 0.0, 0.0)]).unwrap();
        writer.append_chunk(Vec::new()).unwrap();
        writer.append_chunk(vec![rec(43, 0.0, 0.0)]).unwrap();
        writer.finalize(&final_path, 32).unwrap();

        let reader = SidecarReader::open(&final_path).unwrap();
        assert_eq!(reader.len(), 2);
        assert!(reader.get(42).is_some());
        assert!(reader.get(43).is_some());

        let _ = std::fs::remove_file(&final_path);
    }
}
