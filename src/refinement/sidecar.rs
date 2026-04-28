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

use std::fs::File;
use std::io::{self, BufWriter, Write};
use std::path::Path;

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
}
