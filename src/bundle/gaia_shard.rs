//! Per-cell `.zga` Gaia source-record shard.
//!
//! Each cell of a `.zdcl.bundle` carries one `gaia/cell_NNNNN.zga` file holding
//! 104-byte fixed-width [`GaiaRecord`]s sorted ascending by `source_id`. The
//! header format and record layout are specified in `docs/bundle-format.md`
//! under "`gaia/cell_NNNNN.zga` — Gaia source records, sorted by source_id,
//! binsearchable".
//!
//! On little-endian targets (the only mainstream Rust targets), the in-memory
//! `repr(C)` layout of [`GaiaRecord`] is bit-identical to the on-disk layout,
//! so the reader returns a zero-copy `&[GaiaRecord]` via
//! [`bytemuck::cast_slice`] and the writer emits records via
//! [`bytemuck::cast_slice`] in one write.

use std::io::{self, Write};

use bytemuck::{Pod, Zeroable};

#[cfg(target_endian = "big")]
compile_error!(
    "GaiaRecord on-disk layout assumes little-endian; big-endian targets are not supported."
);

/// Magic bytes at the start of every `.zga` file.
pub const MAGIC: &[u8; 8] = b"ZDCLGAIA";

/// On-disk format version.
pub const VERSION: u32 = 1;

/// Wire size of [`GaiaRecord`] in bytes. Asserted at compile time below.
pub const RECORD_SIZE: u32 = 104;

/// Header size in bytes (everything before the first record).
pub const HEADER_SIZE: usize = 32;

/// Canonical 104-byte Gaia source record used throughout the bundle format.
///
/// All multi-byte fields are stored in **little-endian** order on disk. On LE
/// targets, the `repr(C)` in-memory layout matches the on-disk layout exactly,
/// allowing zero-copy reads via [`bytemuck::cast_slice`].
///
/// Optional fields use NaN as a sentinel for "missing"; the [`flags`] bitfield
/// carries fast presence checks so hot-path code can avoid NaN comparisons.
///
/// [`flags`]: GaiaRecord::flags
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Pod, Zeroable)]
pub struct GaiaRecord {
    /// Gaia `source_id`.
    ///
    /// Bit 63 set marks a Hipparcos-supplement record; in that case the low
    /// 31 bits hold the HIP catalog number and [`GaiaRecord::hip_number`]
    /// recovers it. This mirrors `starfield-gaia`'s supplement encoding.
    pub source_id: u64,
    /// Reference epoch of the position, e.g. 2016.0 for Gaia DR3.
    pub ref_epoch: f64,
    /// Right Ascension at `ref_epoch`, degrees, ICRS.
    pub ra: f64,
    /// Declination at `ref_epoch`, degrees, ICRS.
    pub dec: f64,
    /// Proper motion in RA, mas/yr (Gaia convention: includes the cos(dec)
    /// factor). NaN if Gaia did not publish a 5-parameter solution.
    pub pmra: f64,
    /// Proper motion in Dec, mas/yr. NaN if missing.
    pub pmdec: f64,
    /// Parallax, mas. NaN if missing.
    pub parallax: f64,
    /// Radial velocity, km/s. NaN for sources without an RVS solution.
    pub radial_velocity: f64,
    /// Gaia G-band apparent magnitude. Always populated.
    pub phot_g_mean_mag: f64,
    /// 1-σ uncertainty on RA, mas.
    pub sigma_ra: f32,
    /// 1-σ uncertainty on Dec, mas.
    pub sigma_dec: f32,
    /// 1-σ uncertainty on `pmra`, mas/yr.
    pub sigma_pmra: f32,
    /// 1-σ uncertainty on `pmdec`, mas/yr.
    pub sigma_pmdec: f32,
    /// 1-σ uncertainty on parallax, mas.
    pub sigma_parallax: f32,
    /// Pearson correlation between RA and Dec errors, in `[-1, 1]`.
    /// NaN if missing.
    pub ra_dec_corr: f32,
    /// Renormalized Unit Weight Error. RUWE > ~1.4 flags "astrometry
    /// suspect" (binary, blended, high-PM). NaN if missing.
    pub ruwe: f32,
    /// Bitfield of presence and quality flags. See the `FLAG_*` constants.
    pub flags: u32,
}

// Compile-time guarantees on the wire format. If any of these fail the file
// format is silently broken.
const _: () = assert!(std::mem::size_of::<GaiaRecord>() == 104);
const _: () = assert!(std::mem::align_of::<GaiaRecord>() == 8);

impl GaiaRecord {
    /// `pmra`/`pmdec` (and their sigmas) are non-NaN.
    pub const FLAG_HAS_PM: u32 = 1 << 0;
    /// `parallax` (and `sigma_parallax`) is non-NaN.
    pub const FLAG_HAS_PARALLAX: u32 = 1 << 1;
    /// `radial_velocity` is non-NaN.
    pub const FLAG_HAS_RADIAL_VELOCITY: u32 = 1 << 2;
    /// `ra_dec_corr` is non-NaN.
    pub const FLAG_HAS_RA_DEC_CORR: u32 = 1 << 3;
    /// `ruwe` is non-NaN.
    pub const FLAG_HAS_RUWE: u32 = 1 << 4;
    /// Hipparcos supplement record (mirrors `source_id` bit 63).
    pub const FLAG_SOURCE_KIND_SUPPLEMENT: u32 = 1 << 5;
    /// `phot_g_mean_mag` was synthesized from Hipparcos Hp/Tycho via a
    /// per-release polynomial rather than measured by Gaia.
    pub const FLAG_SYNTHETIC_PHOT_G: u32 = 1 << 6;
    /// Proper motion has been extrapolated 25+ years from the Hipparcos
    /// epoch, so positional uncertainty at `ref_epoch` is dominated by PM
    /// drift.
    pub const FLAG_PM_EXTRAPOLATED_LONG: u32 = 1 << 7;

    /// True iff `FLAG_HAS_PM` is set.
    #[inline]
    pub fn has_pm(&self) -> bool {
        self.flags & Self::FLAG_HAS_PM != 0
    }

    /// Typed proper motion in the Gaia DR3 convention, or `None` if the
    /// record carries no PM (2-parameter solution or
    /// Hipparcos-supplement entry with `FLAG_HAS_PM` unset). Mirrors the
    /// upstream `starfield_gaia::GaiaCore::proper_motion` accessor.
    #[inline]
    pub fn proper_motion(&self) -> Option<starfield::ProperMotion> {
        if self.has_pm() {
            Some(starfield::ProperMotion {
                pmra: self.pmra,
                pmdec: self.pmdec,
            })
        } else {
            None
        }
    }

    /// True iff `FLAG_HAS_PARALLAX` is set.
    #[inline]
    pub fn has_parallax(&self) -> bool {
        self.flags & Self::FLAG_HAS_PARALLAX != 0
    }

    /// True iff `FLAG_HAS_RADIAL_VELOCITY` is set.
    #[inline]
    pub fn has_radial_velocity(&self) -> bool {
        self.flags & Self::FLAG_HAS_RADIAL_VELOCITY != 0
    }

    /// True iff `FLAG_HAS_RA_DEC_CORR` is set.
    #[inline]
    pub fn has_ra_dec_corr(&self) -> bool {
        self.flags & Self::FLAG_HAS_RA_DEC_CORR != 0
    }

    /// True iff `FLAG_HAS_RUWE` is set.
    #[inline]
    pub fn has_ruwe(&self) -> bool {
        self.flags & Self::FLAG_HAS_RUWE != 0
    }

    /// True iff `FLAG_SOURCE_KIND_SUPPLEMENT` is set.
    #[inline]
    pub fn is_supplement(&self) -> bool {
        self.flags & Self::FLAG_SOURCE_KIND_SUPPLEMENT != 0
    }

    /// True iff `FLAG_SYNTHETIC_PHOT_G` is set.
    #[inline]
    pub fn is_synthetic_phot_g(&self) -> bool {
        self.flags & Self::FLAG_SYNTHETIC_PHOT_G != 0
    }

    /// True iff `FLAG_PM_EXTRAPOLATED_LONG` is set.
    #[inline]
    pub fn pm_extrapolated_long(&self) -> bool {
        self.flags & Self::FLAG_PM_EXTRAPOLATED_LONG != 0
    }

    /// Recover the Hipparcos HIP number for a supplement record.
    ///
    /// Returns `Some(hip)` when bit 63 of `source_id` is set, where `hip` is
    /// the low 31 bits of `source_id`. Returns `None` for native Gaia
    /// records. Mirrors `starfield-gaia`'s supplement encoding.
    #[inline]
    pub fn hip_number(&self) -> Option<u32> {
        if self.source_id & (1u64 << 63) != 0 {
            Some((self.source_id & 0x7FFF_FFFF) as u32)
        } else {
            None
        }
    }
}

/// Write a `.zga` shard for one cell.
///
/// The caller passes a mutable slice of records; the writer sorts it
/// ascending by `source_id` in place (so binsearch on read is correct), then
/// emits the 32-byte header followed by the records as raw bytes.
///
/// Records are written via `bytemuck::cast_slice`, which on little-endian
/// targets is bit-identical to the on-disk layout specified in
/// `docs/bundle-format.md`.
pub fn write_gaia_shard<W: Write>(
    w: &mut W,
    cell_id: u64,
    records: &mut [GaiaRecord],
) -> io::Result<()> {
    records.sort_by_key(|r| r.source_id);

    // After sorting, duplicate source_ids are necessarily adjacent. The
    // reader uses binary search by source_id; duplicates would silently
    // shadow one another, so reject them at write time.
    for pair in records.windows(2) {
        if pair[0].source_id == pair[1].source_id {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                format!(
                    "duplicate source_id {} in gaia shard input",
                    pair[0].source_id
                ),
            ));
        }
    }

    // Header: 32 bytes, all little-endian.
    let n_records: u32 = records
        .len()
        .try_into()
        .map_err(|_| io::Error::other("n_records exceeds u32::MAX"))?;

    w.write_all(MAGIC)?;
    w.write_all(&VERSION.to_le_bytes())?;
    w.write_all(&RECORD_SIZE.to_le_bytes())?;
    w.write_all(&cell_id.to_le_bytes())?;
    w.write_all(&n_records.to_le_bytes())?;
    w.write_all(&[0u8; 4])?; // reserved padding to 8-byte alignment

    // Records: zero-copy cast to bytes. On LE targets this matches the
    // on-disk layout byte-for-byte.
    let bytes: &[u8] = bytemuck::cast_slice(records);
    w.write_all(bytes)?;

    Ok(())
}

/// Zero-copy view over a parsed `.zga` shard.
///
/// The records slice borrows from the input bytes (typically an mmap), so
/// `parse` is O(1) modulo the alignment + bounds checks.
#[derive(Debug)]
pub struct GaiaShard<'a> {
    cell_id: u64,
    records: &'a [GaiaRecord],
}

impl<'a> GaiaShard<'a> {
    /// Validate the header and produce a zero-copy view over `bytes`.
    ///
    /// Returns `io::Error` (kind `InvalidData`) on:
    /// - slice shorter than the 32-byte header
    /// - bad magic
    /// - unsupported version
    /// - record_size != 104 (rejects future format versions)
    /// - n_records exceeds the slice length
    /// - records range is not 8-byte aligned (mmap'd entries are page-aligned
    ///   so this is satisfied; for ad-hoc `Vec<u8>` the allocator usually
    ///   gives an 8-aligned allocation but is not guaranteed)
    pub fn parse(bytes: &'a [u8]) -> io::Result<Self> {
        if bytes.len() < HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "gaia shard too short: got {} bytes, need at least {} for header",
                    bytes.len(),
                    HEADER_SIZE
                ),
            ));
        }

        if &bytes[0..8] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "gaia shard: bad magic (expected b\"ZDCLGAIA\")",
            ));
        }

        let version = u32::from_le_bytes(bytes[8..12].try_into().unwrap());
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "gaia shard: unsupported version {} (expected {})",
                    version, VERSION
                ),
            ));
        }

        let record_size = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
        if record_size != RECORD_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "gaia shard: unsupported record_size {} (expected {})",
                    record_size, RECORD_SIZE
                ),
            ));
        }

        let cell_id = u64::from_le_bytes(bytes[16..24].try_into().unwrap());
        let n_records = u32::from_le_bytes(bytes[24..28].try_into().unwrap()) as usize;
        // bytes[28..32] is reserved padding; ignored.

        let records_byte_len = n_records
            .checked_mul(RECORD_SIZE as usize)
            .ok_or_else(|| io::Error::other("gaia shard: n_records * 104 overflowed usize"))?;
        let total_needed = HEADER_SIZE
            .checked_add(records_byte_len)
            .ok_or_else(|| io::Error::other("gaia shard: header + records overflowed usize"))?;

        if bytes.len() < total_needed {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "gaia shard: truncated, header claims {} records ({} bytes) but slice is only {} bytes",
                    n_records,
                    total_needed,
                    bytes.len()
                ),
            ));
        }

        let records_bytes = &bytes[HEADER_SIZE..HEADER_SIZE + records_byte_len];
        let records: &[GaiaRecord] = bytemuck::try_cast_slice(records_bytes).map_err(|e| {
            io::Error::other(format!(
                "gaia shard: records slice misaligned or sized wrong: {e}"
            ))
        })?;

        Ok(Self { cell_id, records })
    }

    /// HEALPix cell id this shard covers (defensive duplication of the
    /// filename).
    #[inline]
    pub fn cell_id(&self) -> u64 {
        self.cell_id
    }

    /// Number of records in this shard.
    #[inline]
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True iff this shard holds zero records.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Random access by position, `0..self.len()`.
    #[inline]
    pub fn get(&self, idx: usize) -> Option<&GaiaRecord> {
        self.records.get(idx)
    }

    /// Binsearch by `source_id`. O(log n) cache-line probes.
    #[inline]
    pub fn find(&self, source_id: u64) -> Option<&GaiaRecord> {
        match self
            .records
            .binary_search_by_key(&source_id, |r| r.source_id)
        {
            Ok(idx) => Some(&self.records[idx]),
            Err(_) => None,
        }
    }

    /// All records as a slice, sorted ascending by `source_id`.
    #[inline]
    pub fn records(&self) -> &[GaiaRecord] {
        self.records
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a synthetic record with mostly-real values; flags reflect
    /// presence of the optional fields.
    fn make_record(source_id: u64, mag: f64) -> GaiaRecord {
        GaiaRecord {
            source_id,
            ref_epoch: 2016.0,
            ra: 123.456,
            dec: -7.89,
            pmra: 1.23,
            pmdec: -4.56,
            parallax: 7.89,
            radial_velocity: -10.5,
            phot_g_mean_mag: mag,
            sigma_ra: 0.1,
            sigma_dec: 0.2,
            sigma_pmra: 0.3,
            sigma_pmdec: 0.4,
            sigma_parallax: 0.5,
            ra_dec_corr: 0.05,
            ruwe: 1.05,
            flags: GaiaRecord::FLAG_HAS_PM
                | GaiaRecord::FLAG_HAS_PARALLAX
                | GaiaRecord::FLAG_HAS_RADIAL_VELOCITY
                | GaiaRecord::FLAG_HAS_RA_DEC_CORR
                | GaiaRecord::FLAG_HAS_RUWE,
        }
    }

    #[test]
    fn size_and_align() {
        // Belt-and-suspenders runtime version of the const_ asserts above.
        assert_eq!(std::mem::size_of::<GaiaRecord>(), 104);
        assert_eq!(std::mem::align_of::<GaiaRecord>(), 8);
    }

    #[test]
    fn roundtrip_small() {
        let mut records: Vec<GaiaRecord> = (0..50)
            .map(|i| {
                // Sprinkle some NaN'd RV / parallax fields.
                let mut r = make_record(1_000 + i as u64, 12.0 + i as f64 * 0.01);
                if i % 3 == 0 {
                    r.radial_velocity = f64::NAN;
                    r.flags &= !GaiaRecord::FLAG_HAS_RADIAL_VELOCITY;
                }
                if i % 5 == 0 {
                    r.parallax = f64::NAN;
                    r.sigma_parallax = f32::NAN;
                    r.flags &= !GaiaRecord::FLAG_HAS_PARALLAX;
                }
                r
            })
            .collect();

        let expected = {
            let mut sorted = records.clone();
            sorted.sort_by_key(|r| r.source_id);
            sorted
        };

        let mut buf = Vec::new();
        write_gaia_shard(&mut buf, 42, &mut records).unwrap();

        let shard = GaiaShard::parse(&buf).unwrap();
        assert_eq!(shard.cell_id(), 42);
        assert_eq!(shard.len(), 50);
        assert!(!shard.is_empty());

        for (i, exp) in expected.iter().enumerate() {
            let got = shard.get(i).unwrap();
            // Bit-equal compare: PartialEq on f64 treats NaN != NaN, so we
            // compare the raw byte representation instead.
            let got_bytes: &[u8] = bytemuck::bytes_of(got);
            let exp_bytes: &[u8] = bytemuck::bytes_of(exp);
            assert_eq!(got_bytes, exp_bytes, "mismatch at idx {i}");
        }
    }

    #[test]
    fn find_hits_and_misses() {
        // 200 records at known source_ids: 100, 200, 300, ..., 20000.
        let mut records: Vec<GaiaRecord> = (1..=200).map(|i| make_record(i * 100, 14.0)).collect();

        let mut buf = Vec::new();
        write_gaia_shard(&mut buf, 7, &mut records).unwrap();
        let shard = GaiaShard::parse(&buf).unwrap();

        // Hits.
        for i in 1..=200u64 {
            let id = i * 100;
            let r = shard.find(id).expect("expected hit");
            assert_eq!(r.source_id, id);
        }

        // Misses interspersed with hits.
        for i in 1..=200u64 {
            let id = i * 100 + 1;
            assert!(shard.find(id).is_none(), "unexpected hit for {id}");
        }

        // Below min.
        assert!(shard.find(99).is_none());
        // Above max.
        assert!(shard.find(20_001).is_none());
        // Way outside range.
        assert!(shard.find(0).is_none());
        assert!(shard.find(u64::MAX).is_none());
    }

    #[test]
    fn writer_sorts_input() {
        // Pass records in unsorted order.
        let mut records = vec![
            make_record(500, 12.0),
            make_record(100, 12.0),
            make_record(800, 12.0),
            make_record(200, 12.0),
            make_record(1000, 12.0),
            make_record(50, 12.0),
        ];

        let mut buf = Vec::new();
        write_gaia_shard(&mut buf, 0, &mut records).unwrap();
        let shard = GaiaShard::parse(&buf).unwrap();

        let on_disk: Vec<u64> = shard.records().iter().map(|r| r.source_id).collect();
        assert_eq!(on_disk, vec![50, 100, 200, 500, 800, 1000]);
        // The caller's slice was also sorted in-place.
        let in_caller: Vec<u64> = records.iter().map(|r| r.source_id).collect();
        assert_eq!(in_caller, vec![50, 100, 200, 500, 800, 1000]);
    }

    #[test]
    fn flags_helpers() {
        let mut r = make_record(1, 12.0);
        r.flags = 0;
        assert!(!r.has_pm());
        assert!(!r.has_parallax());
        assert!(!r.has_radial_velocity());
        assert!(!r.has_ra_dec_corr());
        assert!(!r.has_ruwe());
        assert!(!r.is_supplement());
        assert!(!r.is_synthetic_phot_g());
        assert!(!r.pm_extrapolated_long());

        r.flags = GaiaRecord::FLAG_HAS_PM;
        assert!(r.has_pm());
        assert!(!r.has_parallax());

        r.flags = GaiaRecord::FLAG_HAS_PARALLAX | GaiaRecord::FLAG_HAS_RUWE;
        assert!(!r.has_pm());
        assert!(r.has_parallax());
        assert!(r.has_ruwe());
        assert!(!r.has_radial_velocity());

        r.flags = GaiaRecord::FLAG_HAS_RADIAL_VELOCITY | GaiaRecord::FLAG_HAS_RA_DEC_CORR;
        assert!(r.has_radial_velocity());
        assert!(r.has_ra_dec_corr());

        r.flags = GaiaRecord::FLAG_SOURCE_KIND_SUPPLEMENT;
        assert!(r.is_supplement());
        assert!(!r.is_synthetic_phot_g());
        assert!(!r.pm_extrapolated_long());

        r.flags = GaiaRecord::FLAG_SYNTHETIC_PHOT_G;
        assert!(r.is_synthetic_phot_g());
        assert!(!r.is_supplement());

        r.flags = GaiaRecord::FLAG_PM_EXTRAPOLATED_LONG;
        assert!(r.pm_extrapolated_long());

        // All flags set.
        r.flags = u32::MAX;
        assert!(r.has_pm());
        assert!(r.has_parallax());
        assert!(r.has_radial_velocity());
        assert!(r.has_ra_dec_corr());
        assert!(r.has_ruwe());
        assert!(r.is_supplement());
        assert!(r.is_synthetic_phot_g());
        assert!(r.pm_extrapolated_long());
    }

    #[test]
    fn supplement_hip_decoding() {
        // Standard supplement encoding: bit 63 set, HIP in low 31 bits.
        let hip: u32 = 12345;
        let r = GaiaRecord {
            source_id: (1u64 << 63) | hip as u64,
            ref_epoch: 1991.25,
            ra: 0.0,
            dec: 0.0,
            pmra: f64::NAN,
            pmdec: f64::NAN,
            parallax: f64::NAN,
            radial_velocity: f64::NAN,
            phot_g_mean_mag: 5.0,
            sigma_ra: 0.0,
            sigma_dec: 0.0,
            sigma_pmra: 0.0,
            sigma_pmdec: 0.0,
            sigma_parallax: 0.0,
            ra_dec_corr: f32::NAN,
            ruwe: f32::NAN,
            flags: GaiaRecord::FLAG_SOURCE_KIND_SUPPLEMENT,
        };
        assert!(r.is_supplement());
        assert_eq!(r.hip_number(), Some(hip));

        // Native Gaia source: bit 63 clear → no HIP.
        let native = GaiaRecord {
            source_id: 0x1234_5678_9ABC_DEF0,
            ..r
        };
        assert_eq!(native.hip_number(), None);
    }

    #[test]
    fn bad_magic_rejected() {
        let mut buf = Vec::new();
        let mut records = vec![make_record(1, 12.0)];
        write_gaia_shard(&mut buf, 0, &mut records).unwrap();
        // Corrupt the magic.
        buf[0] = b'X';
        let err = GaiaShard::parse(&buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("magic"), "got: {err}");
    }

    #[test]
    fn bad_version_rejected() {
        let mut buf = Vec::new();
        let mut records = vec![make_record(1, 12.0)];
        write_gaia_shard(&mut buf, 0, &mut records).unwrap();
        // Patch the version field to 99.
        let v: u32 = 99;
        buf[8..12].copy_from_slice(&v.to_le_bytes());
        let err = GaiaShard::parse(&buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("version"), "got: {err}");
    }

    #[test]
    fn bad_record_size_rejected() {
        let mut buf = Vec::new();
        let mut records = vec![make_record(1, 12.0)];
        write_gaia_shard(&mut buf, 0, &mut records).unwrap();
        // Patch record_size to the OLD sidecar size (88) — must be rejected.
        let rs: u32 = 88;
        buf[12..16].copy_from_slice(&rs.to_le_bytes());
        let err = GaiaShard::parse(&buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("record_size"), "got: {err}");
    }

    #[test]
    fn truncated_file_rejected() {
        let mut buf = Vec::new();
        let mut records: Vec<GaiaRecord> =
            (0..10).map(|i| make_record(i as u64 + 1, 12.0)).collect();
        write_gaia_shard(&mut buf, 0, &mut records).unwrap();

        // Lop off the last record's worth of bytes; n_records still claims 10.
        let truncated = &buf[..buf.len() - 50];
        let err = GaiaShard::parse(truncated).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("truncated"), "got: {err}");

        // Also: a slice shorter than the header.
        let short = &buf[..16];
        let err = GaiaShard::parse(short).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
    }

    #[test]
    fn nan_round_trip() {
        // All optional float fields NaN; ensure they survive write/read
        // bitwise (no f64 normalization).
        let r = GaiaRecord {
            source_id: 42,
            ref_epoch: 2016.0,
            ra: 0.0,
            dec: 0.0,
            pmra: f64::NAN,
            pmdec: f64::NAN,
            parallax: f64::NAN,
            radial_velocity: f64::NAN,
            phot_g_mean_mag: 12.5,
            sigma_ra: 0.1,
            sigma_dec: 0.2,
            sigma_pmra: f32::NAN,
            sigma_pmdec: f32::NAN,
            sigma_parallax: f32::NAN,
            ra_dec_corr: f32::NAN,
            ruwe: f32::NAN,
            flags: 0,
        };

        let mut records = vec![r];
        let expected = records.clone();

        let mut buf = Vec::new();
        write_gaia_shard(&mut buf, 0, &mut records).unwrap();
        let shard = GaiaShard::parse(&buf).unwrap();

        let got = shard.get(0).unwrap();
        // Compare bit patterns; PartialEq's f64::NAN != f64::NAN would lie.
        let got_bytes: &[u8] = bytemuck::bytes_of(got);
        let exp_bytes: &[u8] = bytemuck::bytes_of(&expected[0]);
        assert_eq!(got_bytes, exp_bytes);

        // Spot-check a few NaN bits via `.is_nan()`.
        assert!(got.pmra.is_nan());
        assert!(got.pmdec.is_nan());
        assert!(got.parallax.is_nan());
        assert!(got.radial_velocity.is_nan());
        assert!(got.sigma_pmra.is_nan());
        assert!(got.ra_dec_corr.is_nan());
        assert!(got.ruwe.is_nan());
        // And non-NaN fields survived too.
        assert_eq!(got.source_id, 42);
        assert_eq!(got.phot_g_mean_mag, 12.5);
    }

    #[test]
    fn duplicate_source_id_rejected() {
        // Two records with the same source_id must be rejected at write
        // time — readers binary-search by source_id and a duplicate would
        // silently shadow one of them.
        let mut records = vec![make_record(42, 12.0), make_record(42, 13.0)];
        let mut buf = Vec::new();
        let err = write_gaia_shard(&mut buf, 0, &mut records).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
        assert!(err.to_string().contains("duplicate"), "got: {err}");
    }

    #[test]
    fn parse_rejects_misaligned_slice() {
        // GaiaRecord requires 8-byte alignment. Build a valid shard, then
        // hand the parser a sub-slice starting one byte in: the records
        // region is no longer 8-aligned and parse must error rather than
        // panic from inside bytemuck.
        let mut records = vec![make_record(1, 12.0), make_record(2, 12.0)];
        let mut prefixed = vec![0u8]; // one stray byte to misalign the rest
        write_gaia_shard(&mut prefixed, 0, &mut records).unwrap();
        let mis = &prefixed[1..];
        // Header parses fine; the bytemuck cast must fail when records
        // start at an offset that puts them on a 1-mod-8 byte boundary.
        // We can't predict alignment of `prefixed`'s allocation, so retry
        // on an explicitly-aligned buffer if the first attempt happens
        // to land aligned.
        let err = match GaiaShard::parse(mis) {
            Ok(_) => {
                // Allocator gave an unlucky address; rebuild with a wider
                // prefix to force misalignment relative to the
                // allocation base.
                let mut records2 = vec![make_record(1, 12.0), make_record(2, 12.0)];
                let mut prefixed2 = vec![0u8; 3];
                write_gaia_shard(&mut prefixed2, 0, &mut records2).unwrap();
                let off = (prefixed2.as_ptr() as usize) & 7;
                // Pick a slice start such that records (at HEADER_SIZE
                // bytes in) land on an odd byte from the underlying
                // allocation.
                let skew = if off == 0 { 1 } else { 8 - off + 1 };
                assert!(
                    skew < prefixed2.len(),
                    "could not construct a misaligned slice"
                );
                let mis2 = &prefixed2[skew..];
                GaiaShard::parse(mis2).expect_err("misaligned slice must error")
            }
            Err(e) => e,
        };
        // Bytemuck reports an alignment error via io::Error::other,
        // which has kind `Other`.
        assert_eq!(err.kind(), io::ErrorKind::Other);
        assert!(err.to_string().contains("misaligned"), "got: {err}");
    }

    #[test]
    fn header_byte_layout() {
        // Pin the exact 32-byte header layout. Empty shard for simplicity
        // (the header is identical regardless of n_records).
        let mut records: Vec<GaiaRecord> = Vec::new();
        let mut buf = Vec::new();
        write_gaia_shard(&mut buf, 0x0123_4567_89AB_CDEF, &mut records).unwrap();
        assert_eq!(buf.len(), HEADER_SIZE);

        let expected: [u8; 32] = [
            // magic "ZDCLGAIA"
            b'Z', b'D', b'C', b'L', b'G', b'A', b'I', b'A', // version = 1 (u32 LE)
            0x01, 0x00, 0x00, 0x00, // record_size = 104 (u32 LE)
            0x68, 0x00, 0x00, 0x00, // cell_id = 0x0123_4567_89AB_CDEF (u64 LE)
            0xEF, 0xCD, 0xAB, 0x89, 0x67, 0x45, 0x23, 0x01, // n_records = 0 (u32 LE)
            0x00, 0x00, 0x00, 0x00, // reserved padding
            0x00, 0x00, 0x00, 0x00,
        ];
        assert_eq!(&buf[..], &expected[..]);
    }

    #[test]
    fn edge_source_ids_roundtrip() {
        // 0 and u64::MAX coexist as source_ids.
        let mut records = vec![make_record(0, 12.0), make_record(u64::MAX, 13.0)];
        let mut buf = Vec::new();
        write_gaia_shard(&mut buf, 1, &mut records).unwrap();
        let shard = GaiaShard::parse(&buf).unwrap();
        assert_eq!(shard.len(), 2);
        assert_eq!(shard.find(0).unwrap().source_id, 0);
        assert_eq!(shard.find(u64::MAX).unwrap().source_id, u64::MAX);

        // A supplement-bit-set id (bit 63 set) coexists with a normal id.
        let supplement_id = 1u64 << 63;
        let mut records2 = vec![make_record(100, 12.0), make_record(supplement_id, 5.0)];
        let mut buf2 = Vec::new();
        write_gaia_shard(&mut buf2, 2, &mut records2).unwrap();
        let shard2 = GaiaShard::parse(&buf2).unwrap();
        assert_eq!(shard2.len(), 2);
        let normal = shard2.find(100).expect("normal id");
        assert_eq!(normal.hip_number(), None);
        let supp = shard2.find(supplement_id).expect("supplement id");
        // hip_number recovers the low 31 bits (zero in this case).
        assert_eq!(supp.hip_number(), Some(0));
    }

    #[test]
    fn parse_rejects_inflated_n_records() {
        // Write a valid 2-record shard, then patch n_records up to a value
        // larger than the slice can hold. Parse must reject rather than
        // walk off the end of the buffer.
        let mut records = vec![make_record(1, 12.0), make_record(2, 13.0)];
        let mut buf = Vec::new();
        write_gaia_shard(&mut buf, 0, &mut records).unwrap();
        let bogus: u32 = 1_000_000;
        buf[24..28].copy_from_slice(&bogus.to_le_bytes());
        let err = GaiaShard::parse(&buf).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        assert!(err.to_string().contains("truncated"), "got: {err}");
    }

    #[test]
    fn empty_shard_roundtrip() {
        let mut records: Vec<GaiaRecord> = Vec::new();
        let mut buf = Vec::new();
        write_gaia_shard(&mut buf, 99, &mut records).unwrap();
        // Header-only file is exactly 32 bytes.
        assert_eq!(buf.len(), HEADER_SIZE);

        let shard = GaiaShard::parse(&buf).unwrap();
        assert_eq!(shard.cell_id(), 99);
        assert!(shard.is_empty());
        assert_eq!(shard.len(), 0);
        assert!(shard.find(0).is_none());
        assert!(shard.records().is_empty());
    }
}
