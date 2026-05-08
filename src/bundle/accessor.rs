//! Storage-agnostic entry access for a `.zdcl.bundle`.
//!
//! A bundle is a logical tree of named entries (relative paths). It can be
//! packaged either as a directory of files on disk or as a `.zip` archive
//! holding the same logical layout. Consumer code reads either form
//! through the [`SubfileAccessor`] trait and the [`EntryBytes`] enum, so
//! the hot path doesn't have to branch on packaging.
//!
//! Two concrete implementations are provided here:
//!
//! - [`FsAccessor`] — backs entries with files in a directory root and
//!   memory-maps each entry on first access (cached for the lifetime of
//!   the accessor).
//! - [`ZipAccessor`] — backs entries with a `zip::ZipArchive<File>` and
//!   decompresses each entry into an owned `Vec<u8>` on every read.
//!
//! See `docs/bundle-format.md` for the broader bundle format design.

use std::fs::File;
use std::io;
use std::num::NonZeroUsize;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use lru::LruCache;
use memmap2::Mmap;
use zip::ZipArchive;

/// Default eviction cap for [`FsAccessor`]'s mmap LRU. Tuned so a
/// 1000-case bench-bundle sweep at ~6,000 cells/case (5° hint on a
/// depth-9 bundle) stays within typical host RAM. Override via
/// [`FsAccessor::open_with_capacity`].
pub const FS_ACCESSOR_DEFAULT_CACHE_CAP: usize = 100_000;

/// Storage-agnostic entry access for a bundle.
///
/// Implementations must be cheap to share across threads (`Send + Sync`).
/// Callers expect that `read_entry` returns something usable as a
/// read-only `&[u8]` regardless of whether the backing storage is an
/// mmap-backed file or a decompressed-zip buffer; the [`EntryBytes`] enum
/// captures that polymorphism with a `Deref<Target = [u8]>` implementation.
pub trait SubfileAccessor: Send + Sync {
    /// True if an entry at this relative path exists.
    fn exists(&self, rel: &str) -> bool;

    /// List all entry names whose relative path starts with `prefix`.
    ///
    /// The returned strings are bundle-relative entry names (e.g.
    /// `"quads/cell_00000.zqd"`), suitable to pass back to
    /// [`Self::read_entry`]. Forward slashes are used regardless of the
    /// host OS so callers can match the bundle spec's path conventions.
    fn list_prefix(&self, prefix: &str) -> io::Result<Vec<String>>;

    /// Return a read-only view of the entry's bytes.
    ///
    /// For [`FsAccessor`] this hands back an `Arc<Mmap>` (cheap clone of
    /// a cached entry; the Arc keeps the mapping alive even if the LRU
    /// cache evicts the slot). For [`ZipAccessor`] this is a freshly
    /// decompressed owned buffer.
    fn read_entry(&self, rel: &str) -> io::Result<EntryBytes>;
}

/// A read-only view of an entry's bytes.
///
/// `Mmap` carries an [`Arc`]-shared memory map; `Owned` carries a freshly
/// decompressed buffer. Both deref to `[u8]`, so consumer code can treat
/// them identically. The `Arc<Mmap>` keeps the mapping alive for as long
/// as any caller holds an `EntryBytes`, even if the originating
/// accessor's LRU cache has since evicted the slot.
pub enum EntryBytes {
    /// Shared memory map backing the entry's bytes.
    Mmap(Arc<Mmap>),
    /// Owned buffer (e.g. decompressed from a zip entry).
    Owned(Vec<u8>),
}

impl Deref for EntryBytes {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            EntryBytes::Mmap(arc) => &arc[..],
            EntryBytes::Owned(buf) => buf.as_slice(),
        }
    }
}

impl AsRef<[u8]> for EntryBytes {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl std::fmt::Debug for EntryBytes {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntryBytes::Mmap(arc) => f
                .debug_struct("EntryBytes::Mmap")
                .field("len", &arc.len())
                .finish(),
            EntryBytes::Owned(buf) => f
                .debug_struct("EntryBytes::Owned")
                .field("len", &buf.len())
                .finish(),
        }
    }
}

/// Filesystem-backed accessor.
///
/// Each entry is mmapped on first access. The mapping is wrapped in
/// `Arc<Mmap>` and stored in a bounded LRU cache; on a hit the accessor
/// hands back an `Arc<Mmap>` clone. Eviction is safe because callers
/// holding an [`EntryBytes::Mmap`] keep the underlying mapping alive
/// through their `Arc`, even after the LRU drops the slot. The cap is
/// configurable via [`Self::open_with_capacity`]; the default is
/// [`FS_ACCESSOR_DEFAULT_CACHE_CAP`].
///
/// # Path-traversal hardening
///
/// `FsAccessor` treats the bundle as untrusted input. Every relative
/// name passed in is validated **before** any filesystem syscall:
/// empty names, absolute paths, `..` components, and NUL bytes are
/// rejected with [`io::ErrorKind::InvalidInput`]. After resolving a
/// candidate path under the root, both root and candidate are
/// canonicalized and the candidate is required to be a descendant of
/// the root. This catches both lexical traversal (`a/../../etc`) and
/// symlinks that point outside the root.
pub struct FsAccessor {
    /// Canonicalized base directory. All resolved entries must live
    /// underneath this path post-canonicalization.
    root: PathBuf,
    cache: Mutex<LruCache<String, Arc<Mmap>>>,
}

impl FsAccessor {
    /// Open a directory root as an accessor with the default
    /// per-process mmap LRU capacity ([`FS_ACCESSOR_DEFAULT_CACHE_CAP`]).
    pub fn open(root: impl AsRef<Path>) -> io::Result<Self> {
        Self::open_with_capacity(root, FS_ACCESSOR_DEFAULT_CACHE_CAP)
    }

    /// Open a directory root with an explicit cache cap. Smaller caps
    /// trade hit rate for VM/RAM bound; larger caps suit long-running
    /// services that stream many distinct entries.
    ///
    /// The path must point at an existing directory; otherwise an
    /// `io::Error` of kind [`io::ErrorKind::NotFound`] (or similar) is
    /// returned. The directory is canonicalized once at construction
    /// time and that canonical path is used as the traversal-check
    /// prefix for every subsequent entry lookup.
    pub fn open_with_capacity(root: impl AsRef<Path>, cap: usize) -> io::Result<Self> {
        let cap = NonZeroUsize::new(cap.max(1)).expect("max(1) ensures non-zero");
        let root = root.as_ref().to_path_buf();
        let meta = std::fs::metadata(&root)?;
        if !meta.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("{} is not a directory", root.display()),
            ));
        }
        // Canonicalize once so we can compare candidates by prefix.
        // This resolves any symlinks in the root path itself; from
        // here on the stored `root` is the trusted base.
        let root = std::fs::canonicalize(&root)?;
        Ok(Self {
            root,
            cache: Mutex::new(LruCache::new(cap)),
        })
    }

    /// Validate a bundle-relative name without touching the filesystem.
    ///
    /// Rejects names that are empty, absolute, contain a `..`
    /// component, or contain a NUL byte. Returns an `InvalidInput`
    /// error so callers can distinguish "bad name" from "missing
    /// file".
    fn validate_name(rel: &str) -> io::Result<()> {
        if rel.is_empty() {
            return Err(invalid_name(rel, "empty name"));
        }
        if rel.contains('\0') {
            return Err(invalid_name(rel, "name contains NUL byte"));
        }
        // Reject obviously-absolute names in either bundle (`/foo`)
        // or host (`Path::is_absolute`) form.
        if rel.starts_with('/') || Path::new(rel).is_absolute() {
            return Err(invalid_name(rel, "absolute path is not allowed"));
        }
        for comp in rel.split('/') {
            if comp == ".." {
                return Err(invalid_name(rel, "`..` component is not allowed"));
            }
        }
        Ok(())
    }

    /// Resolve a bundle-relative path to an absolute filesystem path,
    /// rejecting any name that escapes (or could escape) the root.
    ///
    /// This is the single chokepoint for filesystem access: both
    /// [`SubfileAccessor::exists`] and [`SubfileAccessor::read_entry`]
    /// go through here. The returned path is canonicalized and
    /// guaranteed to live under [`Self::root`].
    fn resolve(&self, rel: &str) -> io::Result<PathBuf> {
        Self::validate_name(rel)?;

        // Bundle paths use forward slashes; `Path::join` handles a
        // forward-slash subpath correctly on the platforms this crate
        // targets.
        let mut candidate = self.root.clone();
        for comp in rel.split('/') {
            if !comp.is_empty() {
                candidate.push(comp);
            }
        }

        // Canonicalize the candidate so any symlinks are resolved
        // before we compare against the root prefix. If the candidate
        // does not exist, propagate the io::Error (NotFound) directly
        // — this preserves the previous "missing entry → NotFound"
        // behaviour for callers like `exists`.
        let canonical = std::fs::canonicalize(&candidate)?;
        if !canonical.starts_with(&self.root) {
            return Err(invalid_name(rel, "resolved path escapes the bundle root"));
        }
        Ok(canonical)
    }
}

/// Construct an `io::Error` for a rejected bundle-relative name.
fn invalid_name(rel: &str, reason: &str) -> io::Error {
    io::Error::new(
        io::ErrorKind::InvalidInput,
        format!("invalid bundle entry name {rel:?}: {reason}"),
    )
}

impl SubfileAccessor for FsAccessor {
    fn exists(&self, rel: &str) -> bool {
        // `resolve` returns Err for invalid names *and* for non-existent
        // files; both should report "doesn't exist" to the caller. On
        // Ok, additionally require that the path is a regular file.
        match self.resolve(rel) {
            Ok(p) => p.is_file(),
            Err(_) => false,
        }
    }

    fn list_prefix(&self, prefix: &str) -> io::Result<Vec<String>> {
        let mut out = Vec::new();
        walk_collect(&self.root, &self.root, prefix, &mut out)?;
        out.sort();
        Ok(out)
    }

    fn read_entry(&self, rel: &str) -> io::Result<EntryBytes> {
        // Validate the name up-front so a malicious raw key can't slip
        // past the cache layer. The cache key is the *raw*
        // bundle-relative name, but we only ever insert via the
        // `resolve` chokepoint, so a hit here always corresponds to a
        // safe path.
        Self::validate_name(rel)?;

        // Fast path: LRU hit. `get` promotes the entry to most-recent.
        {
            let mut cache = self.cache.lock().expect("FsAccessor cache poisoned");
            if let Some(mmap) = cache.get(rel) {
                return Ok(EntryBytes::Mmap(Arc::clone(mmap)));
            }
        }

        // Slow path: resolve (with traversal check), open, and mmap.
        let abs = self.resolve(rel)?;
        let file = File::open(&abs)?;
        // SAFETY: we treat the mapping as immutable for the lifetime of
        // the Arc, which matches the `Mmap` (read-only) contract.
        let mmap = unsafe { Mmap::map(&file)? };
        let arc = Arc::new(mmap);

        let mut cache = self.cache.lock().expect("FsAccessor cache poisoned");
        // Another thread may have raced us; if so, prefer the entry
        // that's already cached so concurrent callers converge on the
        // same Arc when possible. The LRU's `put` returns the previous
        // value if the key was already present.
        let to_return = if let Some(existing) = cache.get(rel) {
            Arc::clone(existing)
        } else {
            cache.put(rel.to_string(), Arc::clone(&arc));
            arc
        };
        Ok(EntryBytes::Mmap(to_return))
    }
}

/// Recursively walk `dir` and collect bundle-relative paths whose
/// rel-path starts with `prefix`.
fn walk_collect(root: &Path, dir: &Path, prefix: &str, out: &mut Vec<String>) -> io::Result<()> {
    let entries = match std::fs::read_dir(dir) {
        Ok(e) => e,
        Err(e) if e.kind() == io::ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(e),
    };
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        let ft = entry.file_type()?;
        if ft.is_dir() {
            walk_collect(root, &path, prefix, out)?;
        } else if ft.is_file() {
            let rel = path
                .strip_prefix(root)
                .map_err(|e| io::Error::other(e.to_string()))?;
            // Always emit forward-slash-separated relative paths.
            let rel_str = rel
                .components()
                .map(|c| c.as_os_str().to_string_lossy().into_owned())
                .collect::<Vec<_>>()
                .join("/");
            if rel_str.starts_with(prefix) {
                out.push(rel_str);
            }
        }
    }
    Ok(())
}

/// Zip-archive-backed accessor.
///
/// `ZipArchive::by_name` borrows the archive mutably, so the underlying
/// archive is wrapped in a `Mutex`. Each `read_entry` decompresses the
/// requested entry into an owned `Vec<u8>` ([`EntryBytes::Owned`]).
pub struct ZipAccessor {
    archive: Mutex<ZipArchive<File>>,
    /// Snapshot of entry names taken at open. Used by `exists` and
    /// `list_prefix` without holding the archive mutex.
    names: Vec<String>,
}

impl ZipAccessor {
    /// Open a zip file as an accessor.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self> {
        let file = File::open(path.as_ref())?;
        let archive = ZipArchive::new(file).map_err(zip_to_io_err)?;
        let names: Vec<String> = archive.file_names().map(|s| s.to_string()).collect();
        Ok(Self {
            archive: Mutex::new(archive),
            names,
        })
    }
}

impl SubfileAccessor for ZipAccessor {
    fn exists(&self, rel: &str) -> bool {
        self.names.iter().any(|n| n == rel)
    }

    fn list_prefix(&self, prefix: &str) -> io::Result<Vec<String>> {
        let mut out: Vec<String> = self
            .names
            .iter()
            .filter(|n| n.starts_with(prefix))
            // Skip "directory" entries that some zip writers emit
            // (trailing-slash names with zero payload).
            .filter(|n| !n.ends_with('/'))
            .cloned()
            .collect();
        out.sort();
        Ok(out)
    }

    fn read_entry(&self, rel: &str) -> io::Result<EntryBytes> {
        use std::io::Read;
        let mut archive = self.archive.lock().expect("ZipAccessor mutex poisoned");
        let mut entry = archive.by_name(rel).map_err(zip_to_io_err)?;
        let mut buf = Vec::with_capacity(entry.size() as usize);
        entry.read_to_end(&mut buf)?;
        Ok(EntryBytes::Owned(buf))
    }
}

fn zip_to_io_err(e: zip::result::ZipError) -> io::Error {
    match e {
        zip::result::ZipError::Io(io_err) => io_err,
        zip::result::ZipError::FileNotFound => {
            io::Error::new(io::ErrorKind::NotFound, "zip entry not found")
        }
        other => io::Error::new(io::ErrorKind::InvalidData, other.to_string()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use std::io::Write;

    /// Logical layout used by every test. Keep it small but representative
    /// (multiple subdirs, varying sizes) so we exercise list_prefix.
    fn sample_entries() -> BTreeMap<&'static str, Vec<u8>> {
        let mut m = BTreeMap::new();
        m.insert("manifest.json", b"{\"format\":\"zdcl-bundle\"}".to_vec());
        m.insert(
            "quads/cell_00000.zqd",
            (0u8..32).cycle().take(256).collect(),
        );
        m.insert(
            "quads/cell_00001.zqd",
            (32u8..64).cycle().take(128).collect(),
        );
        m.insert(
            "gaia/cell_00000.zga",
            (64u8..96).cycle().take(512).collect(),
        );
        m.insert(
            "gaia/cell_00001.zga",
            (96u8..128).cycle().take(64).collect(),
        );
        m
    }

    fn write_fs_layout(root: &Path, entries: &BTreeMap<&'static str, Vec<u8>>) {
        for (rel, bytes) in entries {
            let path = root.join(rel);
            if let Some(parent) = path.parent() {
                std::fs::create_dir_all(parent).expect("create parent dir");
            }
            let mut f = File::create(&path).expect("create entry file");
            f.write_all(bytes).expect("write entry bytes");
        }
    }

    fn write_zip_layout(zip_path: &Path, entries: &BTreeMap<&'static str, Vec<u8>>) {
        let f = File::create(zip_path).expect("create zip file");
        let mut zw = zip::write::ZipWriter::new(f);
        let opts =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        for (rel, bytes) in entries {
            zw.start_file(*rel, opts).expect("start zip entry");
            zw.write_all(bytes).expect("write zip entry bytes");
        }
        zw.finish().expect("finish zip");
    }

    #[test]
    fn fs_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let entries = sample_entries();
        write_fs_layout(tmp.path(), &entries);

        let acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");
        for (rel, bytes) in &entries {
            let got = acc.read_entry(rel).expect("read entry");
            assert_eq!(&got[..], bytes.as_slice(), "byte mismatch for {rel}");
        }
        assert!(acc.exists("manifest.json"));
        assert!(!acc.exists("does/not/exist"));
    }

    #[test]
    fn zip_round_trip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let zip_path = tmp.path().join("bundle.zip");
        let entries = sample_entries();
        write_zip_layout(&zip_path, &entries);

        let acc = ZipAccessor::open(&zip_path).expect("open ZipAccessor");
        for (rel, bytes) in &entries {
            let got = acc.read_entry(rel).expect("read entry");
            assert_eq!(&got[..], bytes.as_slice(), "byte mismatch for {rel}");
        }
        assert!(acc.exists("manifest.json"));
        assert!(!acc.exists("does/not/exist"));
    }

    #[test]
    fn fs_zip_parity() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("bundle_dir");
        std::fs::create_dir(&dir).unwrap();
        let zip_path = tmp.path().join("bundle.zip");
        let entries = sample_entries();
        write_fs_layout(&dir, &entries);
        write_zip_layout(&zip_path, &entries);

        let fs_acc = FsAccessor::open(&dir).expect("open FsAccessor");
        let zip_acc = ZipAccessor::open(&zip_path).expect("open ZipAccessor");

        for rel in entries.keys() {
            let fs_bytes = fs_acc.read_entry(rel).expect("fs read");
            let zip_bytes = zip_acc.read_entry(rel).expect("zip read");
            assert_eq!(
                &fs_bytes[..],
                &zip_bytes[..],
                "fs/zip parity mismatch for {rel}"
            );
        }
    }

    #[test]
    fn fs_cache_hit() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let entries = sample_entries();
        write_fs_layout(tmp.path(), &entries);

        let acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");
        let rel = "quads/cell_00000.zqd";
        let first = acc.read_entry(rel).expect("first read");
        let p1 = first.as_ptr();
        let len1 = first.len();
        // Drop the borrow before the second read so we don't hold the
        // hashmap entry — the cached mmap stays alive in the accessor.
        drop(first);

        let second = acc.read_entry(rel).expect("second read");
        let p2 = second.as_ptr();
        let len2 = second.len();
        assert_eq!(
            p1, p2,
            "cache hit should hand out the same mmap-backed pointer"
        );
        assert_eq!(len1, len2, "cached entry length mismatch");

        // Sanity: it should also still match the original bytes.
        assert_eq!(&second[..], entries[rel].as_slice());
    }

    #[test]
    fn list_prefix_filters() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let dir = tmp.path().join("bundle_dir");
        std::fs::create_dir(&dir).unwrap();
        let zip_path = tmp.path().join("bundle.zip");
        let entries = sample_entries();
        write_fs_layout(&dir, &entries);
        write_zip_layout(&zip_path, &entries);

        let fs_acc = FsAccessor::open(&dir).expect("open FsAccessor");
        let zip_acc = ZipAccessor::open(&zip_path).expect("open ZipAccessor");

        let want_quads: Vec<String> = vec![
            "quads/cell_00000.zqd".to_string(),
            "quads/cell_00001.zqd".to_string(),
        ];
        let want_gaia: Vec<String> = vec![
            "gaia/cell_00000.zga".to_string(),
            "gaia/cell_00001.zga".to_string(),
        ];

        let mut fs_quads = fs_acc.list_prefix("quads/").expect("fs list quads");
        fs_quads.sort();
        assert_eq!(fs_quads, want_quads);
        // Should not include any gaia/ entries.
        assert!(fs_quads.iter().all(|n| !n.starts_with("gaia/")));

        let mut zip_quads = zip_acc.list_prefix("quads/").expect("zip list quads");
        zip_quads.sort();
        assert_eq!(zip_quads, want_quads);
        assert!(zip_quads.iter().all(|n| !n.starts_with("gaia/")));

        let mut fs_gaia = fs_acc.list_prefix("gaia/").expect("fs list gaia");
        fs_gaia.sort();
        assert_eq!(fs_gaia, want_gaia);

        let mut zip_gaia = zip_acc.list_prefix("gaia/").expect("zip list gaia");
        zip_gaia.sort();
        assert_eq!(zip_gaia, want_gaia);
    }

    #[test]
    fn bad_path_fails_cleanly() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let entries = sample_entries();
        write_fs_layout(tmp.path(), &entries);
        let zip_path = tmp.path().join("bundle.zip");
        write_zip_layout(&zip_path, &entries);

        let fs_acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");
        let zip_acc = ZipAccessor::open(&zip_path).expect("open ZipAccessor");

        let err_fs = fs_acc
            .read_entry("does/not/exist")
            .expect_err("fs missing entry should error");
        // Don't pin the exact ErrorKind variant — Linux returns NotFound,
        // some platforms may return InvalidInput when the path resolves
        // weirdly. Just assert that we got an io::Error and didn't panic.
        let _ = err_fs.kind();

        let err_zip = zip_acc
            .read_entry("does/not/exist")
            .expect_err("zip missing entry should error");
        let _ = err_zip.kind();
    }

    /// Lexical traversal via `..` components must be rejected before
    /// the filesystem is touched, regardless of whether the candidate
    /// would have resolved to a real file.
    #[test]
    fn fs_rejects_dotdot_component() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let entries = sample_entries();
        write_fs_layout(tmp.path(), &entries);
        let acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");

        let err = acc
            .read_entry("subdir/../../../etc/passwd")
            .expect_err("`..` in name should be rejected");
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidInput,
            "expected InvalidInput, got {:?}",
            err.kind()
        );
        assert!(
            !acc.exists("subdir/../../../etc/passwd"),
            "exists() should report false for a traversal name"
        );
    }

    /// Absolute paths in the bundle-relative slot must be rejected.
    /// `Path::join` would otherwise discard the root and return the
    /// absolute path as-is.
    #[test]
    fn fs_rejects_absolute_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let entries = sample_entries();
        write_fs_layout(tmp.path(), &entries);
        let acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");

        let err = acc
            .read_entry("/etc/passwd")
            .expect_err("absolute name should be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    /// Names with embedded NUL bytes must be rejected before any
    /// filesystem call, since most syscalls would either truncate or
    /// fail with a less-helpful error.
    #[test]
    fn fs_rejects_nul_byte_in_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let entries = sample_entries();
        write_fs_layout(tmp.path(), &entries);
        let acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");

        let err = acc
            .read_entry("manifest.json\0extra")
            .expect_err("NUL in name should be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    /// Empty names are nonsensical and must be rejected up-front.
    #[test]
    fn fs_rejects_empty_name() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let entries = sample_entries();
        write_fs_layout(tmp.path(), &entries);
        let acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");

        let err = acc
            .read_entry("")
            .expect_err("empty name should be rejected");
        assert_eq!(err.kind(), io::ErrorKind::InvalidInput);
    }

    /// Empty bundle directory: list_prefix("") returns [], exists()
    /// returns false for any name.
    #[test]
    fn fs_empty_bundle_dir() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");
        let listed = acc.list_prefix("").expect("list_prefix");
        assert!(listed.is_empty(), "expected empty listing, got {listed:?}");
        assert!(!acc.exists("anything"));
        assert!(!acc.exists("manifest.json"));
    }

    /// A zero-byte file in the bundle: read_entry returns an empty
    /// EntryBytes without panicking from the mmap path.
    #[test]
    fn fs_zero_byte_file() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let path = tmp.path().join("empty.bin");
        File::create(&path).expect("create empty file");
        let acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");
        let bytes = acc.read_entry("empty.bin").expect("read empty");
        assert_eq!(&bytes[..], b"");
    }

    /// Many entries (≥ 100) round-trip via FsAccessor.
    #[test]
    fn fs_many_entries_roundtrip() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let n: u32 = 128;
        for i in 0..n {
            let path = tmp.path().join(format!("entry_{i:04}.bin"));
            std::fs::write(&path, i.to_le_bytes()).expect("write entry");
        }
        let acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");
        let listed = acc.list_prefix("").expect("list_prefix");
        assert_eq!(listed.len(), n as usize);
        for i in 0..n {
            let rel = format!("entry_{i:04}.bin");
            let got = acc.read_entry(&rel).expect("read entry");
            assert_eq!(&got[..], &i.to_le_bytes()[..]);
        }
    }

    /// FsAccessor must be able to read a hidden file (`.foo`).
    #[test]
    fn fs_hidden_file_reachable() {
        let tmp = tempfile::tempdir().expect("tempdir");
        std::fs::write(tmp.path().join(".foo"), b"hidden").expect("write hidden");
        let acc = FsAccessor::open(tmp.path()).expect("open FsAccessor");
        assert!(acc.exists(".foo"));
        let got = acc.read_entry(".foo").expect("read hidden");
        assert_eq!(&got[..], b"hidden");
    }

    /// ZipAccessor on a zip with a single zero-byte stored entry must
    /// round-trip cleanly.
    #[test]
    fn zip_zero_byte_stored_entry() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let zip_path = tmp.path().join("empty.zip");
        let f = File::create(&zip_path).expect("create zip");
        let mut zw = zip::write::ZipWriter::new(f);
        let opts =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        zw.start_file("empty.bin", opts).expect("start zip entry");
        // No bytes written.
        zw.finish().expect("finish zip");

        let acc = ZipAccessor::open(&zip_path).expect("open ZipAccessor");
        assert!(acc.exists("empty.bin"));
        let got = acc.read_entry("empty.bin").expect("read empty zip entry");
        assert_eq!(&got[..], b"");
    }

    /// ZipAccessor must handle a single archive that mixes Stored and
    /// Deflated compression methods.
    #[test]
    fn zip_mixed_stored_and_deflated() {
        let tmp = tempfile::tempdir().expect("tempdir");
        let zip_path = tmp.path().join("mixed.zip");
        let f = File::create(&zip_path).expect("create zip");
        let mut zw = zip::write::ZipWriter::new(f);

        let stored =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Stored);
        zw.start_file("stored.bin", stored).expect("start stored");
        zw.write_all(b"raw bytes").expect("write stored");

        let deflated =
            zip::write::FileOptions::default().compression_method(zip::CompressionMethod::Deflated);
        zw.start_file("deflated.bin", deflated)
            .expect("start deflated");
        // Use a payload that compresses well so the stored and deflated
        // sizes diverge — confirms the reader actually decompresses.
        let payload: Vec<u8> = std::iter::repeat_n(b'a', 2048).collect();
        zw.write_all(&payload).expect("write deflated");

        zw.finish().expect("finish zip");

        let acc = ZipAccessor::open(&zip_path).expect("open ZipAccessor");
        let stored_got = acc.read_entry("stored.bin").expect("read stored");
        assert_eq!(&stored_got[..], b"raw bytes");
        let deflated_got = acc.read_entry("deflated.bin").expect("read deflated");
        assert_eq!(&deflated_got[..], &payload[..]);
    }

    /// A symlink that lives *inside* the bundle root but points to a
    /// file *outside* the root is the canonical case the canonicalize
    /// step has to catch (lexical checks alone won't notice).
    #[cfg(unix)]
    #[test]
    fn fs_rejects_symlink_escaping_root() {
        use std::os::unix::fs::symlink;

        let tmp = tempfile::tempdir().expect("tempdir");
        // The "outside" file lives in a sibling directory of the
        // bundle root, so a symlink pointing at it has to be caught
        // by the canonicalize-and-prefix-check.
        let outside_dir = tmp.path().join("outside");
        std::fs::create_dir(&outside_dir).expect("create outside dir");
        let secret = outside_dir.join("secret.txt");
        std::fs::write(&secret, b"top secret").expect("write secret");

        let bundle_root = tmp.path().join("bundle");
        std::fs::create_dir(&bundle_root).expect("create bundle root");
        // Symlink at bundle_root/escape -> ../outside/secret.txt
        symlink(&secret, bundle_root.join("escape")).expect("create symlink");

        let acc = FsAccessor::open(&bundle_root).expect("open FsAccessor");

        let err = acc
            .read_entry("escape")
            .expect_err("symlink escaping root should be rejected");
        assert_eq!(
            err.kind(),
            io::ErrorKind::InvalidInput,
            "expected InvalidInput, got {:?}: {}",
            err.kind(),
            err
        );
    }
}
