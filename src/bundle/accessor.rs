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

use std::collections::HashMap;
use std::fs::File;
use std::io;
use std::ops::Deref;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use memmap2::Mmap;
use zip::ZipArchive;

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
    /// For [`FsAccessor`] this is an mmap-backed slice cached for the
    /// lifetime of the accessor; for [`ZipAccessor`] this is a freshly
    /// decompressed owned buffer.
    fn read_entry(&self, rel: &str) -> io::Result<EntryBytes<'_>>;
}

/// A read-only view of an entry's bytes.
///
/// `Mmap` borrows a slice from a memory map owned by the accessor;
/// `Owned` carries a freshly decompressed buffer. Both deref to `[u8]`,
/// so consumer code can treat them identically.
pub enum EntryBytes<'a> {
    /// Borrowed slice into an mmap owned by the accessor.
    Mmap(&'a [u8]),
    /// Owned buffer (e.g. decompressed from a zip entry).
    Owned(Vec<u8>),
}

impl<'a> Deref for EntryBytes<'a> {
    type Target = [u8];

    fn deref(&self) -> &Self::Target {
        match self {
            EntryBytes::Mmap(slice) => slice,
            EntryBytes::Owned(buf) => buf.as_slice(),
        }
    }
}

impl<'a> AsRef<[u8]> for EntryBytes<'a> {
    fn as_ref(&self) -> &[u8] {
        self
    }
}

impl<'a> std::fmt::Debug for EntryBytes<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            EntryBytes::Mmap(slice) => f
                .debug_struct("EntryBytes::Mmap")
                .field("len", &slice.len())
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
/// Each entry is mmapped on first access; subsequent reads return a slice
/// into the cached mmap. The cache lives in a `Mutex<HashMap<...>>`
/// guarded by interior mutability, so the accessor itself is `Send + Sync`
/// and the trait method takes `&self`.
///
/// The mmaps are stored as `Box<Mmap>` to keep their addresses stable
/// across hash-map rehashes, which lets us hand out `&[u8]` slices
/// borrowing from those mmaps for the lifetime of the accessor.
pub struct FsAccessor {
    root: PathBuf,
    cache: Mutex<HashMap<String, Box<Mmap>>>,
}

impl FsAccessor {
    /// Open a directory root as an accessor.
    ///
    /// The path must point at an existing directory; otherwise an
    /// `io::Error` of kind [`io::ErrorKind::NotFound`] (or similar) is
    /// returned.
    pub fn open(root: impl AsRef<Path>) -> io::Result<Self> {
        let root = root.as_ref().to_path_buf();
        let meta = std::fs::metadata(&root)?;
        if !meta.is_dir() {
            return Err(io::Error::new(
                io::ErrorKind::NotADirectory,
                format!("{} is not a directory", root.display()),
            ));
        }
        Ok(Self {
            root,
            cache: Mutex::new(HashMap::new()),
        })
    }

    /// Resolve a bundle-relative path to an absolute filesystem path.
    fn resolve(&self, rel: &str) -> PathBuf {
        // Bundle paths use forward slashes; on Windows we'd have to
        // normalize, but on the platforms this crate targets `Path::join`
        // handles a forward-slash subpath correctly.
        let mut p = self.root.clone();
        for comp in rel.split('/') {
            if !comp.is_empty() {
                p.push(comp);
            }
        }
        p
    }
}

impl SubfileAccessor for FsAccessor {
    fn exists(&self, rel: &str) -> bool {
        self.resolve(rel).is_file()
    }

    fn list_prefix(&self, prefix: &str) -> io::Result<Vec<String>> {
        let mut out = Vec::new();
        walk_collect(&self.root, &self.root, prefix, &mut out)?;
        out.sort();
        Ok(out)
    }

    fn read_entry(&self, rel: &str) -> io::Result<EntryBytes<'_>> {
        // Fast path: cache hit.
        {
            let cache = self.cache.lock().expect("FsAccessor cache poisoned");
            if let Some(mmap) = cache.get(rel) {
                let slice: &[u8] = &mmap[..];
                // SAFETY: the `Box<Mmap>` is stored in the cache and
                // never removed for the lifetime of `self`. Its mapped
                // memory has a stable address (independent of the
                // HashMap's bucket layout, since we go through `Box`).
                // Therefore the slice is valid for `&'_ self`.
                let extended: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(slice) };
                return Ok(EntryBytes::Mmap(extended));
            }
        }

        // Slow path: open the file and mmap it.
        let abs = self.resolve(rel);
        let file = File::open(&abs)?;
        // SAFETY: we treat the mapping as immutable for the lifetime of
        // the accessor, which matches the `Mmap` (read-only) contract.
        let mmap = unsafe { Mmap::map(&file)? };
        let boxed = Box::new(mmap);

        let mut cache = self.cache.lock().expect("FsAccessor cache poisoned");
        // Another thread may have raced us; if so, drop our mapping and
        // use theirs so callers see a consistent ptr per entry.
        let entry = cache.entry(rel.to_string()).or_insert(boxed);
        let slice: &[u8] = &entry[..];
        let extended: &[u8] = unsafe { std::mem::transmute::<&[u8], &[u8]>(slice) };
        Ok(EntryBytes::Mmap(extended))
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

    fn read_entry(&self, rel: &str) -> io::Result<EntryBytes<'_>> {
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
}
