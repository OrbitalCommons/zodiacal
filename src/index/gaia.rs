//! Streaming Gaia DR1 catalog reader.
//!
//! Reads gzipped CSV files directly from the starfield cache directory,
//! exploiting the fact that Gaia source_ids encode HEALPix level-12 pixels
//! in their high bits, so files are spatially ordered.

use std::fs;
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use flate2::read::GzDecoder;
use starfield::catalogs::StarData;

use super::catalog::SpatialCatalog;

/// Gaia HEALPix depth encoded in source_id (level 12).
const GAIA_SOURCE_ID_DEPTH: u8 = 12;
/// Number of bits to shift source_id right to get the HEALPix level-12 pixel.
const GAIA_SOURCE_ID_SHIFT: u32 = 35;

/// Extract the HEALPix level-12 pixel from a Gaia source_id.
fn healpix_from_source_id(source_id: u64) -> u64 {
    source_id >> GAIA_SOURCE_ID_SHIFT
}

/// A spatial catalog backed by Gaia DR1 gzipped CSV files on disk.
///
/// On construction, scans file headers to build a source_id range → file mapping.
/// Queries decompress and parse only the files needed for the requested cells.
pub struct GaiaSpatialCatalog {
    /// Sorted list of (first_source_id, last_source_id, file_path).
    file_ranges: Vec<FileRange>,
    /// Magnitude limit — skip stars fainter than this.
    mag_limit: f64,
}

struct FileRange {
    /// First source_id in this file (from first data row).
    first_source_id: u64,
    /// Path to the gzipped CSV file.
    path: PathBuf,
}

impl GaiaSpatialCatalog {
    /// Build a Gaia spatial catalog from a directory of gzipped CSV files.
    ///
    /// Reads the first data line of each file to determine its source_id range.
    /// This is relatively fast (~seconds for 5000 files) since it only decompresses
    /// the first few bytes of each file.
    pub fn open(dir: &Path, mag_limit: f64) -> std::io::Result<Self> {
        let mut entries: Vec<PathBuf> = fs::read_dir(dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| {
                p.extension().is_some_and(|ext| ext == "gz")
                    && p.file_name()
                        .is_some_and(|n| n.to_string_lossy().starts_with("GaiaSource"))
            })
            .collect();
        entries.sort();

        eprintln!(
            "Scanning {} Gaia files for source_id ranges...",
            entries.len()
        );

        let mut file_ranges = Vec::with_capacity(entries.len());
        for path in entries {
            if let Some(first_id) = read_first_source_id(&path) {
                file_ranges.push(FileRange {
                    first_source_id: first_id,
                    path,
                });
            }
        }

        // Sort by first_source_id so we can binary search.
        file_ranges.sort_by_key(|f| f.first_source_id);

        eprintln!(
            "Indexed {} Gaia files (mag_limit={:.1})",
            file_ranges.len(),
            mag_limit
        );

        Ok(Self {
            file_ranges,
            mag_limit,
        })
    }

    /// Default Gaia cache directory (~/.cache/starfield/gaia/).
    pub fn default_dir() -> Option<PathBuf> {
        dirs_next::cache_dir().map(|d| d.join("starfield").join("gaia"))
    }

    /// Open from the default cache directory.
    pub fn open_default(mag_limit: f64) -> std::io::Result<Self> {
        let dir = Self::default_dir().ok_or_else(|| {
            std::io::Error::new(std::io::ErrorKind::NotFound, "no cache directory found")
        })?;
        Self::open(&dir, mag_limit)
    }

    /// Find files that might contain stars for the given HEALPix level-12 cells.
    fn files_for_cells(&self, cells_12: &[u64]) -> Vec<&Path> {
        if cells_12.is_empty() || self.file_ranges.is_empty() {
            return Vec::new();
        }

        // Convert cells to source_id ranges.
        let mut ranges: Vec<(u64, u64)> = cells_12
            .iter()
            .map(|&cell| {
                let lo = cell << GAIA_SOURCE_ID_SHIFT;
                let hi = ((cell + 1) << GAIA_SOURCE_ID_SHIFT) - 1;
                (lo, hi)
            })
            .collect();
        ranges.sort_unstable();

        // Find files whose source_id range overlaps any of our target ranges.
        let mut result = Vec::new();
        for (i, fr) in self.file_ranges.iter().enumerate() {
            let file_start = fr.first_source_id;
            // File end is approximated as the start of the next file (or u64::MAX).
            let file_end = self
                .file_ranges
                .get(i + 1)
                .map(|next| next.first_source_id - 1)
                .unwrap_or(u64::MAX);

            for &(lo, hi) in &ranges {
                if file_start <= hi && file_end >= lo {
                    result.push(fr.path.as_path());
                    break;
                }
            }
        }

        result
    }

    /// Stream all stars from a file that fall within the given level-12 cells.
    fn read_stars_in_cells(&self, path: &Path, cells_12: &[u64]) -> Vec<StarData> {
        let cell_set: std::collections::HashSet<u64> = cells_12.iter().copied().collect();
        let mut stars = Vec::new();

        let file = match fs::File::open(path) {
            Ok(f) => f,
            Err(_) => return stars,
        };

        let reader = BufReader::new(GzDecoder::new(file));
        let mut lines = reader.lines();

        // Skip header.
        let _ = lines.next();

        for line in lines {
            let line = match line {
                Ok(l) => l,
                Err(_) => continue,
            };

            if let Some(star) = parse_gaia_line(&line, self.mag_limit) {
                let cell = healpix_from_source_id(star.id);
                if cell_set.contains(&cell) {
                    stars.push(star);
                }
            }
        }

        stars
    }

    /// Resolve a depth/cell query to HEALPix level-12 cells.
    fn cells_at_depth_12(depth: u8, cell: u64) -> Vec<u64> {
        if depth == GAIA_SOURCE_ID_DEPTH {
            vec![cell]
        } else if depth < GAIA_SOURCE_ID_DEPTH {
            // Coarser query — enumerate all level-12 children.
            let shift = 2 * (GAIA_SOURCE_ID_DEPTH - depth) as u64;
            let start = cell << shift;
            let count = 1u64 << shift;
            (start..start + count).collect()
        } else {
            // Finer than level 12 — map to parent.
            let shift = 2 * (depth - GAIA_SOURCE_ID_DEPTH) as u64;
            vec![cell >> shift]
        }
    }
}

impl SpatialCatalog for GaiaSpatialCatalog {
    fn stars_in_cell(&self, depth: u8, cell: u64) -> Vec<StarData> {
        let cells_12 = Self::cells_at_depth_12(depth, cell);
        let files = self.files_for_cells(&cells_12);

        let mut stars = Vec::new();
        for path in files {
            stars.extend(self.read_stars_in_cells(path, &cells_12));
        }

        // If query is finer than level 12, filter to exact cell.
        if depth > GAIA_SOURCE_ID_DEPTH {
            stars
                .retain(|s| cdshealpix::nested::hash(depth, s.position.ra, s.position.dec) == cell);
        }

        stars
    }

    fn occupied_cells(&self, depth: u8) -> Vec<u64> {
        // Derive from the file ranges — each file's first source_id tells us
        // which level-12 cell it starts in.
        let mut cells: Vec<u64> = self
            .file_ranges
            .iter()
            .map(|fr| {
                let cell_12 = healpix_from_source_id(fr.first_source_id);
                if depth <= GAIA_SOURCE_ID_DEPTH {
                    let shift = 2 * (GAIA_SOURCE_ID_DEPTH - depth) as u64;
                    cell_12 >> shift
                } else {
                    // Can't determine finer cells without reading files.
                    let shift = 2 * (depth - GAIA_SOURCE_ID_DEPTH) as u64;
                    cell_12 << shift
                }
            })
            .collect();
        cells.sort_unstable();
        cells.dedup();
        cells
    }
}

/// Parse a single Gaia DR1 CSV line into a StarData.
///
/// Returns None if the magnitude exceeds the limit or the line can't be parsed.
/// Key columns (0-indexed): source_id=1, ra=4, dec=6, phot_g_mean_mag=51.
fn parse_gaia_line(line: &str, mag_limit: f64) -> Option<StarData> {
    let fields: Vec<&str> = line.split(',').collect();
    if fields.len() < 52 {
        return None;
    }

    let source_id: u64 = fields[1].parse().ok()?;
    let ra: f64 = fields[4].parse().ok()?;
    let dec: f64 = fields[6].parse().ok()?;
    let mag: f64 = fields[51].parse().ok()?;

    if mag > mag_limit {
        return None;
    }

    Some(StarData::new(source_id, ra, dec, mag, None))
}

/// Read the first source_id from a gzipped Gaia CSV file.
fn read_first_source_id(path: &Path) -> Option<u64> {
    let file = fs::File::open(path).ok()?;
    let reader = BufReader::new(GzDecoder::new(file));
    let mut lines = reader.lines();
    let _ = lines.next()?; // skip header
    let first_line = lines.next()?.ok()?;
    let source_id = first_line.split(',').nth(1)?.parse::<u64>().ok()?;
    Some(source_id)
}

/// List all Gaia CSV files in a directory.
fn list_gaia_files(dir: &Path) -> std::io::Result<Vec<PathBuf>> {
    let mut files: Vec<PathBuf> = fs::read_dir(dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.extension().is_some_and(|ext| ext == "gz")
                && p.file_name()
                    .is_some_and(|n| n.to_string_lossy().starts_with("GaiaSource"))
        })
        .collect();
    files.sort();
    Ok(files)
}

/// Process a single Gaia file: parse all stars under mag_limit, bin into
/// per-cell heaps keeping only the N brightest per cell.
fn process_file(
    path: &Path,
    mag_limit: f64,
    depth: u8,
    max_per_cell: usize,
) -> (HashMap<u64, BinaryHeap<HeapStar>>, u64) {
    let mut heaps: HashMap<u64, BinaryHeap<HeapStar>> = HashMap::new();
    let mut count: u64 = 0;

    let file = match fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return (heaps, 0),
    };

    let reader = BufReader::new(GzDecoder::new(file));
    let mut lines = reader.lines();
    let _ = lines.next(); // skip header

    for line in lines {
        let line = match line {
            Ok(l) => l,
            Err(_) => continue,
        };

        if let Some(star) = parse_gaia_line(&line, mag_limit) {
            count += 1;
            let cell = cdshealpix::nested::hash(depth, star.position.ra, star.position.dec);
            let hs = HeapStar {
                mag: star.magnitude,
                id: star.id,
                ra: star.position.ra,
                dec: star.position.dec,
            };

            let heap = heaps
                .entry(cell)
                .or_insert_with(|| BinaryHeap::with_capacity(max_per_cell + 1));

            if heap.len() < max_per_cell {
                heap.push(hs);
            } else if let Some(faintest) = heap.peek()
                && hs.mag < faintest.mag
            {
                heap.pop();
                heap.push(hs);
            }
        }
    }

    (heaps, count)
}

/// Merge per-file cell heaps into a single set of heaps.
fn merge_heaps(
    dst: &mut HashMap<u64, BinaryHeap<HeapStar>>,
    src: HashMap<u64, BinaryHeap<HeapStar>>,
    max_per_cell: usize,
) {
    for (cell, src_heap) in src {
        let dst_heap = dst
            .entry(cell)
            .or_insert_with(|| BinaryHeap::with_capacity(max_per_cell + 1));

        for hs in src_heap {
            if dst_heap.len() < max_per_cell {
                dst_heap.push(hs);
            } else if let Some(faintest) = dst_heap.peek()
                && hs.mag < faintest.mag
            {
                dst_heap.pop();
                dst_heap.push(hs);
            }
        }
    }
}

use std::collections::{BinaryHeap, HashMap};
use std::sync::atomic::{AtomicU64, Ordering};

use rayon::prelude::*;

use super::builder::{CatalogBuilderConfig, build_index_from_stars_pub};
use super::{Index, IndexStar};

/// Star entry for brightness-ordered heaps (faintest at top for eviction).
#[derive(PartialEq)]
struct HeapStar {
    mag: f64,
    id: u64,
    ra: f64,
    dec: f64,
}

impl Eq for HeapStar {}

impl PartialOrd for HeapStar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapStar {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.mag
            .partial_cmp(&other.mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Build a zodiacal index directly from Gaia DR1 CSV files on disk.
///
/// Processes all files in parallel using rayon, decompressing each file
/// exactly once. Stars are binned into HEALPix cells with per-cell
/// brightness heaps, then the uniformized star set is passed to the
/// standard index builder.
pub fn build_index_from_gaia(
    gaia_dir: &Path,
    mag_limit: f64,
    config: &CatalogBuilderConfig,
) -> std::io::Result<Index> {
    let files = list_gaia_files(gaia_dir)?;
    let n_files = files.len();
    let uni_depth = config.effective_uniformize_depth();
    let max_per_cell = config.max_stars_per_cell;

    eprintln!("Processing {} Gaia files in parallel...", n_files);
    eprintln!(
        "  depth={}, max_per_cell={}, mag_limit={:.1}",
        uni_depth, max_per_cell, mag_limit
    );

    let files_done = AtomicU64::new(0);
    let stars_total = AtomicU64::new(0);

    // Phase 1: parallel file processing — each file produces per-cell heaps.
    let per_file_heaps: Vec<HashMap<u64, BinaryHeap<HeapStar>>> = files
        .par_iter()
        .map(|path| {
            let (heaps, count) = process_file(path, mag_limit, uni_depth, max_per_cell);
            let done = files_done.fetch_add(1, Ordering::Relaxed) + 1;
            stars_total.fetch_add(count, Ordering::Relaxed);
            if done.is_multiple_of(100) || done == n_files as u64 {
                eprintln!("  [{}/{}] files processed...", done, n_files);
            }
            heaps
        })
        .collect();

    let total = stars_total.load(Ordering::Relaxed);
    eprintln!(
        "Parsed {} stars from {} files, merging heaps...",
        total, n_files
    );

    // Phase 2: merge all per-file heaps into one global set.
    let mut global_heaps: HashMap<u64, BinaryHeap<HeapStar>> = HashMap::new();
    for file_heaps in per_file_heaps {
        merge_heaps(&mut global_heaps, file_heaps, max_per_cell);
    }

    // Flatten heaps into sorted star list.
    let mut stars: Vec<IndexStar> = Vec::new();
    for (_, heap) in global_heaps {
        for hs in heap {
            stars.push(IndexStar {
                catalog_id: hs.id,
                ra: hs.ra,
                dec: hs.dec,
                mag: hs.mag,
            });
        }
    }
    stars.sort_by(|a, b| {
        a.mag
            .partial_cmp(&b.mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    eprintln!(
        "Uniformized to {} stars from {} cells",
        stars.len(),
        cdshealpix::nested::n_hash(uni_depth)
    );

    // Phase 3: build quads and KD-trees using the shared core.
    Ok(build_index_from_stars_pub(
        stars,
        config.scale_lower,
        config.scale_upper,
        config.effective_max_quads(),
    ))
}
