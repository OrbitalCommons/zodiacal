//! Spatial catalog abstraction for index building.
//!
//! Provides a trait for catalogs that can return stars grouped by HEALPix cell,
//! plus implementations for in-memory catalogs and streaming Gaia CSV files.

use std::collections::HashMap;

use starfield::catalogs::{StarCatalog, StarData};

/// A star catalog that can be queried by HEALPix cell.
///
/// This is the primary interface consumed by the index builder. Implementations
/// can be backed by in-memory data or stream from disk on demand.
pub trait SpatialCatalog {
    /// Return stars within a single HEALPix cell at the given depth.
    fn stars_in_cell(&self, depth: u8, cell: u64) -> Vec<StarData>;

    /// Return all cells that contain at least one star at the given depth.
    fn occupied_cells(&self, depth: u8) -> Vec<u64>;
}

/// Wraps any `StarCatalog` into a `SpatialCatalog` by binning all stars
/// into HEALPix cells at construction time.
pub struct InMemorySpatialCatalog {
    /// Stars grouped by (depth, cell) → Vec<StarData>.
    /// We store at the finest depth requested and aggregate for coarser queries.
    stars: Vec<StarData>,
}

impl InMemorySpatialCatalog {
    /// Build from any starfield `StarCatalog`.
    pub fn from_catalog(catalog: &impl StarCatalog) -> Self {
        let stars: Vec<StarData> = catalog.star_data().collect();
        Self { stars }
    }

    /// Build from an iterator of `StarData`.
    pub fn from_star_data(iter: impl Iterator<Item = StarData>) -> Self {
        Self {
            stars: iter.collect(),
        }
    }

    /// Number of stars in the catalog.
    pub fn len(&self) -> usize {
        self.stars.len()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.stars.is_empty()
    }
}

impl SpatialCatalog for InMemorySpatialCatalog {
    fn stars_in_cell(&self, depth: u8, cell: u64) -> Vec<StarData> {
        self.stars
            .iter()
            .filter(|s| cdshealpix::nested::hash(depth, s.position.ra, s.position.dec) == cell)
            .cloned()
            .collect()
    }

    fn occupied_cells(&self, depth: u8) -> Vec<u64> {
        let mut cells: Vec<u64> = self
            .stars
            .iter()
            .map(|s| cdshealpix::nested::hash(depth, s.position.ra, s.position.dec))
            .collect();
        cells.sort_unstable();
        cells.dedup();
        cells
    }
}

/// Pre-indexed in-memory spatial catalog.
///
/// Stars are binned into HEALPix cells at a fixed depth during construction,
/// making cell queries O(1) lookups instead of full scans.
pub struct IndexedSpatialCatalog {
    depth: u8,
    cells: HashMap<u64, Vec<StarData>>,
}

impl IndexedSpatialCatalog {
    /// Build from any starfield `StarCatalog` at a given HEALPix depth.
    pub fn from_catalog(catalog: &impl StarCatalog, depth: u8) -> Self {
        let mut cells: HashMap<u64, Vec<StarData>> = HashMap::new();
        for s in catalog.star_data() {
            let cell = cdshealpix::nested::hash(depth, s.position.ra, s.position.dec);
            cells.entry(cell).or_default().push(s);
        }
        Self { cells, depth }
    }

    /// Build from an iterator of `StarData` at a given HEALPix depth.
    pub fn from_star_data(iter: impl Iterator<Item = StarData>, depth: u8) -> Self {
        let mut cells: HashMap<u64, Vec<StarData>> = HashMap::new();
        for s in iter {
            let cell = cdshealpix::nested::hash(depth, s.position.ra, s.position.dec);
            cells.entry(cell).or_default().push(s);
        }
        Self { cells, depth }
    }

    /// Number of stars across all cells.
    pub fn len(&self) -> usize {
        self.cells.values().map(|v| v.len()).sum()
    }

    /// Whether the catalog is empty.
    pub fn is_empty(&self) -> bool {
        self.cells.is_empty()
    }

    /// The HEALPix depth this catalog was indexed at.
    pub fn depth(&self) -> u8 {
        self.depth
    }
}

impl SpatialCatalog for IndexedSpatialCatalog {
    fn stars_in_cell(&self, depth: u8, cell: u64) -> Vec<StarData> {
        if depth == self.depth {
            // Exact match — direct lookup.
            self.cells.get(&cell).cloned().unwrap_or_default()
        } else if depth < self.depth {
            // Coarser query — aggregate all child cells.
            let shift = 2 * (self.depth - depth) as u64;
            let child_start = cell << shift;
            let child_end = (cell + 1) << shift;
            let mut result = Vec::new();
            for (&c, stars) in &self.cells {
                if c >= child_start && c < child_end {
                    result.extend_from_slice(stars);
                }
            }
            result
        } else {
            // Finer query — find the parent cell and filter.
            let shift = 2 * (depth - self.depth) as u64;
            let parent = cell >> shift;
            self.cells
                .get(&parent)
                .map(|stars| {
                    stars
                        .iter()
                        .filter(|s| {
                            cdshealpix::nested::hash(depth, s.position.ra, s.position.dec) == cell
                        })
                        .cloned()
                        .collect()
                })
                .unwrap_or_default()
        }
    }

    fn occupied_cells(&self, depth: u8) -> Vec<u64> {
        if depth == self.depth {
            let mut cells: Vec<u64> = self.cells.keys().copied().collect();
            cells.sort_unstable();
            cells
        } else if depth < self.depth {
            // Coarser — map each stored cell to its parent and dedup.
            let shift = 2 * (self.depth - depth) as u64;
            let mut cells: Vec<u64> = self.cells.keys().map(|&c| c >> shift).collect();
            cells.sort_unstable();
            cells.dedup();
            cells
        } else {
            // Finer — expand each stored cell to its children and filter.
            let mut cells = Vec::new();
            for stars in self.cells.values() {
                for s in stars {
                    cells.push(cdshealpix::nested::hash(
                        depth,
                        s.position.ra,
                        s.position.dec,
                    ));
                }
            }
            cells.sort_unstable();
            cells.dedup();
            cells
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_stars() -> Vec<StarData> {
        // Stars at known positions spread across the sky
        vec![
            StarData::new(1, 10.0, 20.0, 5.0, None),
            StarData::new(2, 10.1, 20.1, 6.0, None),
            StarData::new(3, 180.0, -30.0, 4.0, None),
            StarData::new(4, 270.0, 60.0, 7.0, None),
        ]
    }

    #[test]
    fn in_memory_roundtrip() {
        let catalog = InMemorySpatialCatalog::from_star_data(make_test_stars().into_iter());
        assert_eq!(catalog.len(), 4);

        let cells = catalog.occupied_cells(2);
        assert!(!cells.is_empty());

        // All stars should be recoverable
        let mut total = 0;
        for &cell in &cells {
            total += catalog.stars_in_cell(2, cell).len();
        }
        assert_eq!(total, 4);
    }

    #[test]
    fn indexed_exact_depth() {
        let catalog = IndexedSpatialCatalog::from_star_data(make_test_stars().into_iter(), 4);
        assert_eq!(catalog.len(), 4);

        let cells = catalog.occupied_cells(4);
        let mut total = 0;
        for &cell in &cells {
            total += catalog.stars_in_cell(4, cell).len();
        }
        assert_eq!(total, 4);
    }

    #[test]
    fn indexed_coarser_query() {
        let catalog = IndexedSpatialCatalog::from_star_data(make_test_stars().into_iter(), 4);

        // Query at coarser depth should still return all stars
        let cells = catalog.occupied_cells(2);
        let mut total = 0;
        for &cell in &cells {
            total += catalog.stars_in_cell(2, cell).len();
        }
        assert_eq!(total, 4);
    }

    #[test]
    fn indexed_finer_query() {
        let catalog = IndexedSpatialCatalog::from_star_data(make_test_stars().into_iter(), 2);

        // Query at finer depth should still return all stars
        let cells = catalog.occupied_cells(4);
        let mut total = 0;
        for &cell in &cells {
            total += catalog.stars_in_cell(4, cell).len();
        }
        assert_eq!(total, 4);
    }
}
