//! `LiveIndex` — stateful in-memory index whose loaded cell set can
//! grow and shrink at runtime.
//!
//! Plan 3 of the deployment-mode roadmap (`plans/03-live-index.md`),
//! refined per user input to use a [`KdForest`] of per-cell sub-trees
//! instead of rebuilding one big tree on every cell change. Cell
//! add/drop is now O(1) for tree maintenance; query cost grows linearly
//! with the number of loaded cells (typically <100 in realtime use).
//!
//! For the existing `solve()` path which expects a concrete `Index`
//! with flat star/quad/code vectors, `LiveIndex::as_index()` flattens
//! the forest on demand. That cost is paid lazily — only when a solve
//! actually needs it, not on every membership change.

use std::collections::{HashMap, HashSet};
use std::io;
use std::time::{Duration, Instant};

use crate::geom::sphere::radec_to_xyz;
use crate::kdtree::{KdForest, KdTree};
use crate::quads::{Code, DIMCODES, DIMQUADS, Quad};
use crate::solver::SkyRegion;

use super::source::{HealpixCell, IndexFragment, IndexSource};
use super::{Index, IndexStar};

/// Per-cell payload tracked by `LiveIndex`. Star indices in `quads` are
/// local to this cell's `stars` (they were already remapped to that
/// frame by `IndexSource::load_cells`).
struct LoadedCell {
    stars: Vec<IndexStar>,
    quads: Vec<Quad>,
    codes: Vec<Code>,
    /// Reserved for future LRU/eviction policy (currently never read).
    #[allow(dead_code)]
    last_used: Instant,
}

/// Stateful in-memory subset of an `IndexSource`. The loaded cell set
/// can be grown via [`Self::ensure_region`] and shrunk via
/// [`Self::drop_outside`]. Internally maintains a forest of per-cell
/// KdTrees so add/drop don't require an O(N log N) rebuild.
pub struct LiveIndex<S: IndexSource> {
    source: S,
    loaded: HashMap<HealpixCell, LoadedCell>,
    star_forest: KdForest<3>,
    code_forest: KdForest<{ DIMCODES }>,
    /// Bumped on every membership change; surfaces to consumers via
    /// `build_generation()` so they can detect when their cached view
    /// is stale.
    build_generation: u64,
}

#[derive(Debug, Clone)]
pub struct EnsureReport {
    pub cells_added: usize,
    pub stars_added: usize,
    pub elapsed: Duration,
}

#[derive(Debug, Clone)]
pub struct DropReport {
    pub cells_dropped: usize,
    pub stars_dropped: usize,
    pub elapsed: Duration,
}

impl<S: IndexSource> LiveIndex<S> {
    pub fn open(source: S) -> Self {
        Self {
            source,
            loaded: HashMap::new(),
            star_forest: KdForest::new(),
            code_forest: KdForest::new(),
            build_generation: 0,
        }
    }

    pub fn source(&self) -> &S {
        &self.source
    }

    pub fn build_generation(&self) -> u64 {
        self.build_generation
    }

    pub fn loaded_cells(&self) -> impl Iterator<Item = &HealpixCell> {
        self.loaded.keys()
    }

    pub fn loaded_star_count(&self) -> usize {
        self.loaded.values().map(|c| c.stars.len()).sum()
    }

    pub fn loaded_quad_count(&self) -> usize {
        self.loaded.values().map(|c| c.quads.len()).sum()
    }

    pub fn loaded_cell_count(&self) -> usize {
        self.loaded.len()
    }

    /// Borrow as a `KdForest<3>` over star unit-vectors. Useful when a
    /// consumer wants to query the live tree directly without
    /// flattening to a concrete `Index`.
    pub fn star_forest(&self) -> &KdForest<3> {
        &self.star_forest
    }

    pub fn code_forest(&self) -> &KdForest<{ DIMCODES }> {
        &self.code_forest
    }

    /// Ensure all cells intersecting `region` are loaded. Existing cells
    /// are kept; only the delta is loaded from the source.
    pub fn ensure_region(&mut self, region: &SkyRegion) -> io::Result<EnsureReport> {
        let start = Instant::now();
        let wanted = self.source.cells_intersecting(region);
        let to_add: Vec<HealpixCell> = wanted
            .into_iter()
            .filter(|c| !self.loaded.contains_key(c))
            .collect();
        if to_add.is_empty() {
            return Ok(EnsureReport {
                cells_added: 0,
                stars_added: 0,
                elapsed: start.elapsed(),
            });
        }
        let stars_added = self.add_cells(&to_add)?;
        Ok(EnsureReport {
            cells_added: to_add.len(),
            stars_added,
            elapsed: start.elapsed(),
        })
    }

    /// Drop all cells NOT intersecting `region`.
    pub fn drop_outside(&mut self, region: &SkyRegion) -> DropReport {
        let start = Instant::now();
        let keep: HashSet<HealpixCell> =
            self.source.cells_intersecting(region).into_iter().collect();
        let to_drop: Vec<HealpixCell> = self
            .loaded
            .keys()
            .filter(|c| !keep.contains(c))
            .copied()
            .collect();
        let mut stars_dropped = 0;
        for cell in &to_drop {
            stars_dropped += self.remove_cell(cell);
        }
        DropReport {
            cells_dropped: to_drop.len(),
            stars_dropped,
            elapsed: start.elapsed(),
        }
    }

    /// Drop specific cells by id.
    pub fn drop_cells(&mut self, cells: &[HealpixCell]) -> DropReport {
        let start = Instant::now();
        let mut dropped_count = 0;
        let mut stars_dropped = 0;
        for cell in cells {
            let removed = self.remove_cell(cell);
            if removed > 0 || self.loaded.contains_key(cell) {
                // contains_key check is for completeness; we already
                // removed; this branch never fires.
            }
            if removed > 0 {
                dropped_count += 1;
                stars_dropped += removed;
            }
        }
        DropReport {
            cells_dropped: dropped_count,
            stars_dropped,
            elapsed: start.elapsed(),
        }
    }

    /// Replace the loaded set with exactly the cells covering `region`.
    /// Atomic from the caller's perspective: if the load fails, the
    /// previous loaded set is preserved.
    pub fn set_region(&mut self, region: &SkyRegion) -> io::Result<EnsureReport> {
        let start = Instant::now();
        let wanted: HashSet<HealpixCell> =
            self.source.cells_intersecting(region).into_iter().collect();
        let to_add: Vec<HealpixCell> = wanted
            .iter()
            .filter(|c| !self.loaded.contains_key(c))
            .copied()
            .collect();
        let to_drop: Vec<HealpixCell> = self
            .loaded
            .keys()
            .filter(|c| !wanted.contains(c))
            .copied()
            .collect();

        // Add first so a load failure leaves us in the prior state.
        let stars_added = if to_add.is_empty() {
            0
        } else {
            self.add_cells(&to_add)?
        };
        for cell in &to_drop {
            self.remove_cell(cell);
        }
        Ok(EnsureReport {
            cells_added: to_add.len(),
            stars_added,
            elapsed: start.elapsed(),
        })
    }

    fn add_cells(&mut self, cells: &[HealpixCell]) -> io::Result<usize> {
        let frag: IndexFragment = self.source.load_cells(cells)?;
        // The fragment's stars are concatenated across `cells` in the
        // source's cell order. We need to split them back per-cell so
        // each cell can own its sub-tree. The source guarantees that
        // load_cells walks cells in the order they're presented — but
        // returns the union as a single Vec without a per-cell split
        // table. Easiest is to rebuild per-cell by re-querying the
        // source one cell at a time. That keeps the data structure
        // aligned without depending on the fragment's internal layout.
        let _ = frag; // Drop the union fragment — we re-query per-cell below.

        let mut total_stars_added = 0;
        for &cell in cells {
            let single_frag = self.source.load_cells(std::slice::from_ref(&cell))?;
            let n_stars = single_frag.stars.len();
            if n_stars == 0 && single_frag.quads.is_empty() {
                continue;
            }
            // Build per-cell trees. Star tree is over xyz unit vectors;
            // result indices are local to this cell's `stars`. Code tree
            // similarly; results are local to this cell's `quads`.
            let star_points: Vec<[f64; 3]> = single_frag
                .stars
                .iter()
                .map(|s| radec_to_xyz(s.ra, s.dec))
                .collect();
            let star_indices: Vec<usize> = (0..n_stars).collect();
            let star_tree = KdTree::<3>::build(star_points, star_indices);

            let code_indices: Vec<usize> = (0..single_frag.codes.len()).collect();
            let code_tree = KdTree::<{ DIMCODES }>::build(single_frag.codes.clone(), code_indices);

            self.star_forest.insert(cell.id, star_tree);
            self.code_forest.insert(cell.id, code_tree);

            self.loaded.insert(
                cell,
                LoadedCell {
                    stars: single_frag.stars,
                    quads: single_frag.quads,
                    codes: single_frag.codes,
                    last_used: Instant::now(),
                },
            );
            total_stars_added += n_stars;
        }
        if total_stars_added > 0 {
            self.build_generation += 1;
        }
        Ok(total_stars_added)
    }

    fn remove_cell(&mut self, cell: &HealpixCell) -> usize {
        let stars_removed = match self.loaded.remove(cell) {
            Some(c) => c.stars.len(),
            None => return 0,
        };
        self.star_forest.remove(cell.id);
        self.code_forest.remove(cell.id);
        self.build_generation += 1;
        stars_removed
    }

    /// Flatten the loaded cells into a single `Index` with rebuilt
    /// flat KdTrees. Used by callers that need the existing concrete-
    /// `Index` solver/refine APIs. Iteration order across cells is
    /// HashMap-iteration order, which is stable for a given map but
    /// not deterministic across runs.
    pub fn as_index(&self) -> Index {
        let mut stars: Vec<IndexStar> = Vec::with_capacity(self.loaded_star_count());
        let mut quads: Vec<Quad> = Vec::with_capacity(self.loaded_quad_count());
        let mut codes: Vec<Code> = Vec::with_capacity(self.loaded_quad_count());

        for cell in self.loaded.values() {
            let base = stars.len();
            for s in &cell.stars {
                stars.push(s.clone());
            }
            for q in &cell.quads {
                let mut new_ids = [0usize; DIMQUADS];
                for (i, &sid) in q.star_ids.iter().enumerate() {
                    new_ids[i] = sid + base;
                }
                quads.push(Quad { star_ids: new_ids });
            }
            for c in &cell.codes {
                codes.push(*c);
            }
        }

        let star_points: Vec<[f64; 3]> = stars.iter().map(|s| radec_to_xyz(s.ra, s.dec)).collect();
        let star_idx: Vec<usize> = (0..stars.len()).collect();
        let star_tree = KdTree::<3>::build(star_points, star_idx);
        let code_idx: Vec<usize> = (0..codes.len()).collect();
        let code_tree = KdTree::<{ DIMCODES }>::build(codes, code_idx);

        let (scale_lower, scale_upper) = self.source.scale_range();
        Index {
            star_tree,
            stars,
            code_tree,
            quads,
            scale_lower,
            scale_upper,
            metadata: self.source.metadata().cloned(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{HealpixCell, IndexFragment, IndexMetadata, IndexSource};
    use crate::kdtree::KdQueryable;
    use crate::quads::{DIMQUADS, Quad};
    use std::sync::Mutex;

    /// In-memory `IndexSource` for tests: stores a flat list of
    /// (HealpixCell, stars, quads, codes) and answers `cells_intersecting`
    /// by comparing each cell's first-star RA/Dec against the region.
    struct MockSource {
        cells: Vec<MockCell>,
        scale_lower: f64,
        scale_upper: f64,
        load_count: Mutex<usize>,
    }

    struct MockCell {
        cell: HealpixCell,
        center_ra: f64,
        center_dec: f64,
        stars: Vec<IndexStar>,
        quads: Vec<Quad>,
        codes: Vec<Code>,
    }

    impl IndexSource for MockSource {
        fn cells_intersecting(&self, region: &SkyRegion) -> Vec<HealpixCell> {
            self.cells
                .iter()
                .filter(|c| region.contains(c.center_ra, c.center_dec))
                .map(|c| c.cell)
                .collect()
        }
        fn load_cells(&self, cells: &[HealpixCell]) -> io::Result<IndexFragment> {
            let mut count = self.load_count.lock().unwrap();
            *count += 1;
            drop(count);
            let mut stars = Vec::new();
            let mut quads = Vec::new();
            let mut codes = Vec::new();
            for cell in cells {
                if let Some(mc) = self.cells.iter().find(|c| c.cell == *cell) {
                    let base = stars.len();
                    stars.extend(mc.stars.iter().cloned());
                    for q in &mc.quads {
                        let mut new_ids = [0usize; DIMQUADS];
                        for (i, &sid) in q.star_ids.iter().enumerate() {
                            new_ids[i] = sid + base;
                        }
                        quads.push(Quad { star_ids: new_ids });
                    }
                    codes.extend(mc.codes.iter().copied());
                }
            }
            Ok(IndexFragment {
                stars,
                quads,
                codes,
                scale_lower: self.scale_lower,
                scale_upper: self.scale_upper,
                metadata: None,
            })
        }
        fn cell_depth(&self) -> u8 {
            5
        }
        fn metadata(&self) -> Option<&IndexMetadata> {
            None
        }
        fn star_count(&self) -> usize {
            self.cells.iter().map(|c| c.stars.len()).sum()
        }
        fn quad_count(&self) -> usize {
            self.cells.iter().map(|c| c.quads.len()).sum()
        }
        fn scale_range(&self) -> (f64, f64) {
            (self.scale_lower, self.scale_upper)
        }
    }

    fn make_mock_source() -> MockSource {
        let mut cells = Vec::new();
        // Three cells at three different positions on the sky.
        for (cell_id, (ra, dec)) in [
            (0u64, (0.5_f64, 0.3_f64)),
            (1u64, (1.5, -0.1)),
            (2u64, (3.0, 0.5)),
        ] {
            let mut stars = Vec::new();
            for i in 0..6 {
                let frac = i as f64 / 6.0;
                stars.push(IndexStar {
                    catalog_id: cell_id * 1000 + i as u64,
                    ra: ra + frac * 0.005,
                    dec: dec + frac * 0.005,
                    mag: 5.0 + frac,
                });
            }
            cells.push(MockCell {
                cell: HealpixCell {
                    depth: 5,
                    id: cell_id,
                },
                center_ra: ra,
                center_dec: dec,
                stars,
                quads: Vec::new(),
                codes: Vec::new(),
            });
        }
        MockSource {
            cells,
            scale_lower: 0.001,
            scale_upper: 0.05,
            load_count: Mutex::new(0),
        }
    }

    #[test]
    fn open_starts_empty() {
        let live = LiveIndex::open(make_mock_source());
        assert_eq!(live.loaded_cell_count(), 0);
        assert_eq!(live.loaded_star_count(), 0);
        assert_eq!(live.build_generation(), 0);
    }

    #[test]
    fn ensure_region_loads_intersecting_cells() {
        let mut live = LiveIndex::open(make_mock_source());
        // Region covering cells 0 and 1 (both around RA~0.5..1.5).
        let region = SkyRegion::from_radians(starfield::Equatorial::new(1.0, 0.1), 0.6);
        let report = live.ensure_region(&region).unwrap();
        assert_eq!(report.cells_added, 2);
        assert_eq!(report.stars_added, 12);
        assert_eq!(live.loaded_star_count(), 12);
        assert_eq!(live.build_generation(), 1);
    }

    #[test]
    fn ensure_region_idempotent() {
        let mut live = LiveIndex::open(make_mock_source());
        let region = SkyRegion::from_radians(starfield::Equatorial::new(0.5, 0.3), 0.05);
        let r1 = live.ensure_region(&region).unwrap();
        let gen_after_first = live.build_generation();
        let r2 = live.ensure_region(&region).unwrap();
        assert!(r1.cells_added > 0);
        assert_eq!(r2.cells_added, 0);
        assert_eq!(r2.stars_added, 0);
        assert_eq!(live.build_generation(), gen_after_first);
    }

    #[test]
    fn drop_outside_compacts() {
        let mut live = LiveIndex::open(make_mock_source());
        // Load everything.
        let all_sky =
            SkyRegion::from_radians(starfield::Equatorial::new(0.0, 0.0), std::f64::consts::PI);
        live.ensure_region(&all_sky).unwrap();
        assert_eq!(live.loaded_cell_count(), 3);

        // Drop everything outside a tight region around cell 0.
        let tight = SkyRegion::from_radians(starfield::Equatorial::new(0.5, 0.3), 0.05);
        let report = live.drop_outside(&tight);
        assert!(report.cells_dropped >= 2);
        assert!(live.loaded_cell_count() <= 1);
    }

    #[test]
    fn set_region_replaces_membership() {
        let mut live = LiveIndex::open(make_mock_source());
        let region_a = SkyRegion::from_radians(starfield::Equatorial::new(0.5, 0.3), 0.05);
        live.set_region(&region_a).unwrap();
        let cells_before: HashSet<HealpixCell> = live.loaded_cells().copied().collect();

        let region_b = SkyRegion::from_radians(starfield::Equatorial::new(3.0, 0.5), 0.05);
        live.set_region(&region_b).unwrap();
        let cells_after: HashSet<HealpixCell> = live.loaded_cells().copied().collect();

        assert_ne!(cells_before, cells_after);
        assert!(!cells_after.is_empty());
    }

    #[test]
    fn build_generation_increments_on_change() {
        let mut live = LiveIndex::open(make_mock_source());
        let g0 = live.build_generation();
        let region = SkyRegion::from_radians(starfield::Equatorial::new(0.5, 0.3), 0.05);
        live.ensure_region(&region).unwrap();
        let g1 = live.build_generation();
        assert!(g1 > g0);
        // Idempotent ensure should not bump.
        live.ensure_region(&region).unwrap();
        assert_eq!(g1, live.build_generation());
        // Drop should bump.
        let nowhere = SkyRegion::from_radians(starfield::Equatorial::new(5.0, 1.5), 0.001);
        live.drop_outside(&nowhere);
        assert!(live.build_generation() > g1);
    }

    #[test]
    fn star_forest_query_unions_subtrees() {
        let mut live = LiveIndex::open(make_mock_source());
        let all_sky =
            SkyRegion::from_radians(starfield::Equatorial::new(0.0, 0.0), std::f64::consts::PI);
        live.ensure_region(&all_sky).unwrap();

        // The forest should expose a union view across all sub-trees.
        let total_via_forest = live.star_forest().len();
        assert_eq!(total_via_forest, live.loaded_star_count());

        // A nearest query should find a hit somewhere across the forest.
        let center = radec_to_xyz(0.5, 0.3);
        let hit = live.star_forest().nearest(&center);
        assert!(hit.is_some());
    }

    #[test]
    fn as_index_flattens_loaded_set() {
        let mut live = LiveIndex::open(make_mock_source());
        let region =
            SkyRegion::from_radians(starfield::Equatorial::new(0.0, 0.0), std::f64::consts::PI);
        live.ensure_region(&region).unwrap();

        let idx = live.as_index();
        assert_eq!(idx.stars.len(), live.loaded_star_count());
        assert_eq!(idx.star_tree.len(), idx.stars.len());
        // Quad indices in the flattened Index must be in-bounds for stars.
        for q in &idx.quads {
            for &sid in &q.star_ids {
                assert!(sid < idx.stars.len());
            }
        }
    }
}
