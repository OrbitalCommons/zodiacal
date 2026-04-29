# Plan 3: LiveIndex (add/drop API)

## Goal

A stateful in-memory index whose set of loaded cells can grow and shrink at
runtime. Internally maintains a `KdForest` of per-cell sub-trees so cell
add/drop is O(1) for tree maintenance — no full rebuild. Wraps an
`IndexSource` (plan 2) so it works with any backing store.

**Design update from the original plan**: rather than rebuilding the KD-tree
on every cell change (originally planned: full O(N log N) rebuild), we now
keep per-cell sub-trees in a `KdForest` and union queries across them. Cell
add = "build a small tree, append to forest"; cell drop = "remove the named
sub-tree." Per-query cost grows linearly with the number of loaded sub-trees
(typically <100 in realtime use); rebuild cost is gone. For the existing
`solve()` API which expects a flat `Index`, `LiveIndex::as_index()` flattens
on demand and pays rebuild cost only when actually called — not on every
membership change.

## Non-goals

- No background loading. `ensure_region` blocks until the new cells are in.
  (Background prefetch is a v2 enhancement; see `plans/05-realtime-solver.md`.)
- No per-cell reference counting. The current API treats "loaded set" as a
  single owner — tracking who asked for which cell adds complexity that the
  realtime cases don't actually need (one orchestrator drives membership).
- No incremental KD-tree updates. Full rebuild on every membership change.
  See cost analysis below.

## API surface

```rust
// src/index/live.rs (new file)

pub struct LiveIndex<S: IndexSource> {
    source: S,
    loaded: HashMap<HealpixCell, LoadedCell>,
    star_tree: KdTree<3>,
    code_tree: KdTree<{ DIMCODES }>,
    stars: Vec<IndexStar>,
    quads: Vec<Quad>,
    scale_lower: f64,
    scale_upper: f64,
    metadata: Option<IndexMetadata>,
    /// Tracks build version — incremented on every (re)build of trees.
    build_generation: u64,
}

struct LoadedCell {
    star_range: std::ops::Range<usize>,   // index into self.stars
    quad_range: std::ops::Range<usize>,   // (currently full-range in v1)
    last_used: Instant,                   // for future LRU eviction
}

impl<S: IndexSource> LiveIndex<S> {
    pub fn open(source: S) -> Self;

    /// Ensure all cells intersecting `region` are loaded. No-op for cells
    /// already loaded. Triggers a tree rebuild iff any cells were added.
    pub fn ensure_region(&mut self, region: &SkyRegion) -> io::Result<EnsureReport>;

    /// Drop all cells NOT intersecting `region`. Triggers a tree rebuild
    /// iff any cells were dropped.
    pub fn drop_outside(&mut self, region: &SkyRegion) -> DropReport;

    /// Drop specific cells by ID.
    pub fn drop_cells(&mut self, cells: &[HealpixCell]) -> DropReport;

    /// Replace the loaded set with exactly the cells covering `region`.
    /// Atomic from the caller's perspective: either succeeds or leaves
    /// state unchanged.
    pub fn set_region(&mut self, region: &SkyRegion) -> io::Result<EnsureReport>;

    pub fn loaded_cells(&self) -> impl Iterator<Item = &HealpixCell>;
    pub fn loaded_star_count(&self) -> usize;
    pub fn build_generation(&self) -> u64;

    /// Borrow as a regular `Index` for the existing solver/tweak/refine code.
    pub fn as_index(&self) -> &Index;
}

#[derive(Debug, Clone)]
pub struct EnsureReport {
    pub cells_added: usize,
    pub stars_added: usize,
    pub trees_rebuilt: bool,
    pub elapsed: Duration,
}

#[derive(Debug, Clone)]
pub struct DropReport {
    pub cells_dropped: usize,
    pub stars_dropped: usize,
    pub trees_rebuilt: bool,
    pub elapsed: Duration,
}
```

`as_index()` is the bridge to existing code — `solve()`, `tweak_solution()`,
and `refine_solution()` all take `&Index`, so anything that works against
`Index` works against `LiveIndex` via this borrow.

## Algorithm

### `ensure_region`

1. `wanted = source.cells_intersecting(region)`.
2. `to_add = wanted - self.loaded.keys()`.
3. If `to_add.is_empty()`: return early.
4. `fragment = source.load_cells(&to_add)?`.
5. Append fragment's stars to `self.stars`, recording `star_range` per cell.
6. Append fragment's quads/codes to `self.quads`/internal `codes` vector,
   shifting star indices by the original star_count.
7. Rebuild both KD-trees from scratch. Increment `build_generation`.
8. Return `EnsureReport`.

### `drop_outside`

1. `keep_cells = source.cells_intersecting(region).into_iter().collect::<HashSet>`.
2. `drop_set = self.loaded.keys() - keep_cells`.
3. If `drop_set.is_empty()`: return early.
4. **Compaction:** rebuild `self.stars` and `self.quads` from only the kept
   cells (in their existing order to preserve quad indices). Update each
   surviving cell's `star_range` and `quad_range`.
5. Rebuild both KD-trees. Increment `build_generation`.
6. Return `DropReport`.

### `set_region` (atomic replace)

1. `wanted = source.cells_intersecting(region)`.
2. Compute `(to_add, to_drop)` deltas vs. `self.loaded`.
3. If `to_add` requires a `source.load_cells` call, do it FIRST (failure path
   leaves state untouched).
4. Apply drops + adds in one shot, rebuild trees once.
5. Return `EnsureReport` (and discard the implicit drop report — caller
   typically only cares about new state, not delta).

### Tree rebuild cost analysis

`KdTree::build` is O(N log N). Rebuild times:

| Loaded stars | Estimated rebuild time |
|---|---|
| 1 k | < 1 ms |
| 10 k | ~5 ms |
| 100 k | ~50 ms |
| 1 M | ~500 ms |

For realtime modes with prior-bounded cell sets (typically < 100 k stars),
rebuild is well within tick-rate budget. For server mode loading the full
sky once, it's a one-time ~few-second cost (acceptable).

If rebuild ever becomes a bottleneck, the next move is incremental KD-tree
updates — but tracked as a future-only concern; not in v1.

### Memory model

`LiveIndex` owns `Vec<IndexStar>`, `Vec<Quad>`, two KD-trees. On a tree
rebuild we drop the old trees and build new ones — peak memory is briefly 2×.
For the budget where this matters (loaded set > 1M stars) we'd want to
build the new trees first, then atomically swap. v1 takes the simpler path.

## Backwards compatibility

Strictly additive. Existing `Index` and its API continue to work. Anything
that takes `&Index` accepts `LiveIndex::as_index()`.

## Tests

Unit tests in `src/index/live.rs::tests`:

- **`open_empty`**: open against a source with no stars; loaded set is empty.
- **`ensure_region_loads_cells`**: ensure a region; verify expected cells appear.
- **`ensure_region_idempotent`**: ensure same region twice; second call is no-op (`cells_added == 0`, `trees_rebuilt == false`).
- **`drop_outside_compacts`**: load 10 cells, drop_outside a region intersecting only 3; verify stars/quads compact correctly.
- **`set_region_atomic`**: stub a source that errors on a specific cell; verify failed `set_region` leaves prior state untouched.
- **`build_generation_increments_on_change`**: verify `build_generation` bumps when membership changes, stays the same when no-op.
- **`as_index_works_with_solve`**: synthetic scenario; load cells, run `solve()` against `as_index()`, verify it returns a solution.

In-memory `IndexSource` test fixture (`MockSource`) for these tests, so we
don't need a real file.

## Effort estimate

| Step | Effort |
|---|---|
| `LiveIndex` struct + `open` | 0.5 day |
| `ensure_region` + tree rebuild | 0.5 day |
| `drop_outside` + compaction | 0.5 day |
| `set_region` atomic replace | 0.5 day |
| `MockSource` test fixture | 0.5 day |
| Tests (7) | 0.5 day |
| Integration test against real `ZdclFile` | 0.5 day |
| **Total** | **~3 days** |

## Dependencies

- **Plan 2** must land first (`IndexSource` trait + `ZdclFile`).
- Independent of plans 1, 4, 5.

## Open questions

- **Should `LiveIndex` track cell access for LRU?** Today: no — caller is
  responsible for issuing drops. The `LoadedCell.last_used` field is reserved
  for a future enhancement. Recommendation: add the field, don't use it in v1,
  but document that it's reserved for an `evict_lru(n)` API.
- **Concurrent reads while ensure/drop is mutating?** v1 takes `&mut self` for
  membership changes and `&self` for read access via `as_index`. The borrow
  checker enforces no concurrent mutation. For multi-threaded server use,
  callers wrap in `RwLock<LiveIndex<S>>`. Don't bake locks in.
- **Should `ensure_region` accept a `padding_rad` arg like plan 1?** Yes — the
  same boundary-quad concern applies. Default: 0; callers add their own.
- **Build provenance.** When `ensure_region` adds cells, the `metadata` block
  is the source's metadata, not a per-cell or per-snapshot metadata. That's
  fine for v1 since metadata is per-file. If we ever support multi-source
  composition, metadata becomes per-cell.
