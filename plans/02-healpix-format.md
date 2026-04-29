# Plan 2: HEALPix-grouped on-disk layout (file format v3)

## Goal

Re-organize the `.zdcl` file so stars are sorted by HEALPix cell, and add a
header table mapping `cell_id → (byte_offset, star_count)`. Loading a
`SkyRegion` then becomes O(region) on disk — for a 1° hint that's ~MB of I/O,
vs ~GB for plan 1.

## Non-goals

- No change to the in-memory `Index` API surface (KdTrees, etc.). Loaders just
  populate `Index` more efficiently.
- No quad-cell grouping in v1 of the new format — quads remain ungrouped and
  load fully (see "Cross-cutting decisions" in `plans/README.md`). Quad-by-cell
  is a future v4 if quad I/O becomes the bottleneck.

## API surface

```rust
// src/index/source.rs (new file)

pub trait IndexSource: Send + Sync {
    fn cells_intersecting(&self, region: &SkyRegion) -> Vec<HealpixCell>;
    fn load_cells(&self, cells: &[HealpixCell]) -> io::Result<IndexFragment>;
    fn cell_depth(&self) -> u8;
    fn metadata(&self) -> Option<&IndexMetadata>;
}

pub struct ZdclFile {
    mmap: Mmap,
    cell_table: Vec<CellEntry>,   // sorted by cell_id
    cell_depth: u8,
    metadata: Option<IndexMetadata>,
    quads_offset: usize,           // start of quads block
    n_quads: usize,
    codes_offset: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct HealpixCell {
    pub depth: u8,
    pub id: u64,
}

#[derive(Clone, Copy, Debug)]
struct CellEntry {
    cell_id: u64,
    file_offset: u64,    // bytes from start of file to this cell's first star
    star_count: u32,
}

#[derive(Debug)]
pub struct IndexFragment {
    pub stars: Vec<IndexStar>,
    pub quads: Vec<Quad>,        // all quads — see non-goals
    pub codes: Vec<Code>,        // parallel with quads
    pub scale_lower: f64,
    pub scale_upper: f64,
    pub metadata: Option<IndexMetadata>,
}

impl ZdclFile {
    pub fn open(path: &Path) -> io::Result<Self>;
    pub fn star_count(&self) -> usize;
    pub fn quad_count(&self) -> usize;
}

impl IndexSource for ZdclFile { ... }
```

`Index::load` is reimplemented in terms of `ZdclFile::load_full()` (which
delegates to `load_cells(all_cells)`). Backwards-compatible with v2 files via
an internal version-dispatch loader.

## File format v3

```
Offset                    Field                                Notes
-----------------------------------------------------------------------------
0                         magic = b"ZDCL"                      4 bytes
4                         version: u32 = 3                     4 bytes (was 2)
8                         metadata_len: u64                    8 bytes
16                        metadata bytes (UTF-8 JSON)          metadata_len bytes
16+meta_len               cell_depth: u8                       NEW: 1 byte
16+meta_len+1             reserved: [u8; 7]                    NEW: 7 bytes (zero)
16+meta_len+8             n_cells: u64                         NEW: 8 bytes
16+meta_len+16            cell_table:                          NEW: n_cells × 20 bytes
                            [(cell_id u64,                       (sorted by cell_id ascending)
                              file_offset u64,
                              star_count u32)] × n_cells
... aligned to 8 ...
star_block_offset         n_stars: u64                         (kept for reader convenience)
+8                        n_quads: u64
+16                       scale_lower: f64
+24                       scale_upper: f64
+32                       stars: [IndexStar] × n_stars         32 bytes each, GROUPED BY CELL
                                                                in cell_id ascending order
star_block_offset
  + 32 + n_stars*32       quads: [(star_ids: [u32; 4])]        16 bytes each
+ n_quads*16              codes: [(code: [f64; 4])]            32 bytes each
```

**Key invariants:**

- Stars are sorted by `cell_id` (computed from each star's `(ra, dec)` at
  build time using `cdshealpix::nested::hash(depth, ra_rad, dec_rad)`).
- `cell_table[k].file_offset` points to the first star of cell `k` within the
  star block, so loading a cell is `read_at(offset, star_count * 32)`.
- `cell_table` itself is sorted by `cell_id`, allowing binary search by cell.
- Cells with zero stars are omitted from the table (most depths leave many
  cells empty for sparse catalogs).
- Quads still refer to stars by their global compact index — i.e., the offset
  *within the star block*, which equals
  `cell_table[c].file_offset_index + intra_cell_idx`.

### Memory map layout intent

Designed so a `mmap`ed reader can cheaply:
- Parse the header (small, sequential, hot).
- Binary-search the cell table for the first cell ≥ `region`'s low cell ID.
- Read a contiguous range of cells via one or two `pread`/`copy_from_slice`
  operations.
- Borrow the quad and code blocks as `&[u8]` views without copying.

## Algorithm

### Builder (write side)

1. Existing builder runs as today, producing a `Vec<IndexStar>` and `Vec<Quad>`.
2. **New step**: assign a cell ID to each star using
   `cdshealpix::nested::hash(cell_depth, star.ra, star.dec)`.
3. Sort stars by `(cell_id, mag)`. Within a cell, brightest first stays a
   useful invariant for the matcher.
4. Recompute quad star indices to point to the new positions.
5. Build a `Vec<CellEntry>` by walking the sorted star list.
6. Write file in v3 layout.

### Reader (`ZdclFile::open`)

1. Open file, mmap it.
2. Parse header; reject if magic mismatch or version not in {2, 3}.
   - **v2 files**: synthesize a single virtual cell containing all stars, with
     `cell_table = [(cell_id=0, file_offset=star_block_start, star_count=n_stars)]`
     and `cell_depth = 0`. Loading any region returns the full set. This keeps
     the IndexSource API uniform across format versions.
3. Parse metadata JSON if `metadata_len > 0`.
4. Read cell table into a `Vec<CellEntry>`. For 12 × 4^5 = 12288 cells × 20
   bytes ≈ 240 KB — fits in L2.
5. Compute and stash `quads_offset` and `codes_offset`.

### Loading cells (`ZdclFile::load_cells`)

1. For each requested `HealpixCell`, binary-search `cell_table` for the entry
   (or skip if absent — empty cell).
2. Group consecutive cells into contiguous byte ranges (a "run-length" merge)
   to avoid many tiny reads.
3. For each run, do one mmap-slice copy into a `Vec<IndexStar>`.
4. Quads + codes: load all (per non-goal). Walk the quads block, drop any quad
   whose stars aren't in the loaded set, remap star indices.
5. Return `IndexFragment`.

### `cells_intersecting`

Use `cdshealpix::nested::cone_search(depth, region.center.ra, region.center.dec, region.radius_rad)` (or equivalent for the API the version exposes) to enumerate cells overlapping the region.

## Backwards compatibility

- **v3 reader** must accept v2 files (synthesize a single-cell layout).
- **v2 reader** cannot read v3 files. The version dispatch correctly errors
  with "unsupported version: 3" via the existing check.
- Old `Index::load(path)` continues to work for v2; for v3 it delegates to
  `ZdclFile::open(path)?.load_full()`.

## Tests

Unit tests in `src/index/source.rs::tests`:

- **`v3_roundtrip`**: build a small index, write as v3, read back, verify
  per-field equality.
- **`v3_loads_v2_files_via_synthesis`**: write a known v2 file, open with
  `ZdclFile`, verify load_full returns expected content.
- **`cells_intersecting_disjoint_region`**: query a region with no stars →
  empty cell list.
- **`cells_intersecting_full_sky`**: query a region with radius ≥ π → all
  populated cells.
- **`load_cells_empty_request`**: pass empty cell list → empty fragment, no IO.
- **`load_cells_runs_merge`**: pass 10 consecutive cells, verify one read
  rather than 10 (use a tracking BufReader if needed).
- **`load_cells_drops_quads_outside`**: load 1 cell of 100, verify resulting
  quads only reference loaded stars, dropped count is reported.

Integration test:

- **`v3_sparse_load_matches_full_load_for_full_region`**: build, write v3,
  load full vs load with all-sky region — should match bit-for-bit on stars
  + quads + codes.
- **`v3_solve_smoke`**: build v3 of the existing synthetic scenario, solve
  via `Index::from(ZdclFile::open(path)?.load_full())`, verify it solves.

Migration path test:

- **`upgrade_v2_to_v3_via_round_trip`**: load a real v2 file (use the existing
  `scratch/gaia_index.zdcl.bak`), re-save as v3, verify load + solve still
  works. Useful as a one-shot CLI migration helper too.

## Effort estimate

| Step | Effort |
|---|---|
| File format design + spec doc | 0.5 day |
| Builder change to sort + emit cell table | 1 day |
| `ZdclFile::open` + cell-table parsing | 1 day |
| `IndexSource` trait + `load_cells` impl | 1.5 days |
| `cells_intersecting` (cdshealpix integration) | 0.5 day |
| v2 backwards-compat path | 0.5 day |
| Unit + integration tests | 1 day |
| `Index::load` reimplementation in terms of `ZdclFile` | 0.5 day |
| Migration CLI helper (`zodiacal upgrade-index <path>`) | 0.5 day |
| Docs + CHANGELOG | 0.5 day |
| **Total** | **~7 days (1 week)** |

## Dependencies

- `cdshealpix` is already a dependency (used by index builder).
- Plan 1 is independent. Either can land first.

## Open questions

- **Cell depth choice.** README recommends depth 5 (3072 cells, ~5° per cell).
  Verify this is right for typical FOVs:
  - 0.5° FOV → 1 cell load worst case.
  - 5° FOV → ~4-12 cells.
  - 30° FOV → ~50-100 cells.
  - All seem fine. Could also expose as a build-time flag for users who want
    finer granularity for tiny FOVs.
- **Cell table encoding when n_cells is large.** At depth 8 there are
  786,432 cells × 20 bytes ≈ 15 MB. Still fits in RAM but starts being
  noticeable. For depths > 6 we should compress (e.g., delta-encode `cell_id`
  since they're sorted, store `file_offset` as deltas too). Defer until we
  actually want depth > 6.
- **Endianness.** All multi-byte fields are little-endian (matches v2). Add
  to spec doc.
- **Migration story for the existing 12 GB zodiacal-web indexes.** Need a
  one-shot CLI that takes a v2 `.zdcl` and emits the v3 form. Estimate ~10
  min per index given disk I/O. Should be safe to run in-place (write to
  `.tmp` then atomic-rename).
