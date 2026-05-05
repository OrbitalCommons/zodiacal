# `.zdcl.bundle` — proposed multi-band, HEALPix-sharded index format

## Motivation

The current `.zdcl` v3 file is HEALPix-grouped *internally* (cell table + cell-grouped stars/quads) but is a single binary file holding one scale band's quads alongside the star block. Two requirements are pushing against that design:

1. **Multi-band indexing** to recover the README's 98.5 % solve-rate baseline (12 narrow scale bands, astrometry.net-style). A single file with one quad block needs to grow either to N quad blocks (v4 single-file) or split into multiple files.
2. **Region-loading still has to be fast.** Ground / space deployment modes already lean on `IndexSource::load_cells` to stream only the cells overlapping a FOV. Whatever multi-band layout we pick must preserve that.

This doc specifies a directory-or-zip "bundle" layout that delivers both, with a per-cell file convention that mirrors how `starfield-datasources` already ships the bycell excerpt.

## Top-level layout

A bundle is identified by its top-level path. The internal logical layout is identical in either packaging form; the reader abstracts the difference behind a `SubfileAccessor` trait (see Reader API):

- **Directory:** `path/to/index.zdcl.bundle/` — files on disk, mmap-friendly. Hot-path serving.
- **Zip:** `path/to/index.zdcl.bundle.zip` — same logical entries packed in a zip archive. Convenient for distribution; entries decompress on read so it's a colder access path than the directory form, but the API is identical.

The same internal layout in both:

```
index.zdcl.bundle/   (or as entries inside the .zip)
├── manifest.json               # the canonical commit point — present iff bundle is finalized
├── quads/
│   ├── cell_00000.zqd          # ALL bands' quads + codes for cell 0, in one file
│   ├── cell_00001.zqd
│   └── …
└── gaia/
    ├── cell_00000.zga          # per-star Gaia records for cell 0, sorted by source_id
    ├── cell_00001.zga
    └── …
```

- One file per cell for quads (multi-band, with band-table at the head), one file per cell for the per-star Gaia records. The gaia per-cell file is the canonical per-star store, and the quad file's local indices point into it (see file format below).
- A quad goes into the file of the HEALPix cell containing its **centroid** (unit-vector mean of the 4 member stars). For the current cell-driven builder, all four members already live in the same cell, so centroid-cell == anchor-cell; once the builder gains cross-cell support (#65 follow-up), a straddle quad lands in whichever cell its centroid falls in, no matter where its anchors live.
- All bands' quads for the same cell sit in **one** file with a band-table at the head — both because it cuts file count by `n_bands` (often 10–12×) and because region-load + multi-band-solve naturally reads "everything that quad-matters for this cell" with one mmap.
- `cell_NNNNN` is zero-padded decimal at the bundle's `cell_depth`. Width is computed from `cell_depth` at write time (`12·4^depth` decimal-digits + 1 for safety) and read from the manifest, not assumed.
- Empty cells are simply absent from disk — no zero-byte files. The manifest's `populated_cells` block tells the reader what exists.
- File extensions: `.zqd` (quads + codes, multi-band per cell), `.zga` (per-star Gaia records, per cell). Distinct so a quick `find . -name '*.zqd'` always tells you "all per-cell quad shards."

### `cell_depth` is a build-time knob

The bundle's HEALPix sharding depth is **configurable at build time** — it does *not* have to match the depth of the source excerpt the bundle was built from. Trade-offs:

| `cell_depth` | n_cells | Cell area | At G ≤ 20 (112 M stars) | Typical use |
|---:|---:|---:|---:|---|
| 4 | 3 072 | ~13.4 sq.deg | ~36 500 stars/cell, ~1.2 MB/cell | Coarse server-mode bundles |
| **5** | **12 288** | **~3.4 sq.deg** | **~9 100 stars/cell, ~290 KB/cell** | **Default; matches bycell excerpt** |
| 6 | 49 152 | ~0.84 sq.deg | ~2 300 stars/cell, ~73 KB/cell | Ground tracker (~degree FOV) |
| 7 | 196 608 | ~0.21 sq.deg | ~570 stars/cell, ~18 KB/cell | Space tracker (narrow FOV) |
| 8 | 786 432 | ~0.052 sq.deg | ~140 stars/cell, ~4.5 KB/cell | Very narrow FOV / dense per-FOV cell sweep |

The deeper the depth, the smaller each per-cell file and the more of them. Pick by the FOV size of the consuming pipeline: a 0.3°-FOV camera benefits from depth 7 (FOV touches a handful of cells), while a wide-field server load can take depth 4–5 happily. The manifest records the chosen depth so the reader can compute filename widths and dispatch lookups.

When the source excerpt is sharded at one depth and the bundle is built at another:

- **Coarser bundle** (`bundle_depth < source_depth`): the builder aggregates `4^(source_depth - bundle_depth)` source cells per output cell during `stars_in_cell`. The cell-source adapter expands one bundle cell into a list of source cells and concatenates their stars.
- **Finer bundle** (`bundle_depth > source_depth`): the builder iterates source cells, then partitions each one's stars by HEALPix hash at the deeper bundle depth. Each star is re-hashed at write time.
- **Equal** (default; bycell excerpt at level 5 → bundle at level 5): no remapping, identity passthrough — the fast path.

## Build pipeline

A bundle is produced by **two phases**: a parallel build phase that emits HEALPix per-cell shards into a temporary work directory, and a single-threaded tidy phase that packages the work dir into the final folder or zip artifact and writes the manifest as the commit point.

```
                (parallel)                   (single-threaded)
[bycell excerpt] ──build──▶ [work_dir/...]   ──tidy──▶ [final folder or .zip]
                                                                │
                                                       manifest.json is
                                                       the commit edge
```

### Phase 1 — parallel shard build

Workers iterate cells in parallel via the cell-driven builder (#70). For each `cell_id` a worker:

1. Pulls the cell's stars via the `CellStarSource` adapter (e.g. `LazyExcerptSource` reading from the bycell excerpt).
2. Brightness-truncates to `max_stars_per_cell`.
3. Builds quads for every band over the same star buffer (one I/O hit, N bands of quad emission).
4. Writes two intermediate files into the **work directory**:
   - `work_dir/quads/cell_NNNNN.zqd.part.HHHHHHHH`
   - `work_dir/gaia/cell_NNNNN.zga.part.HHHHHHHH`
5. fsyncs each, then atomically `rename(2)`s onto the canonical names `cell_NNNNN.zqd` and `cell_NNNNN.zga` within `work_dir/`.
6. Updates the build manifest (`work_dir/.build-manifest.json`, the in-build resume primitive — distinct from the final bundle `manifest.json`) to mark this cell complete.

`HHHHHHHH` is an 8-hex-digit per-worker-invocation nonce (e.g. `xxh3-64` of pid + thread_id + nanos + cell_id) so concurrent workers never collide on a partial filename. If two workers race the same cell, the rename winner commits and the loser deletes its `.part.*` (build is deterministic per (cell, config), so the losing file would be byte-equivalent anyway).

A crashed build is identified at re-run by **the absence of the final bundle manifest** in the eventual output path; the work_dir survives, partial files survive, and the existing `BuildManifest` skip-completed-cells logic resumes from where it left off. No lost work.

### Phase 2 — tidy sweep into final artifact

When all cells are committed, a single-threaded finalize step:

1. Sweeps `work_dir/quads/` and `work_dir/gaia/` for any leftover `*.part.*` files (crashed-mid-rename droppings) and removes them.
2. Constructs the final `manifest.json` from the build-manifest's totals and the user-supplied experiment / build-metadata fields.
3. Packages the work dir into the chosen output form:
   - **Folder output** (`--output path/to/index.zdcl.bundle`): `mv work_dir path.partial && rename(path.partial, path)`. Cheap — usually a single atomic directory rename if work_dir was on the same filesystem as the output. Otherwise a recursive copy + atomic rename.
   - **Zip output** (`--output path/to/index.zdcl.bundle.zip`): stream-zip the work dir's contents into `path.partial.zip` (entry order: `manifest.json` last; per-cell files in cell-id order for predictable seeks), fsync, then `rename(path.partial.zip, path.zip)`.
4. The **final manifest is the commit edge** — folder rename or zip rename is what makes the bundle "exist." Pre-tidy crashes leave the work_dir intact; tidy-mid crashes leave a `.partial` artifact that the next finalize will overwrite.
5. After successful tidy, the work_dir can be removed (or kept around for debug — the user-facing CLI should default to keep, with `--prune-work-dir` opt-in).

### Atomicity & resume summary

| Event | What's on disk afterwards | Recovery |
|---|---|---|
| Worker crashes mid-cell | `cell_NNNNN.{zqd,zga}.part.HHHHHHHH` orphaned in work_dir | Resume re-runs that cell's build; partials are swept on next tidy. |
| Worker crashes after rename, before manifest update | Canonical shard exists, build-manifest doesn't list the cell | Resume re-runs the cell (deterministic; produces same bytes). |
| All workers done, builder process killed before tidy | Work_dir complete, final bundle manifest absent | Re-run; build-manifest sees all cells done; tidy phase runs. |
| Tidy crashes mid-rename | `path.partial` (or `.partial.zip`) on disk; no final manifest | Re-run tidy; existing `.partial` is overwritten. |
| Tidy completes | Final folder/zip with `manifest.json` as last entry | Bundle is committed; reader sees a consistent artifact. |

**Verification.** The reader's first check on `open` is "does `manifest.json` parse and pass schema validation?" If yes, every per-cell file referenced by the manifest's `populated_cells` set is expected to exist; the reader can optionally validate every file's magic + version + cell_id + size on open (`bundle.verify()` — opt-in, since region-load shouldn't pay it). For stronger integrity than structural checks, see the format-version-bump note in the critique (per-block `xxh3-64` payload hashes).

## Manifest

`manifest.json` is the **commit point**: a bundle is considered finalized iff the manifest is present and parseable. Any partial state during a build leaves the manifest absent or stale-from-the-prior-build.

```json
{
  "format": "zdcl-bundle",
  "format_version": 1,
  "cell_depth": 5,
  "n_cells": 12288,
  "experiment": "G≤20 DR3 bycell, 12 bands √2 at 10-600 arcsec, 100 quads/cell/band, max_stars_per_cell=10000, max_reuse=8 — built 2026-05-05 from build-from-excerpt-series",
  "build_metadata": {
    "tool": "zodiacal-tools 0.2.0",
    "build_started_utc": "2026-05-05T01:08:57Z",
    "build_finished_utc": "2026-05-05T02:18:42Z",
    "source": {
      "kind": "starfield-datasources-bycell",
      "release": "Dr3",
      "path": "~/.cache/starfield/gaia-excerpts/dr3-mag20-bycell"
    }
  },
  "gaia": {
    "n_total": 112080742,
    "record_size": 104,
    "schema_version": 1,
    "max_stars_per_cell": 10000,
    "mag_limit": 20.0,
    "populated_cells": 12288
  },
  "bands": [
    {
      "label": "band_00",
      "scale_lower_arcsec": 10.0,
      "scale_upper_arcsec": 14.142,
      "quads_per_cell": 100,
      "max_reuse": 8,
      "n_quads_total": 1228800,
      "populated_cells": 12288
    },
    {
      "label": "band_01",
      "scale_lower_arcsec": 14.142,
      "scale_upper_arcsec": 20.0,
      "quads_per_cell": 100,
      "max_reuse": 8,
      "n_quads_total": 1228000,
      "populated_cells": 12283
    }
    /* … band 02 … band 11 … */
  ]
}
```

The free-text `experiment` field is for human ops notes — a one-liner summarizing what variant of build params produced this bundle. The structured `build_metadata` is for tooling.

## Per-cell file formats

### `quads/cell_NNNNN.zqd` — quads + codes for one cell, all bands

One file per HEALPix cell holds **every band's quads** for that cell, with a band table at the head pointing at each band's quad/code blocks. Indices are **into this cell's `cell_NNNNN.zga` file** — local indices, not global. (Cross-cell quads are not supported by the current cell-driven builder; once they are, a straddle quad lives in the file of whichever cell contains its centroid.)

```
magic         8 B  "ZDCLQUAD"
version       4 B  u32 LE = 1
reserved      4 B
cell_id       8 B  u64 LE                 — defensive duplication; filename is the source of truth
n_bands       4 B  u32 LE                 — must equal manifest.bands.len()
reserved      4 B  → align band table to 8

band_table    n_bands × 24 B
              for each band:
                band_idx     4 B  u32 LE
                n_quads      4 B  u32 LE
                quads_offset 8 B  u64 LE  — byte offset of this band's quads block, from file start
                codes_offset 8 B  u64 LE  — byte offset of this band's codes block, from file start

padding to 8-byte boundary

per band, in band_idx order, no padding between blocks:
  quads_block  n_quads × 16 B  (4 × u32 local star indices)
  codes_block  n_quads × 32 B  (4 × f64)
```

The reader pattern is:

```rust
let mmap = bundle.mmap_quad_shard(cell_id)?;     // one mmap per cell, all bands
let band_5 = mmap.band(5);                       // O(1) slice via the band table
for (quad, code) in band_5.iter() { … }
```

Loading "all bands' quads for one cell" is a single mmap. Loading "one band's quads across all cells in a region" is N mmaps (one per cell), with a constant-time slice into each. Both common paths are cheap.

The 24-byte-per-band table is small (288 B for 12 bands) and lets the writer emit blocks in any order — useful when the cell-driven builder runs band-N quad emission in parallel and finalizes the per-cell file by stitching in-memory band buffers. Empty bands (n_quads == 0) take only 24 B in the table; the offsets point at zero-length blocks and the reader handles that as a no-op slice.

#### Quad addressing & band recovery

Band membership is **not** stored on individual quad records; it lives in the band table at the head of the cell file (one entry per band, one band per block). That keeps the quad record at a clean 16 B and the codes record at 32 B. Consequently:

**Backing out band from a "quad number" within a cell.** Given an intra-cell index `i ∈ [0, total_quads_in_cell)` (where `total = Σ band_table[k].n_quads`), the band a quad belongs to is the unique `k` such that `cumsum[k] ≤ i < cumsum[k+1]`, where `cumsum[k] = Σ_{j<k} band_table[j].n_quads`. O(log n_bands) with a sorted cumsum, O(1) if the reader caches it on file open.

Equivalently, given `(band_idx, intra_band_idx)` you can compute the file offset directly:

```
quad_record_offset = band_table[band_idx].quads_offset + intra_band_idx × 16
code_record_offset = band_table[band_idx].codes_offset + intra_band_idx × 32
```

**Unique keys for a quad.** Several equivalent identifiers, ordered most to least specific:

| Key | Granularity | Use |
|---|---|---|
| `(cell_id, band_idx, intra_band_idx)` | The canonical address within the bundle. | Reader API, "give me quad X for inspection." |
| `(cell_id, intra_cell_idx)` | Equivalent; collapses band+intra-band into one position via the cumsum above. | Compact in-memory references; survives future band-layout changes. |
| `sorted([local_idx_a, local_idx_b, local_idx_c, local_idx_d])` | The 4 star positions in this cell's `gaia/cell_NNNNN.zga` file. | Implicit dedup key during build (`HashSet<[usize; 4]>` in `build_quads_for_cell`). |
| `sorted([source_id_a, source_id_b, source_id_c, source_id_d])` | The 4 Gaia source_ids of the member stars. | Globally unique across rebuilds; useful for cross-build verification ("did this bundle keep the same specific quad?"). |

**What's *not* unique.** The 4 indices in their *encoded order* (the sequence stored in the quad record) is not a key — `compute_canonical_code` reorders members during canonical encoding. Code-comparing quads requires sorting the index tuple first. The 4-D code itself is also not unique: many distinct quads land at similar codes (the whole reason solver matching uses a tolerance).

**The natural reader-API key.** `(cell_id, intra_cell_idx)` because it (1) survives any future change to band layout (adding bands, splitting bands, flat layouts), (2) trivially recovers `band_idx` via the cumsum, and (3) gives band-agnostic consumers a flat numbering they can iterate.

For the **builder**, the natural emission order is per-band (build all of band 0 for cell N, then all of band 1, …). With that order, `intra_cell_idx == sum_of_n_quads_in_lower_bands + intra_band_idx`, so the two addressing schemes coincide on disk for free.

### `gaia/cell_NNNNN.zga` — Gaia source records, sorted by source_id, binsearchable

This file replaces the prior split between a verifier-facing stars block and a refinement-facing sidecar. One canonical 104-byte record per star covers both: enough fields for the verifier's brightness prior, full astrometric covariance for weighted refinement (#47), and the universal Gaia DR3 quality flag.

```
magic         8 B  "ZDCLGAIA"
version       4 B  u32 LE = 1
record_size   4 B  u32 LE = 104
cell_id       8 B  u64 LE
n_records     4 B  u32 LE
reserved      4 B  → align to 8
records       n_records × 104 B  (sorted ascending by source_id)
```

#### Record layout — 104 B fixed-width

```
offset  size   field                        notes
------  ----   --------------------------   ------------------------------------------------------
   0    8 B   source_id           u64 LE   Gaia source_id; also indexed in
                                            the quad file via local idx.
   8    8 B   ref_epoch           f64 LE   Epoch of the position; J2016.0 for DR3.
  16    8 B   ra                  f64 LE   degrees, ICRS, at ref_epoch.
  24    8 B   dec                 f64 LE   degrees, ICRS, at ref_epoch.
  32    8 B   pmra                f64 LE   mas/yr, with cos(dec) factor applied (Gaia convention).
                                            NaN if Gaia did not publish a 5-parameter solution.
  40    8 B   pmdec               f64 LE   mas/yr; NaN if missing.
  48    8 B   parallax            f64 LE   mas; NaN if missing.
  56    8 B   radial_velocity     f64 LE   km/s; NaN for the ~98 % of DR3 sources without RVS.
  64    8 B   phot_g_mean_mag     f64 LE   Gaia G-band apparent magnitude. Always present.
  72    4 B   sigma_ra            f32 LE   mas; 1-σ along the RA direction.
  76    4 B   sigma_dec           f32 LE   mas; 1-σ along Dec.
  80    4 B   sigma_pmra          f32 LE   mas/yr.
  84    4 B   sigma_pmdec         f32 LE   mas/yr.
  88    4 B   sigma_parallax      f32 LE   mas.
  92    4 B   ra_dec_corr         f32 LE   Pearson correlation in [-1, +1]. NaN if Gaia did
                                            not publish a 5-parameter solution. Without
                                            this, the (RA, Dec) error covariance is treated
                                            as diagonal and any chi-square is wrong.
  96    4 B   ruwe                f32 LE   Renormalized Unit Weight Error. RUWE > ~1.4 flags
                                            "astrometry suspect" (binary, blended, high-PM).
                                            NaN if missing.
 100    4 B   flags               u32 LE   bitfield (presence flags + future quality bits)
 ------------
 104 B total
```

#### `flags` bitfield

```
bit  0   has_pm                  pmra/pmdec/sigma_pm* are non-NaN
bit  1   has_parallax            parallax/sigma_parallax are non-NaN
bit  2   has_radial_velocity     radial_velocity is non-NaN
bit  3   has_ra_dec_corr         ra_dec_corr is non-NaN
bit  4   has_ruwe                ruwe is non-NaN
bit  5   …                        reserved for ipd_frac_multi_peak / duplicated_source / etc.
                                  if a future schema bump promotes them.
bits 6..31  reserved.
```

Cheap presence checks via bitfield avoid hot-path NaN comparisons; the float fields still carry NaN as a sentinel for self-describing reads.

#### Why 104 B (not 96, not 128)

- **96 B** would carry only `phot_g_mean_mag` on top of today's 88 B. Sufficient to drop `stars/`, but leaves the weighted-refinement covariance and the Gaia quality flag *missing forever* unless we bump format-version later — and retrofitting the covariance after consumers exist is expensive.
- **128 B** would tack on photometric extras (BP, RP, sigma_RV, n_obs_al, ipd_frac_multi_peak) that aren't needed by zodiacal itself; downstream code can re-pull those from a separate excerpt query if it cares. The 24 B cost is real (45 % bigger than today, 33 % fewer records per page during binsearch).
- **104 B** carries exactly the fields zodiacal *uses*: brightness for the verifier, full 2×2 astrometric covariance for refinement, RUWE for quality screening. Stops there.

The cell-level pivot table is omitted — at typical depths each cell has at most a few × 10⁵ records, so a direct binsearch over the mmapped record array is one cache-line probe per O(log n) step. A pivot table would be wasted bookkeeping at this granularity.

## Reader API

The reader works against any object that knows how to enumerate and read entries by relative path. That abstraction is the `SubfileAccessor` trait:

```rust
/// Storage-agnostic entry access for a bundle. Two concrete impls:
///   - `FsAccessor`  — entries are files on disk under a directory root;
///                     `read_entry` returns an `Mmap` (mmap-backed slice).
///   - `ZipAccessor` — entries live in a zip archive; `read_entry` returns
///                     an owned `Bytes` (decompressed copy of the entry).
pub trait SubfileAccessor: Send + Sync {
    /// True if an entry at this relative path exists.
    fn exists(&self, rel: &str) -> bool;

    /// Cheap "list everything under `prefix/`" — used at open to know
    /// which cells are populated and which aren't (without a manifest
    /// round-trip if the manifest has been deleted).
    fn list_prefix(&self, prefix: &str) -> io::Result<Vec<String>>;

    /// Return a read-only byte slice for `rel`. The returned `EntryBytes`
    /// is either a borrowed `&[u8]` from an mmap (FsAccessor hot path) or
    /// an owned `Bytes` from a decompressed zip entry. Either way, hot
    /// code can binsearch / cast / slice it identically.
    fn read_entry(&self, rel: &str) -> io::Result<EntryBytes<'_>>;
}

pub enum EntryBytes<'a> {
    Mmap(&'a [u8]),
    Owned(bytes::Bytes),
}
```

`FsAccessor` mmaps each entry on first access and caches the mmap; `ZipAccessor` decompresses on each `read_entry`. The hot-path verifier and refinement code see `&[u8]` either way and don't have to care.

```rust
pub struct ZdclBundle {
    accessor: Box<dyn SubfileAccessor>,
    manifest: Manifest,
    /// Per-cell entry-bytes cached on first access.
    cell_cache: Mutex<HashMap<u64, CellEntries>>,
}

impl ZdclBundle {
    /// Auto-detect dir vs zip from the path; build the appropriate
    /// `SubfileAccessor`; parse `manifest.json`.
    pub fn open(path: impl AsRef<Path>) -> io::Result<Self>;

    pub fn manifest(&self) -> &Manifest;
    pub fn cell_depth(&self) -> u8;
    pub fn bands(&self) -> &[BandInfo];

    /// Slice every band over `region`. Returns a `MultiBandFragment` whose
    /// gaia records are the union of `region`'s cells and whose quads are
    /// partitioned by band.
    pub fn load_region(&self, region: &SkyRegion) -> io::Result<MultiBandFragment>;

    /// Single band. Useful when the solver dispatches one band's
    /// code-tree at a time.
    pub fn load_region_band(&self, band_idx: usize, region: &SkyRegion) -> io::Result<IndexFragment>;

    /// Per-source-id Gaia record lookup. `cell_hint` short-circuits the
    /// cell-from-source_id derivation; if `None`, the bundle derives the
    /// cell from the source_id's HEALPix prefix and the bundle's
    /// `cell_depth`.
    pub fn gaia_get(&self, source_id: u64, cell_hint: Option<u64>) -> io::Result<Option<GaiaRecord>>;
    pub fn gaia_get_many(&self, source_ids: &[u64]) -> io::Result<Vec<Option<GaiaRecord>>>;

    /// Optional structural-integrity check (magic + version + cell_id +
    /// size) over every populated cell. Skipped on the normal `open` path.
    pub fn verify(&self) -> io::Result<VerifyReport>;
}
```

`MultiBandFragment` holds the union of region cells' gaia records plus a `Vec<IndexFragment>` (one per band, all referencing the same gaia records via local index). The solver receives `&[&Index]` exactly as today; constructing those is `bundle.bands().iter().map(|b| &b.index).collect()`.

The directory and zip forms differ only in the `EntryBytes` variant returned and the access cost (mmap = lazy + fast; zip-decompress = eager + slower). The folder form is the hot-path serving target; the zip form is the distribution artifact.

---

## Critique

### What this gets right

- **Multi-band as first-class.** Solvers expecting `&[&Index]` already work; building a bundle is the only new construction path.
- **One canonical per-star store.** Each cell has exactly one `cell_NNNNN.zga` carrying the 104 B Gaia record, referenced by every band's quads via local index. No duplication across bands the way `build-index-series` produces today (~12× star-block redundancy).
- **The 104 B record covers both hot-path and refinement.** Verifier reads `phot_g_mean_mag` for its prior; refinement reads the full 2×2 astrometric covariance via `ra_dec_corr`; quality-aware consumers filter on `ruwe`. No "load stars for solving, load sidecar for refinement" split.
- **Cell-sharded gaia records.** A FOV that touches K cells reads K gaia shards instead of binsearching one global sidecar. Per-cell shards are small enough that direct binsearch beats a pivot table.
- **Region-load is unchanged in shape.** "Open manifest, list cells in region, read their entries" is exactly the existing pattern.
- **Two-phase build with a clean commit edge.** Parallel shard build into a work_dir; single tidy phase packages it; the final manifest is the commit point. Crash anywhere along the way, restart picks up where it left off; the only "is this bundle valid" question reduces to "is the final `manifest.json` parseable?"
- **Folder vs zip is a packaging-only choice.** The reader sees both via the `SubfileAccessor` trait; consumer code never branches on which form it's reading.

### What I'd push back on / open questions

1. **Cross-cell quads are still unsupported.** This isn't new — the cell-driven builder already drops quads whose backbones straddle cell boundaries — but the per-cell file format makes it harder to fix later. To support cross-cell quads we'd either need to (a) duplicate the quad in both cells' files, (b) introduce a "cross-cell quads" file type that lists referencing cell IDs explicitly, or (c) keep per-cell files only for "interior" quads and a global file for boundary quads. None of these is impossible but the per-cell-only assumption is now baked into more places.

2. **Sidecar binsearch within a cell, not globally.** Today the sidecar is a single global file with a pivot table — *any* `source_id` resolves with one binsearch. With per-cell shards, the reader has to know *which cell* the source_id belongs to before opening the right shard. Two ways:
   - Derive from source_id's HEALPix-12 prefix → cell at depth N. This is robust for Gaia DR3 (source_ids encode HEALPix-12) but couples the layout to Gaia's id scheme.
   - Caller passes the cell hint (the index already knows what cell each matched star came from). Cleaner but requires plumbing changes through `RefinementCatalog::load_sidecar_filtered`.
   - This was foreshadowed in #66.

3. **Format-version drift.** We've now got: a manifest version, a quad-shard version, a gaia-shard version. Each can evolve independently. That's actually a feature compared to the v3 monolith, but it means more code paths and more migration stories.

4. **Free-text `experiment` is too permissive.** A free-text field is good for ops notes but tooling will want structured fields anyway (e.g., "show all bundles built from G ≤ 19 catalog with ≥ 100 quads/cell"). My instinct: keep the free-text field for human notes *and* add a structured `build_params` block in `build_metadata` that captures the salient knobs (mag_limit, max_stars_per_cell, scale-band list, max_reuse, source kind). Then tooling has a canonical place to look without parsing prose.

5. **No payload-level integrity check.** Structural checks (magic, version, cell_id, sizes) catch truncation and config drift; they don't catch silent bit-rot or a per-record byte that flipped without changing size. A future format-version bump could add per-block `xxh3-64` payload hashes (8 B in the band table per band, 8 B in the gaia header per cell). Until then, `bundle.verify()` is structural-only.

6. **Backward compat with v3 single files.** `Index::load(path)` today handles v1/v2/v3 transparently (after #72). Should it grow to also auto-detect "this path is a bundle" and delegate? Probably yes, with a clear "single-file vs bundle" branch at the top. `path.is_dir() || path.ends_with(".zip")` → bundle; else → single-file v3 reader.

10. **Versioning of the bundle format itself.** `format_version: 1` in the manifest. Future changes (e.g. adding a per-cell distortion-model table, or a per-band quad-density adaptive scheme) bump the version. Bundles before v2 stay readable as v1; readers ship a known-version-handler table. Same pattern as the v3 layout in `source.rs`.

### Summary recommendation

Ship this as **v1 of the bundle format** and gate it behind a new builder subcommand (`build-from-excerpt-series`) without breaking the v3 single-file path. The first deliverable is enough Rust to:

- write a bundle from the cell-driven builder loop, one band per scale bin,
- read a bundle into a multi-band view that the existing `solve()` API can consume.

The shard-bucketing optimization (item #1 above) and the cross-cell-quads design (item #3) can wait until profiling or use-case shows they matter.
