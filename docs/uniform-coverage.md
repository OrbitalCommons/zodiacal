# Uniform sky coverage without a coarse magnitude cutoff

## The problem

A blind solver only works on FOVs whose stars match quads in the index. If the
index has zero quads in a region, the solver returns nothing for any image
pointed there — no matter how good the centroids are.

The analysis tool (`scripts/analyze_index.py`) on the production
`scratch/gaia_g14.zdcl` (1 M quads, 16.8 M stars at G ≤ 14) revealed exactly
that pathology:

- Quads sat in **133 / 12,288 HEALPix cells (1.1 %)** at level 5 — almost
  entirely the galactic plane and a handful of dense clusters (Hyades,
  Pleiades, Sagittarius bulge).
- One single G = 7.6 star appeared in **257,870 quads** by itself; the
  brightness-first quad builder sank a quarter of the entire 1 M-quad budget
  into one region around it.
- A 0.5 ° FOV at random sky pointings landed **zero quads in 99.8 %** of cases.
  Uniform expectation was 19 quads / FOV.
- Where the index *did* reach, depth was fine — the dimmest used star saturated
  near G ≈ 14 in nearly every populated cell. So depth-where-reached wasn't
  the problem; **sky coverage** was.

## Why a coarser magnitude cutoff is the wrong fix

The intuitive response is "set the magnitude limit deeper, so dark patches
of sky have enough stars to be indexable." That's true but insufficient by
itself, because the *builder*, not the catalog, was the constraint:

- The legacy builder (`build_index` in `src/index/builder.rs`) sorts every
  star in the catalog by brightness, then iterates anchors from brightest to
  faintest.
- Each anchor emits O(K²) quads where K is the count of nearby stars within
  scale_upper. In dense regions K is hundreds; in the galactic plane it can
  saturate the quad budget within the first few thousand anchors.
- A single global `max_quads` cap then short-circuits *all* remaining
  anchors. Faint anchors in dark cells never get tried.

So even at G ≤ 20 the legacy path would still concentrate quads in the
brightest regions. Going deeper without changing the iteration scheme just
adds more bright-region quads.

## The two-level fix

### Level 1: iterate by HEALPix cell, not by global brightness

`src/index/cell_builder.rs::build_index_cell_driven` walks `0..cell_count` at
the source's HEALPix depth (level 5 = 12,288 cells, ≈ 3.36 sq.deg each).
For each cell, it:

1. Pulls that cell's stars from the source (after the source-side mag gate).
2. Truncates to the brightest N per cell (`max_stars_per_cell`).
3. Runs `build_quads_for_cell` to emit up to `quads_per_cell` quads using only
   that cell's stars.
4. Commits the per-cell artifact + sidecar chunk + manifest entry.

Sky uniformity is now **a structural property, not a hope**: every cell gets
its own quad budget. The `--quads-per-cell` flag is the "minimum quads per
cell" knob — best-effort, capped only by what the cell's star geometry can
actually support.

### Level 2: deep magnitude limit so every cell *can* fill its budget

A high-galactic-latitude void with only a handful of bright stars in it can't
geometrically produce 100 valid quads, no matter what `--quads-per-cell`
asks. The fix is to push the magnitude limit deep enough that even sparse
regions have enough scale-valid stars.

For DR3:

| Mag limit | Stars / cell (median) | Sufficient for 100 quads/cell? |
|---:|---:|---|
| G ≤ 14 | ~1,400 | Yes in dense regions; marginal at high latitude |
| G ≤ 16 | ~7,000 | Yes essentially everywhere |
| G ≤ 20 | ~85,000 | Far beyond — `max_stars_per_cell` truncation kicks in |

The G ≤ 20 bycell excerpt at
`~/.cache/starfield/gaia-excerpts/dr3-mag20-bycell/` (1.06 B rows, level-5
sharded, one file per cell) is the input. With `--max-stars-per-cell 2000
--quads-per-cell 100`, every populated cell saturates both budgets exactly,
and the resulting index is sky-uniform by construction.

The point of going deep is *not* "use 1 B stars in the index" — it's "have
enough stars per cell that the per-cell quad budget is what's binding, not
the catalog depth."

## Supporting infrastructure

The cell-driven path has three corollary requirements that the legacy in-RAM
builder didn't:

### Streaming sidecar (`SidecarStreamWriter`)

The legacy `write_sidecar` materializes every `SidecarRecord` in RAM before
sorting and writing. At 88 B per record, G ≤ 17 is the practical ceiling on a
64 GB box. To go deeper, the sidecar needs to stream.

`SidecarStreamWriter` (`src/refinement/sidecar.rs`) takes per-cell chunks,
sorts each one in RAM, dumps to a numbered temp file, and at finalize-time
runs an external k-way merge into the on-disk format. Peak RAM is bounded by
the largest per-cell chunk (≈ tens of MB at level 5), not the whole catalog.
Final on-disk bytes are byte-identical to what the in-RAM writer would have
produced for the same input set — verified by a roundtrip test.

### Resumable per-cell manifest (`BuildManifest`)

Long deep builds (~10–60 min) shouldn't lose all progress on a crash or
ctrl-C. `src/index/build_manifest.rs::BuildManifest` is a JSON file with
`completed_cells: BTreeSet<u32>`, atomically rewritten (tmp + fsync + rename)
after every per-cell commit. Per-cell artifacts live in
`work_dir/cell_artifacts/cell_NNNNN.idx-tmp` and sidecar chunks in
`work_dir/sidecar_chunks/chunk_NNNN.sidecar-tmp`. A re-run with the same
`--work-dir` skips already-committed cells and continues.

### Parallel cell processing

Each cell's work is independent — independent input file, independent quad
build, independent artifact path. The orchestrator runs the cell loop via
`rayon::par_iter` so all cores are used. Shared state is a brief
`Mutex<BuildManifest>` per per-cell commit, plus an atomic counter inside
`SidecarStreamWriter` for chunk-index allocation. Speedup on a 16-core box is
~10× wall-clock; the bottleneck is then disk bandwidth on the chunk + sidecar
writes.

### Output stability

Parallelism is only safe if the output bytes don't depend on cell completion
order. The cell-driven builder guarantees this:

- Per-cell index artifacts are named by `cell_id` (deterministic).
- `finalize_index` iterates the manifest's `BTreeSet<u32>` of completed cells,
  which is sorted by cell_id regardless of insertion order.
- The sidecar's chunk paths embed atomically-allocated chunk indices, and
  `finalize` sorts chunks by path before the k-way merge, so the merge sees a
  reproducible chunk order.
- The merge emits records in `source_id` order, independent of which thread
  produced which chunk.

The `parallel_build_matches_sequential` test runs the same synthetic source
through a 1-thread rayon pool and a 4-thread pool and asserts the resulting
`.zdcl` and `.zdcl.gaia` are byte-identical.

## What this fixes vs. doesn't

It fixes:

- The 1.1 % sky coverage. Every populated HEALPix cell now hosts up to
  `--quads-per-cell` quads.
- The 99.8 % empty-FOV rate at random pointings. Coverage becomes uniform up
  to per-cell geometric capacity.
- The `max_reuse` blowout. Per-cell brightness ordering + per-cell reuse cap
  means no single bright star eats hundreds of thousands of quad slots.
- The G ≤ 17 RAM wall. Streaming sidecar + per-cell-bounded peak memory mean
  G ≤ 20 (and beyond) is tractable on a workstation.

It doesn't fix:

- Cells with truly insufficient star geometry — a void at the south galactic
  pole with only a few faint stars *will* fall short of `--quads-per-cell` no
  matter what we do. The `BuildSummary` reports per-cell counts so this is
  visible.
- Cross-cell quads. The current per-cell builder uses only the focus cell's
  stars; quads whose backbones straddle a HEALPix boundary at the source's
  depth are dropped. At level 5 (cells ~1.8°) this loss is small for typical
  quad scales (30″–30′) but non-zero. Adding neighbour context is a
  follow-up.

## How to build a sky-uniform deep index

```
zodiacal-tools build-from-excerpt \
  --excerpt-dir ~/.cache/starfield/gaia-excerpts/dr3-mag20-bycell \
  --output-prefix scratch/gaia_g20 \
  --work-dir   scratch/gaia_g20.work \
  --mag-limit         20.0  \
  --max-stars-per-cell 2000  \
  --quads-per-cell      100  \
  --max-reuse             8
```

Knobs:

- `--mag-limit` — go deep. The whole point of the cell-driven path is that
  going deeper *doesn't* concentrate the budget in bright regions, so deep
  is now free.
- `--max-stars-per-cell` — brightness-truncates each cell's input. Bounds
  per-cell RAM and per-cell quad-build cost. ~2000 is plenty at level 5.
- `--quads-per-cell` — the "minimum quads per cell" target. ~30 quads / sq.deg
  → 100 quads / cell at level 5 is the standard density (rough astrometry.net
  rule of thumb: ≥50 candidate quads per typical FOV).
- `--max-reuse` — cap on per-star quad participation (per-cell). Default 8
  matches `CatalogBuilderConfig`.

## Verifying coverage

Run `scripts/analyze_index.py <path>.zdcl` on the resulting file. The
expected change vs. `scratch/gaia_g14.zdcl`:

- `scale_coverage.png` — per-cell quad density should now be a tight
  distribution near `--quads-per-cell` rather than a heavy-tailed long-tail.
- `sky_density.png` — quad-centroid map fills the sphere uniformly instead
  of clumping along the galactic plane.
- `per_cell_depth.png` — colour map looks roughly uniform; depth still
  saturates at the catalog mag limit because the per-cell heap pulled stars
  brightness-ordered.
- `fov_coverage.png` — empty-FOV rate drops from ~99 % to ~0 %, the survival
  curve moves to the right.

## References

- Cell-driven builder: `src/index/cell_builder.rs`
- Streaming sidecar: `src/refinement/sidecar.rs::SidecarStreamWriter`
- Build manifest: `src/index/build_manifest.rs::BuildManifest`
- Adapter to `LazyLoadingCatalog`: `zodiacal-tools/src/build_from_excerpt.rs`
- CLI: `zodiacal-tools build-from-excerpt`
- Visual analyzer: `scripts/analyze_index.py`
- Upstream cell reader: `starfield-gaia` PR
  [#48](https://github.com/OrbitalCommons/starfield-datasources/pull/48)
  (`LazyLoadingCatalog::entries_in_cell` + `cell_count`)
