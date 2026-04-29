# Plan 1: Sparse load (Tier 1)

## Goal

Add `Index::load_in_region(path, &SkyRegion) -> io::Result<Index>` that returns
an `Index` containing only stars within the given region (and the quads that
reference those stars). Same on-disk file format. The win is **memory** — disk
I/O is unchanged because we still scan the whole file.

## Non-goals

- No file format change. (That's plan 2.)
- No incremental add/drop. (That's plan 3.)
- No background async load. The function blocks until the in-region `Index` is
  ready.

## API surface

```rust
// src/index/mod.rs

impl Index {
    /// Load only stars within `region`, plus the quads referencing those stars.
    /// Quads with any star outside `region` are dropped entirely.
    ///
    /// `region` should be padded by at least `expected_quad_scale_upper` to
    /// avoid dropping quads whose backbone straddles the region boundary —
    /// see [`load_in_region_padded`] for the explicit version.
    pub fn load_in_region(
        path: &Path,
        region: &SkyRegion,
    ) -> io::Result<Index>;

    /// Load with explicit padding around `region`. Padding is added to the
    /// region's radius before any star or quad acceptance test.
    pub fn load_in_region_padded(
        path: &Path,
        region: &SkyRegion,
        padding_rad: f64,
    ) -> io::Result<Index>;
}
```

The existing `Index::load(path)` stays as the "full sky" path. No change to
its signature or behavior.

## Algorithm

1. Open the file with `BufReader`. Read magic, version, metadata as today.
2. Read `n_stars`, `n_quads`, `scale_lower`, `scale_upper` as today.
3. **Stars phase:**
   - Compute `region_center_xyz = radec_to_xyz(region.center.ra, region.center.dec)` once.
   - Compute `radius_sq = 2.0 * (1.0 - (region.radius_rad + padding_rad).cos())` (chord-length squared).
   - For each of `n_stars` records:
     - Read 32 bytes (catalog_id u64, ra f64, dec f64, mag f64).
     - Compute `xyz = radec_to_xyz(ra, dec)`.
     - If `dot(xyz, region_center_xyz) >= 1.0 - radius_sq/2` (i.e., chord-distance squared ≤ radius_sq): keep, push into a `kept_indices: Vec<usize>` and a parallel `IndexStar` list. Map old index → new compact index in a `HashMap<usize, usize>`.
     - Otherwise: discard.
4. **Quads phase (raw star indices):**
   - Read each quad's 4 u32 star indices.
   - If all 4 indices are in the kept set: rewrite to compact indices and push.
   - Otherwise: discard. (Track skipped count for diagnostics.)
   - Track which raw quad indices were kept so we can correctly read their codes.
5. **Codes phase:**
   - For each raw quad in original order, read the 4 f64 code components.
   - If the corresponding quad was kept (via parallel-index tracking), push the code into the kept-codes list.
   - Otherwise: discard the bytes.
6. Build `KdTree<3>` over kept star unit-vectors and `KdTree<{ DIMCODES }>` over kept codes.
7. Return `Index` with `metadata: original_metadata` (unchanged, since the file's metadata still describes the build).

### Why filter at quads-read-time, not later

Reading and discarding bytes is faster than reading-then-allocating-then-filtering. The bytes per kept-or-discarded quad are 16 (star_ids) + 32 (code) = 48 — sequential reads are cheap, but allocations aren't.

### Edge cases

- **Empty result.** `region` doesn't intersect any catalog stars → `Index` with empty `stars`/`quads` vectors. Should not error; the solver will simply find no matches.
- **All-sky region.** `region.radius_rad ≥ π` → equivalent to `Index::load`. Don't special-case in code — let the chord-distance test be the truth source. (Document the equivalence.)
- **Anti-meridian / pole regions.** `radec_to_xyz` is well-defined everywhere; chord-distance test is rotation-invariant. No special handling needed.
- **Quads with renumbered indices > u32::MAX.** Can't happen — kept quads are a subset, and the original index already capped at u32 indices.

## File format / data flow

No file format change. We rely on the v2 format's sequential structure:

```
[header] → [stars: n_stars × 32B] → [quads: n_quads × 16B] → [codes: n_quads × 32B]
```

The streaming loader can discard records as it reads them, so peak memory is
the kept-set size, not the file size.

## Backwards compatibility

Strictly additive. `Index::load` unchanged; new methods take a `SkyRegion`
which is already `pub`.

## Tests

Unit tests in `src/index/store.rs::tests`:

- **`load_in_region_full_sky_matches_full_load`**: build a small index, save
  it. `load_in_region` with a region radius of 2π should produce identical
  output to `load`.
- **`load_in_region_drops_outside_stars`**: build an index with stars at known
  positions in two distinct sky patches. `load_in_region` centered on patch A
  should yield only patch-A stars.
- **`load_in_region_drops_quads_with_outside_member`**: build with quads that
  span the boundary; verify they're dropped from the result.
- **`load_in_region_padding_keeps_boundary_quads`**: same as above but with
  enough padding to capture boundary quads. Verify they appear.
- **`load_in_region_compact_indices`**: dropped quads' star indices are
  remapped correctly; loaded `Index` passes a basic `solve` smoke test.

Integration test in `src/refinement/tests.rs` or new file:

- **`sparse_load_then_solve`**: load a synthetic 1000-star index built across
  a wide region, sparse-load a tight region, run a synthetic solve. Verify it
  finds a solution and the WCS matches the truth.

## Effort estimate

| Step | Effort |
|---|---|
| Implement `load_in_region` + `load_in_region_padded` | 0.5 day |
| Unit tests (5) | 0.5 day |
| Integration test | 0.5 day |
| Profile + verify memory drop on real index | 0.5 day |
| Docs + CHANGELOG | 0.5 day |
| **Total** | **~3 days** |

## Dependencies

None — built entirely on existing types. Can land before plan 2 lands.

## Open questions

- **Default padding for `load_in_region`.** The padded version is explicit, but
  `load_in_region` (un-padded) needs a default. Options:
  - 0 (caller is responsible) — simplest, surprises nobody who passes a tight region.
  - The index's own `scale_upper` — auto-pads enough to keep all valid quads.
  - Recommendation: **default to 0**, document the trade-off. Add `load_in_region_padded` for the convenience case.
- **Should the resulting `Index` carry a marker that it's a regional load?** E.g., `metadata.region_filter: Option<SkyRegion>`. Useful for diagnostics ("why is my full-sky solver only matching one patch?"). Recommendation: add it — small cost, real value.
