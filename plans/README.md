# Deployment-mode plans

Five staged plans that take zodiacal from "single monolithic `Index::load`" to
supporting three operational profiles:

- **Server**: load full sky once, serve many concurrent solves.
- **Realtime telescope**: ground station with `(lat, lon, time)`; load only
  what's currently above the horizon, add/drop cells as Earth rotates.
- **Realtime star tracker**: spacecraft with ephemeris + attitude estimate;
  load only what's near boresight, slide as the platform moves.

The first three plans build the substrate (sparse load, file format, live
index). Plans 4-5 layer the operational orchestration on top.

| # | Plan | Scope | Effort |
|---|---|---|---|
| 1 | [Sparse load (Tier 1)](01-sparse-load.md) | `Index::load_in_region` — read whole file, drop out-of-region stars before building KD-trees. No file format change. | ~3 days |
| 2 | [HEALPix-grouped layout (Tier 2)](02-healpix-format.md) | File format v3: stars sorted by HEALPix cell, header carries cell→offset table. Disk I/O drops to O(region). | ~1 week |
| 3 | [LiveIndex](03-live-index.md) | Stateful in-memory index supporting `ensure_region` / `drop_outside`. Tree rebuild on cell changes. | ~3 days |
| 4 | [Pointing sources](04-pointing-sources.md) | `PointingSource` trait + `GroundStation` (lat/lon→zenith) and `SpacecraftBoresight` (ephemeris+quaternion→boresight). | ~1 week |
| 5 | [RealtimeSolver](05-realtime-solver.md) | Orchestrator: ties LiveIndex + PointingSource + refinement; `tick()` to refresh, `solve()` to produce a refined solution. | ~1 week |

**Total:** 3-4 engineer-weeks for the full stack. Each plan can land independently
behind its predecessors.

## Reading order

If skimming for design context: **README → plan 5 → plans 1-4 in dependency order.**

If implementing: top-to-bottom (1 → 5). Plans 1, 2, and 3 are strict
prerequisites for 5; plan 4 is independent of 1-3 but consumed by 5.

## Open cross-cutting decisions

These appear in multiple plans; settling them once will avoid churn:

- **HEALPix depth for cell grouping.** The existing index builder uses
  `uniformize_depth` (typically 6-8) for *star selection*. Cell-grouping
  granularity is a separate choice — finer means smaller per-cell loads
  but larger header tables and more cells to track. Default
  recommendation: depth 5 (3072 cells, ~5° per cell) — coarse enough that
  a 1° hint loads ~1-2 cells, fine enough that the cell table stays small
  (3072 × 16 bytes ≈ 50 KB).
- **Quad/code locality.** Quads can span cells. Either
  (a) duplicate quads into every cell touching them (~3-4× quad-block
  growth), or (b) keep quads ungrouped (load all quads always; for a 31M
  star d8 index that's still only ~25 MB of quad data). **Recommendation:
  start with (b)**, revisit if quad load dominates.
- **Tree rebuild policy.** KD-trees are O(N log N) to build. For a
  realtime cadence with ~10k stars in scope, ~ms-scale rebuild is fine.
  No need for incremental KD-tree updates.

## Status

| # | Plan | Status |
|---|---|---|
| 1 | Sparse load | not started |
| 2 | HEALPix layout | not started |
| 3 | LiveIndex | not started |
| 4 | Pointing sources | not started, depends on issue #44 (terrestrial observer state) |
| 5 | RealtimeSolver | not started |
