# Regional bundle solving

`ZdclBundle::solve_region(&sources, image_size, &region, &config)` replaces
`load_region -> fragments -> Index -> solve`. It returns a
`Result<RegionalSolveReport, RegionalSolveError>`:

- `report.solve.outcome`: `Solved`, `TimedOut`, or `NoMatch`.
- `report.solve.stats`: verification/rejection diagnostics.
- `report.profile`: separate load, index-build, solve, and total elapsed times,
  plus band, star, and quad counts. Star count is across loaded cells once,
  not multiplied by the number of bands.
- `report.indexes`: loaded indexes in manifest band order, retaining the meaning
  of solution `index_id` and matched reference-star indices for diagnostics or
  refinement. Drop these when only the WCS is needed, or reuse them with
  `solve_checked` for repeated frames in the same region.

The loading region selects overlapping HEALPix cells; it does not restrict the
accepted WCS center. Set `config.within` explicitly to require a center within a
region. The supplied config is never rewritten. Timeout applies to the search,
not loading or index construction, and remains cooperative rather than a hard
wall-clock deadline.

Invalid sources, image dimensions, region or configuration return typed
`InvalidInput` errors before I/O. Bundle loading/format failures return `Load`
with the original I/O error available. Keep the bundle handle open to reuse its
shard cache. Existing manual load and solve APIs remain available.

Aperture1 can replace its bundle orchestration and stage timers with this call.
Retain source filtering before the mission cap, hint freshness/camera checks,
mission time/size limits, camera quaternion/row-parity conventions, and FSM
behavior downstream.
