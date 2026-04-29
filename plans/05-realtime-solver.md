# Plan 5: RealtimeSolver (orchestrator)

## Goal

Tie `LiveIndex` (plan 3) and `PointingSource` (plan 4) together with the
existing solver and refinement pipelines. Expose a `tick()` entry point that
refreshes the loaded cells and a `solve()` entry point that produces a
refined solution using the currently-loaded index.

This is the front door for both realtime modes (telescope, star tracker).
The server mode (mode 1) doesn't need this — it's a thin convenience over
the existing `Index`/`solve` API.

## Non-goals

- No background prefetch / async loading. v1 is synchronous; `tick()` blocks.
  Background prefetch is a v2 enhancement (see "Open questions").
- No multi-platform orchestration (e.g., array of star trackers sharing a
  cache). Single platform per `RealtimeSolver` instance.
- No persistence of solver state across restarts.

## API surface

```rust
// src/realtime.rs (new module)

pub struct RealtimeSolver<S: IndexSource, P: PointingSource> {
    live: LiveIndex<S>,
    pointing: P,
    solver_config: SolverConfig,
    refinement_config: Option<RefinementConfig>,
    catalog: Option<RefinementCatalog>,  // optional sidecar-backed
    refresh_policy: RefreshPolicy,
    last_refresh: Option<RefreshSnapshot>,
    /// User-supplied padding added to PointingSource::current_region.
    region_padding_rad: f64,
}

pub enum RefreshPolicy {
    /// Refresh on every `solve()` call. Simple, predictable.
    EveryTick,

    /// Skip refresh if the new region center is within `angular_threshold`
    /// of the previous refresh's region center AND the time since last
    /// refresh is < `max_age`.
    OnPointingDelta {
        angular_threshold: f64,    // radians
        max_age: Duration,
    },

    /// Refresh only at fixed intervals.
    OnInterval { period: Duration },
}

#[derive(Debug, Clone)]
struct RefreshSnapshot {
    when: Instant,
    region: SkyRegion,
    cells_loaded: usize,
    stars_loaded: usize,
}

#[derive(Debug)]
pub struct RealtimeOutput {
    pub solution: Option<RefinedSolution>,
    pub refresh: Option<EnsureReport>,
    pub solve_elapsed: Duration,
    pub refresh_elapsed: Duration,
    /// Snapshot of the loaded cell set used for this solve.
    pub build_generation: u64,
}

impl<S: IndexSource, P: PointingSource> RealtimeSolver<S, P> {
    pub fn builder(source: S, pointing: P) -> RealtimeSolverBuilder<S, P>;

    /// Refresh loaded cells per the refresh policy. No-op if policy decides
    /// nothing has changed enough.
    pub fn tick(&mut self, t: Time) -> io::Result<Option<EnsureReport>>;

    /// Run a full solve at time `t`. Calls `tick(t)` first, then `solve`,
    /// then (if configured) `refine_solution`. Returns the refined solution
    /// plus diagnostics.
    pub fn solve(
        &mut self,
        sources: &[DetectedSource],
        image_size: (f64, f64),
        t: Time,
    ) -> io::Result<RealtimeOutput>;

    pub fn loaded_cell_count(&self) -> usize;
    pub fn last_refresh(&self) -> Option<&RefreshSnapshot>;
}

pub struct RealtimeSolverBuilder<S, P> { ... }

impl<S, P> RealtimeSolverBuilder<S, P> {
    pub fn solver_config(self, c: SolverConfig) -> Self;
    pub fn refinement_config(self, c: RefinementConfig) -> Self;
    pub fn catalog(self, c: RefinementCatalog) -> Self;
    pub fn refresh_policy(self, p: RefreshPolicy) -> Self;
    pub fn region_padding_deg(self, deg: f64) -> Self;
    pub fn build(self) -> RealtimeSolver<S, P>;
}
```

The builder pattern keeps the constructor sane (refinement is optional, catalog
is optional, padding has a sensible default, etc.). For users who don't need
refinement, omit `.catalog()` and `.refinement_config()` — `solve` returns a
`RefinedSolution` with `n_iterations: 0` and the raw `SipWcs` from tweak.

## Data flow

```
┌──────────────┐    ┌──────────────┐     ┌───────────────┐
│ PointingSource│───▶│ current_region│────▶│ RefreshPolicy  │
└──────────────┘    └──────────────┘     │  decides:     │
                                          │   refresh?    │
                                          └──────┬────────┘
                                                 │
                                                 ▼
                                          ┌──────────────┐
                                          │ LiveIndex.   │
                                          │ set_region() │
                                          └──────┬───────┘
                                                 │
            ┌────────────────────────────────────┘
            ▼
┌──────────────────────┐    ┌──────────────┐     ┌──────────────────┐
│ DetectedSources       │───▶│ solve(...)   │────▶│ tweak_solution   │
└──────────────────────┘    │ (existing)   │     │ (existing)       │
                            └──────────────┘     └────────┬─────────┘
                                                          │
                                  ┌───────────────────────┘
                                  ▼
                            ┌──────────────┐    ┌──────────────────┐
                            │ if refinement │───▶│ refine_solution  │
                            │   configured: │    │ (existing)       │
                            └──────────────┘    └────────┬─────────┘
                                                          │
                                                          ▼
                                                  ┌──────────────┐
                                                  │ RealtimeOutput│
                                                  └──────────────┘
```

`PointingSource::observer_state(t)` feeds into the `ObservationContext` for
refinement automatically — no need for the caller to specify it separately.

## Algorithm

### `tick`

1. Consult `refresh_policy` + `last_refresh` to decide whether to refresh:
   - `EveryTick`: always.
   - `OnPointingDelta`: compute candidate region, compare angular distance
     of its center to `last_refresh.region.center`. If both above threshold
     AND `now - last_refresh.when > max_age`, refresh. (Requires evaluating
     `pointing.current_region(t)` even on no-op ticks.)
   - `OnInterval`: refresh if `now - last_refresh.when >= period`.
2. If refresh: `region = pointing.current_region(t)`; pad by `region_padding_rad`;
   call `live.set_region(&padded_region)`. Update `last_refresh`.
3. Return `Some(report)` on refresh, `None` on skip.

### `solve`

1. `tick(t)?` — may load/drop cells.
2. `(solution, _stats) = solve(sources, &[live.as_index()], image_size, &solver_config)`.
3. If `solution.is_none()`: return `RealtimeOutput { solution: None, ... }`.
4. Run `tweak_solution` against the live index (preserving existing semantics).
5. If `refinement_config` and `catalog` are both set:
   - Build `ObservationContext` from `pointing.observer_state(t)` (or fall
     back to a stub if `None`).
   - Call `refine_solution(...)`.
6. Return `RealtimeOutput`.

### Refresh-policy choice rationale

| Policy | Best for | Risk |
|---|---|---|
| `EveryTick` | low solve rate (< 1 Hz), debugging | wasted I/O |
| `OnPointingDelta` | telescope cadence (~Hz), star tracker | tuning sensitivity |
| `OnInterval` | streaming with no attitude estimator | stale cells if scope moves fast |

Default: `OnPointingDelta { angular_threshold: 0.5° in radians, max_age: 60s }`.

## Backwards compatibility

Brand-new module. No impact on existing code.

`RefinedSolution` is the existing type from PR #49. `EnsureReport` is from
plan 3. `SkyRegion`, `SolverConfig`, `RefinementConfig` are existing.

## Tests

Unit tests in `src/realtime.rs::tests`:

- **`tick_no_refresh_when_policy_skips`**: configure `OnPointingDelta` with
  large threshold; tick repeatedly with same time → no refresh after first.
- **`tick_refresh_when_pointing_changes`**: mock pointing source returning
  shifting regions; verify refresh fires when angular delta > threshold.
- **`solve_returns_none_when_no_match`**: empty sources → `solution: None`,
  `solve_elapsed > 0`, `refresh_elapsed > 0`.
- **`solve_with_refinement_calls_refine`**: synthetic scenario; verify
  `solution.is_some()` and `n_iterations > 0`.
- **`solve_without_catalog_skips_refinement`**: omit catalog; verify
  refinement is skipped, `wcs` is from tweak directly.
- **`build_generation_propagates`**: verify `RealtimeOutput.build_generation`
  matches `LiveIndex.build_generation()` at solve time.

Integration tests:

- **`ground_station_simulated_dawn_to_dusk`**: ground station + LiveIndex over
  a synthetic full-sky source. Tick at 1-minute intervals across 12 simulated
  hours. Verify cells track zenith, stars are loaded/dropped reasonably,
  no errors.
- **`star_tracker_attitude_sweep`**: spacecraft + LiveIndex; mock attitude
  source rotates the boresight at 0.1°/s for 60 simulated seconds. Verify
  cells follow, refinement converges on each frame.
- **`star_tracker_with_real_zodiacal_web_indexes`** (gated, manual): use the
  actual `gaia_jbt_*.zdcl` indexes converted to v3 (plan 2's CLI), verify
  end-to-end works at realtime cadence.

## Effort estimate

| Step | Effort |
|---|---|
| `RealtimeSolver` struct + builder | 1 day |
| `tick` + refresh policies | 1 day |
| `solve` + tween-with-refinement glue | 1.5 days |
| Unit tests (6) | 1 day |
| Integration tests (2 synthetic) | 1.5 days |
| Manual test against real indexes | 0.5 day |
| Docs + example binary (`examples/realtime_telescope.rs`) | 1 day |
| **Total** | **~7-8 days (1 week+)** |

## Dependencies

Strict prerequisites:

- **Plan 2** — `IndexSource` + `ZdclFile`.
- **Plan 3** — `LiveIndex`.
- **Plan 4** — `PointingSource`.

Soft dependencies:

- **Issue #44** — terrestrial observer state (only blocks ground-mode
  refinement, not the orchestrator itself).
- **Refinement work from PR #49** — already landed locally (in flight as PR).

## Open questions

- **Background prefetch.** For predictable motion (sidereal tracking, smooth
  attitude maneuvers) we could pre-load cells the boresight is about to
  enter. Adds a thread, a channel, and lifetime complexity. Recommendation:
  measure first — if `tick()` latency dominates the solve loop on real data,
  add prefetch as a v2 enhancement.
- **Telemetry / metrics export.** Real systems want to monitor "% time spent
  in refresh vs solve vs refine", "stars currently loaded", "cell churn rate
  per minute". Recommendation: emit these via a `MetricSink` trait that
  defaults to no-op; ship a `LogMetricSink` for development. Don't bake in a
  specific metrics library.
- **Failure modes.** What happens when:
  - `pointing.current_region(t)` returns an empty/degenerate region?
    → Skip refresh, log warning, retain previous loaded set.
  - `live.set_region` fails partway through (I/O error from source)?
    → `set_region` is atomic per plan 3 — old state preserved. Propagate
       the error from `tick()`. Caller decides whether to retry or bail.
  - `solve` returns no match with a tight hint, fast clock?
    → Return `RealtimeOutput { solution: None, ... }` — caller handles.
  - Sources are stale (dropped frame, retransmit)?
    → Caller's job to deduplicate before passing in. Not our concern.
- **Time source.** `tick(t)` and `solve(t)` take `Time` explicitly so callers
  can replay logged data. For live use, callers pass `Time::now()` or
  equivalent. Recommendation: keep the explicit parameter; don't bake in
  "wall clock now."
- **Concurrent access.** Single mutable owner. Multiple solver clients on
  one platform should share the underlying `Arc<RwLock<LiveIndex>>` and
  build separate `RealtimeSolver`s, or wrap the whole `RealtimeSolver` in
  `Arc<Mutex<...>>`. Document but don't implement locking ourselves.
- **What if the catalog needs refresh too?** The Gaia sidecar (issue #45) is
  also keyed by region. As cells are loaded/dropped from `LiveIndex`, the
  corresponding sidecar entries should follow. Recommendation: extend
  `RefinementCatalog` to support live cell membership the same way; treat as
  a follow-up to plan 5 (or rolled into a v2 of plan 5).
