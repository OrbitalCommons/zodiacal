# Ground-telescope mode

Realtime plate solving for a fixed-location telescope as the visible
sky rotates with Earth. The loaded index tracks the zenith — only
HEALPix cells currently observable above the horizon stay resident in
RAM. New cells are loaded from the v3 `.zdcl` file as the zenith drifts;
cells that have set are dropped.

## When to use

- Telescope automation pipelines where the field of view is bounded
  by the local horizon at any given instant.
- Long-duration imaging campaigns where you'd rather not hold the full
  sky resident.
- Robotic telescopes with a known mount geometry.
- Anywhere the pointing changes on Earth-rotation timescales.

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  Process                                                   │
│                                                            │
│  ZdclFile (mmap, ~/path/to/gaia_g14.zdcl)                  │
│      │                                                     │
│      ▼                                                     │
│  LiveIndex<ZdclFile>     ◄────── ensure_region() per tick  │
│      │   ┌──────────────────┐                              │
│      ├──►│ KdForest<3>      │  per-cell sub-trees;         │
│      │   │  ┌─┐ ┌─┐ ┌─┐ ┌─┐ │  add/drop on cell change     │
│      │   │  └─┘ └─┘ └─┘ └─┘ │  (no full rebuild)           │
│      │   └──────────────────┘                              │
│      │                                                     │
│      ▼                                                     │
│  RealtimeSolver<ZdclFile, GroundStation>                   │
│      │                                                     │
│      │   on tick(t):                                       │
│      │     region = GroundStation::current_region(t)       │
│      │             = (zenith_radec(lat, lon, t),           │
│      │                90° - min_altitude)                  │
│      │     live.set_region(region)  ← cell delta           │
│      │                                                     │
│      ▼                                                     │
│  solve(sources) → Solution                                 │
└────────────────────────────────────────────────────────────┘
```

## End-to-end example

```rust
use std::path::Path;
use std::time::Duration;

use starfield::time::Timescale;
use zodiacal::index::{LiveIndex, ZdclFile};
use zodiacal::pointing::GroundStation;
use zodiacal::realtime::{RealtimeSolver, RefreshPolicy};
use zodiacal::solver::{SolverConfig, VerifyConfig};

fn main() -> std::io::Result<()> {
    // Open the v3 index lazily — no stars are loaded yet, just the
    // cell table from the file header.
    let source = ZdclFile::open(Path::new("indexes/gaia_g14.zdcl"))?;

    // Mount a station: latitude/longitude in degrees, plus a minimum
    // altitude cutoff (stars below this altitude get pruned from the
    // returned region).
    let pointing = GroundStation::from_degrees(
        /* lat   */ 34.225,    // Mt. Wilson, e.g.
        /* lon   */ -118.057,
        /* min_alt */ 30.0,    // 30° above horizon
    );

    // The orchestrator. OnPointingDelta means we only re-load cells
    // when the zenith has drifted >0.1 rad (~5.7°) or the snapshot
    // is older than 60s. Tune these to your refresh budget.
    let mut rt = RealtimeSolver::new(source, pointing).with_refresh_policy(
        RefreshPolicy::OnPointingDelta {
            angular_threshold_rad: 0.1,
            max_age: Duration::from_secs(60),
        },
    ).with_solver_config(SolverConfig {
        verify: VerifyConfig {
            log_odds_accept: 30.0,
            ..VerifyConfig::default()
        },
        ..SolverConfig::default()
    });

    let ts = Timescale::default();
    loop {
        let now = ts.tt(/* your current time */ chrono::Utc::now());

        // Pull a frame from your camera, extract sources... (your code)
        let sources = extract_sources_for_current_frame();
        let image_size = (4096.0, 4096.0);

        // tick + solve in one call. tick() refreshes the LiveIndex;
        // the solver runs against the now-current loaded set.
        let out = rt.solve(&sources, image_size, &now)?;
        match out.solution {
            Some(sol) => {
                let (ra, dec) = sol.wcs.field_center();
                println!("solved: ra={ra:.3} dec={dec:.3}");
            }
            None => eprintln!("no solution this frame"),
        }

        std::thread::sleep(Duration::from_secs(5)); // your frame rate
    }
}

# fn extract_sources_for_current_frame() -> Vec<zodiacal::extraction::DetectedSource> { vec![] }
```

## How `KdForest` makes this affordable

The original Plan-3 design called for a unified `KdTree<3>` over all
loaded stars, rebuilt on every cell add/drop. For ground telescopes
that's expensive: every minute or two as a cell rises or sets, you'd
pay an O(N log N) rebuild.

`KdForest` stores per-cell sub-trees instead. Add/drop becomes:

```rust
// Loading a new cell that just rose above the horizon.
let new_cell_stars: Vec<[f64; 3]> = /* from ZdclFile::load_cells */;
let sub_tree = KdTree::<3>::build(new_cell_stars, indices);
forest.insert(cell_id, sub_tree);   // O(1) insert into Vec

// Dropping a cell that just set.
forest.remove(cell_id);              // O(C) for the Vec retain
```

Per-cell tree build is **O(C log C)** where C is stars-per-cell — at
HEALPix depth 5 with a typical mag-limit, that's 100s of stars and
takes microseconds. Dropping a cell is just a `Vec::retain` — no tree
walk, no allocation.

### Concrete cost comparison

For a ground station at 34° latitude with `min_altitude_deg = 30°`,
the visible region has radius 60°. At cell depth 5 (3072 cells over the
full sky), that's ~770 cells visible at any instant. A typical mag-14
catalog yields ~5,500 stars per cell, so ~4.2 M stars resident.

| Operation | Plan-3 unified-tree (rebuild) | KdForest (delta-only) | Win |
|---|---|---|---|
| Cell rises (1 cell added) | rebuild over 4.2 M stars: ~1.5 s | build 1 sub-tree of 5.5 K stars: ~3 ms | **500×** |
| Cell sets (1 cell dropped) | rebuild over 4.2 M stars: ~1.5 s | `Vec::retain` removing 1 entry: <1 µs | **>10⁶×** |
| Solve (after refresh) | flat tree query: ~0.5 ms | forest query (770 sub-trees): ~5 ms | 0.1× |

The forest pays a **10× per-query** cost (each query fans out across
sub-trees) in exchange for **near-free membership changes**. That's
the right trade for a realtime loop where membership churns every few
minutes.

If your refresh cadence becomes much faster than per-frame solves
(e.g., a slewing follow-up scope), the forest is even more dominant.
If it becomes much slower (a stare-mode survey), consider periodically
flattening to a unified tree via `LiveIndex::as_index()` for the hot
path and rebuilding the forest in the background.

## Memory expectations

For a 34° latitude station with 30° min altitude, depth-5 cell grouping,
a mag-14 base catalog:

| Component | RSS |
|---|---|
| `ZdclFile` mmap (header + cell table only) | ~50 KB |
| `LiveIndex` cells (770 visible × ~5,500 stars × 32 B) | ~135 MB |
| KD-tree node overhead (~50 B/star) | ~210 MB |
| Per-cell quad+code metadata | ~5 MB |
| **Total resident** | **~350 MB** |

vs. server mode (~5 GB for the same source data) — ~14× smaller
working set because we never page in the cells we can't see.

The `.zdcl.gaia` sidecar mmap is demand-paged just like server mode;
typically <50 MB resident even for high-rate refinement workloads.

## Performance expectations

Per-frame budget at 1 Hz, 1 s solve cadence:

| Stage | Time | Notes |
|---|---|---|
| `tick()` (no membership change) | <100 µs | Just a `current_region` call + same-cells comparison. |
| `tick()` (1–5 cells change) | 5–25 ms | Per-cell file read + sub-tree build. |
| `solve` (blind, mag-14 region) | 10–50 ms | Forest fan-out × per-tree query cost. |
| `refine_solution` (50 stars, no observer state for ground) | 5 ms | Apparent-place chain. |
| **Total per-frame budget** | **~100 ms p99** | Well within a 1 Hz loop. |

For 10 Hz (planetary follow-up scopes), you'd want
`RefreshPolicy::OnPointingDelta` with a tight `angular_threshold_rad`
to skip refreshes when the pointing hasn't changed appreciably.

## Sharding & scaling

### Single station, single index

The most common case. One `LiveIndex<ZdclFile>` over the deepest
single-band index you can build (mag 14 fits comfortably; mag 16 if
your station has 16+ GB RAM).

### Multi-band scale series

For wide-field plus narrow-field cameras on the same mount, hold one
`LiveIndex` per scale band. `solve()` accepts a slice of `&Index`, so
you can call `as_index()` on each `LiveIndex` and feed all the views in:

```rust
let v0 = live_band_0.as_index();
let v1 = live_band_1.as_index();
let v2 = live_band_2.as_index();
let (sol, _) = solve(&sources, &[&v0, &v1, &v2], image_size, &config);
```

`as_index()` is cached on `LiveIndex::build_generation`, so the cost is
paid only when membership has actually changed.

### Multiple stations

Each station gets its own `RealtimeSolver` instance. They can share
the underlying `ZdclFile` (it's `Sync`) — no extra disk space needed.

### What to monitor

| Metric | Why |
|---|---|
| Cells loaded count | Sanity-check the geometry (~770 for 30°-altitude horizon at 34° lat) |
| `RealtimeOutput.refresh_elapsed` | Per-tick refresh cost; watch for OS page-cache thrash |
| `RealtimeOutput.solve_elapsed` | Per-frame solve cost; should track the loaded star count |
| `LiveIndex::build_generation` jumps | Indicates membership change rate; useful for tuning refresh policy |
| Sidecar mmap RSS | Refinement working set |

## Pitfalls

- **Refresh policy choice matters.** `EveryTick` is the safe default
  but pays the `current_region` + diff cost on every solve. Switch to
  `OnPointingDelta` once you know your zenith-drift cadence.
- **`min_altitude_deg` affects both refraction and load.** Lower
  values load more cells (bigger memory) and pull in stars near the
  horizon where atmospheric refraction is significant. 20–30° is the
  sweet spot for most astronomy.
- **`observer_state` returns `None` in v1.** Ground-mode refinement
  currently skips parallax/aberration corrections (they're <100 mas
  for typical fields, so the refinement still works at arcsec
  precision). [Issue #44](https://github.com/OrbitalCommons/zodiacal/issues/44)
  tracks the terrestrial → BCRS state pipeline; once that lands,
  ground refinement matches space-mode precision.
- **Don't `as_index()` per solve unless membership actually changed.**
  The `RealtimeSolver` already caches by `build_generation` for you;
  if you call `as_index()` directly, gate it the same way.
- **Polar orbits / equatorial mounts.** `GroundStation` assumes a
  fixed location. If your mount slews to RA/Dec on demand (rather
  than tracking the zenith), use the spacecraft pattern instead with
  a constant ephemeris and an attitude source that returns your
  current target.

## Open issues

- [#44](https://github.com/OrbitalCommons/zodiacal/issues/44) —
  terrestrial → BCRS observer state for refinement.
- [#46](https://github.com/OrbitalCommons/zodiacal/issues/46) —
  `solve_and_refine` convenience wiring.
