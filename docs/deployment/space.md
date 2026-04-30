# Spacecraft star-tracker mode

Realtime plate solving for a spacecraft platform with a known
ephemeris and a (possibly noisy) attitude estimate. The loaded index
follows the boresight as the spacecraft slews; only HEALPix cells
near the current FOV stay resident. Refinement gets observer state
for free from the ephemeris, so parallax + aberration corrections
land naturally without extra plumbing.

## When to use

- Star tracker on a satellite, probe, lander, or balloon.
- Any platform where boresight pointing is known (or estimated) and
  changes faster than Earth-rotation timescales.
- Hard memory constraints (flight hardware, embedded targets).
- Refinement at the 10 mas level — spacecraft mode is the natural
  home for this since the ephemeris already provides the BCRS
  observer state refinement needs.

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│  Process / FSW task                                          │
│                                                              │
│  ZdclFile (mmap)         EphemerisSource    AttitudeSource   │
│      │                        │                  │           │
│      ▼                        │                  │           │
│  LiveIndex<ZdclFile>          │                  │           │
│      │ ┌──────────────────┐   │                  │           │
│      │ │ KdForest<3>      │   │                  │           │
│      │ │  ┌─┐ ┌─┐ ┌─┐     │   │                  │           │
│      │ │  └─┘ └─┘ └─┘     │   │                  │           │
│      │ └──────────────────┘   │                  │           │
│      │                        │                  │           │
│      │              ┌─────────▼──────────────────▼───────┐   │
│      │              │ SpacecraftBoresight<E, A>          │   │
│      │              │   current_region(t) =              │   │
│      │              │     attitude · boresight_body      │   │
│      │              │     → SkyRegion(detector + pad)    │   │
│      │              │   observer_state(t) =              │   │
│      │              │     ephemeris.state_at(t)          │   │
│      │              │     → ObserverState::Barycentric   │   │
│      │              └────────────────┬───────────────────┘   │
│      │                               │                       │
│      ▼                               ▼                       │
│  RealtimeSolver<ZdclFile, SpacecraftBoresight<E, A>>         │
│      │                                                       │
│      │  on tick(t): set_region(boresight FOV)                │
│      │                                                       │
│      │  on solve():                                          │
│      │    1. tick (refresh loaded cells)                     │
│      │    2. solve(sources) → blind WCS                      │
│      │    3. refine_solution(wcs, catalog,                   │
│      │       ObservationContext { time: t,                   │
│      │                            observer: state })         │
│      │       → 10 mas WCS                                    │
│      ▼                                                       │
│  RefinedSolution                                             │
└──────────────────────────────────────────────────────────────┘
```

## End-to-end example

```rust
use std::path::Path;
use std::time::Duration;

use nalgebra::UnitQuaternion;
use starfield::time::{Time, Timescale};
use zodiacal::index::{LiveIndex, ZdclFile};
use zodiacal::pointing::{
    AttitudeSource, BcrsState, EphemerisSource, SpacecraftBoresight,
};
use zodiacal::realtime::{RealtimeSolver, RefreshPolicy};
use zodiacal::refinement::{
    ObservationContext, ObserverState, RefinementCatalog,
    RefinementConfig, SidecarReader,
};
use zodiacal::solver::SolverConfig;

// Bring your own ephemeris/attitude impls. These are typically
// thin wrappers over anise/SPICE/TLE/Kalman code.
struct AniseEphemeris { /* anise::Almanac + spk handle */ }

impl EphemerisSource for AniseEphemeris {
    fn state_at(&self, _t: &Time) -> Option<BcrsState> {
        // Look up spacecraft barycentric state from your kernel.
        // Stub for the example; in production this is an anise/SPICE call.
        Some(BcrsState {
            position_au: [0.998, 0.012, 0.005],
            velocity_au_per_day: [-0.000_3, 0.017, 0.000_1],
        })
    }
}

struct KalmanAttitude { /* your filter state */ }

impl AttitudeSource for KalmanAttitude {
    fn quaternion_at(&self, _t: &Time) -> Option<UnitQuaternion<f64>> {
        // Return current best estimate. None during initialization
        // → pointing source falls back to full-sky region (blind solve).
        Some(UnitQuaternion::identity())
    }
}

fn main() -> std::io::Result<()> {
    // Open the v3 index — header only, no stars resident yet.
    let source = ZdclFile::open(Path::new("indexes/gaia_g14.zdcl"))?;

    // Construct the pointing source from caller-supplied ephemeris
    // and attitude estimators.
    let pointing = SpacecraftBoresight::new(
        AniseEphemeris { /* ... */ },
        KalmanAttitude { /* ... */ },
        /* detector_half_angle */ 5_f64.to_radians(),  // 10° diagonal FOV
        /* fov_padding         */ 5_f64.to_radians(),  // pad for slew
    );

    // For star tracker cadence (~10 Hz), prefer OnPointingDelta with
    // a tight angular threshold — refresh only when the boresight has
    // actually moved.
    let mut rt = RealtimeSolver::new(source, pointing).with_refresh_policy(
        RefreshPolicy::OnPointingDelta {
            angular_threshold_rad: 0.02,                 // ~1.1°
            max_age: Duration::from_secs(5),
        },
    );

    // Optional: refinement plumbing.
    let sidecar = SidecarReader::open(Path::new("indexes/gaia_g14.zdcl.gaia"))?;
    let refinement_config = RefinementConfig::default();

    let ts = Timescale::default();
    loop {
        // Get current frame from the camera, extract sources, current time.
        let now = ts.tt(/* current TT */ chrono::Utc::now());
        let sources = read_camera_frame_sources();
        let image_size = (1024.0, 1024.0);

        // Tick + blind solve.
        let out = rt.solve(&sources, image_size, &now)?;
        let Some(blind) = out.solution else {
            continue; // no match this frame — stay silent or report
        };

        // Refine using the ephemeris-supplied observer state.
        let observer = rt.pointing().observer_state(&now)
            .expect("ephemeris must be available for refinement");
        let obs_ctx = ObservationContext { time: now.clone(), observer };

        let matched_ids: Vec<u64> = collect_matched_source_ids(&blind);
        let catalog = RefinementCatalog::from_sidecar_filtered(
            &sidecar, &matched_ids,
        );
        let refined = zodiacal::refinement::refine_solution(
            &blind.wcs.into(), &sources, rt.live_index().as_index().star_tree.as_ref(),
            &catalog, &obs_ctx, &refinement_config,
        )?;

        publish_attitude_update(refined);
    }
}

# fn read_camera_frame_sources() -> Vec<zodiacal::extraction::DetectedSource> { vec![] }
# fn collect_matched_source_ids(_: &zodiacal::solver::Solution) -> Vec<u64> { vec![] }
# fn publish_attitude_update(_: zodiacal::refinement::RefinedSolution) {}
```

(Code shown is illustrative — exact API for sidecar→catalog wiring
depends on whether you've integrated the convenience helpers from
[#46](https://github.com/OrbitalCommons/zodiacal/issues/46).)

## How `KdForest` makes this affordable

Spacecraft slews can change the loaded cell set faster than ground
stations do. A typical maneuver might rotate the boresight 10° in
seconds; at HEALPix depth 5 (~5° per cell) that's 1–4 cell additions
and removals per second.

With unified-tree rebuilds (the original Plan-3 design), each
membership change would pay an O(N log N) rebuild over the entire
loaded set. For a typical FOV-only resident set of ~50 K stars that's
~30 ms per rebuild — multiplied by 4 rebuilds/sec = 120 ms/sec spent
just on KD-tree maintenance, which would dominate a 10 Hz solve loop.

`KdForest` reduces this to per-cell sub-tree builds:

| Operation | Unified-tree (rebuild) | KdForest (delta-only) | Win |
|---|---|---|---|
| Slew brings 1 new cell into FOV | ~30 ms (rebuild over 50 K) | ~0.1 ms (build 1 sub-tree of ~150 stars) | **300×** |
| Slew drops 1 cell out of FOV | ~30 ms | <1 µs (`Vec::retain`) | **>10⁵×** |
| Rapid sweep (5 cells/sec churn) | 150 ms/sec lost to rebuilds | <1 ms/sec | **150×** |
| Solve fanout cost | 0 (single tree) | ~0.5 ms (~30 sub-trees) | 0.5× |

The forest's per-query cost (5 sub-trees × ~0.1 ms = 0.5 ms) is
strictly smaller than the rebuild cost it replaces (30 ms once per
slew). Both scale linearly, but the constants are dramatically in
favor of the forest at flight cadence.

### Pre-fetch is the next optimization

For predictable maneuvers (sidereal tracking, planned attitude profiles),
the orchestrator can pre-load cells the boresight is about to enter.
Not implemented in v1 — see "Open issues" below.

## Memory expectations

For a 5°-half-angle detector + 5° padding (so a 10°-radius region),
depth-5 cell grouping, mag-14 base catalog:

| Component | RSS |
|---|---|
| `ZdclFile` mmap (header + cell table) | ~50 KB |
| `LiveIndex` cells (~13 visible × ~150 stars × 32 B) | ~60 KB |
| KD-tree node overhead (~50 B/star) | ~100 KB |
| Per-cell quad+code metadata | ~20 KB |
| Refinement catalog (50 matched stars × 88 B) | ~5 KB |
| **Total resident** | **~250 KB** |

Yes, kilobytes. Spacecraft mode hits the absolute minimum because
the FOV is genuinely tiny relative to the full sky. This is the
mode where the difference between "load full sky" and "load only
what you can see" is most dramatic.

## Performance expectations

Per-frame budget at 10 Hz on a flight-class CPU (single core, ARM
Cortex-A78 class). Real numbers depend heavily on the host
platform; treat these as order-of-magnitude.

| Stage | Time | Notes |
|---|---|---|
| `tick()` (no membership change) | ~20 µs | Quaternion math + same-cell comparison. |
| `tick()` (1 cell change) | ~1 ms | mmap fault + sub-tree build (cold cache). |
| `solve` (blind, 13-cell region) | ~1 ms | Forest fan-out is small; dense indexes match fast. |
| Refinement (50 matched stars) | ~3 ms | Apparent-place + weighted re-fit; observer state from ephemeris. |
| **Total per-frame budget** | **~5 ms** | Well within 100 ms / 10 Hz. |

For 100 Hz tracking (some agile tactical systems), the limiting
factor becomes the cell-load latency on the I/O path. Pre-fetching
predicted-future cells in a background thread would close that gap.

## Sharding & scaling

### Single index, narrow FOV

The most common spacecraft case. One `LiveIndex<ZdclFile>` over the
deepest catalog the platform's memory budget supports (mag 14 fits
in <1 MB resident; mag 16 is ~10 MB).

### Multi-band scale series

For cameras with very different angular scales (e.g., a wide-field
fine-guidance + a narrow-field science detector), hold one
`LiveIndex` per scale band over different `ZdclFile`s. Each
maintains its own forest; they share the underlying mmap if they
point at the same file.

### Embedded / no_std considerations

`LiveIndex` and `KdForest` are pure-Rust with no I/O on the hot path
(disk reads happen at `set_region` time, never during `solve`).
Memory is the only concern — see budget above.

`ZdclFile` uses `memmap2` which requires libc-level mmap support;
on bare-metal targets you'd need to swap the backing for a `Read +
Seek` impl over your storage layer. Not currently abstracted; see
[`IndexSource`](../../src/index/source.rs) for the trait you'd
re-implement.

### What to monitor

| Metric | Why |
|---|---|
| Boresight angular delta per frame | Inputs the refresh-policy threshold |
| Cells loaded count | Should track 10–30; spikes indicate slew transients |
| `RealtimeOutput.refresh_elapsed` | Cell-load tail latency; watch for I/O blocking |
| `RealtimeOutput.solve_elapsed` | Forest-query tail latency |
| Refinement residual_rms_mas | The actual precision you're achieving |
| Attitude estimator update rate vs solve rate | If filter outpaces solver, your tick may pre-empt useful work |

## Pitfalls

- **`SpacecraftBoresight` defaults boresight to +Z body frame.** Most
  detectors are aligned this way, but if yours isn't, set
  `boresight_body` explicitly via the builder.
- **Quaternion convention.** Body-to-inertial: multiplying a body-frame
  vector by the quaternion gives its inertial-frame coordinates.
  Mismatch with your filter's convention will silently give wrong
  pointing. Verify with a known maneuver in commissioning.
- **`AttitudeSource::quaternion_at` returning `None`** falls back to a
  full-sky region (blind solve). That's the right behavior during
  estimator startup, but if it persists in flight you're paying
  full-sky load cost every refresh — instrument and alarm on it.
- **Time accuracy matters for refinement.** A 1-second time error
  produces ~15″ of error from Earth's BCRS velocity alone. The
  pointing path tolerates seconds of slop (cell-load is angular-scale
  forgiving); refinement does not. If your time source is rough, use
  it for `tick()` and pass a more precise time to `refine_solution`.
- **EOPs are not needed.** Spacecraft mode skips Earth orientation
  parameters entirely (the ephemeris gives BCRS state directly).
  This is one fewer dependency to ship to flight than ground mode
  needs.
- **Refresh policy choice matters more here than ground.** With a
  10 Hz solve loop and slow-slew operations, `OnPointingDelta` with
  ~degree-scale threshold can skip 90% of refreshes for free.

## Open issues

- [#43](https://github.com/OrbitalCommons/zodiacal/issues/43) —
  Solar/planetary gravitational deflection in refinement (currently
  skipped; matters for fields close to the Sun).
- [#46](https://github.com/OrbitalCommons/zodiacal/issues/46) —
  `solve_and_refine` convenience wiring.
- [#47](https://github.com/OrbitalCommons/zodiacal/issues/47) —
  Weighted least squares in the refit (uses Gaia covariance).
- Background prefetch of predicted-future cells (no issue yet; see
  ["Pre-fetch is the next optimization"](#pre-fetch-is-the-next-optimization)).
