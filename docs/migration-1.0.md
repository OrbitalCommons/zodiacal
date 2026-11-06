# Migrating to Zodiacal 1.0

The solver has one public result model: invalid input is an error; valid attempts
produce an explicit outcome and diagnostics. The tuple compatibility APIs are removed.

| Removed API | Replacement |
| --- | --- |
| `solver::solve` | `solver::solve_checked` |
| `solver::solve_with_callback` | `solver::solve_checked_with_callback` |
| `solver::solve_image` | `solver::solve_image_checked` (`image-processing`) |
| `SolveStats::timed_out` | `SolveOutcome::TimedOut` |
| `SolveStats::invalid_input` / `invalid_config` | `SolveInputError` returned through `Result` |
| `RealtimeOutput::solution` | `RealtimeOutput::solve.outcome` and `.solve.stats` |

```rust
use zodiacal::solver::{solve_checked, SolveOutcome, SolverConfig};

let report = solve_checked(&sources, &indexes, image_size, &config)?;
match report.outcome {
    SolveOutcome::Solved(solution) => use_wcs(&solution.wcs),
    SolveOutcome::TimedOut => report_timeout(),
    SolveOutcome::NoMatch => report_no_match(),
}
// report.stats contains verification diagnostics.
```

Construct sources with `DetectedSource::new(x, y, flux)?` and read them through
`x()`, `y()`, and `flux()`. JSON's `x/y/flux` shape is unchanged, but invalid
coordinates/flux no longer deserialize. `sources_from_points` also returns a
`Result`. Image bounds are inclusive zero-based pixel-center coordinates through
`width - 1` and `height - 1`. If your application drops invalid detections, retain
that policy: discard failed constructions, filter with `validate_in_image`, then
apply your source cap. The solver does not silently discard sources.

Use `.validated()?` on SolverConfig/VerifyConfig and
`SkyRegion::try_from_radians`/`try_from_degrees` for checked construction. Solver
entry points revalidate after mutation. Public config fields and existing region
constructors remain available. Ordered infinite log-odds thresholds still support
exhaustive verification.

`ZdclBundle::solve_region` replaces manual load/build/solve orchestration and
reports separate stage timings. Its loading region selects cells; `config.within`
separately constrains accepted centers. Realtime solving likewise returns a full
SolveReport; its `io::Result` retains invalid-input errors as the underlying cause.

For camera geometry, `geometric_center_axes()` returns the midpoint boresight and
measured column/row tangents. Existing `field_center()` and `field_radius()` keep
their legacy conventions. Quaternion serialization, row-parity encoding, hint
freshness, mission bounds, source ordering/caps and FSM policy stay in applications.

Index formats and the existing manual loading APIs are unchanged. This release
does not add new legacy-format readers. Timeout remains a cooperative budget with
checks at outer search phases, not a hard wall-clock deadline.
