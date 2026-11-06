# Checked solving

Use `SolverConfig { ... }.validated()?` and
`VerifyConfig { ... }.validated()?` to finish checked construction. Existing
public fields and default configs remain supported. `validate()` checks an
existing config, and every solver entry point validates again to catch later
mutation. Use `SkyRegion::try_from_radians` or `try_from_degrees` for checked
region construction; legacy constructors remain available.

The checks enforce algorithmic constraints rather than mission policy: finite
positive ordered scale bounds, at least four quad-building sources, finite
nonnegative code tolerance, finite epoch, a valid celestial region, a positive
finite match radius, distractor fraction strictly between zero and one, ordered
non-NaN log-odds thresholds, and positive minimum matches. Infinite ordered
log-odds thresholds remain supported for disabling early stopping. Direct
low-level verification callers should validate their VerifyConfig before use.

`solver::solve_checked` returns `Result<SolveReport, input::SolveInputError>`.
Invalid sources and invalid configuration have separate typed error variants.
For valid attempts, match `report.outcome` on `SolveOutcome::Solved(solution)`,
`TimedOut`, or `NoMatch`. `Solved` owns a boxed `Solution`; diagnostics are in
`report.stats`. `solve_checked_with_callback` also exposes verification progress.

Version 1.0 removes `solve`, `solve_with_callback`, and the tuple-returning
`solve_image`. Use the checked APIs above and `solve_image_checked` (with the
`image-processing` feature). `SolveStats` contains diagnostics only: errors
come through `Result`, and timeout/no-match/success through `SolveOutcome`.
No search or callbacks occur on rejected input.

Timeout is a cooperative search budget, not a hard deadline. Input validation
and index loading are outside the budget; source sorting and search setup are
inside it. Checks occur at outer search phases, so a phase may process many
candidates before checking again and can exceed the requested budget. Empty indexes or fewer than four sources produce `NoMatch`
without searching, even with zero timeout. Comparing elapsed time avoids
arithmetic overflow even for `Duration::MAX`.

Aperture1 can remove config validation formulas and Option/timed_out inference.
Keep mission limits, hint freshness and camera checks, and FSM transitions in
the adapter. Source filtering policy remains as described in source-validation.md.
