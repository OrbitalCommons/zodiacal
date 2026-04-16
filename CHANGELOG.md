# Changelog

## 0.2.0

### Added
- `solve_with_callback()` for live solve progress visibility — calls a user-provided
  `FnMut(&SolveStats)` after each quad attempt with zero overhead when unused (#40)
- `SkyRegion` type and `SolverConfig::within` to constrain solves to a known sky area
- `SolverConfig::with_scale_range()` builder with ordering validation
- Feature-gated dependencies for lean library use:
  - `image-processing`: gates ndarray, extraction functions, `solve_image()`
  - `cli`: gates clap, glob, image, indicatif
  - `fits`: gates fitsio-pure (default)
- Replaced custom HEALPix implementation with `cdshealpix` crate (#36)
- Made `indicatif` optional behind `cli` feature with no-op progress shim (#34)
- CI feature matrix testing 5 feature combinations

### Changed
- Default features are now just `["fits"]` — library consumers no longer pull in
  CLI, image processing, or progress bar dependencies
- Binary requires `--features cli` to build
- HEALPix operations now delegate to `cdshealpix` crate (CDS Strasbourg)

## 0.0.2

Initial tagged release with blind plate solving, quad-based index building,
TAN WCS fitting, Bayesian verification, and SIP distortion refinement.
