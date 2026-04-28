# Changelog

## 0.3.0

### Added
- `refinement` module: high-precision astrometric refinement using starfield's
  apparent-place pipeline (proper motion, parallax, light-time, stellar
  aberration). Targets ~10 mas absolute astrometry on noise-free synthetic data.
- `RefinementCatalog` + `GaiaAstrometry` + `ObservationContext` types for
  feeding full Gaia astrometry through to the refinement step.
- Flat sorted-by-source_id binary catalog sidecar (`SidecarRecord`,
  `SidecarReader`, `write_sidecar`) with mmap + in-memory pivot table +
  galloping-search batch lookup. Designed for the `.zdcl.gaia` companion
  to existing index files (#45).
- `RefinementCatalog::load_sidecar_filtered()` for loading only the rows
  needed for a given FOV from a sidecar file.

### Changed
- Bumped `starfield` dependency from `0.9.1` to `0.11` to pick up the
  apparent-place pipeline (`Position::observe`, `Position::apparent`,
  `Star::observe_from`).
- `solver::try_quad`: `SolverConfig::scale_range` and `SolverConfig::within`
  are now applied **before** `fit_tan_wcs` rather than after. Eliminates a
  ~12× pessimization observed when a wide scale_range hint causes every
  index band to enter the inner fit loop (#48). The `within` filter pads
  by `index.scale_upper` to keep boundary candidates intact.

### Dependencies
- Added `nalgebra = "0.32"` (matches starfield's pin) for sidecar interop.
- Added `memmap2 = "0.9"` for the sidecar reader.

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
