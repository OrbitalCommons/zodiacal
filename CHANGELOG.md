# Changelog

## 0.4.1

### Changed
- Exclude `test_cases/` from the published crate (#58). The 0.4.0 package
  shipped 1036 files / 10.3 MiB uncompressed; ~1000 of those were
  `test_cases/*.json` solver fixtures with no value to downstream library
  users. 0.4.1 drops the same surface to ~36 files / a few hundred KB.
  No code or API change.

## 0.4.0

Five-step deployment-mode roadmap: the index now supports sparse loading,
HEALPix-grouped on-disk layout, runtime add/drop of cells, pluggable pointing
sources for ground stations and spacecraft, and a realtime orchestrator that
ties it all together. Server, ground-telescope, and star-tracker deployment
profiles can all be assembled from the new pieces.

### Added
- **Plan 1 — Sparse load (#52)**: `Index::load_in_region(path, &SkyRegion)` and
  `Index::load_in_region_padded(path, &SkyRegion, padding_rad)` read the existing
  v2 file format sequentially but only retain stars within the region (and quads
  referencing those stars). Same disk I/O, dramatically lower memory.
- **Plan 2 — HEALPix file format v3 (#53)**: stars are sorted by HEALPix cell
  and the file header carries a sorted `(cell_id, star_offset, star_count)`
  table. New `IndexSource` trait abstracts cell-based access; `ZdclFile`
  implements it via mmap. v1/v2 files are accepted transparently as a single
  virtual cell. `Index::save_v3(path, cell_depth)` emits the new format;
  `From<IndexFragment> for Index` bridges sparse loads back into the existing
  solver path.
- **Plan 3 — `LiveIndex` with `KdForest` (#54)**: stateful in-memory index
  whose loaded cell set can be grown via `ensure_region(&SkyRegion)` and
  shrunk via `drop_outside(&SkyRegion)` / `set_region(&SkyRegion)`. New
  `KdQueryable` trait + `KdForest<DIM>` container hold per-cell sub-trees so
  cell add/drop is O(1) for tree maintenance — no full rebuild on every
  membership change. `as_index()` flattens to the flat-`Index` solver API on
  demand.
- **Plan 4 — `PointingSource` (#55)**: `PointingSource` trait with two
  concrete implementations:
  - `GroundStation { latitude, longitude, min_altitude }` — zenith RA from
    Local Sidereal Time, Dec from latitude, region radius =
    `90° - min_altitude`.
  - `SpacecraftBoresight<E: EphemerisSource, A: AttitudeSource>` — pure
    quaternion math; rotates a body-frame boresight (default `+Z`) into the
    inertial frame and returns the FOV cap. Caller-supplied `EphemerisSource`
    and `AttitudeSource` traits keep the implementation library-agnostic.
- **Plan 5 — `RealtimeSolver` orchestrator (#56)**: `RealtimeSolver<S, P>`
  ties `LiveIndex` + `PointingSource` + the existing solver into one
  realtime-friendly `tick()` / `solve()` interface. Three refresh policies
  (`EveryTick`, `OnPointingDelta`, `OnInterval`) control when the loaded set
  is re-synced. The internal flat-`Index` snapshot is cached by
  `LiveIndex::build_generation`, so the steady-state solve cost is a pointer
  dereference rather than an O(N log N) tree rebuild.

### Internal
- `KdQueryable` trait + `KdForest::range_search_tagged` /
  `TaggedSearchResult` for callers that need to know which sub-tree a hit
  came from.
- `IndexSource::scale_range()` accessor so consumers can pre-filter by quad
  scale band without first loading a fragment.
- v3 reader hardening: validates `cell_depth <= 29`, bounds `n_cells` against
  both `12 × 4^cell_depth` and remaining file size before allocating, and
  validates every `(star_offset, star_count)` range against `n_stars` at
  open time. Uses checked arithmetic on file offsets to prevent panics on
  adversarial input.

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
