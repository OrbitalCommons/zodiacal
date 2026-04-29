# zodiacal

A blind astrometry plate solver written in Rust, inspired by [astrometry.net](https://astrometry.net).

Given an astronomical image (or a list of detected source positions), zodiacal determines the sky coordinates — pointing, rotation, and plate scale — using only the pattern of stars in the image. No prior knowledge of the field is required.

## How It Works

Zodiacal implements the geometric hashing approach described in [Lang et al. (2010)](https://arxiv.org/abs/0910.2233):

1. **Source extraction** — detect bright point sources in the image
2. **Quad formation** — form 4-star asterisms from the brightest sources and compute scale-invariant hash codes
3. **Code matching** — search prebuilt index files for matching asterisms using a kd-tree
4. **Hypothesis fitting** — compute a TAN-WCS (tangent-plane) projection from matched star correspondences
5. **Bayesian verification** — evaluate the alignment against known catalog positions using a log-odds decision framework

The solver uses multi-scale indexes with per-index scale filtering to efficiently search across different angular scales.

## Results

Tested on 1,000 simulated fields (9568x6380 px, plate scale 0.1296"/px, FOV ~20'x13'):

| Configuration | Solved | Rate | Median Time |
|---------------|--------|------|-------------|
| Mag 16 catalog, 4 bands (3x factor) | 962/1000 | 96.2% | 0.62s |
| Mag 16 + verify catalog (82M stars) | 967/1000 | 96.7% | — |
| Mag 19 catalog, 4 bands (3x factor) | 972/1000 | 97.2% | — |
| Mag 19 catalog, 12 bands (sqrt(2) factor) | **985/1000** | **98.5%** | 1.07s |

Best configuration solve time distribution (985 solved):
- 48% solved in under 1 second
- 83% solved in under 5 seconds
- 93% solved in under 10 seconds
- 99% solved in under 30 seconds
- 60-second timeout for all cases

The 15 remaining failures are quad-matching timeouts where the solver exhausts the search budget without finding the correct asterism match. Analysis shows these are a mix of sparse fields (few detected sources) and fields where the correct quad is buried among too many candidates.

For comparison, the reference astrometry.net system reports 99.9% on SDSS fields (2048x2048 px, 0.396"/px, wider FOV ~13.5').

## Building Indexes

Zodiacal requires prebuilt index files (`.zdcl`) containing star positions and quad hash codes. These are built from a [starfield](https://github.com/OrbitalCommons/starfield) binary catalog.

### Catalog Generation

Star catalogs are built using the `gaia_filter` tool from the starfield crate, which filters cached Gaia DR3 source files by magnitude:

```bash
# Build a magnitude-19 catalog (~565M stars, ~17 GB)
cd /path/to/starfield
cargo build --release --example gaia_filter
./target/release/examples/gaia_filter --magnitude 19.0 --output gaia_mag19.bin --threads 8 all
```

The `all` argument processes all cached Gaia source files. Brighter magnitude limits produce smaller, faster-to-load catalogs:

| Magnitude Limit | Stars | File Size |
|----------------|-------|-----------|
| 16.0 | ~83M | ~2.5 GB |
| 19.0 | ~565M | ~17 GB |

### Building a Multi-Scale Index Series

The recommended approach builds a series of narrow-band indexes, each covering a different angular scale range:

```bash
zodiacal build-index-series \
  -c gaia_mag19.bin \
  -o my_index \
  --scale-lower 10.0 \
  --scale-upper 600.0 \
  --max-stars-per-cell 30 \
  --passes 16 \
  --max-reuse 8 \
  --max-depth 8
```

This produces index files `my_index_00.zdcl` through `my_index_11.zdcl`, with 12 bands at a sqrt(2) scale factor. Each band covers a narrow range of quad angular sizes (e.g., 80-113", 113-160"), keeping the code trees small to reduce false-positive matches.

Key parameters:
- **`--scale-lower`/`--scale-upper`**: angular scale range in arcseconds
- **`--scale-factor`**: ratio between adjacent bands (default sqrt(2) ~1.414; use 3.0 for fewer, wider bands)
- **`--max-stars-per-cell`**: brightest stars per HEALPix cell for uniformization (default 30)
- **`--max-depth`**: maximum HEALPix depth (default 8 = 786,432 cells; limits memory usage)
- **`--passes`**: number of quad-building passes per cell
- **`--max-reuse`**: maximum times a star can be reused across quads

### Building a Single Index

For quick testing or narrow-scale applications:

```bash
zodiacal build-index \
  -c gaia_mag16.bin \
  -o test_index.zdcl \
  --scale-lower 30.0 \
  --scale-upper 600.0
```

## Solving Images

### Single Image

```bash
zodiacal solve image.png \
  -i my_index_00.zdcl \
  -i my_index_01.zdcl \
  -i my_index_02.zdcl \
  --timeout 60
```

### Batch Solving

Process a directory of images or JSON source files:

```bash
zodiacal batch-solve images/ \
  -i my_index_00.zdcl -i my_index_01.zdcl -i my_index_02.zdcl \
  --pattern "*.json" \
  --timeout 60 \
  --failures-dir /tmp/failures
```

The `--failures-dir` option writes diagnostic JSON for each unsolved case, useful for analyzing failure modes.

### From Pre-Extracted Sources

The solver accepts JSON source lists directly (skips extraction):

```bash
zodiacal solve sources.json \
  -i my_index_00.zdcl -i my_index_01.zdcl
```

### Extracting Sources

Extract sources from an image and write them as JSON for later solving:

```bash
zodiacal extract image.png -o sources.json
```

## Sources JSON Format

Zodiacal uses a JSON format for exchanging detected source lists between tools. The `extract` command produces this format, and the solver can consume it.

### Schema

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `image_width` | number | yes | Image width in pixels |
| `image_height` | number | yes | Image height in pixels |
| `ra_deg` | number | no | Known RA of field center (degrees) |
| `dec_deg` | number | no | Known Dec of field center (degrees) |
| `plate_scale_arcsec` | number | no | Known plate scale (arcsec/pixel) |
| `sources` | array | yes | Detected sources |
| `sources[].x` | number | yes | Source x position (pixels) |
| `sources[].y` | number | yes | Source y position (pixels) |
| `sources[].flux` | number | yes | Source brightness (ADU) |

### Example

```json
{
  "image_width": 9568,
  "image_height": 6380,
  "ra_deg": 265.47,
  "dec_deg": 44.31,
  "plate_scale_arcsec": 0.1296,
  "sources": [
    { "x": 4821.3, "y": 3190.7, "flux": 54210.0 },
    { "x": 1023.5, "y": 892.1, "flux": 38450.0 }
  ]
}
```

The `ra_deg`, `dec_deg`, and `plate_scale_arcsec` fields are optional hints. When present they can speed up solving by constraining the search space. When absent they are omitted from the JSON entirely.

## Library Use

The CLI is a thin wrapper over the `zodiacal` crate. Embedding the solver into your own application gives you access to several features the CLI doesn't expose, plus three deployment-mode building blocks (see below).

### Core API

```rust
use zodiacal::index::Index;
use zodiacal::solver::{solve, SolverConfig};
use zodiacal::extraction::DetectedSource;

let index = Index::load("my_index_00.zdcl".as_ref())?;
let sources: Vec<DetectedSource> = /* from extract_sources() or your own pipeline */;
let (solution, stats) = solve(&sources, &[&index], (image_w, image_h), &SolverConfig::default());
```

### High-Precision Refinement (10 mas absolute astrometry)

After the blind solve produces a TAN+SIP solution, the `refinement` module re-fits the WCS using each matched catalog star's **apparent direction** at the observation time — applying proper motion, parallax, light-time, and stellar aberration via the [starfield](https://github.com/OrbitalCommons/starfield) apparent-place pipeline.

```rust
use zodiacal::refinement::{refine_solution, ObservationContext, ObserverState, RefinementConfig};

let obs = ObservationContext {
    time: starfield::time::Timescale::default().tt_jd(jd_tt, None),
    observer: ObserverState::Barycentric {
        position_au: spacecraft_pos,
        velocity_au_per_day: spacecraft_vel,
    },
};
let refined = refine_solution(&tweaked_wcs, &sources, &index, &gaia_catalog, &obs, &RefinementConfig::default())?;
println!("residual RMS: {:.2} mas", refined.residual_rms_mas);
```

The full Gaia astrometry needed for refinement (RA/Dec/PM/parallax/RV + per-component sigmas) lives in a `RefinementCatalog`. For real datasets you'd populate it from the **Gaia sidecar** (`my_index.zdcl.gaia`) — a flat sorted-by-source_id binary file colocated with the index, accessed via mmap + galloping search:

```rust
use zodiacal::refinement::RefinementCatalog;

let catalog_ids: Vec<u64> = matched_sources.iter().map(|s| s.catalog_id).collect();
let gaia = RefinementCatalog::load_sidecar_filtered(
    "my_index_00.zdcl.gaia".as_ref(),
    &catalog_ids,
)?;
```

## Deployment Modes

The library exposes three building blocks that compose into different operational profiles:

### Mode 1 — Server (full sky, batch)

Load every index once, share across many concurrent solves. The standard `Index::load` + `solve` API works as-is. Wrap in `Arc<Index>` to share across threads.

```rust
use std::sync::Arc;
let index = Arc::new(Index::load("my_index_00.zdcl".as_ref())?);
// dispatch solves across rayon / tokio / thread pool
```

### Mode 2 — Realtime telescope (slewing zenith)

A ground telescope's visible sky changes as Earth rotates. `LiveIndex` tracks which HEALPix cells of the index need to be resident, `GroundStation` provides the current zenith from `(lat, lon, time)`, and `RealtimeSolver` ticks them in sync:

```rust
use zodiacal::index::{LiveIndex, ZdclFile};
use zodiacal::pointing::GroundStation;
use zodiacal::realtime::{RealtimeSolver, RefreshPolicy};
use std::time::Duration;

let source = ZdclFile::open("my_index_v3.zdcl".as_ref())?;
let pointing = GroundStation::from_degrees(34.0, -118.0, /* min altitude */ 30.0);
let mut rt = RealtimeSolver::new(source, pointing)
    .with_refresh_policy(RefreshPolicy::OnPointingDelta {
        angular_threshold_rad: 0.1,
        max_age: Duration::from_secs(60),
    });
// later, per frame:
let out = rt.solve(&detected_sources, image_size, &now)?;
```

### Mode 3 — Realtime star tracker (spacecraft)

Same orchestrator, with a `SpacecraftBoresight` driving the loaded region from an attitude estimate + ephemeris. Bring your own `EphemerisSource` (`anise`, SPICE, TLE via `starfield::sgp4lib`, custom Kalman) and `AttitudeSource`:

```rust
use zodiacal::pointing::{SpacecraftBoresight, EphemerisSource, AttitudeSource};

let pointing = SpacecraftBoresight::new(my_ephemeris, my_attitude_estimator,
    /* detector half-angle */ 0.05_f64.to_radians(),
    /* fov padding */ 0.05_f64.to_radians());
let mut rt = RealtimeSolver::new(zdcl_file_v3, pointing);
```

Spacecraft mode also feeds `observer_state(t)` into refinement automatically — you get parallax + aberration corrections for free using the same ephemeris.

### Sparse loading (any mode)

For server-mode callers with prior knowledge of where they're pointing, `Index::load_in_region` reads the same v2 file format but keeps only stars inside a `SkyRegion`. Same disk I/O as a full load, dramatically lower resident memory.

```rust
use zodiacal::solver::SkyRegion;
use starfield::Equatorial;
let region = SkyRegion::from_degrees(Equatorial::new(0.5, 0.3), 5.0);
let small_index = Index::load_in_region_padded(path, &region, /* pad rad */ 0.01)?;
```

## File Format

`.zdcl` index files are versioned. Both the original streaming format and the new HEALPix-grouped layout are accepted by all current readers.

| Version | Layout | Sparse cell load | Notes |
|---|---|---|---|
| v1 | streaming, no metadata | no | legacy |
| v2 | streaming + length-prefixed JSON metadata | no | written by `Index::save` |
| v3 | HEALPix-cell-grouped + cell table in header | **yes** (via `ZdclFile`) | written by `Index::save_v3` |

`ZdclFile::open` accepts v1/v2/v3 transparently; older files appear as a single virtual cell so the `IndexSource` API works uniformly.

## Architecture

The blind solve pipeline:

```
Image → Source Extraction → Quad Formation → Code Matching → WCS Fitting → Verification
                                                  ↑
                                          Index Files (.zdcl)
                                                  ↑
                                     Star Catalog → HEALPix Uniformization → Quad Building
```

Optional post-solve refinement chain:

```
Tweaked WCS + Matched Sources + Gaia sidecar + Observation Context
                                  ↓
                         Apparent place per matched star
                         (PM + parallax + light-time + aberration)
                                  ↓
                         Weighted re-fit of TAN parameters
                                  ↓
                          Refined Solution (~10 mas RMS)
```

For realtime modes, `LiveIndex` sits between the file and the solver, holding only the HEALPix cells currently relevant. `RealtimeSolver` coordinates pointing-source updates with `LiveIndex` membership changes and the cached snapshot used for solving.

Key design decisions:
- **HEALPix uniformization** at index build, **HEALPix grouping** in v3 file format: stars are distributed across sky cells to ensure uniform coverage and to enable cell-targeted reads.
- **Multi-scale indexes**: narrow angular-scale bands keep kd-trees small, reducing false-positive matches.
- **Dual-parity matching**: field codes are tried in both orientations (original + x/y swapped) to handle image flips.
- **Pre-fit filters**: `SolverConfig::scale_range` and `SolverConfig::within` reject candidates before the LSQ fit, eliminating ~10× pessimization on wide hints.
- **`KdForest` for live indexes**: per-cell sub-trees so cell add/drop is O(1) — no full rebuild on every realtime membership change.
- **Generation-cached snapshots**: `RealtimeSolver` caches the flat `Index` view by `LiveIndex::build_generation`, so steady-state solve cost is a pointer-deref.

## Dependencies

- [starfield](https://github.com/OrbitalCommons/starfield) — star catalogs, coordinate systems, time scales, apparent-place pipeline
- [cdshealpix](https://crates.io/crates/cdshealpix) — HEALPix tessellation for index uniformization and v3 cell layout
- [memmap2](https://crates.io/crates/memmap2) — mmap-backed sidecar and v3 index readers
- [nalgebra](https://crates.io/crates/nalgebra) — quaternion + vector math (used by refinement and pointing-source)
- [ndarray](https://crates.io/crates/ndarray) (optional, `image-processing` feature) — N-dimensional arrays
- [clap](https://crates.io/crates/clap) (optional, `cli` feature) — CLI argument parsing
- [image](https://crates.io/crates/image) (optional, `cli` feature) — image loading
- [rayon](https://crates.io/crates/rayon) — parallelism
- [serde](https://crates.io/crates/serde) / [serde_json](https://crates.io/crates/serde_json) — serialization

## License

See [LICENSE](LICENSE) for details.
