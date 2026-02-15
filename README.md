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

## Architecture

The solver pipeline:

```
Image → Source Extraction → Quad Formation → Code Matching → WCS Fitting → Verification
                                                  ↑
                                          Index Files (.zdcl)
                                                  ↑
                                     Star Catalog → HEALPix Uniformization → Quad Building
```

Key design decisions:
- **HEALPix uniformization**: stars are distributed across sky cells to ensure uniform index coverage regardless of stellar density
- **Multi-scale indexes**: narrow angular-scale bands keep kd-trees small, reducing false-positive matches
- **Dual-parity matching**: field codes are tried in both orientations (original + x/y swapped) to handle image flips
- **Incremental quad loop**: quads are generated in two phases (astrometry.net style) to avoid redundancy
- **Per-index scale filtering**: indexes whose band can't overlap with the field's backbone angular scale are skipped entirely

## Dependencies

- [starfield](https://github.com/OrbitalCommons/starfield) — star catalogs, coordinate systems, star finding
- [ndarray](https://crates.io/crates/ndarray) — N-dimensional arrays
- [clap](https://crates.io/crates/clap) — CLI argument parsing
- [image](https://crates.io/crates/image) — image loading
- [rayon](https://crates.io/crates/rayon) — parallelism
- [serde](https://crates.io/crates/serde) / [serde_json](https://crates.io/crates/serde_json) — serialization

## License

See [LICENSE](LICENSE) for details.
