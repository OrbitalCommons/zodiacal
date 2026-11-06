# Field-center residuals (#67)

The one-pixel declination bias is a comparison-point mismatch in the frozen
`set1-legacy` fixtures, not a reason to shift every solved WCS. Their recorded
truth maps to approximately `(W/2, H/2 - 1)`. The newer PNG fixtures map to
`(W/2, H/2)`. Keep `field_center()` at its documented legacy point and evaluate
`pixel_to_radec(x, y)` at the actual calibrated or renderer boresight when
comparing pointing truth.

## Measurements

Measured 2026-11-06 with the 1.1.0 production solver, the first 100 numbered
cases of **each** set at zodiacal-test-cases revision
`a89a00c3c1c32251a125de35992c1ad985f45258`, and the twelve historical
`mag19_fine_00..11.zdcl` indexes. The diagnostic uses a ±25% scale hint,
a 0.5-degree truth-region hint, 3-second cooperative timeout, 3-pixel verification
radius, log-odds threshold 100 and at least 10 matches. It performs the initial
TAN solve without subsequent refitting. This is a pointing-convention experiment,
not a new blind-solve-rate baseline or a sweep of all 12,000 fixtures.

All three sets solved 97/100 cases. No solved cases were clipped from the
statistics. Medians of the truth's pixel offset from `(W/2, H/2)`:

| Set | X offset (pixels) | Y offset (pixels) |
|---|---:|---:|
| set1-legacy | +0.00345 | -0.99793 |
| set2-dr3-mag19 | +0.00417 | +0.03457 |
| set3-dr3-mag19-galaxies-10k | -0.03472 | +0.02957 |

Median pointing error / p95 in **milliarcseconds**, using
`hypot(ΔRA × cos(truth Dec), ΔDec)` and nearest-rank p95:

| Set | `(W/2, H/2)` | Geometric midpoint | `(W/2, H/2 - 1)` |
|---|---:|---:|---:|
| set1-legacy | 129.3 / 133.4 | 91.7 / 99.0 | **2.6 / 12.2** |
| set2-dr3-mag19 | **22.0 / 108.4** | 98.9 / 177.3 | 136.0 / 186.6 |
| set3-dr3-mag19-galaxies-10k | **22.0 / 143.0** | 93.6 / 162.1 | 134.2 / 194.3 |

For set1, the signed median Dec residual changes from +129.27 mas to -0.27 mas
when evaluating the flipped boresight. A global geometric-center substitution
would add a half-pixel RA offset and would make the newer fixtures worse.
The newer sets' remaining scatter is outside this investigation's scope.

[Raw measurements](data/field-center-residuals.csv) retain all 300 attempts.
[Index SHA-256 checksums](data/field-center-indexes.sha256) identify the local
historical inputs; those large binary files are not included in this repository.

## Why one pixel appears

Zero-based pixel-center rows flip as `y' = H - 1 - y`. Therefore a renderer
boresight at `H/2` moves to `H/2 - 1`, **not** `H/2`. In contrast, the geometric
midpoint `(H - 1)/2` is invariant under the flip. The regression test checks this
for odd/even dimensions and rotated WCSs.

The exact [set3 renderer revision's sensor mapping](https://github.com/CosmicFrontierLabs/focalplane/blob/baee07c6e9e2ac30e637d21ee7db104313b52fff/simulator/src/hardware/sensor_array.rs)
places the commanded boresight at `W/2, H/2`; its image deposition uses integer
pixel centers. The [legacy fixture provenance](https://github.com/OrbitalCommons/zodiacal-test-cases/blob/a89a00c3c1c32251a125de35992c1ad985f45258/set1-legacy/README.md)
records a FITS extraction pipeline with a vertical flip. The measured legacy
offset agrees with that transformation. The retired renderer's exact revision
and original FITS files are unavailable, so this does not establish which
historical component first introduced the convention mismatch.

Do not remove the FITS row flip or apply a constant declination subtraction.
The correct comparison follows the selected pixel through the WCS, including
rotation and parity. Frozen JSON centroids do not change when the FITS loader
changes.

## Reproduce and compare the intended boresight

```sh
cargo run --release --example center_residuals -- \
  ../zodiacal-test-cases 100 scratch/mag19_fine_*.zdcl > residuals.csv
```

`bench-indexes` and `bench-bundle` accept `--boresight-offset-x` and
`--boresight-offset-y`, in pixels relative to `(W/2, H/2)`. Both default to zero
and affect only reported coordinates/residuals, not solving, region loading or
verification. Accepted and rejected residuals use the same point; bundle trace
JSON records it as `solved.boresight_pixel`.

- For the frozen set1 fixtures, use `--boresight-offset-y -1`.
- For set2/set3, keep the defaults.
- To measure the geometric midpoint, use both offsets set to `-0.5`.
- For other images, use their calibrated boresight convention; do not infer it
  from the filename or apply the set1 offset automatically.
