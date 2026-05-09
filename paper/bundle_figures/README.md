# Bundle-format figures (§ "Spatial Sharding and Region-Restricted Solving")

These scripts regenerate the depth-9, $G \le 20$ bundle figures used in
the paper. They consume artifacts produced by:

- `zodiacal-tools build-from-excerpt-series` (the bundle build itself)
- `zodiacal-tools bench-bundle` against `set2-dr3-mag19` at four
  pointing-hint radii (0.5°, 1.0°, 2.0°, 4.0°)

By default they read from `/home/meawoppl/scratch/bundle-build/bench/`;
edit the `OUT` paths at the top of each script if your scratch lives
elsewhere.

| script | output PNG (in `paper/figures/`) |
|---|---|
| `plot_radii_v2.py`         | `d9_solve_time_by_radius_cdf.png`, `d9_load_time_by_radius_cdf.png`, `d9_accuracy_by_radius_cdf.png` |
| `plot_radec_residuals.py`  | `d9_radec_residual_hists_r1.png`                                |
| `plot_failures_mollweide.py` | `d9_failures_mollweide.png`                                   |
| `plot_quads_mollweide.py`  | `d9_quads_per_cell_mollweide.png` (caches scan to `quads_per_cell_scan.npz`) |

The PNGs themselves are checked in alongside the paper as a frozen
snapshot of the run that produced the figures cited in the text.
