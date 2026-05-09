# Building the paper's figures

`paper/make_figures.py` regenerates every figure from data already checked in. A fresh clone of this branch has everything you need:

```bash
python3 paper/make_figures.py
```

This writes `<name>.pdf` and `<name>.png` for each figure into `paper/figures/` (which is gitignored — figures are derived, not source). The `.tex` files reference `figures/<name>.pdf`.

## Subset / list

```bash
python3 paper/make_figures.py --list                    # print figure names
python3 paper/make_figures.py --only solve_cdf          # one figure
python3 paper/make_figures.py --only hist_4band,solve_cdf
```

## Inputs

| Input | Default | Override | Source |
|---|---|---|---|
| Batch-solve logs | `paper/data/*.log` | `--logs-dir` | Tracked alongside this script. |
| Test-case JSONs | `<repo>/test_cases/` | `--cases-dir` | 1000 synthetic field truths + source lists. After #102 these will live in [`OrbitalCommons/zodiacal-test-cases`](https://github.com/OrbitalCommons/zodiacal-test-cases) under `set1-legacy/`. |

The three logs are:

- `4band_results.log` — 4-band ($\alpha = 3$) batch-solve, RA/Dec at 4 decimals.
- `12band_results.log` — 12-band ($\alpha = \sqrt{2}$) batch-solve.
- `12band_hires.log` — 12-band rerun with RA/Dec logged at 8 decimals (sub-mas precision); used by every residual figure.

## Figures

| Name | Inputs | Used in |
|---|---|---|
| `hist_4band` | `4band_results.log` | §results — 4-band time histogram |
| `hist_12band` | `12band_results.log` | §results — 12-band time histogram |
| `hist_12band_hires` | `12band_hires.log` | (staged) |
| `solve_cdf` | `4band_results.log`, `12band_results.log` | §results — CDF comparing both configs |
| `hist_residuals` | `12band_hires.log` + `<case>.json` | (staged) — raw position-residual histogram |
| `hist_residuals_corrected` | same | (staged) — residual after subtracting the +0.131″ Dec bias (see [zodiacal#67](https://github.com/OrbitalCommons/zodiacal/issues/67)) |
| `residual_scatter` | same | (staged) — (ΔRA·cos δ, Δδ) scatter |

"Staged" = the figure is generated but not yet referenced from `zodiacal.tex`.

## Regenerating the input logs (rare)

The committed logs are the canonical figure inputs; you should not need to rerun. If you do (e.g., after a solver change that you want to land in the paper), regenerate via `zodiacal batch-solve` against the test-case directory and overwrite the file under `paper/data/`. Approximate runtime per log on an 8-core x86 box: 10–30 minutes for 1000 cases.

## Adding a new figure

1. Add a `def fig_<name>(...)` next to the existing ones in `make_figures.py`.
2. Register it in the `figures()` dict.
3. Reference it from `zodiacal.tex` as `figures/<name>.pdf`.
4. If it needs a new input file, drop it under `paper/data/` and update this table.

## Outputs are not committed

`paper/figures/` is in `.gitignore`. Always regenerate from sources; never edit the produced PDFs/PNGs.
