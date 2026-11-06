# HST index-strategy loop (2026-10)

Notes on a 7-batch, 30-run sweep of bundle-build and solver settings against
real Hubble data, made to find an index and solver configuration for HST
F606W fields with a 1° pointing hint. This file records the exact
configuration so the runs can be reproduced or extended, the full results,
and what they imply.

**Bottom line:** going from the production IMX455 bundle and solver defaults
to HST-tuned settings raised correct solves from **19.1% to 35.8%**, and cut
wrong solves from **72 to 12**, on 800 solvable HST fields.

## 1. Test data

- **Set:** `hubble-f606w` from
  [zodiacal-test-cases](https://github.com/OrbitalCommons/zodiacal-test-cases),
  the 2026-10 re-extraction (zodiacal-test-cases#5). All 75,673 cases were
  re-extracted with the NaN-safe extractor (#144): 5σ, at most 200 sources.
  Pointing and plate scale come from the current MAST FITS headers, which are
  aligned to Gaia, so the header WCS is good ground truth.
- **Gaia census:** `hubble-f606w/_gaia_census.csv` counts Gaia DR3 G ≤ 20
  stars inside each field's inscribed and circumscribed circles.
- **Dev sample:** cases with `n_sources ≥ 10` and `in_g20 ≥ 5` (at least 5
  Gaia stars inside the inscribed circle), which leaves 42,470 solvable
  cases. Sampled **200 per instrument**, seed 0 → **800 cases**: ACS/WFC,
  WFC3/UVIS, WFPC2/PC, ACS/HRC.
  - Observations span 1994–2026. Fields span 29″ (HRC) to 215″
    (ACS/WFC), at 0.025–0.10″/px.
  - Because the sample is limited to fields with Gaia stars, **solve rates
    here are against fields that are solvable in principle, not the raw
    75k**.
- **Catalog:** the starfield Gaia DR3 G ≤ 20 HEALPix-5 bycell excerpt
  (`dr3-mag20-bycell`, rebuilt 2026-10 via the starfield-datastore
  pull-through cache).
- **Reduced excerpt:** to make each build take minutes to a couple of hours
  instead of most of a day, bundles were built only from the **1,751
  level-5 cells** that overlap any dev case's bench region (1° radius plus
  the field radius). That's 82 GB of excerpt, or 448,256 depth-9 cells.
  Inside those cells every bundle is identical to a full-sky build, so the
  benchmark sees the same distractors it would with a full-sky bundle.

## 2. Harness

`scripts/index_strategy_loop.py`:

```bash
# once: dev sample + reduced excerpt (symlinks)
scripts/index_strategy_loop.py prepare --loop-dir ~/scratch/index-loop \
    --cases ~/scratch/hst-reextract/hubble-f606w \
    --census ~/scratch/hst-reextract/_gaia_census.csv \
    --excerpt ~/.cache/starfield/gaia-excerpts/dr3-mag20-bycell \
    --radius-deg 1.0 --per-inst 200

# per strategy: build -> bench -> score -> append to results.md
scripts/index_strategy_loop.py run --loop-dir ~/scratch/index-loop \
    --cases ~/scratch/hst-reextract/hubble-f606w --tools <zodiacal-tools> \
    --name <run> --timeout-secs 10 \
    --bench-extra="--log-odds-accept 40 --min-matches 8" \
    [--keep-bundle] -- <build-from-excerpt-series args>
```

Benchmark-only runs reuse a kept bundle by symlinking it into
`runs/<name>/bundle.zdcl.bundle` before `run`.

Every `bench-bundle` call used `--radius-deg 1.0 --scale-hint`:
- a 1° hint around the header pointing;
- plate scale ±25%;
- `obs_epoch` auto-filled from `hst.t_min_mjd`/`t_max_mjd`, so catalog
  proper motion is propagated to each observation's date.

**Scoring:** a case is **correct** if it solved and the solved centre is
within **2″** of the header pointing. It's **wrong** if it solved farther
away. Wrong solves were overwhelmingly false positives, median about 0.76°
away.

**Tools:**
- `zodiacal-tools` at `76081ac` for batch 1.
- From batch 2 on, the `bench-bundle-solver-knobs` branch
  (#152, head `f94cd36`), which adds `--match-radius-pix`,
  `--max-field-stars`, `--code-tolerance`, `--log-odds-accept` and
  `--min-matches`.

**Machine:** cfl-test-bench (80-core Ampere Altra, 62 GB RAM, no swap).
Builds ran under `systemd-run --user` with `MemoryMax=28G`,
`OOMPolicy=continue` and per-strategy retries. Builds resume from the work
directory after an out-of-memory kill. Three builder out-of-memory kills
happened over the sweep: the control, the depth-10 build and the combined
build. Two early batch-2 retries were a harness argument bug, not memory.

## 3. Results

Columns:
- **Correct:** share of the 800 cases solved within 2″ of the header.
- **Wrong:** number of solves more than 2″ away.
- **Instrument columns:** correct rate per instrument.
- **Δt:** correct rate by time between the observation and Gaia's 2016
  epoch.
- **Build min:** wall-clock build time. It's 0 for benchmark-only runs, and
  covers only the last attempt when a build was resumed after an
  out-of-memory kill.

Unless the args say otherwise, the defaults were:
- **Build:** depth 9, 8 bands, `--max-reuse 8`, G ≤ 20.
- **Solver:** log-odds 20, at least 5 matches, 3 px match radius, 50 field
  stars, code tolerance 0.002, 10 s timeout.

| run | correct | wrong | ACS/WFC | WFC3/UVIS | WFPC2/PC | ACS/HRC | Δt<5y | 5–15y | >15y | median ms | bundle GB | build min | args (build / bench) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| b1-control-d9-60-480-q80 | 19.1% (153) | 72 | 46.5% | 24.5% | 5.5% | 0.0% | 39.7% | 12.0% | 6.9% | 30.8 | 31.55 | 23.6 | `--cell-depth 9 --bands 7 --scale-lower 60 --scale-upper 480 --quads-per-cell 80 --max-stars-per-cell 2500` (production IMX455 settings) |
| b1-d9-8-150-q200 | 33.5% (268) | 291 | 56.0% | 38.0% | 13.5% | 26.5% | 54.3% | 27.1% | 15.3% | 36.2 | 40.16 | 139.3 | `--scale-lower 8 --scale-upper 150 --quads-per-cell 200 --max-stars-per-cell 5000` |
| b1-d10-8-150-q100 | 33.8% (270) | 300 | 55.5% | 39.5% | 13.5% | 26.5% | 54.3% | 27.1% | 18.1% | 38.1 | 51.22 | 154.9 | `--cell-depth 10 --scale-lower 8 --scale-upper 150 --quads-per-cell 100 --max-stars-per-cell 1500` |
| b1-d9-15-300-q200 | 28.5% (228) | 344 | 58.0% | 37.5% | 14.0% | 4.5% | 55.7% | 18.3% | 18.1% | 50.2 | 44.53 | 124.8 | `--scale-lower 15 --scale-upper 300 --quads-per-cell 200 --max-stars-per-cell 5000` |
| b2-lo40 | 34.4% (275) | 56 | 57.0% | 39.0% | 14.0% | 27.5% | 55.2% | 27.5% | 19.4% | 665.7 | 40.16 | 0 | b1-d9-8-150-q200 bundle · `--log-odds-accept 40` |
| b2-lo80 | 25.6% (205) | 9 | 47.0% | 27.0% | 7.5% | 21.0% | 44.3% | 19.7% | 11.1% | 1159.5 | 40.16 | 0 | same bundle · `--log-odds-accept 80` |
| b2-lo40-mm8 | 28.2% (226) | 13 | 49.0% | 29.0% | 12.5% | 22.5% | 46.1% | 22.0% | 18.1% | 941.1 | 40.16 | 0 | same bundle · `--log-odds-accept 40 --min-matches 8` |
| b2-lo80-mm8-r6 | 26.2% (210) | 44 | 46.0% | 27.5% | 9.5% | 22.0% | 43.4% | 20.6% | 13.9% | 879.5 | 40.16 | 0 | same bundle · `--log-odds-accept 80 --min-matches 8 --match-radius-pix 6` |
| b2-lo80-mm8-fs100 | 27.0% (216) | 8 | 50.0% | 27.5% | 9.5% | 21.0% | 46.1% | 20.6% | 13.9% | 10001.2 | 40.16 | 0 | same bundle · `--log-odds-accept 80 --min-matches 8 --max-field-stars 100` |
| b2-lo120-mm10 | 20.8% (166) | 7 | 33.5% | 24.0% | 6.0% | 19.5% | 34.2% | 16.7% | 8.3% | 1302.2 | 40.16 | 0 | same bundle · `--log-odds-accept 120 --min-matches 10` |
| b3-d9-15-300-q200-v | 23.2% (186) | 12 | 50.0% | 28.5% | 13.5% | 1.0% | 46.1% | 14.0% | 19.4% | 1843.9 | 44.53 | 0 | b1-d9-15-300 bundle · `--log-odds-accept 40 --min-matches 8` |
| b3-d9-8-150-q100-v | 25.9% (207) | 11 | 45.5% | 26.5% | 10.5% | 21.0% | 43.4% | 19.8% | 15.3% | 991.4 | 34.03 | 121.1 | `8–150″ q100` · lo40 mm8 |
| b3-d9-8-150-q400-v | 31.4% (251) | 12 | 50.0% | 32.0% | 17.5% | 26.0% | 47.9% | 25.1% | 25.0% | 796.4 | 50.63 | 144.2 | `8–150″ q400` · lo40 mm8 |
| b3-d9-8-150-q200-m19-v | 24.5% (196) | 14 | 34.0% | 25.0% | 15.0% | 24.0% | 32.9% | 21.8% | 18.1% | 566.4 | 26.28 | 116.4 | `8–150″ q200 --mag-limit 19` · lo40 mm8 |
| b3-d9-8-150-q200-m18-v | 17.4% (139) | 9 | 23.5% | 20.5% | 7.5% | 18.0% | 22.4% | 16.1% | 11.1% | 511.3 | 15.83 | 106.9 | `8–150″ q200 --mag-limit 18` · lo40 mm8 |
| b4-d9-4-150-q400-v | 32.5% (260) | 16 | 50.5% | 32.5% | 16.0% | 31.0% | 47.5% | 27.3% | 23.6% | 663.5 | 47.33 | 178.5 | `4–150″ q400` · lo40 mm8 |
| b4-d9-8-100-q400-v | 30.8% (246) | 10 | 47.5% | 32.0% | 16.5% | 27.0% | 45.7% | 25.5% | 22.2% | 291.9 | 47.97 | 128.7 | `8–100″ q400` · lo40 mm8 |
| b4-d9-8-150-q400-mps10k-v | 31.1% (249) | 12 | 50.0% | 31.0% | 17.5% | 26.0% | 47.5% | 24.9% | 25.0% | 807.3 | 53.99 | 140.8 | `8–150″ q400 --max-stars-per-cell 10000` · lo40 mm8 |
| b4-d9-8-150-q600-v | 32.6% (261) | 12 | 51.0% | 34.0% | 18.0% | 27.5% | 49.3% | 26.5% | 25.0% | 769.8 | 59.89 | 144.5 | `8–150″ q600` · lo40 mm8 |
| b5-d9-4-150-q600-v | 33.4% (267) | 16 | 51.5% | 34.5% | 15.5% | 32.0% | 48.9% | 28.1% | 23.6% | 656.4 | 55.25 | ~180* | `4–150″ q600` · lo40 mm8 (**kept bundle** for b5–b7) |
| b5-t30 | 33.9% (271) | 17 | 51.5% | 34.5% | 16.5% | 33.0% | 48.9% | 28.7% | 25.0% | 657.1 | 55.25 | 0 | kept · lo40 mm8, `--timeout-secs 30` |
| b5-ct001 | 33.8% (270) | 14 | 51.5% | 34.5% | 16.5% | 32.5% | 48.9% | 28.5% | 25.0% | 465.4 | 55.25 | 0 | kept · lo40 mm8 `--code-tolerance 0.001` |
| b5-ct004 | 32.9% (263) | 15 | 51.5% | 33.5% | 14.5% | 32.0% | 48.9% | 27.3% | 23.6% | 1169.5 | 55.25 | 0 | kept · lo40 mm8 `--code-tolerance 0.004` |
| b5-fs30 | 28.4% (227) | 10 | 46.0% | 28.0% | 9.0% | 30.5% | 42.9% | 24.2% | 13.9% | 126.5 | 55.25 | 0 | kept · lo40 mm8 `--max-field-stars 30` |
| b5-fs75 | 34.4% (275) | 16 | 53.5% | 35.0% | 17.0% | 32.0% | 50.2% | 28.7% | 26.4% | 2838.7 | 55.25 | 0 | kept · lo40 mm8 `--max-field-stars 75` |
| b6-fs75-ct001 | 34.9% (279) | 14 | 53.5% | 35.0% | 18.5% | 32.5% | 50.2% | 29.5% | 26.4% | 2304.1 | 55.25 | 0 | kept · lo40 mm8 fs75 ct0.001 |
| b6-fs100-ct001 | 34.9% (279) | 14 | 53.5% | 35.0% | 18.5% | 32.5% | 50.2% | 29.3% | 27.8% | 6834.0 | 55.25 | 0 | kept · lo40 mm8 fs100 ct0.001 |
| b6-fs75-ct0005 | 35.1% (281) | 13 | 53.5% | 35.5% | 19.0% | 32.5% | 51.1% | 29.3% | 27.8% | 1913.8 | 55.25 | 0 | kept · lo40 mm8 fs75 ct0.0005 |
| **b7-fs75-ct00025** | **35.8% (286)** | 12 | 54.0% | 36.0% | 20.5% | 32.5% | 51.6% | 30.1% | 27.8% | 1695.2 | 55.25 | 0 | kept · lo40 mm8 **fs75 ct0.00025** |
| b7-fs75-ct0001 | 34.9% (279) | 9 | 54.0% | 35.5% | 19.5% | 30.5% | 51.6% | 28.9% | 26.4% | 1611.6 | 55.25 | 0 | kept · lo40 mm8 fs75 ct0.0001 |
| b7-fs50-ct0005 | 33.6% (269) | 11 | 51.0% | 34.5% | 16.5% | 32.5% | 48.9% | 28.3% | 25.0% | 402.0 | 55.25 | 0 | kept · lo40 mm8 fs50 ct0.0005 |

\* The build was out-of-memory killed once at about 14 GB; the recorded 58
minutes covers only the resumed attempt.

## 4. Findings

1. **Quad scales must fit HST fields.** 60–480″ (production) gives HRC 0% and
   PC 5.5%. 8–150″ gives 26.5% and 13.5%. 4–150″ lifts HRC to 31–33%.
   15–300″ is best for ACS/WFC (58%) but drops HRC to 4.5%.
2. **Default verification is unsafe on real data.** At log-odds 20 and 5
   matches, more than half the solves on the 8–150″ bundle were wrong, all
   false positives within the 1° hint. Log-odds 40 with at least 8 matches
   cuts wrong solves to about 1.5% of cases. Log-odds 80 or 120 removes the
   last few but loses many correct solves.
3. **Code tolerance should be about 8× tighter than the default for HST.**
   Accuracy and speed both improve from 0.002 down to **0.00025**; at 0.0001
   real matches start being missed. Sub-pixel HST centroids give near-exact
   codes, so a loose tolerance mostly adds false candidates. It also suggests
   proper motion isn't significantly perturbing quad codes.
4. **Gaia depth matters:** G ≤ 19 and G ≤ 18 cost 13% and 38% of correct
   solves, almost all of it on ACS/WFC (49% → 34% → 23.5%).
5. **More quads per cell helps, with diminishing returns:** 100 → 200 → 400
   → 600 gives 25.9 → 28.2 → 31.4 → 32.6%, and the bundle grows about 25%
   per doubling.
6. **Field stars: 75 is the sweet spot.** 30 loses 5 points (mostly PC), 75
   adds about 1 point over 50, and 100 adds nothing over 75 while taking 3×
   as long.
7. **No effect:** cell depth 10 versus 9 (only a bigger and slower build),
   10,000 versus 5,000 stars per cell, and a 30 s versus 10 s timeout (+4
   solves).
8. **Worse:** a 6 px match radius adds false positives without recovering
   correct solves.
9. **Time from 2016:** correct rate falls with |Δt| (about 52% → 30% → 28%),
   but most of that is the instrument mix (WFPC2 and HRC are old). Within
   ACS/WFC the effect is modest.

## 5. Recommended HST configuration

**Bundle build:**

```
--cell-depth 9 --bands 8 --scale-lower 4 --scale-upper 150
--quads-per-cell 600 --max-stars-per-cell 5000 --max-reuse 8 --mag-limit 20
```

This is about 55 GB for the 22% of the sky covered here, so roughly 250 GB
for the full sky.

**Solver:**

| | Accurate | Fast |
|---|---|---|
| `--max-field-stars` | 75 | 50 |
| `--code-tolerance` | 0.00025 | 0.0005 |
| `--log-odds-accept` / `--min-matches` | 40 / 8 | 40 / 8 |
| `--match-radius-pix` / timeout | 3 / 10 s | 3 / 10 s |
| **Correct / wrong / median** | **35.8% / 12 / 1.70 s** | 33.6% / 11 / 0.40 s |

**Per instrument (accurate):** ACS/WFC 54.0%, WFC3/UVIS 36.0%, ACS/HRC
32.5%, WFPC2/PC 20.5%.

## 6. Not yet explored / next steps

- **`bench-triage` on the ~515 still-failing dev cases,** to separate fields
  with no usable quad from extraction problems (galaxies and saturated
  stars dominating the brightest 200 sources).
- **Leaving no-proper-motion (2-parameter) Gaia stars out of quads.** This
  needs a builder option.
- **A smarter quad search order,** so that 75–100 field stars don't cost
  seconds.
- **Revisiting `bench-bundle` defaults for real data:** log-odds 20 → 40,
  code tolerance 0.002 → 0.00025.
- **Confirming on a larger, independent sample** (a different seed, or the
  full 42k solvable subset) before treating the absolute numbers as final.

## 7. Reproducing

- **Batch scripts and per-run commands:** each `runs/<name>/` has
  `build.cmd`, `bench.cmd`, `bench.csv`, `run.json` and `score.json`.
  These live on cfl-test-bench under `~/scratch/index-loop/` (not committed).
- **To regenerate:**
  1. Run `prepare` with the arguments in §2. It's deterministic given the
     census CSV and seed 0.
  2. Re-run any row's build and bench arguments with `run`.
