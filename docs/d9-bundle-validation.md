# Depth-9 G≤20 bundle validation

Empirical results from validating the depth-9 G≤20 bundle (built per
the recommendations in [`bundle-build-lessons.md`]) against the
1000-case `set2-dr3-mag19` test corpus, plus a position-hint radius
sweep to characterize how solve time scales with hint quality.

## Bundle vs the depth-8 baseline

| metric | depth-8 (prior) | depth-9 (this build) |
|---|---|---|
| Cells | 786,432 | 3,145,728 (4×) |
| Cell size | 13.7' | 6.87' |
| Bands | 7 × `[60", 600"]` | 7 × `[60", 480"]` |
| `quads_per_cell` | 100 | 80 |
| `max_stars_per_cell` | 10000 | 2500 |
| Total stars | 900,101,054 | 898,790,763 |
| Total quads | ~543M | ~1.6B (~3×) |
| Bundle on disk | 116 GB (27 q + 89 g) | 171 GB (78 q + 93 g) |
| Build wall-clock | ~4 hr | ~14 hr (mostly tidy IO) |

Build was launched per the recipe in `bundle-build-lessons.md`:

```
zodiacal-tools build-from-excerpt-series \
  --cell-depth 9 --bands 7 \
  --scale-lower 60.0 --scale-upper 480.0 \
  --quads-per-cell 80 --max-stars-per-cell 2500 \
  --mag-limit 20.0
```

Memory ceiling held at ~7.6 GB peak RSS through both build and tidy
phases — the partition-locality + bounded-loads + pre-truncate fixes
landed in PRs #114/#115 are durable for this scale.

## Solve quality: 1.41° regional, scale-hinted

Apples-to-apples comparison: same `bench-bundle --radius-deg 1.4142
--scale-hint` invocation against `set2-dr3-mag19`.

| metric | depth-8 | **depth-9** |
|---|---|---|
| Solved | 919 / 1000 (91.9%) | **1000 / 1000 (100.0%)** |
| Avg load | 512 ms | 645 ms |
| **Avg solve** | **175 ms** | **8.8 ms** |
| Median solve | 0.8 ms | 2.8 ms |
| p95 solve | 54.8 ms | 35.1 ms |
| p99 solve | 265.7 ms | 105 ms |
| Wall-clock | 695 s | 661 s |

The headline is the **20× drop in average solve time** (175 ms → 8.8 ms)
alongside the closure of the 8.1% failure gap. Depth-8 failures spent
their time in the solver verifying hundreds-to-thousands of code-tolerance
matches that all scored at the log-odds floor (none were the truth);
depth-9 has enough findable quads per FOV that the solver locks onto the
correct hypothesis on its first few attempts.

The depth-8 failures were diagnosed by `bench-triage` (PR #116):
catalog-vs-detection coverage was perfect (50/50 within 3 px, median
~1.5 px residual), and `n_quads_on_image = 0` for all 81 failures —
the bundle's quad set didn't include any 4-tuple whose stars all fell
inside the FOV. Going to depth-9 (cell ≤ ½ × min FOV dimension) made
several cells fully-internal to every FOV, which trivially fixes
findability.

## Solve time vs position-hint radius

`bench-bundle --radius-deg R --scale-hint` sweep, depth-9 bundle:

| radius | n solved | median | p95 | p99 | avg load |
|---|---|---|---|---|---|
| 1.0° | 1000 / 1000 | **1.90 ms** | 20.0 ms | 54.6 ms | 268 ms |
| 1.41° | 1000 / 1000 | 2.80 ms | 35.1 ms | 105 ms | 645 ms |
| 2.0° | 548 / 548* | 4.85 ms | 63.3 ms | 241 ms | 1123 ms |
| 5.0° | DNF (crash at case 92) | — | — | — | 10802 ms |

\* See "Sweep limitations" below.

Solve time scales roughly with hint area (∝ r²) — a wider hint loads a
larger region, so the solver wades through more spurious code-tolerance
matches before locking on. Solve **success rate stays at 100%** at
every radius the sweep reached.

Plots: `solve_time_by_radius_hist.png` (three stacked histograms),
`solve_time_by_radius_cdf.png` (overlaid CDFs), in
`/home/meawoppl/scratch/bundle-build/bench/`.

### Sweep limitations: bundle reader cache leak

The 5° run crashed at case 92 with `Cannot allocate memory (os error
12)`; the 2° run made it to case 549 before the same failure. The 1°
and 1.41° runs completed cleanly (1000 cases each).

Root cause: `ZdclBundle::cell_cache` (in `src/bundle/reader.rs`) is a
plain `Mutex<HashMap<u32, Arc<CellEntries>>>` — once a cell's quad +
gaia bytes are loaded, they stay in memory for the process lifetime.
There's no eviction, so the cumulative-cells-touched footprint grows
monotonically with sweep length × per-case cell count.

Rough scaling for the depth-9 bundle (3.14M cells, ~30 KB/cell average
on plane fields):

| radius | per-case cells | cumulative after 1000 cases | when it crashes |
|---|---|---|---|
| 5°   | ~6,000 | ~1.5M cells (~45 GB) | case 92 |
| 2°   | ~960   | ~500k cells (~15 GB) | case 549 |
| 1.41° | ~480 | ~250k cells (~7 GB) | finishes |
| 1°   | ~240   | ~150k cells (~4 GB) | finishes |

The fix is straightforward (LRU eviction, or a byte-cap eviction) and
is filed as [issue #119]. Without it, larger-radius benches and any
long-running query process will eventually OOM.

## What's next

- **Field-test against set3-dr3-mag20** (a deeper test corpus) when it
  exists — the current `set2-dr3-mag19` only goes to G≤19, and the
  bundle is built to G≤20, so we're not exercising the bundle's full
  faint-end coverage.
- **Cap the bundle reader's `cell_cache`** (issue #119) so 5° benches
  and long-running query servers don't OOM.
- **Add a fully-blind harness** (no `--radius-deg`, no `--scale-hint`)
  to characterize cold-start solve performance. Today's `bench-bundle`
  always loads a region, so all numbers here are upper bounds.

[`bundle-build-lessons.md`]: ./bundle-build-lessons.md
[issue #119]: https://github.com/OrbitalCommons/zodiacal/issues/119
