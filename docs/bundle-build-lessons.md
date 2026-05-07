# Lessons from the depth-8 G≤20 bundle build

A field journal from the first production bundle build (786,432 cells × 7
bands × Gaia DR3 G≤20). What broke, what we changed, and what to do
differently next time. Reference these notes before kicking off a larger
build.

## What we shipped

| field | value |
|---|---|
| Cell depth | 8 (~13.7' cells) |
| Bands | 7 × `[60", 600"]` log-spaced |
| Mag limit | G ≤ 20 |
| `quads_per_cell` | 100 |
| `max_stars_per_cell` | 10000 |
| Stars indexed | 900,101,054 |
| Quads indexed | ~543M (across 7 bands) |
| Bundle on disk | 116 GB (27 GB quads / 89 GB gaia) |
| End-to-end build wall-clock (after fixes) | ~3:18 from resume + ~50 min initial = ~4 hr |

## Solve performance against `set2-dr3-mag19` (1000 cases)

Regional + scale-hinted (`--radius-deg 1.4142`, `--scale-hint`, *not* a
fully blind solve):

- 919 / 1000 solved (91.9%)
- Median solve time 0.8 ms, p95 54.8 ms, p99 265.7 ms
- All 81 failures: catalog matches detections perfectly (50/50 within 3
  px, median residual ~1.5 px) but **no bundle quad has all 4 stars
  projecting on-image** (`n_quads_on_image = 0`). The bundle's
  `quads_per_cell = 100` selection per HEALPix-8 cell happens to not
  include any 4-tuple whose 4 stars all fall inside the FOV at those
  pointings.

## Lessons

### Sizing — pick `cell_depth` from the FOV, not from intuition

The most consequential parameter we got wrong. At depth 8 the cell is
13.7'; the test images are 13.77' × 20.65'. The cell is *equal* to FOV
height and *narrower* than FOV width, so cells almost never fit
**fully** inside the FOV. That's the entire 8.1% failure mode.

**Rule:** pick `cell_depth` so the cell is **smaller than half the FOV
in both dimensions**. For an IMX455-class instrument at 0.13"/px (FOV
21' × 14'), that is depth 9 (6.87'). For a wider instrument, scale
accordingly.

Related knobs once `cell_depth` is right:

- `scale_upper` should be `≪` FOV diagonal so quads fit at any
  rotation. Setting it near the cell size (we had 600" ≈ 10' against
  a 13.7' cell) is borderline; aim for ≤ ½ × FOV diagonal.
- `max_stars_per_cell` should scale with cell area. Going depth 8 → 9
  shrinks cells 4×; drop the cap proportionally (10000 → 2500) to keep
  per-cell density invariant and slash cached-partition memory.

### Memory — three compounding fixes, all necessary

The depth-8 mag-20 build OOMed at exactly **57 GB anon-rss** twice
before we plugged everything that was leaking. Order of fixes (they
compound; remove any one and a 62 GB host can't finish):

1. **Sort `to_process` by source-partition key** + group siblings into
   one `par_iter` chunk per source cell ([PR #114]). Random per-cell
   access across 80 rayon workers makes every worker hold a different
   multi-GB partition concurrently; sorted access lets the cache work.
2. **Pre-truncate cached partitions during build, not after read** +
   drop the unused 88-byte `SidecarRecord` field from `CellStar`
   ([PR #115]). For galactic-plane source partitions a single bundle
   cell can carry millions of stars before the orchestrator's
   `truncate(max_stars_per_cell)` discards 99% of them; truncate inside
   `BycellExcerptSource::load_and_partition` instead.
3. **Bound concurrent in-flight CSV parses** with a Condvar permit pool
   (`MAX_CONCURRENT_PARSES = 4`, in PR #115). Each parse holds ~5 GB
   `Dr3Catalog` plus ~5 GB un-truncated buckets transiently; a burst of
   80 cache misses across rayon workers stacks ~40+ GB of transient
   memory. The permit cap keeps it bounded.

For the *next* dense build, **(3) is the load-bearing fix**. If you go
denser (mag-21, finer cell depth), tune the permit cap by
`permits × per-parse-peak < free_ram`.

### Manifest hot path — actor + interval flush

The manifest mutex was 76% of worker time before we replaced it with an
mpsc channel + dedicated actor thread ([PR #111], [PR #112]). Two
related things that bit us:

- `BuildManifest::commit_cell` shouldn't touch disk; let the actor flush
  on a wall-clock cadence ([PR #113]). With a per-cell save the actor
  was rewriting a 17 MB JSON blob ~10×/sec near the end of the build.
- `cell_stats` must be `BTreeMap<u32, CellStats>`, not a sorted `Vec`,
  once N grows past a few thousand. The original `Vec<(u32, CellStats)>`
  with `sort_by_key` on every insert and `iter().find()` on readback was
  O(N²) at depth 8.

### Build layout and disk

- **Budget 4–5× the final bundle size of free disk during a build.**
  Work-dir + `.partial` + the existing `.bundle` if you're keeping it.
  Our depth-8 build peaked at ~232 GB on disk during tidy.
- **The tidy phase is overwhelmingly IO-bound.** Copying 786k shards
  from work-dir to the final bundle ate ~2 hours at depth 8 — close to
  half the total wall-clock. Anything that elides the work-dir →
  bundle copy (build directly into the bundle layout, or stream into a
  zip output) is high leverage on big builds.
- `gaia/` is ~3× the size of `quads/` at mag-20 (89 GB vs 27 GB). The
  104-byte `GaiaRecord` per star dominates. A "gaia-lite" record kind
  for non-supplement stars (drop sigmas/correlations) would compress
  significantly, at the cost of refinement quality.

### Operational — long-running builds need a non-shell parent

Run all multi-hour builds via:

```
systemd-run --user --no-block --unit=NAME \
  --working-directory=/path/to/repo \
  --property=StandardOutput=append:/path/to/log \
  --property=StandardError=append:/path/to/log \
  -- /path/to/binary args…
```

Anything spawned from an interactive shell inherits that session's
cgroup and is reaped if `agent-portal.service` (or whatever else hosts
the shell) blips. We lost a build at 49% to an unrelated agent-portal
restart; the systemd-run wrapper made the next attempt durable.

Handles after launch: `systemctl --user status NAME`,
`systemctl --user stop NAME`, `journalctl --user -u NAME`.

### Test harnesses we have, and the gap

- `bench-bundle` is regional + optionally scale-hinted; it's not a
  blind solve, so its solve rate is an upper bound. The 91.9% number
  here is "given you know the field is within 1.4°, can the bundle
  solve it"; cold-start blind performance is unmeasured.
- `bench-triage` (PR #116) is the right shape for failure analysis.
  Its `n_quads_on_image` + `n_findable_quads` columns localize whether
  a failure is catalog-side, bundle-side, or solver-side. Run it as
  part of every new build's acceptance.
- The test-case JSONs from `OrbitalCommons/zodiacal-test-cases` carry
  `(ra, dec, plate_scale)` but **not the renderer's CD-matrix
  orientation**; we had to brute-force-discover that
  `focalplane motion_simulator` outputs
  `cd = [[+scale, 0], [0, -scale]]` (image y points south). Future
  test corpora should pin the truth WCS in the JSON; raise an issue
  against the test-cases repo when generating the next set.

## Recommended args for the next big build (IMX455-class FOV)

```
--cell-depth 9                # cell ≈ 6.87' < ½ × min(FOV dimensions)
--bands 7
--scale-lower 60.0
--scale-upper 480.0           # ≈ ½ FOV diagonal
--quads-per-cell 80           # ~9 cells fully inside FOV → ~5000 candidates per pointing
--max-stars-per-cell 2500     # cell area is 4× smaller than depth 8
--max-reuse 8
--mag-limit 20.0
```

Estimated bundle size ~160 GB, build wall-clock ~3 hr (depending on
tidy IO). **Validate by building one HEALPix-5 cell's worth of bundle
first** (~20 min) and re-running the previously-failing 81 test cases
against it before committing to the full build.

[PR #111]: https://github.com/OrbitalCommons/zodiacal/pull/111
[PR #112]: https://github.com/OrbitalCommons/zodiacal/pull/112
[PR #113]: https://github.com/OrbitalCommons/zodiacal/pull/113
[PR #114]: https://github.com/OrbitalCommons/zodiacal/pull/114
[PR #115]: https://github.com/OrbitalCommons/zodiacal/pull/115
[PR #116]: https://github.com/OrbitalCommons/zodiacal/pull/116
