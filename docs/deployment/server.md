# Server mode

A solver service: full-sky index resident in memory, many concurrent
solves in flight, predictable latency and throughput. This is the
simplest deployment — most of the realtime machinery (LiveIndex,
PointingSource, RealtimeSolver) is unnecessary.

## When to use

- Backend for an HTTP plate-solve service.
- Batch processing of historical data.
- Multi-tenant scenarios where queries can land anywhere on the sky.
- Anything where the field of view is unpredictable or changes per
  request.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Process                                                    │
│                                                             │
│  Arc<Index> ◄──── load once at startup                      │
│      │                                                      │
│      │   share across N solver workers                      │
│      ▼                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                   │
│  │ Worker 1 │  │ Worker 2 │  │ Worker N │  rayon thread     │
│  │ solve()  │  │ solve()  │  │ solve()  │  pool / tokio /   │
│  └──────────┘  └──────────┘  └──────────┘  custom queue     │
│                                                             │
│  Optional: Arc<SidecarReader> for refinement                │
└─────────────────────────────────────────────────────────────┘
```

## End-to-end example

```rust
use std::path::Path;
use std::sync::Arc;
use rayon::prelude::*;

use zodiacal::extraction::DetectedSource;
use zodiacal::index::Index;
use zodiacal::refinement::{
    apparent_radec, refine_solution, ObservationContext, ObserverState,
    RefinementCatalog, RefinementConfig, SidecarReader,
};
use zodiacal::solver::{solve, SolverConfig};

struct ServerState {
    indexes: Vec<Arc<Index>>,
    sidecar: Option<Arc<SidecarReader>>,
    solver_config: SolverConfig,
    refinement_config: Option<RefinementConfig>,
}

impl ServerState {
    fn new(index_paths: &[&Path], sidecar_path: Option<&Path>) -> std::io::Result<Self> {
        // Build phase — pay this once at startup.
        let indexes: Vec<Arc<Index>> = index_paths
            .iter()
            .map(|p| Index::load(p).map(Arc::new))
            .collect::<std::io::Result<_>>()?;
        let sidecar = sidecar_path
            .map(|p| SidecarReader::open(p).map(Arc::new))
            .transpose()?;
        Ok(Self {
            indexes,
            sidecar,
            solver_config: SolverConfig::default(),
            refinement_config: None, // opt-in per request
        })
    }

    fn solve_one(
        &self,
        sources: &[DetectedSource],
        image_size: (f64, f64),
    ) -> Option<zodiacal::solver::Solution> {
        let refs: Vec<&Index> = self.indexes.iter().map(|a| &**a).collect();
        let (sol, _stats) = solve(sources, &refs, image_size, &self.solver_config);
        sol
    }

    fn solve_batch(
        &self,
        requests: Vec<(Vec<DetectedSource>, (f64, f64))>,
    ) -> Vec<Option<zodiacal::solver::Solution>> {
        // rayon — each thread holds its own &Vec<&Index> view; the underlying
        // Arc<Index> instances are shared without contention (KdTree is
        // immutable after build).
        requests
            .into_par_iter()
            .map(|(sources, image_size)| self.solve_one(&sources, image_size))
            .collect()
    }
}

fn main() -> std::io::Result<()> {
    let server = ServerState::new(
        &[
            Path::new("indexes/gaia_jbt_00.zdcl"),
            Path::new("indexes/gaia_jbt_01.zdcl"),
            // ... more bands
        ],
        Some(Path::new("indexes/gaia_jbt_00.zdcl.gaia")),
    )?;
    // ...serve requests by calling server.solve_batch(...) or server.solve_one(...)
    Ok(())
}
```

## Memory expectations

| Workload | Resident |
|---|---|
| Single d8 index loaded | ~5 GB (1 GB on disk × ~5× in-memory expansion for KD-trees) |
| Single d8 sidecar mmapped | ~0 KB demand-faulted (counted as cache, not RSS) |
| 7 narrow-scale bands (`gaia_jbt_*.zdcl`) loaded | ~35 GB |
| Per-request transient | ~few MB (sources, candidate WCS) |

Use `mmap` for the sidecar (already what `SidecarReader::open` does) so
the OS can manage resident pages — the sidecar's only consulted at
refinement time, not on every solve, so demand-paging is the right
default.

## Performance expectations

Numbers from end-to-end benchmarks against the real `gaia_d8` index
(31 M stars) on a 64-core x86_64 host. **All times are p50.**

| Operation | Cold cache | Warm cache | Notes |
|---|---|---|---|
| `Index::load` | ~5 s | ~5 s | One-time, dominated by KD-tree build. |
| `solve` (blind) | 60 µs | 60 µs | Surprisingly fast — first quad usually wins. |
| `solve` (10° hint) | 190 µs | 190 µs | 3× slower than blind — see "the hint paradox" below. |
| `solve` (1° hint) | 6 ms | 6 ms | 100× slower than blind. |
| `refine_solution` (50 matched stars) | 5 ms | 5 ms | Apparent-place + weighted re-fit. |

### The hint paradox

A counterintuitive result: tight `within` hints make solves **slower**,
not faster, on dense indexes. The reason: with no hint, the solver
returns on the first verify-passing quad match (which is almost always
the truth match for a noise-free field). With a 1° `within`, candidates
whose centroid falls outside the padded region get pre-fit-rejected, so
the solver iterates deeper through code-space neighbors before finding
one whose geometry passes the hint.

This is **correct behavior** — the hint is doing its job by rejecting
false positives, not optimizing for raw throughput. For server use:
- If you have a tight prior, set `within` to leverage it for correctness.
- If you don't, omit it — blind solves are faster on dense data.

See the bench tooling at `benches/attitude_hint.rs` for reproducible
measurements.

## Why `KdForest` doesn't help here

In server mode you load the full sky once and never change the loaded
set. A single unified `KdTree<3>` over all stars is the right structure
— there's no membership churn to amortize. `KdForest`'s benefit (cheap
add/drop) is wasted if you never add or drop.

If your "server" actually rotates between regions (e.g., one process
per observatory time slice), promote it to ground mode — the per-cell
sub-tree machinery would start to pay off.

## Sharding & scaling

### Multi-band indexes

The recommended index build is a series of narrow angular-scale bands
(`gaia_jbt_00.zdcl` through `gaia_jbt_06.zdcl` covering 60″ to 600″ at
√2 spacing). Pass them all to `solve`:

```rust
let indexes_owned = vec![
    Arc::new(Index::load("gaia_jbt_00.zdcl".as_ref())?),
    Arc::new(Index::load("gaia_jbt_01.zdcl".as_ref())?),
    // ...
];
let refs: Vec<&Index> = indexes_owned.iter().map(|a| &**a).collect();
solve(&sources, &refs, image_size, &config)
```

The solver skips bands whose scale range can't overlap the field's
backbone angular distance — wide-field requests don't pay the
narrow-band cost and vice versa.

### Horizontal scaling

`Index` is `Send + Sync` (KD-trees are immutable after build). Wrap
in `Arc<Index>` and share across:

- **rayon** thread pool for CPU-bound batch work
- **tokio** async runtime if you're serving HTTP — wrap `solve` in
  `spawn_blocking` since it's CPU-bound
- **process pool** for tenant isolation; each process loads its own
  copy

Because `Arc<Index>` is shared without contention, going from 1 to 64
solver workers is linear scaling up to the bench-measured saturation
point of ~80 logical cores (rayon's default pool size on the test box).

### What to monitor

| Metric | Why |
|---|---|
| `solve` p50 / p95 / p99 latency | Tail latency surfaces dense-region slowdowns |
| `n_verified` per solve | A spike means false-positive-heavy region |
| RSS | Should be flat after startup; growth = leak |
| Sidecar mmap page-cache hit rate | If refinement starts eating disk I/O |
| Verify rejection log-odds distribution | Helps tune `log_odds_accept` |

## Pitfalls

- **Don't reload `Index` per request.** Tree build is ~5 s. Build at
  startup and share via `Arc`.
- **Don't pre-load the sidecar into memory.** `SidecarReader::open`
  uses mmap — that's the right move. Eagerly reading 2.7 GB into a
  `Vec<SidecarRecord>` doubles your memory budget without speeding up
  the only access pattern (point lookups by source_id).
- **Set `verify.log_odds_accept` per your false-positive tolerance.**
  Defaults are conservative; production servers often want to tighten
  to 30+ to reject ambiguous matches.
- **Pin solver thread count if you're sharing the box.** rayon's
  default uses all logical cores; for a multi-tenant server consider
  `rayon::ThreadPoolBuilder` to isolate.
