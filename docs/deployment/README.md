# Deployment modes

Zodiacal's library API supports three operational profiles. They share the
same substrate (the v3 `.zdcl` file format, the `IndexSource` trait, the
`KdForest` data structure, the refinement pipeline) but compose those
pieces differently to match different runtime constraints.

| Mode | Use case | Memory | Solve cadence | Doc |
|---|---|---|---|---|
| Server | Batch / multi-tenant solver service | Full sky resident (~4 GB at G≤19) | Many in flight | [server.md](server.md) |
| Ground | Telescope automation, follows zenith | Visible-cap resident (~50 MB) | Per-frame, ~Hz | [ground.md](ground.md) |
| Space | Spacecraft star tracker | FOV resident (~10 MB) | Per-frame, ~10 Hz | [space.md](space.md) |

## Pick by deployment characteristics

```
Is your physical pointing changing during operation?
├─ No → Server mode
│       Load full index once (Arc<Index>), share across solver workers,
│       fan out solves with rayon/tokio.
│
└─ Yes → Realtime mode
         │
         What drives the pointing?
         ├─ Earth rotation (telescope) → Ground mode
         │     PointingSource = GroundStation (lat, lon, time → zenith).
         │     Refresh cadence ~minutes.
         │
         └─ Spacecraft attitude (star tracker) → Space mode
               PointingSource = SpacecraftBoresight<Ephemeris, Attitude>.
               Refresh cadence ~seconds; observer_state feeds refinement.
```

## What changes between modes

The library types you actually instantiate:

| Component | Server | Ground | Space |
|---|---|---|---|
| `Index::load(path)` | ✓ | — | — |
| `Arc<Index>` shared | ✓ | — | — |
| `ZdclFile::open(path)` | — | ✓ | ✓ |
| `LiveIndex<ZdclFile>` | — | ✓ | ✓ |
| `KdForest<3>` for stars | — (single tree fine) | ✓ (per-cell sub-trees) | ✓ |
| `PointingSource` impl | — | `GroundStation` | `SpacecraftBoresight` |
| `RealtimeSolver` orchestrator | — | ✓ | ✓ |
| Refinement | optional, in-line | optional, in-line | natural — observer_state from ephemeris |
| Sidecar (`.zdcl.gaia`) | mmap once | mmap once | mmap once |

## Cross-cutting performance notes

The numbers below come from end-to-end benchmarks against the real
`gaia_d8` index (31 M stars, full sky, 1 GB on disk) at 1″/px image
scale. Hardware: 64-core x86_64.

| Operation | Cost | Notes |
|---|---|---|
| `Index::load` (full sky) | ~5–8 s | Streams once, rebuilds two KD-trees. One-time. |
| `ZdclFile::open` (mmap) | <1 ms | Just reads the header + cell table. |
| `LiveIndex::ensure_region` (cold cache) | ~5–10 ms / cell | First touch faults pages; small per-cell tree build. |
| `LiveIndex::ensure_region` (warm cache) | <1 ms / cell | OS page cache hit; tree build is the only cost. |
| `KdForest::insert` (one cell add) | O(N\_cell log N\_cell), typically ~µs | New sub-tree only; no full rebuild. |
| `KdForest::remove` (one cell drop) | O(1) | Vec retain. |
| `solve` (blind, well-formed field) | 0.1–10 ms | Wide range — depends on index density and quad code matches. |
| `refine_solution` (post-solve) | ~50 ms | One apparent-place call per matched star. |

The KdForest data structure is the architectural enabler for ground/space:
it lets `LiveIndex` add/drop cells without rebuilding the unified
KD-tree on every membership change. See each mode doc for concrete
breakdown.

## Sharding strategy

The v3 file format groups stars by HEALPix cell at build time. That
gives every deployment mode the same on-disk layout but different
runtime access patterns:

- **Server**: `ZdclFile::open` → `load_full()` once → flat `Index`. Cell
  boundaries are invisible at runtime. The grouping lets you build the
  index once and serve everyone, including future realtime callers, from
  the same file.
- **Realtime (ground/space)**: `ZdclFile::open` → `LiveIndex` →
  `cells_intersecting(region)` per `tick()` → only relevant cells
  resident in RAM. Disk I/O drops from O(file) to O(region).

Build-side sharding (multiple narrow-band index files like
`my_index_00.zdcl` ... `my_index_06.zdcl` covering different angular
scale ranges) is **independent** of cell grouping and handled at the
solver level via `solve(sources, &[&i0, &i1, &i2, ...], ...)`. All
three modes can mix these — server mode loads N files into N
`Arc<Index>` instances and passes them all to `solve`; realtime modes
hold N `LiveIndex` instances over N `ZdclFile`s.

## See also

- [Plan 1–5](https://github.com/OrbitalCommons/zodiacal/issues?q=label%3Aplans) — the staged design docs that produced this stack.
- [`zodiacal-tools build-from-shards`](../../zodiacal-tools/) — the CLI that produces the `.zdcl` index + `.zdcl.gaia` sidecar consumed by all three modes.
