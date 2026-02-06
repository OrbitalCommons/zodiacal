# Zodiacal — Blind Astrometry Implementation Plan

## Overview

A pure-Rust blind plate solver inspired by astrometry.net. Takes an `Array2<f32>` image (or a pre-extracted point cloud), identifies the region of sky it depicts, and returns a WCS solution mapping pixels to celestial coordinates.

No FITS I/O, no web UI, no network services — just the core math.

---

## Dependency Map (DAG)

```
                    ┌──────────────────────────────────┐
                    │        (A) Geometric Types        │
                    │  TAN projection, CD matrix, SIP   │
                    └──────┬──────────┬────────────────┘
                           │          │
              ┌────────────┘          └───────────────┐
              ▼                                       ▼
 ┌─────────────────────┐                ┌──────────────────────┐
 │  (B) Star Extraction │                │  (C) Quad Codec      │
 │  starfield DAO/IRAF  │                │  code computation,   │
 │  → point cloud out   │                │  invariant hashing   │
 └─────────┬────────────┘                └──────────┬───────────┘
           │                                        │
           │                       ┌────────────────┤
           │                       ▼                ▼
           │          ┌──────────────────┐  ┌───────────────────┐
           │          │  (D) KD-Tree     │  │  (E) Index Builder │
           │          │  generic N-dim   │  │  catalog → quads   │
           │          │  range search    │  │  → code KD-tree    │
           │          └────────┬─────────┘  └───────┬───────────┘
           │                   │                    │
           │                   └──────┬─────────────┘
           ▼                          ▼
 ┌──────────────────┐     ┌───────────────────────┐
 │ (F) Field Quads  │     │  (G) Index Store      │
 │ build quads from │     │  serialize/load index  │
 │ detected stars   │     │  files from disk       │
 └────────┬─────────┘     └───────────┬────────────┘
          │                           │
          └──────────┬────────────────┘
                     ▼
          ┌─────────────────────┐
          │  (H) Solver Loop    │
          │  match field quads  │
          │  against index      │
          └─────────┬───────────┘
                    │
           ┌───────┴────────┐
           ▼                ▼
 ┌──────────────────┐ ┌──────────────────┐
 │ (I) WCS Fitting  │ │ (J) Verification │
 │ least-squares    │ │ Bayesian match   │
 │ TAN from corr.   │ │ scoring          │
 └────────┬─────────┘ └────────┬─────────┘
          │                    │
          └──────┬─────────────┘
                 ▼
       ┌──────────────────┐
       │ (K) SIP Tweaker  │
       │ polynomial dist. │
       │ refinement       │
       └──────────────────┘
```

### Parallelizable Work Streams

Components that can be developed **in parallel** (no dependency between them):

| Stream | Components | Notes |
|--------|-----------|-------|
| **Stream 1** | (A) Geometric Types | Foundation — start first |
| **Stream 2** | (B) Star Extraction | Wraps starfield, independent |
| **Stream 3** | (C) Quad Codec + (D) KD-Tree | Pure math, no other deps |
| **Stream 4** | (I) WCS Fitting | Needs (A), but can stub types |

After Stream 1 completes, (E), (F), (G), (H), (J) become unblocked and depend on the foundations.

---

## Step-by-Step Implementation

### Step 1 — Project Skeleton & Geometric Types `(A)`

**Module:** `src/geom/`

**Purpose:** Core types for WCS projections, the coordinate math that everything else builds on.

#### 1.1 TAN Projection (`src/geom/tan.rs`)

```rust
pub struct TanWcs {
    /// Reference point on sky (RA, Dec) in radians
    pub crval: [f64; 2],
    /// Reference point in pixel coordinates
    pub crpix: [f64; 2],
    /// CD matrix: linear pixel→intermediate-world-coords
    /// cd[0] = [cd1_1, cd1_2], cd[1] = [cd2_1, cd2_2]
    pub cd: [[f64; 2]; 2],
    /// Image dimensions (width, height) in pixels
    pub image_size: [f64; 2],
}
```

**Methods:**

| Method | Description |
|--------|-------------|
| `pixel_to_xyz(px, py) → [f64; 3]` | Pixel → unit vector on celestial sphere |
| `xyz_to_pixel(xyz) → Option<(f64, f64)>` | Unit vector → pixel (None if behind tangent plane) |
| `pixel_to_radec(px, py) → (f64, f64)` | Pixel → (RA, Dec) in radians |
| `radec_to_pixel(ra, dec) → Option<(f64, f64)>` | (RA, Dec) → pixel coords |
| `pixel_scale() → f64` | Approximate degrees/pixel from CD matrix |
| `field_center() → (f64, f64)` | RA, Dec of image center |
| `field_radius() → f64` | Angular radius enclosing the image |

**Math:** Standard gnomonic (TAN) projection:
- Forward: `(u, v) = (px - crpix) → (x, y) = CD · (u, v)` → project onto sphere
- Inverse: sphere → tangent plane → CD⁻¹ → pixel

**Tests:**
- Round-trip pixel→radec→pixel for grid of points
- Known solution: Polaris at center, specific CD matrix
- Edge cases: poles, 0/360° RA wrap

#### 1.2 SIP Distortion (`src/geom/sip.rs`)

```rust
pub struct SipWcs {
    pub tan: TanWcs,
    /// Forward distortion: pixel → corrected pixel
    pub a: Vec<Vec<f64>>,  // a[p][q] for u^p * v^q
    pub b: Vec<Vec<f64>>,
    pub a_order: usize,
    pub b_order: usize,
    /// Inverse distortion: corrected → pixel
    pub ap: Vec<Vec<f64>>,
    pub bp: Vec<Vec<f64>>,
    pub ap_order: usize,
    pub bp_order: usize,
}
```

**Methods:** Same interface as `TanWcs` but with distortion polynomials applied in the pipeline:
```
pixel (u,v) → distort: (U,V) = (u + Σ a[p][q]·uᵖvᵠ, v + Σ b[p][q]·uᵖvᵠ) → CD → sky
```

**Implementation detail:** Distortion evaluated via Horner's method for numerical stability.

#### 1.3 Spherical Geometry Utilities (`src/geom/sphere.rs`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `radec_to_xyz` | `(ra, dec) → [f64; 3]` | Spherical to unit vector |
| `xyz_to_radec` | `([f64; 3]) → (f64, f64)` | Unit vector to (RA, Dec) |
| `angular_distance` | `(xyz_a, xyz_b) → f64` | Great-circle distance |
| `star_midpoint` | `(a, b) → [f64; 3]` | Midpoint on sphere |
| `star_coords` | `(point, ref_point) → (f64, f64)` | Gnomonic projection of point relative to reference |

These mirror astrometry.net's `starutil.h` functions but without the C baggage.

---

### Step 2 — Star Extraction Wrapper `(B)`

**Module:** `src/extraction.rs`

**Purpose:** Bridge between raw `Array2<f32>` images and the point cloud the solver needs.

```rust
pub struct DetectedSource {
    pub x: f64,
    pub y: f64,
    pub flux: f64,
}

pub struct ExtractionConfig {
    pub fwhm: f64,
    pub threshold_sigma: f64,
    pub max_sources: usize,
    pub sort_by_brightness: bool,
}

pub fn extract_sources(
    image: &Array2<f32>,
    config: &ExtractionConfig,
) -> Vec<DetectedSource>
```

**Implementation:** Delegate to starfield's `DAOStarFinder` or `IRAFStarFinder`:
1. Run starfield star finder on the image
2. Filter by sharpness/roundness to reject artifacts
3. Sort by flux (brightest first — solver processes brightest stars first)
4. Truncate to `max_sources`
5. Return `Vec<DetectedSource>`

**Also accept pre-extracted point clouds:**

```rust
pub fn sources_from_points(xy: &[(f64, f64)]) -> Vec<DetectedSource>
```

This allows users who already have source lists to skip extraction.

---

### Step 3 — KD-Tree `(D)`

**Module:** `src/kdtree.rs`

**Purpose:** Generic N-dimensional KD-tree with range search. Used for both:
- 3D star positions on the unit sphere (star tree)
- 4D+ code space (code tree)

```rust
pub struct KdTree<const DIM: usize> {
    nodes: Vec<KdNode<DIM>>,
    points: Vec<[f64; DIM]>,
    indices: Vec<usize>,      // original index of each point
}

pub struct KdNode<const DIM: usize> {
    split_dim: usize,
    split_val: f64,
    left: Option<usize>,
    right: Option<usize>,
    // Leaf nodes store index ranges
    point_start: usize,
    point_end: usize,
}
```

**Key Methods:**

| Method | Description |
|--------|-------------|
| `build(points, indices) → KdTree` | Construct tree with median splitting |
| `range_search(query, radius²) → Vec<(usize, f64)>` | All points within L2 distance, returns (index, dist²) |
| `nearest(query) → (usize, f64)` | Single nearest neighbor |
| `range_count(query, radius²) → usize` | Count without collecting (for verification) |

**Design decisions:**
- Use const generics `DIM` so the compiler specializes for 3D (stars) and 4D (codes)
- Flat array layout (cache-friendly)
- No dynamic allocation during search (reuse result buffer)
- Support `serde` for serialization to/from index files

**Tests:**
- Brute-force comparison for small datasets
- Known geometric configurations
- Performance benchmark vs brute force at N=100K

---

### Step 4 — Quad Codec `(C)`

**Module:** `src/quads.rs`

**Purpose:** Compute geometric hash codes from sets of 4 stars. This is the core pattern-matching primitive.

#### 4.1 Code Computation

```rust
pub const DIMQUADS: usize = 4;
pub const DIMCODES: usize = 2 * (DIMQUADS - 2); // = 4

/// A quad is defined by 4 star indices
pub struct Quad {
    pub star_ids: [usize; DIMQUADS],
}

/// The geometric hash of a quad — a point in code space
pub type Code = [f64; DIMCODES];

/// Compute the code for a quad given star positions on the unit sphere.
///
/// Stars A, B form the "backbone". C, D are projected into a coordinate
/// system defined by A-B, producing scale/rotation-invariant coordinates.
pub fn compute_code(star_xyz: &[[f64; 3]; DIMQUADS]) -> Code
```

**Algorithm** (mirrors `quad_compute_star_code` in astrometry.net):

1. Let A = star_xyz[0], B = star_xyz[1]
2. Compute midpoint M of A and B on the sphere
3. Project A and B onto tangent plane at M → (Ax, Ay), (Bx, By)
4. Compute rotation angle θ that aligns AB with x-axis
5. Compute scale = |AB|²
6. For each star C, D (indices 2, 3):
   - Project onto tangent plane at M → (Cx, Cy)
   - Translate: (Cx - Ax, Cy - Ay)
   - Rotate by θ and scale by 1/|AB|
   - Store as code[2*(i-2)], code[2*(i-2)+1]

#### 4.2 Invariant Enforcement

```rust
/// Enforce ordering constraints on a code so that equivalent
/// geometric configurations produce the same code.
///
/// Constraints:
///   1. code[0] (cx) ≤ code[2] (dx)
///   2. mean of x-coords ≤ 0.5
///
/// Returns (reordered_code, reordered_star_ids, parity)
pub fn enforce_invariants(
    code: Code,
    star_ids: [usize; DIMQUADS],
) -> (Code, [usize; DIMQUADS], bool)
```

#### 4.3 Permutation Enumeration

```rust
/// Given backbone stars A, B and candidate stars for C, D slots,
/// enumerate all valid quads and their codes.
///
/// Yields (code, star_ids) for each valid assignment of
/// candidates to C, D positions.
pub fn enumerate_field_quads(
    backbone: (usize, usize),
    candidates: &[usize],
    positions_xy: &[(f64, f64)],       // pixel positions
    reference_point: [f64; 3],          // (not needed for field quads, only index)
    cos_theta: f64,
    sin_theta: f64,
    scale_inv: f64,
) -> Vec<(Code, [usize; DIMQUADS])>
```

**Tests:**
- Rotation invariance: rotate star pattern, codes should match
- Scale invariance: scale star pattern, codes should match
- Mirror invariance: flip pattern, codes should match (with parity)
- Known quad from astrometry.net test data

---

### Step 5 — Index Builder `(E)`

**Module:** `src/index/builder.rs`

**Purpose:** Offline process that takes a star catalog and produces a searchable index.

#### 5.1 Index Structure

```rust
pub struct Index {
    /// Star positions as unit vectors on the celestial sphere
    pub star_tree: KdTree<3>,
    /// Star metadata (magnitude, ID, etc.)
    pub star_data: Vec<IndexStar>,
    /// Code KD-tree for quad matching
    pub code_tree: KdTree<DIMCODES>,
    /// Quad definitions (which star IDs form each quad)
    pub quads: Vec<Quad>,
    /// Scale range this index covers (in radians)
    pub scale_range: (f64, f64),
    /// Number of stars per quad
    pub dimquads: usize,
}

pub struct IndexStar {
    pub catalog_id: u64,
    pub ra: f64,    // radians
    pub dec: f64,   // radians
    pub mag: f64,
}
```

#### 5.2 Build Pipeline

```rust
pub struct IndexBuilderConfig {
    /// Min/max angular size of quads to generate (radians)
    pub scale_range: (f64, f64),
    /// Maximum number of stars to include (brightest first)
    pub max_stars: usize,
    /// Maximum number of quads to generate
    pub max_quads: usize,
    /// Number of stars per quad
    pub dimquads: usize,
}

pub fn build_index(
    catalog: &impl StarCatalog,  // starfield trait
    config: &IndexBuilderConfig,
) -> Index
```

**Algorithm:**

1. **Select stars:** Take the `max_stars` brightest from the catalog
2. **Build star tree:** Insert 3D unit vectors into `KdTree<3>`
3. **Generate quads:**
   - For each star A (by brightness):
     - Range-search star tree for neighbors within `scale_range.1`
     - For each neighbor B (where B > A in index):
       - Compute AB angular distance
       - Skip if outside `scale_range`
       - Find all stars within the AB circle (candidate C, D)
       - For each valid combination of C, D:
         - Compute code
         - Enforce invariants
         - Store quad + code
4. **Build code tree:** Insert all codes into `KdTree<DIMCODES>`
5. **Return Index**

**Deduplication:** Use a hash set on sorted star_ids to avoid duplicate quads.

**Memory concern:** For a full Hipparcos catalog (~118K stars), the number of possible quads is enormous. Use `max_quads` to cap it and prioritize quads formed from brighter stars (they're more likely to be detected).

---

### Step 6 — Index Serialization `(G)`

**Module:** `src/index/store.rs`

**Purpose:** Save and load index files. Custom binary format (no FITS dependency).

```rust
impl Index {
    pub fn save(&self, path: &Path) -> Result<(), Error>
    pub fn load(path: &Path) -> Result<Index, Error>
}
```

**Format:** Simple binary with header:

```
[magic: 4 bytes "ZDCL"]
[version: u32]
[dimquads: u32]
[num_stars: u64]
[num_quads: u64]
[scale_lower: f64]
[scale_upper: f64]
[star_data: num_stars × IndexStar]
[star_tree: serialized KdTree<3>]
[quad_data: num_quads × Quad]
[code_tree: serialized KdTree<DIMCODES>]
```

Use `bincode` or manual byte serialization. Prefer simplicity over compression for v1.

---

### Step 7 — WCS Fitting `(I)`

**Module:** `src/fitting.rs`

**Purpose:** Given matched star correspondences (pixel ↔ sky), compute the WCS transformation.

#### 7.1 TAN Fitting (Linear)

```rust
/// Fit a TAN WCS from corresponding pixel and sky positions.
///
/// Uses SVD-based least-squares to find the CD matrix that best
/// maps pixel offsets to tangent-plane coordinates.
///
/// Requires at least 3 correspondences (6 unknowns: crval[2], cd[2][2], crpix[2]),
/// but crpix is typically fixed at image center.
pub fn fit_tan_wcs(
    star_xyz: &[[f64; 3]],   // reference star unit vectors
    field_xy: &[(f64, f64)],  // pixel coordinates
    image_size: (f64, f64),
) -> Result<TanWcs, FitError>
```

**Algorithm:**
1. Compute tangent point (crval) as the centroid of star_xyz projected to RA/Dec
2. Set crpix = image center
3. Project all stars onto tangent plane at crval → (ξᵢ, ηᵢ)
4. Solve least-squares: `[ξ; η] = CD · [u; v]` where `(u, v) = (px - crpix_x, py - crpix_y)`
5. SVD decomposition to find CD matrix
6. Return TanWcs

**Dependencies:** `nalgebra` for SVD. No external LAPACK needed.

#### 7.2 SIP Fitting (Polynomial Distortion)

```rust
/// Fit SIP distortion coefficients on top of an existing TAN WCS.
///
/// Given a TAN solution and additional star correspondences,
/// fits polynomial correction terms.
pub fn fit_sip(
    tan: &TanWcs,
    star_xyz: &[[f64; 3]],
    field_xy: &[(f64, f64)],
    order: usize,            // polynomial order (typically 2-5)
) -> Result<SipWcs, FitError>
```

**Algorithm:**
1. For each correspondence, compute residual between TAN prediction and actual pixel
2. Set up Vandermonde matrix for polynomial terms: u^p · v^q for p+q ≤ order
3. Solve least-squares for A, B coefficients
4. Compute inverse coefficients AP, BP by fitting the reverse direction

---

### Step 8 — Verification `(J)`

**Module:** `src/verify.rs`

**Purpose:** Given a candidate WCS solution, score it against all detected sources using Bayesian log-odds.

```rust
pub struct VerifyConfig {
    /// Positional tolerance in pixels (how close a match must be)
    pub match_radius_pix: f64,
    /// Expected fraction of field sources that are noise/artifacts
    pub distractor_fraction: f64,
    /// Log-odds threshold to accept a solution
    pub log_odds_accept: f64,
    /// Log-odds threshold to bail early (definitely wrong)
    pub log_odds_bail: f64,
}

pub struct VerifyResult {
    pub log_odds: f64,
    pub n_matched: usize,
    pub n_distractor: usize,
    pub n_conflict: usize,
    pub matched_pairs: Vec<(usize, usize)>,  // (field_idx, catalog_idx)
}

pub fn verify_solution(
    wcs: &TanWcs,
    field_sources: &[DetectedSource],
    index: &Index,
    config: &VerifyConfig,
) -> VerifyResult
```

**Algorithm** (mirrors astrometry.net's verify.c):

1. Project all index stars in the field of view onto pixel coordinates using the candidate WCS
2. Build a KD-tree over projected reference star pixel positions
3. For each field source:
   a. Search for nearest reference star within `match_radius_pix`
   b. If found: compute log-likelihood ratio `log(p_match / p_distractor)`
      - `p_match = 1 / (π · σ²)` (Gaussian positional model)
      - `p_distractor = 1 / (image_area)` (uniform random)
   c. If not found: mark as distractor
4. Sum log-odds across all field sources
5. Early termination: bail if running log-odds drops below `log_odds_bail`
6. Accept if final log-odds exceeds `log_odds_accept`

---

### Step 9 — The Solver Loop `(H)`

**Module:** `src/solver.rs`

**Purpose:** The main blind solver that ties everything together.

```rust
pub struct SolverConfig {
    /// Pixel scale range to search (arcsec/pixel)
    pub scale_range: Option<(f64, f64)>,
    /// RA/Dec hint (radians) and search radius
    pub position_hint: Option<([f64; 2], f64)>,
    /// Maximum number of field stars to use for quad building
    pub max_field_stars: usize,
    /// Code matching tolerance (in code-space L2 distance²)
    pub code_tolerance: f64,
    /// Verification config
    pub verify: VerifyConfig,
}

pub struct Solution {
    pub wcs: TanWcs,
    pub verify: VerifyResult,
    pub field_sources_used: usize,
}

pub fn solve(
    sources: &[DetectedSource],
    indexes: &[Index],
    config: &SolverConfig,
) -> Option<Solution>
```

**Algorithm** (mirrors astrometry.net's solver.c):

```
1. Sort sources by brightness (should already be sorted)
2. Truncate to config.max_field_stars
3. Build field star KD-tree (2D, pixel coords) for inbox checks

4. For each index in indexes:
   a. Compute acceptable quad scale range in pixels
      (from index.scale_range and config.scale_range)

5. For each "new" field star (index N = 2, 3, 4, ...):
   For each pair (A, B) where A < B < N:
     a. Compute pixel distance d(A,B)
     b. Skip if d² outside acceptable scale range
     c. Compute rotation angle θ for A→B baseline
     d. Find candidate C, D stars:
        - Stars with index < N
        - That fall inside the AB circle
        - (Use field KD-tree range search)
     e. Skip if fewer than (DIMQUADS - 2) candidates

     f. For each index:
        i.  Check if AB scale matches this index's range
        ii. For each valid permutation of C, D from candidates:
            - Compute field quad code
            - Enforce invariants
            - Search index.code_tree within config.code_tolerance
            - For each code match:
              · Retrieve reference quad star positions
              · Fit TAN WCS from 4 correspondences
              · Verify against all sources
              · If verify.log_odds > threshold:
                  RETURN Solution
```

**Performance notes:**
- The outer loop processes stars by brightness — brightest first
- This means the most reliable matches are tried first
- The pquad cache (precomputed AB pair data) avoids redundant distance calculations
- Early termination when a good solution is found

---

### Step 10 — SIP Tweaker `(K)`

**Module:** `src/tweak.rs`

**Purpose:** After finding a TAN solution, refine it with SIP polynomial distortion.

```rust
pub fn tweak_solution(
    initial_wcs: &TanWcs,
    field_sources: &[DetectedSource],
    index: &Index,
    sip_order: usize,
    iterations: usize,
) -> Result<SipWcs, FitError>
```

**Algorithm:**
1. Start with the TAN WCS from the solver
2. Project all reference stars in FOV onto pixels
3. Match field sources to projected references (nearest within tolerance)
4. Fit SIP coefficients from matched pairs using `fit_sip()`
5. Re-project references with new SIP, re-match
6. Repeat for `iterations` rounds (typically 3-5 converge)
7. Return refined SipWcs

---

## Testing Strategy

### Unit Tests (per module)

| Module | Test Strategy |
|--------|---------------|
| `geom/tan` | Round-trip pixel↔sky, known WCS solutions, edge cases at poles |
| `geom/sip` | Compare against known SIP headers, distortion symmetry |
| `geom/sphere` | Distance calculations, midpoint, coordinates near poles |
| `kdtree` | Brute-force equivalence, empty tree, single point, degenerate dims |
| `quads` | Rotation/scale/flip invariance, known code values |
| `fitting` | Fit from synthetic correspondences, residual < ε |
| `verify` | Synthetic field with known matches + distractors |

### Integration Tests

| Test | Description |
|------|-------------|
| **Synthetic solve** | Generate a fake star field from catalog, add noise, solve blind |
| **Round-trip** | Build index → generate image from index → solve → verify WCS matches |
| **Scale sweep** | Test solving at various pixel scales (narrow to wide field) |
| **Distortion** | Apply known SIP distortion, verify tweaker recovers it |

### Comparison Tests (optional, via starfield pybridge)

If useful, compare against astrometry.net's own solver on the same inputs using the Python bridge infrastructure from starfield.

---

## Module / File Layout

```
src/
├── lib.rs              — Public API re-exports
├── geom/
│   ├── mod.rs
│   ├── tan.rs          — TanWcs (Step 1.1)
│   ├── sip.rs          — SipWcs (Step 1.2)
│   └── sphere.rs       — Spherical geometry utilities (Step 1.3)
├── extraction.rs       — Star extraction wrapper (Step 2)
├── kdtree.rs           — Generic KD-tree (Step 3)
├── quads.rs            — Quad codec (Step 4)
├── index/
│   ├── mod.rs
│   ├── builder.rs      — Index construction (Step 5)
│   └── store.rs        — Serialization (Step 6)
├── fitting.rs          — WCS fitting (Step 7)
├── verify.rs           — Bayesian verification (Step 8)
├── solver.rs           — Main solver loop (Step 9)
└── tweak.rs            — SIP refinement (Step 10)
```

---

## Crate Dependencies

| Crate | Purpose |
|-------|---------|
| `nalgebra` | SVD, matrix ops for WCS fitting |
| `ndarray` | `Array2<f32>` image type |
| `starfield` | Star catalogs, source extraction, coordinates |
| `serde` + `bincode` | Index serialization |
| `log` | Diagnostics |

---

## Implementation Order (Recommended)

**Phase 1 — Foundation (can be parallelized)**
1. `geom/sphere.rs` — sphere utilities
2. `geom/tan.rs` — TAN projection
3. `kdtree.rs` — KD-tree
4. `quads.rs` — quad codec

**Phase 2 — Infrastructure**
5. `extraction.rs` — starfield wrapper
6. `index/builder.rs` — index construction
7. `index/store.rs` — serialization
8. `fitting.rs` — WCS fitting

**Phase 3 — Solver**
9. `verify.rs` — verification
10. `solver.rs` — main loop

**Phase 4 — Polish**
11. `geom/sip.rs` — SIP distortion type
12. `tweak.rs` — SIP refinement
13. Integration tests with synthetic data

---

## Open Questions / Future Work

- **Index tiling:** For all-sky solving, indexes should be tiled by healpix region. Defer to v2.
- **Parallelism:** The solver loop over indexes is embarrassingly parallel. Add rayon later.
- **GPU:** KD-tree range search could benefit from GPU. Way out of scope for v1.
- **Proper motion:** For high-precision work, stars need epoch propagation. Starfield handles this — integrate when needed.
- **Multi-extension indexes:** astrometry.net packs multiple scales into one file. Start with one-scale-per-file.
