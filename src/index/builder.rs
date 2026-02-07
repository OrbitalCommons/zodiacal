use std::collections::{BinaryHeap, HashMap, HashSet};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
use rayon::prelude::*;
use starfield::catalogs::{StarCatalog, StarData};

use crate::geom::sphere::{angular_distance, radec_to_xyz, star_midpoint};
use crate::healpix;
use crate::kdtree::KdTree;
use crate::quads::{Code, DIMCODES, DIMQUADS, Quad, compute_canonical_code};

use super::{Index, IndexStar};

/// Configuration for building an index.
pub struct IndexBuilderConfig {
    /// Minimum angular size of quads (radians).
    pub scale_lower: f64,
    /// Maximum angular size of quads (radians).
    pub scale_upper: f64,
    /// Maximum number of stars to use (brightest first).
    /// Used by `build_index` for direct star lists.
    pub max_stars: usize,
    /// Maximum number of quads to generate.
    pub max_quads: usize,
}

/// Configuration for the HEALPix-guided catalog index builder.
pub struct CatalogBuilderConfig {
    /// Minimum angular size of quads (radians).
    pub scale_lower: f64,
    /// Maximum angular size of quads (radians).
    pub scale_upper: f64,
    /// Maximum brightest stars to keep per HEALPix cell during uniformization.
    pub max_stars_per_cell: usize,
    /// Maximum total quads to generate. If None, auto-computed as
    /// `passes * npix(quad_depth)` to ensure every cell gets adequate coverage.
    pub max_quads: Option<usize>,
    /// HEALPix depth for star uniformization (auto-computed if None).
    pub uniformize_depth: Option<u8>,
    /// HEALPix depth for quad building cells (auto-computed if None).
    pub quad_depth: Option<u8>,
    /// Number of quad-building passes per cell.
    pub passes: usize,
    /// Maximum times a star can appear in quads (limits over-representation).
    pub max_reuse: usize,
}

impl Default for CatalogBuilderConfig {
    fn default() -> Self {
        Self {
            scale_lower: (30.0 / 3600.0_f64).to_radians(),
            scale_upper: (1800.0 / 3600.0_f64).to_radians(),
            max_stars_per_cell: 10,
            max_quads: None,
            uniformize_depth: None,
            quad_depth: None,
            passes: 16,
            max_reuse: 8,
        }
    }
}

impl CatalogBuilderConfig {
    /// Compute the effective uniformization depth (auto or explicit).
    fn effective_uniformize_depth(&self) -> u8 {
        self.uniformize_depth
            .unwrap_or_else(|| healpix::depth_for_scale(self.scale_upper * 2.0))
    }

    /// Compute the effective quad-building depth (auto or explicit).
    pub fn effective_quad_depth(&self) -> u8 {
        self.quad_depth
            .unwrap_or_else(|| healpix::depth_for_scale(self.scale_upper * 2.0))
    }

    /// Compute recommended max_quads based on HEALPix geometry.
    ///
    /// Returns `passes * npix(quad_depth)` — enough quads to fill
    /// every cell with the requested number of passes.
    pub fn recommended_max_quads(&self) -> usize {
        let n_cells = healpix::npix(self.effective_quad_depth()) as usize;
        self.passes * n_cells
    }

    /// Resolve the effective max_quads (explicit or auto-computed).
    pub fn effective_max_quads(&self) -> usize {
        self.max_quads
            .unwrap_or_else(|| self.recommended_max_quads())
    }
}

/// Convert an angular distance (radians) to squared chord distance on the unit sphere.
fn angular_to_chord_sq(theta: f64) -> f64 {
    2.0 * (1.0 - theta.cos())
}

/// Given 4 star positions, find the pair with maximum angular distance
/// and return (star_xyz, star_ids) reordered so that pair is [0],[1].
fn canonical_quad_order(
    star_xyz: &[[f64; 3]; DIMQUADS],
    star_ids: [usize; DIMQUADS],
) -> ([[f64; 3]; DIMQUADS], [usize; DIMQUADS]) {
    let mut best_pair = (0, 1);
    let mut best_dist = 0.0f64;

    for i in 0..DIMQUADS {
        for j in (i + 1)..DIMQUADS {
            let d = angular_distance(star_xyz[i], star_xyz[j]);
            if d > best_dist {
                best_dist = d;
                best_pair = (i, j);
            }
        }
    }

    let (ai, bi) = best_pair;
    let mut others: Vec<usize> = (0..DIMQUADS).filter(|&i| i != ai && i != bi).collect();
    others.sort_by_key(|&i| star_ids[i]);

    let order = [ai, bi, others[0], others[1]];
    let new_xyz: [[f64; 3]; DIMQUADS] = std::array::from_fn(|i| star_xyz[order[i]]);
    let new_ids: [usize; DIMQUADS] = std::array::from_fn(|i| star_ids[order[i]]);
    (new_xyz, new_ids)
}

fn spinner_style() -> ProgressStyle {
    ProgressStyle::with_template("{spinner:.cyan} {prefix:.bold} {wide_msg}")
        .unwrap()
        .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏✓")
}

fn bar_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "{spinner:.cyan} {prefix:.bold} [{bar:40.cyan/dim}] {pos}/{len} {per_sec} {eta} {wide_msg}",
    )
    .unwrap()
    .progress_chars("━╸─")
    .tick_chars("⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏✓")
}

fn finish_spinner(pb: &ProgressBar, msg: &str) {
    pb.set_style(ProgressStyle::with_template("{prefix:.bold.green} {wide_msg}").unwrap());
    pb.finish_with_message(msg.to_string());
}

fn finish_bar(pb: &ProgressBar, msg: &str) {
    pb.set_style(ProgressStyle::with_template("{prefix:.bold.green} {wide_msg}").unwrap());
    pb.finish_with_message(msg.to_string());
}

/// Build an index from a list of stars, with parallel quad generation and progress bars.
///
/// Stars should be provided as `(catalog_id, ra_radians, dec_radians, magnitude)`.
/// They will be sorted by magnitude (ascending = brightest first) and truncated
/// to `max_stars`.
pub fn build_index(stars: &[(u64, f64, f64, f64)], config: &IndexBuilderConfig) -> Index {
    let mp = MultiProgress::new();

    let pb_sort = mp.add(ProgressBar::new_spinner());
    pb_sort.set_style(spinner_style());
    pb_sort.set_prefix("✦ Stars");
    pb_sort.set_message("sorting by brightness...");
    pb_sort.enable_steady_tick(std::time::Duration::from_millis(80));

    let mut sorted: Vec<(u64, f64, f64, f64)> = stars.to_vec();
    sorted.sort_by(|a, b| a.3.partial_cmp(&b.3).unwrap_or(std::cmp::Ordering::Equal));
    sorted.truncate(config.max_stars);

    let index_stars: Vec<IndexStar> = sorted
        .iter()
        .map(|&(id, ra, dec, mag)| IndexStar {
            catalog_id: id,
            ra,
            dec,
            mag,
        })
        .collect();

    finish_spinner(
        &pb_sort,
        &format!("✓ {} stars selected (brightest first)", index_stars.len()),
    );

    build_index_from_stars(
        index_stars,
        config.scale_lower,
        config.scale_upper,
        config.max_quads,
        &mp,
    )
}

/// Core index builder that takes pre-selected IndexStars.
fn build_index_from_stars(
    index_stars: Vec<IndexStar>,
    scale_lower: f64,
    scale_upper: f64,
    max_quads: usize,
    mp: &MultiProgress,
) -> Index {
    // --- Build star KD-tree ---
    let pb_tree = mp.add(ProgressBar::new_spinner());
    pb_tree.set_style(spinner_style());
    pb_tree.set_prefix("✦ Star tree");
    pb_tree.set_message("building 3D KD-tree...");
    pb_tree.enable_steady_tick(std::time::Duration::from_millis(80));

    let xyzs: Vec<[f64; 3]> = index_stars
        .iter()
        .map(|s| radec_to_xyz(s.ra, s.dec))
        .collect();

    let star_points = xyzs.clone();
    let star_indices: Vec<usize> = (0..xyzs.len()).collect();
    let star_tree = KdTree::<3>::build(star_points, star_indices);

    finish_spinner(&pb_tree, &format!("✓ KD-tree built ({} nodes)", xyzs.len()));

    if xyzs.len() < DIMQUADS {
        let code_tree = KdTree::<{ DIMCODES }>::build(vec![], vec![]);
        return Index {
            star_tree,
            stars: index_stars,
            code_tree,
            quads: vec![],
            scale_lower,
            scale_upper,
        };
    }

    // --- Parallel quad generation ---
    let n_stars = xyzs.len();
    let pb_quads = mp.add(ProgressBar::new(n_stars as u64));
    pb_quads.set_style(bar_style());
    pb_quads.set_prefix("✦ Quads");
    pb_quads.set_message("generating quads...");
    pb_quads.enable_steady_tick(std::time::Duration::from_millis(80));

    let quad_count = AtomicUsize::new(0);
    let done = AtomicBool::new(false);
    let chord_sq_upper = angular_to_chord_sq(scale_upper);

    let batches: Vec<Vec<([usize; DIMQUADS], Quad, Code)>> = (0..n_stars)
        .into_par_iter()
        .map(|a_idx| {
            if done.load(Ordering::Relaxed) {
                pb_quads.inc(1);
                return Vec::new();
            }

            let a_xyz = xyzs[a_idx];
            let neighbors = star_tree.range_search(&a_xyz, chord_sq_upper);
            let mut local = Vec::new();
            let mut local_seen: HashSet<[usize; DIMQUADS]> = HashSet::new();

            for nb in &neighbors {
                if done.load(Ordering::Relaxed) {
                    break;
                }

                let b_idx = nb.index;
                if b_idx <= a_idx {
                    continue;
                }

                let b_xyz = xyzs[b_idx];
                let ab_dist = angular_distance(a_xyz, b_xyz);
                if ab_dist < scale_lower || ab_dist > scale_upper {
                    continue;
                }

                let mid = star_midpoint(a_xyz, b_xyz);
                let cd_radius_sq = angular_to_chord_sq(ab_dist);
                let candidates = star_tree.range_search(&mid, cd_radius_sq);

                let candidate_ids: Vec<usize> = candidates
                    .iter()
                    .map(|c| c.index)
                    .filter(|&idx| idx != a_idx && idx != b_idx)
                    .collect();

                for (ci, &c_idx) in candidate_ids.iter().enumerate() {
                    for &d_idx in &candidate_ids[(ci + 1)..] {
                        let mut key = [a_idx, b_idx, c_idx, d_idx];
                        key.sort();

                        if !local_seen.insert(key) {
                            continue;
                        }

                        let raw_xyz = [a_xyz, b_xyz, xyzs[c_idx], xyzs[d_idx]];
                        let raw_ids = [a_idx, b_idx, c_idx, d_idx];
                        let (ordered_xyz, ordered_ids) = canonical_quad_order(&raw_xyz, raw_ids);
                        let (code, canonical_ids, _) =
                            compute_canonical_code(&ordered_xyz, ordered_ids);

                        local.push((
                            key,
                            Quad {
                                star_ids: canonical_ids,
                            },
                            code,
                        ));
                    }
                }
            }

            let batch_size = local.len();
            let prev = quad_count.fetch_add(batch_size, Ordering::Relaxed);
            if prev + batch_size >= max_quads {
                done.store(true, Ordering::Relaxed);
            }

            pb_quads.inc(1);
            local
        })
        .collect();

    // --- Merge & deduplicate ---
    let pb_dedup = mp.add(ProgressBar::new_spinner());
    pb_dedup.set_style(spinner_style());
    pb_dedup.set_prefix("✦ Dedup");
    pb_dedup.set_message("merging and deduplicating quads...");
    pb_dedup.enable_steady_tick(std::time::Duration::from_millis(80));

    let total_raw: usize = batches.iter().map(|b| b.len()).sum();
    let mut seen: HashSet<[usize; DIMQUADS]> = HashSet::with_capacity(total_raw);
    let mut quads: Vec<Quad> = Vec::with_capacity(total_raw.min(max_quads));
    let mut codes: Vec<Code> = Vec::with_capacity(total_raw.min(max_quads));

    for batch in batches {
        for (key, quad, code) in batch {
            if seen.insert(key) {
                quads.push(quad);
                codes.push(code);
                if quads.len() >= max_quads {
                    break;
                }
            }
        }
        if quads.len() >= max_quads {
            break;
        }
    }

    finish_bar(
        &pb_quads,
        &format!("✓ {} stars scanned ({} raw candidates)", n_stars, total_raw),
    );
    finish_spinner(
        &pb_dedup,
        &format!("✓ {} unique quads (from {} raw)", quads.len(), total_raw),
    );

    // --- Build code KD-tree ---
    let pb_code_tree = mp.add(ProgressBar::new_spinner());
    pb_code_tree.set_style(spinner_style());
    pb_code_tree.set_prefix("✦ Code tree");
    pb_code_tree.set_message(format!(
        "building {}D KD-tree over {} codes...",
        DIMCODES,
        codes.len()
    ));
    pb_code_tree.enable_steady_tick(std::time::Duration::from_millis(80));

    let code_indices: Vec<usize> = (0..codes.len()).collect();
    let code_tree = KdTree::<{ DIMCODES }>::build(codes, code_indices);

    finish_spinner(
        &pb_code_tree,
        &format!("✓ code KD-tree built ({} entries)", quads.len()),
    );

    Index {
        star_tree,
        stars: index_stars,
        code_tree,
        quads,
        scale_lower,
        scale_upper,
    }
}

/// A star tuple wrapped for use in a max-heap keyed by magnitude.
///
/// We want to keep the *brightest* (lowest magnitude) stars, so the heap
/// should evict the *faintest* (highest magnitude). A max-heap with magnitude
/// as the key naturally puts the faintest star at the top for eviction.
#[derive(PartialEq)]
struct HeapStar {
    mag: f64,
    id: u64,
    ra: f64,
    dec: f64,
}

impl Eq for HeapStar {}

impl PartialOrd for HeapStar {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for HeapStar {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.mag
            .partial_cmp(&other.mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// Build an index from a starfield `StarCatalog` using HEALPix-guided uniformization.
///
/// Streams through the catalog, keeping the N brightest stars per HEALPix cell
/// to ensure spatially uniform coverage. Then builds quads from the uniformized set.
pub fn build_index_from_catalog(
    catalog: &impl StarCatalog,
    config: &CatalogBuilderConfig,
) -> Index {
    let mp = MultiProgress::new();

    let uni_depth = config.effective_uniformize_depth();
    let n_cells = healpix::npix(uni_depth);
    let max_per_cell = config.max_stars_per_cell;

    // --- Phase 1: HEALPix uniformization ---
    let pb_uni = mp.add(ProgressBar::new_spinner());
    pb_uni.set_style(spinner_style());
    pb_uni.set_prefix("✦ Uniformize");
    pb_uni.set_message(format!(
        "streaming catalog (depth {}, {} cells, {} per cell)...",
        uni_depth, n_cells, max_per_cell
    ));
    pb_uni.enable_steady_tick(std::time::Duration::from_millis(80));

    // Per-cell heaps: keep brightest max_per_cell stars per HEALPix pixel
    let mut cell_heaps: HashMap<u64, BinaryHeap<HeapStar>> = HashMap::new();
    let mut total_streamed: u64 = 0;

    for s in catalog.star_data() {
        total_streamed += 1;
        if total_streamed.is_multiple_of(1_000_000) {
            pb_uni.set_message(format!(
                "streamed {}M stars ({} cells populated)...",
                total_streamed / 1_000_000,
                cell_heaps.len()
            ));
        }

        let pixel = healpix::lon_lat_to_nested(s.position.ra, s.position.dec, uni_depth);
        let heap = cell_heaps
            .entry(pixel)
            .or_insert_with(|| BinaryHeap::with_capacity(max_per_cell + 1));

        let star = HeapStar {
            mag: s.magnitude,
            id: s.id,
            ra: s.position.ra,
            dec: s.position.dec,
        };

        if heap.len() < max_per_cell {
            heap.push(star);
        } else if let Some(faintest) = heap.peek()
            && star.mag < faintest.mag
        {
            heap.pop();
            heap.push(star);
        }
    }

    // Flatten all heaps into a single sorted star list
    let mut stars: Vec<IndexStar> = Vec::new();
    for (_, heap) in cell_heaps {
        for hs in heap {
            stars.push(IndexStar {
                catalog_id: hs.id,
                ra: hs.ra,
                dec: hs.dec,
                mag: hs.mag,
            });
        }
    }
    stars.sort_by(|a, b| {
        a.mag
            .partial_cmp(&b.mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    finish_spinner(
        &pb_uni,
        &format!(
            "✓ {} stars from {}M catalog (depth {}, {} per cell)",
            stars.len(),
            total_streamed / 1_000_000,
            uni_depth,
            max_per_cell
        ),
    );

    // --- Phase 2: Build star KD-tree ---
    let pb_tree = mp.add(ProgressBar::new_spinner());
    pb_tree.set_style(spinner_style());
    pb_tree.set_prefix("✦ Star tree");
    pb_tree.set_message("building 3D KD-tree...");
    pb_tree.enable_steady_tick(std::time::Duration::from_millis(80));

    let xyzs: Vec<[f64; 3]> = stars.iter().map(|s| radec_to_xyz(s.ra, s.dec)).collect();
    let star_points = xyzs.clone();
    let star_indices: Vec<usize> = (0..xyzs.len()).collect();
    let star_tree = KdTree::<3>::build(star_points, star_indices);

    finish_spinner(&pb_tree, &format!("✓ KD-tree built ({} nodes)", xyzs.len()));

    // --- Phase 3: Per-cell quad building ---
    let quad_depth = config.effective_quad_depth();
    let n_quad_cells = healpix::npix(quad_depth);
    let max_quads = config.effective_max_quads();
    let quads_per_cell = (max_quads as u64 / n_quad_cells).max(1) as usize;

    let pb_quads = mp.add(ProgressBar::new(n_quad_cells));
    pb_quads.set_style(bar_style());
    pb_quads.set_prefix("✦ Quads");
    pb_quads.set_message(format!(
        "per-cell quad building (depth {}, {} cells, {} per cell)...",
        quad_depth, n_quad_cells, quads_per_cell
    ));
    pb_quads.enable_steady_tick(std::time::Duration::from_millis(80));

    // Assign each star to its quad-depth cell.
    let mut cell_stars: HashMap<u64, Vec<usize>> = HashMap::new();
    for (idx, star) in stars.iter().enumerate() {
        let cell = healpix::lon_lat_to_nested(star.ra, star.dec, quad_depth);
        cell_stars.entry(cell).or_default().push(idx);
    }

    // Track how many times each star is used in quads (for max_reuse).
    let star_use_count: Vec<AtomicUsize> = (0..stars.len()).map(|_| AtomicUsize::new(0)).collect();

    // Process cells in parallel.
    let cell_ids: Vec<u64> = (0..n_quad_cells).collect();
    let all_cell_quads: Vec<Vec<(Quad, [f64; DIMCODES])>> = cell_ids
        .par_iter()
        .map(|&cell_id| {
            pb_quads.inc(1);

            // Gather star indices from this cell + neighbors.
            let mut neighbor_cells = healpix::neighbours(cell_id, quad_depth);
            neighbor_cells.push(cell_id);

            let mut local_star_indices: Vec<usize> = Vec::new();
            for &nc in &neighbor_cells {
                if let Some(indices) = cell_stars.get(&nc) {
                    local_star_indices.extend_from_slice(indices);
                }
            }
            local_star_indices.sort_unstable();
            local_star_indices.dedup();

            if local_star_indices.len() < DIMQUADS {
                return Vec::new();
            }

            let mut cell_quads: Vec<(Quad, [f64; DIMCODES])> = Vec::new();
            let mut seen: HashSet<[usize; DIMQUADS]> = HashSet::new();

            // Try to build quads from pairs (A, B) where A is in this cell.
            let this_cell_stars: Vec<usize> = cell_stars.get(&cell_id).cloned().unwrap_or_default();

            'outer: for &a_idx in &this_cell_stars {
                if cell_quads.len() >= quads_per_cell {
                    break;
                }

                let a_xyz = xyzs[a_idx];

                for &b_idx in &local_star_indices {
                    if b_idx == a_idx {
                        continue;
                    }

                    let b_xyz = xyzs[b_idx];
                    let ab_dist = angular_distance(a_xyz, b_xyz);
                    if ab_dist < config.scale_lower || ab_dist > config.scale_upper {
                        continue;
                    }

                    let mid = star_midpoint(a_xyz, b_xyz);
                    let cd_radius_sq = angular_to_chord_sq(ab_dist);
                    let candidates: Vec<usize> = local_star_indices
                        .iter()
                        .copied()
                        .filter(|&idx| {
                            idx != a_idx && idx != b_idx && {
                                let d = xyzs[idx];
                                let dx = d[0] - mid[0];
                                let dy = d[1] - mid[1];
                                let dz = d[2] - mid[2];
                                dx * dx + dy * dy + dz * dz < cd_radius_sq
                            }
                        })
                        .collect();

                    for ci in 0..candidates.len() {
                        for di in (ci + 1)..candidates.len() {
                            let c_idx = candidates[ci];
                            let d_idx = candidates[di];

                            let mut key = [a_idx, b_idx, c_idx, d_idx];
                            key.sort();
                            if !seen.insert(key) {
                                continue;
                            }

                            // Check reuse limits.
                            let can_use = [a_idx, b_idx, c_idx, d_idx].iter().all(|&idx| {
                                star_use_count[idx].load(Ordering::Relaxed) < config.max_reuse
                            });
                            if !can_use {
                                continue;
                            }

                            let raw_xyz = [a_xyz, b_xyz, xyzs[c_idx], xyzs[d_idx]];
                            let raw_ids = [a_idx, b_idx, c_idx, d_idx];
                            let (ordered_xyz, ordered_ids) =
                                canonical_quad_order(&raw_xyz, raw_ids);
                            let (code, canonical_ids, _) =
                                compute_canonical_code(&ordered_xyz, ordered_ids);

                            for &idx in &canonical_ids {
                                star_use_count[idx].fetch_add(1, Ordering::Relaxed);
                            }

                            cell_quads.push((
                                Quad {
                                    star_ids: canonical_ids,
                                },
                                code,
                            ));

                            if cell_quads.len() >= quads_per_cell {
                                break 'outer;
                            }
                        }
                    }
                }
            }

            cell_quads
        })
        .collect();

    // Merge all cell quads.
    let mut quads: Vec<Quad> = Vec::new();
    let mut codes: Vec<[f64; DIMCODES]> = Vec::new();
    let mut global_seen: HashSet<[usize; DIMQUADS]> = HashSet::new();

    for cell_batch in all_cell_quads {
        for (quad, code) in cell_batch {
            let mut key = quad.star_ids;
            key.sort();
            if global_seen.insert(key) {
                quads.push(quad);
                codes.push(code);
            }
        }
    }

    finish_bar(
        &pb_quads,
        &format!("✓ {} unique quads from {} cells", quads.len(), n_quad_cells),
    );

    // --- Phase 4: Build code KD-tree ---
    let pb_code_tree = mp.add(ProgressBar::new_spinner());
    pb_code_tree.set_style(spinner_style());
    pb_code_tree.set_prefix("✦ Code tree");
    pb_code_tree.set_message(format!(
        "building {}D KD-tree over {} codes...",
        DIMCODES,
        codes.len()
    ));
    pb_code_tree.enable_steady_tick(std::time::Duration::from_millis(80));

    let code_indices: Vec<usize> = (0..codes.len()).collect();
    let code_tree = KdTree::<{ DIMCODES }>::build(codes, code_indices);

    finish_spinner(
        &pb_code_tree,
        &format!("✓ code KD-tree built ({} entries)", quads.len()),
    );

    Index {
        star_tree,
        stars,
        code_tree,
        quads,
        scale_lower: config.scale_lower,
        scale_upper: config.scale_upper,
    }
}

/// Build an index from pre-collected `StarData` entries.
pub fn build_index_from_star_data(stars: &[StarData], config: &IndexBuilderConfig) -> Index {
    let tuples: Vec<(u64, f64, f64, f64)> = stars
        .iter()
        .map(|s| (s.id, s.position.ra, s.position.dec, s.magnitude))
        .collect();
    build_index(&tuples, config)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::sphere::radec_to_xyz;
    use std::collections::HashSet;

    fn make_small_catalog() -> Vec<(u64, f64, f64, f64)> {
        // 10 stars in a small patch of sky around RA=1.0, Dec=0.5
        let base_ra = 1.0;
        let base_dec = 0.5;
        let offsets = [
            (0.00, 0.00),
            (0.01, 0.00),
            (0.00, 0.01),
            (0.01, 0.01),
            (0.005, 0.005),
            (0.002, 0.008),
            (0.008, 0.002),
            (0.003, 0.003),
            (0.007, 0.007),
            (0.004, 0.009),
        ];
        offsets
            .iter()
            .enumerate()
            .map(|(i, &(dra, ddec))| (i as u64, base_ra + dra, base_dec + ddec, i as f64))
            .collect()
    }

    #[test]
    fn small_catalog_builds_index() {
        let catalog = make_small_catalog();
        let config = IndexBuilderConfig {
            scale_lower: 0.001,
            scale_upper: 0.02,
            max_stars: 10,
            max_quads: 1000,
        };

        let index = build_index(&catalog, &config);

        assert_eq!(index.stars.len(), 10);
        assert!(
            !index.quads.is_empty(),
            "expected some quads to be generated"
        );
        assert_eq!(index.star_tree.len(), 10);
        assert_eq!(index.code_tree.len(), index.quads.len());
    }

    #[test]
    fn scale_filtering() {
        let catalog = make_small_catalog();
        let scale_lower = 0.005;
        let scale_upper = 0.008;
        let config = IndexBuilderConfig {
            scale_lower,
            scale_upper,
            max_stars: 10,
            max_quads: 1000,
        };

        let index = build_index(&catalog, &config);

        for quad in &index.quads {
            let a = &index.stars[quad.star_ids[0]];
            let b = &index.stars[quad.star_ids[1]];
            let a_xyz = radec_to_xyz(a.ra, a.dec);
            let b_xyz = radec_to_xyz(b.ra, b.dec);
            let dist = angular_distance(a_xyz, b_xyz);
            assert!(
                dist <= scale_upper * 3.0,
                "quad backbone distance {dist} unexpectedly large (scale_upper = {scale_upper})"
            );
        }
    }

    #[test]
    fn no_duplicate_quads() {
        let catalog = make_small_catalog();
        let config = IndexBuilderConfig {
            scale_lower: 0.001,
            scale_upper: 0.02,
            max_stars: 10,
            max_quads: 1000,
        };

        let index = build_index(&catalog, &config);

        let mut seen: HashSet<[usize; DIMQUADS]> = HashSet::new();
        for quad in &index.quads {
            let mut key = quad.star_ids;
            key.sort();
            assert!(
                seen.insert(key),
                "duplicate quad found: {:?}",
                quad.star_ids
            );
        }
    }

    #[test]
    fn empty_catalog() {
        let config = IndexBuilderConfig {
            scale_lower: 0.001,
            scale_upper: 0.02,
            max_stars: 10,
            max_quads: 1000,
        };

        let index = build_index(&[], &config);
        assert!(index.stars.is_empty());
        assert!(index.quads.is_empty());
        assert!(index.star_tree.is_empty());
        assert!(index.code_tree.is_empty());
    }

    #[test]
    fn single_star_catalog() {
        let catalog = vec![(1u64, 1.0, 0.5, 3.0)];
        let config = IndexBuilderConfig {
            scale_lower: 0.001,
            scale_upper: 0.02,
            max_stars: 10,
            max_quads: 1000,
        };

        let index = build_index(&catalog, &config);
        assert_eq!(index.stars.len(), 1);
        assert!(index.quads.is_empty());
    }

    #[test]
    fn max_quads_limit() {
        let catalog = make_small_catalog();
        let max_quads = 3;
        let config = IndexBuilderConfig {
            scale_lower: 0.001,
            scale_upper: 0.02,
            max_stars: 10,
            max_quads,
        };

        let index = build_index(&catalog, &config);
        assert!(
            index.quads.len() <= max_quads,
            "generated {} quads, expected at most {max_quads}",
            index.quads.len()
        );
    }

    #[test]
    fn code_tree_search_finds_known_quad() {
        let catalog = make_small_catalog();
        let config = IndexBuilderConfig {
            scale_lower: 0.001,
            scale_upper: 0.02,
            max_stars: 10,
            max_quads: 1000,
        };

        let index = build_index(&catalog, &config);
        assert!(!index.quads.is_empty());

        let quad = &index.quads[0];
        let star_xyz: [[f64; 3]; DIMQUADS] = std::array::from_fn(|i| {
            let s = &index.stars[quad.star_ids[i]];
            radec_to_xyz(s.ra, s.dec)
        });
        let (code, _, _) = compute_canonical_code(&star_xyz, quad.star_ids);

        let results = index.code_tree.range_search(&code, 1e-10);
        assert!(
            !results.is_empty(),
            "code tree search should find the known quad"
        );

        let found_quad_idx = results[0].index;
        let mut found_ids = index.quads[found_quad_idx].star_ids;
        found_ids.sort();
        let mut expected_ids = quad.star_ids;
        expected_ids.sort();
        assert_eq!(found_ids, expected_ids);
    }

    #[test]
    fn stars_sorted_by_brightness() {
        let catalog = vec![
            (1u64, 1.0, 0.5, 5.0),
            (2, 1.001, 0.5, 1.0),
            (3, 1.002, 0.5, 3.0),
            (4, 1.003, 0.5, 0.5),
        ];
        let config = IndexBuilderConfig {
            scale_lower: 0.0001,
            scale_upper: 0.01,
            max_stars: 10,
            max_quads: 100,
        };

        let index = build_index(&catalog, &config);
        assert_eq!(index.stars[0].catalog_id, 4); // mag 0.5 (brightest)
        assert_eq!(index.stars[1].catalog_id, 2); // mag 1.0
        assert_eq!(index.stars[2].catalog_id, 3); // mag 3.0
        assert_eq!(index.stars[3].catalog_id, 1); // mag 5.0
    }

    #[test]
    fn max_stars_truncation() {
        let catalog = make_small_catalog();
        let config = IndexBuilderConfig {
            scale_lower: 0.001,
            scale_upper: 0.02,
            max_stars: 5,
            max_quads: 1000,
        };

        let index = build_index(&catalog, &config);
        assert_eq!(index.stars.len(), 5);
        assert_eq!(index.star_tree.len(), 5);
    }
}
