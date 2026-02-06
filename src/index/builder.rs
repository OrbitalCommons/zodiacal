use std::collections::HashSet;

use starfield::catalogs::{StarCatalog, StarData};

use crate::geom::sphere::{angular_distance, radec_to_xyz, star_midpoint};
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
    pub max_stars: usize,
    /// Maximum number of quads to generate.
    pub max_quads: usize,
}

/// Convert an angular distance (radians) to squared chord distance on the unit sphere.
///
/// For two unit vectors separated by angle theta, the chord distance squared is
/// `2 * (1 - cos(theta))`, which equals the squared L2 distance between the vectors.
fn angular_to_chord_sq(theta: f64) -> f64 {
    2.0 * (1.0 - theta.cos())
}

/// Build an index from a list of stars.
///
/// Stars should be provided as `(catalog_id, ra_radians, dec_radians, magnitude)`.
/// They will be sorted by magnitude (ascending = brightest first) and truncated
/// to `max_stars`.
pub fn build_index(stars: &[(u64, f64, f64, f64)], config: &IndexBuilderConfig) -> Index {
    // Step 1: Sort by magnitude (ascending = brightest first), take first max_stars
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

    // Step 2: Convert to 3D unit vectors, build star KD-tree
    let xyzs: Vec<[f64; 3]> = index_stars
        .iter()
        .map(|s| radec_to_xyz(s.ra, s.dec))
        .collect();

    let star_points = xyzs.clone();
    let star_indices: Vec<usize> = (0..xyzs.len()).collect();
    let star_tree = KdTree::<3>::build(star_points, star_indices);

    if xyzs.len() < DIMQUADS {
        let code_tree = KdTree::<{ DIMCODES }>::build(vec![], vec![]);
        return Index {
            star_tree,
            stars: index_stars,
            code_tree,
            quads: vec![],
            scale_lower: config.scale_lower,
            scale_upper: config.scale_upper,
        };
    }

    // Step 3: Generate quads
    let chord_sq_upper = angular_to_chord_sq(config.scale_upper);
    let mut quads: Vec<Quad> = Vec::new();
    let mut codes: Vec<Code> = Vec::new();
    let mut seen: HashSet<[usize; DIMQUADS]> = HashSet::new();

    'outer: for a_idx in 0..xyzs.len() {
        let a_xyz = xyzs[a_idx];

        // Find all neighbors of A within scale_upper
        let neighbors = star_tree.range_search(&a_xyz, chord_sq_upper);

        for nb in &neighbors {
            let b_idx = nb.index;
            if b_idx <= a_idx {
                continue;
            }

            let b_xyz = xyzs[b_idx];
            let ab_dist = angular_distance(a_xyz, b_xyz);
            if ab_dist < config.scale_lower || ab_dist > config.scale_upper {
                continue;
            }

            // Find C, D candidates: stars near the midpoint of A,B within distance ab_dist
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
                    if !seen.insert(key) {
                        continue;
                    }

                    let star_xyz = [a_xyz, b_xyz, xyzs[c_idx], xyzs[d_idx]];
                    let star_ids = [a_idx, b_idx, c_idx, d_idx];
                    let (code, canonical_ids, _) = compute_canonical_code(&star_xyz, star_ids);

                    quads.push(Quad {
                        star_ids: canonical_ids,
                    });
                    codes.push(code);

                    if quads.len() >= config.max_quads {
                        break 'outer;
                    }
                }
            }
        }
    }

    // Step 4: Build code KD-tree
    let code_indices: Vec<usize> = (0..codes.len()).collect();
    let code_tree = KdTree::<{ DIMCODES }>::build(codes, code_indices);

    // Step 5: Return Index
    Index {
        star_tree,
        stars: index_stars,
        code_tree,
        quads,
        scale_lower: config.scale_lower,
        scale_upper: config.scale_upper,
    }
}

/// Build an index from a starfield `StarCatalog`.
///
/// Extracts stars via the `star_data()` iterator. RA/Dec from `StarData.position`
/// are in radians (matching our internal convention).
pub fn build_index_from_catalog(catalog: &impl StarCatalog, config: &IndexBuilderConfig) -> Index {
    let stars: Vec<(u64, f64, f64, f64)> = catalog
        .star_data()
        .map(|s| (s.id, s.position.ra, s.position.dec, s.magnitude))
        .collect();
    build_index(&stars, config)
}

/// Build an index from pre-collected `StarData` entries.
///
/// Useful when you've already filtered or transformed catalog data.
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
                dist >= scale_lower - 1e-10 && dist <= scale_upper + 1e-10,
                "quad backbone distance {dist} outside [{scale_lower}, {scale_upper}]"
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

        // Recompute the code for the first quad and search for it
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
