//! Generic N-dimensional KD-tree for fast spatial search.
//!
//! Uses const generics so the compiler generates specialized code for each
//! dimensionality (3D star positions on the unit sphere, 4D quad codes, etc.).

/// Result of a spatial search: original index and squared Euclidean distance.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub index: usize,
    pub dist_sq: f64,
}

/// Internal node representation stored in a flat array.
#[derive(Debug, Clone)]
enum Node {
    /// Interior node: split dimension, split value, left child index, right child index.
    Split {
        dim: usize,
        value: f64,
        left: usize,
        right: usize,
    },
    /// Leaf node: range [start..end) into the points/indices arrays.
    Leaf { start: usize, end: usize },
}

/// Maximum number of points in a leaf node before we split.
const LEAF_SIZE: usize = 16;

/// A KD-tree for fast range and nearest-neighbor search in N dimensions.
pub struct KdTree<const DIM: usize> {
    nodes: Vec<Node>,
    points: Vec<[f64; DIM]>,
    indices: Vec<usize>,
}

impl<const DIM: usize> KdTree<DIM> {
    /// Build a KD-tree from points. `indices` maps each point to its original ID.
    pub fn build(points: Vec<[f64; DIM]>, indices: Vec<usize>) -> Self {
        assert_eq!(points.len(), indices.len());

        if points.is_empty() {
            return KdTree {
                nodes: Vec::new(),
                points,
                indices,
            };
        }

        let n = points.len();
        let mut tree = KdTree {
            nodes: Vec::new(),
            points,
            indices,
        };

        let mut order: Vec<usize> = (0..n).collect();
        tree.build_recursive(&mut order, 0, n);

        let old_points = tree.points.clone();
        let old_indices = tree.indices.clone();
        for (new_pos, &old_pos) in order.iter().enumerate() {
            tree.points[new_pos] = old_points[old_pos];
            tree.indices[new_pos] = old_indices[old_pos];
        }

        tree
    }

    fn build_recursive(&mut self, order: &mut [usize], start: usize, end: usize) -> usize {
        let count = end - start;

        if count <= LEAF_SIZE {
            let node_idx = self.nodes.len();
            self.nodes.push(Node::Leaf { start, end });
            return node_idx;
        }

        let split_dim = self.pick_split_dim(&order[start..end]);

        let median_pos = start + count / 2;
        self.nth_element(order, start, end, median_pos, split_dim);
        let split_value = self.points[order[median_pos]][split_dim];

        let node_idx = self.nodes.len();
        self.nodes.push(Node::Leaf { start: 0, end: 0 });

        let left = self.build_recursive(order, start, median_pos);
        let right = self.build_recursive(order, median_pos, end);

        self.nodes[node_idx] = Node::Split {
            dim: split_dim,
            value: split_value,
            left,
            right,
        };

        node_idx
    }

    fn pick_split_dim(&self, order: &[usize]) -> usize {
        let mut best_dim = 0;
        let mut best_spread = f64::NEG_INFINITY;

        for d in 0..DIM {
            let mut lo = f64::INFINITY;
            let mut hi = f64::NEG_INFINITY;
            for &idx in order {
                let v = self.points[idx][d];
                if v < lo {
                    lo = v;
                }
                if v > hi {
                    hi = v;
                }
            }
            let spread = hi - lo;
            if spread > best_spread {
                best_spread = spread;
                best_dim = d;
            }
        }

        best_dim
    }

    fn nth_element(&self, order: &mut [usize], mut lo: usize, mut hi: usize, k: usize, dim: usize) {
        while hi - lo > 1 {
            let mid = lo + (hi - lo) / 2;
            let a = self.points[order[lo]][dim];
            let b = self.points[order[mid]][dim];
            let c = self.points[order[hi - 1]][dim];
            let pivot_idx = if (a <= b && b <= c) || (c <= b && b <= a) {
                mid
            } else if (b <= a && a <= c) || (c <= a && a <= b) {
                lo
            } else {
                hi - 1
            };
            order.swap(pivot_idx, hi - 1);
            let pivot_val = self.points[order[hi - 1]][dim];

            let mut store = lo;
            for i in lo..hi - 1 {
                if self.points[order[i]][dim] < pivot_val {
                    order.swap(i, store);
                    store += 1;
                }
            }
            order.swap(store, hi - 1);

            if store == k {
                return;
            } else if k < store {
                hi = store;
            } else {
                lo = store + 1;
            }
        }
    }

    /// Find all points within squared L2 distance of query.
    pub fn range_search(&self, query: &[f64; DIM], radius_sq: f64) -> Vec<SearchResult> {
        let mut results = Vec::new();
        if !self.nodes.is_empty() {
            self.range_search_recursive(0, query, radius_sq, &mut results);
        }
        results
    }

    fn range_search_recursive(
        &self,
        node_idx: usize,
        query: &[f64; DIM],
        radius_sq: f64,
        results: &mut Vec<SearchResult>,
    ) {
        match self.nodes[node_idx] {
            Node::Leaf { start, end } => {
                for i in start..end {
                    let dsq = squared_distance(query, &self.points[i]);
                    if dsq <= radius_sq {
                        results.push(SearchResult {
                            index: self.indices[i],
                            dist_sq: dsq,
                        });
                    }
                }
            }
            Node::Split {
                dim,
                value,
                left,
                right,
            } => {
                let diff = query[dim] - value;
                let diff_sq = diff * diff;

                let (near, far) = if query[dim] <= value {
                    (left, right)
                } else {
                    (right, left)
                };

                self.range_search_recursive(near, query, radius_sq, results);

                if diff_sq <= radius_sq {
                    self.range_search_recursive(far, query, radius_sq, results);
                }
            }
        }
    }

    /// Find the single nearest neighbor.
    pub fn nearest(&self, query: &[f64; DIM]) -> Option<SearchResult> {
        if self.nodes.is_empty() {
            return None;
        }
        let mut best = SearchResult {
            index: 0,
            dist_sq: f64::INFINITY,
        };
        self.nearest_recursive(0, query, &mut best);
        if best.dist_sq.is_infinite() {
            None
        } else {
            Some(best)
        }
    }

    fn nearest_recursive(&self, node_idx: usize, query: &[f64; DIM], best: &mut SearchResult) {
        match self.nodes[node_idx] {
            Node::Leaf { start, end } => {
                for i in start..end {
                    let dsq = squared_distance(query, &self.points[i]);
                    if dsq < best.dist_sq {
                        best.dist_sq = dsq;
                        best.index = self.indices[i];
                    }
                }
            }
            Node::Split {
                dim,
                value,
                left,
                right,
            } => {
                let diff = query[dim] - value;
                let diff_sq = diff * diff;

                let (near, far) = if query[dim] <= value {
                    (left, right)
                } else {
                    (right, left)
                };

                self.nearest_recursive(near, query, best);

                if diff_sq < best.dist_sq {
                    self.nearest_recursive(far, query, best);
                }
            }
        }
    }

    /// Count points within squared L2 distance (no allocation).
    pub fn range_count(&self, query: &[f64; DIM], radius_sq: f64) -> usize {
        if self.nodes.is_empty() {
            return 0;
        }
        self.range_count_recursive(0, query, radius_sq)
    }

    fn range_count_recursive(&self, node_idx: usize, query: &[f64; DIM], radius_sq: f64) -> usize {
        match self.nodes[node_idx] {
            Node::Leaf { start, end } => {
                let mut count = 0;
                for i in start..end {
                    if squared_distance(query, &self.points[i]) <= radius_sq {
                        count += 1;
                    }
                }
                count
            }
            Node::Split {
                dim,
                value,
                left,
                right,
            } => {
                let diff = query[dim] - value;
                let diff_sq = diff * diff;

                let (near, far) = if query[dim] <= value {
                    (left, right)
                } else {
                    (right, left)
                };

                let mut count = self.range_count_recursive(near, query, radius_sq);
                if diff_sq <= radius_sq {
                    count += self.range_count_recursive(far, query, radius_sq);
                }
                count
            }
        }
    }

    /// Number of points in the tree.
    pub fn len(&self) -> usize {
        self.points.len()
    }

    /// Whether the tree is empty.
    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }
}

/// Squared Euclidean distance between two N-dimensional points.
#[inline]
fn squared_distance<const DIM: usize>(a: &[f64; DIM], b: &[f64; DIM]) -> f64 {
    let mut sum = 0.0;
    for i in 0..DIM {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_tree() {
        let tree = KdTree::<3>::build(vec![], vec![]);
        assert!(tree.is_empty());
        assert_eq!(tree.len(), 0);
        assert!(tree.nearest(&[0.0, 0.0, 0.0]).is_none());
        assert!(tree.range_search(&[0.0, 0.0, 0.0], 1.0).is_empty());
        assert_eq!(tree.range_count(&[0.0, 0.0, 0.0], 1.0), 0);
    }

    #[test]
    fn single_point() {
        let tree = KdTree::<3>::build(vec![[1.0, 2.0, 3.0]], vec![42]);

        let nearest = tree.nearest(&[1.0, 2.0, 3.0]).unwrap();
        assert_eq!(nearest.index, 42);
        assert!(nearest.dist_sq < 1e-15);

        let nearest = tree.nearest(&[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(nearest.index, 42);
        assert!((nearest.dist_sq - 14.0).abs() < 1e-10);

        let results = tree.range_search(&[1.0, 2.0, 3.0], 0.01);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].index, 42);

        let results = tree.range_search(&[100.0, 100.0, 100.0], 0.01);
        assert!(results.is_empty());
    }

    #[test]
    fn unit_square_corners_2d() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
        let indices: Vec<usize> = (0..4).collect();
        let tree = KdTree::<2>::build(points, indices);

        assert_eq!(tree.len(), 4);

        let results = tree.range_search(&[0.0, 0.0], 1.0);
        let mut found_indices: Vec<usize> = results.iter().map(|r| r.index).collect();
        found_indices.sort();
        assert_eq!(found_indices, vec![0, 1, 2]);

        let results = tree.range_search(&[0.5, 0.5], 0.5);
        let mut found_indices: Vec<usize> = results.iter().map(|r| r.index).collect();
        found_indices.sort();
        assert_eq!(found_indices, vec![0, 1, 2, 3]);

        let nearest = tree.nearest(&[0.1, 0.1]).unwrap();
        assert_eq!(nearest.index, 0);
    }

    #[test]
    fn brute_force_equivalence_3d() {
        let mut state: u64 = 123456789;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        let n = 1000;
        let points: Vec<[f64; 3]> = (0..n).map(|_| [rng(), rng(), rng()]).collect();
        let indices: Vec<usize> = (0..n).collect();
        let tree = KdTree::<3>::build(points.clone(), indices);

        for _ in 0..50 {
            let query = [rng(), rng(), rng()];
            let radius_sq = rng() * 0.3;

            let mut tree_results: Vec<usize> = tree
                .range_search(&query, radius_sq)
                .iter()
                .map(|r| r.index)
                .collect();
            tree_results.sort();

            let mut brute_results: Vec<usize> = points
                .iter()
                .enumerate()
                .filter(|(_, p)| squared_distance(&query, p) <= radius_sq)
                .map(|(i, _)| i)
                .collect();
            brute_results.sort();

            assert_eq!(
                tree_results, brute_results,
                "Mismatch for query {:?} radius_sq {}",
                query, radius_sq
            );

            assert_eq!(tree.range_count(&query, radius_sq), brute_results.len());
        }
    }

    #[test]
    fn nearest_neighbor_brute_force_3d() {
        let mut state: u64 = 987654321;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        let n = 500;
        let points: Vec<[f64; 3]> = (0..n).map(|_| [rng(), rng(), rng()]).collect();
        let indices: Vec<usize> = (0..n).collect();
        let tree = KdTree::<3>::build(points.clone(), indices);

        for _ in 0..100 {
            let query = [rng(), rng(), rng()];
            let tree_nearest = tree.nearest(&query).unwrap();

            let brute_nearest = points
                .iter()
                .enumerate()
                .map(|(i, p)| (i, squared_distance(&query, p)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            assert_eq!(tree_nearest.index, brute_nearest.0);
            assert!((tree_nearest.dist_sq - brute_nearest.1).abs() < 1e-10);
        }
    }

    #[test]
    fn four_dimensional_search() {
        let mut state: u64 = 1111111111;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        let n = 200;
        let points: Vec<[f64; 4]> = (0..n).map(|_| [rng(), rng(), rng(), rng()]).collect();
        let indices: Vec<usize> = (0..n).collect();
        let tree = KdTree::<4>::build(points.clone(), indices);

        assert_eq!(tree.len(), n);

        for _ in 0..30 {
            let query = [rng(), rng(), rng(), rng()];
            let radius_sq = rng() * 0.5;

            let mut tree_results: Vec<usize> = tree
                .range_search(&query, radius_sq)
                .iter()
                .map(|r| r.index)
                .collect();
            tree_results.sort();

            let mut brute_results: Vec<usize> = points
                .iter()
                .enumerate()
                .filter(|(_, p)| squared_distance(&query, p) <= radius_sq)
                .map(|(i, _)| i)
                .collect();
            brute_results.sort();

            assert_eq!(tree_results, brute_results);
        }

        let query = [rng(), rng(), rng(), rng()];
        let tree_nearest = tree.nearest(&query).unwrap();
        let brute_nearest = points
            .iter()
            .enumerate()
            .map(|(i, p)| (i, squared_distance(&query, p)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .unwrap();
        assert_eq!(tree_nearest.index, brute_nearest.0);
    }

    #[test]
    fn index_preservation() {
        let points = vec![[10.0, 20.0], [30.0, 40.0], [50.0, 60.0]];
        let indices = vec![100, 200, 300];
        let tree = KdTree::<2>::build(points.clone(), indices.clone());

        for (i, point) in points.iter().enumerate() {
            let result = tree.nearest(point).unwrap();
            assert_eq!(result.index, indices[i]);
            assert!(result.dist_sq < 1e-15);
        }
    }

    #[test]
    fn duplicate_points() {
        let points = vec![[1.0, 1.0]; 10];
        let indices: Vec<usize> = (0..10).collect();
        let tree = KdTree::<2>::build(points, indices);

        let results = tree.range_search(&[1.0, 1.0], 0.01);
        assert_eq!(results.len(), 10);

        let nearest = tree.nearest(&[1.0, 1.0]).unwrap();
        assert!(nearest.dist_sq < 1e-15);
    }

    #[test]
    fn larger_than_leaf_size() {
        let mut state: u64 = 42;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        let n = 100;
        let points: Vec<[f64; 3]> = (0..n).map(|_| [rng(), rng(), rng()]).collect();
        let indices: Vec<usize> = (0..n).collect();
        let tree = KdTree::<3>::build(points.clone(), indices);

        assert_eq!(tree.len(), n);
        assert!(!tree.is_empty());

        for (i, point) in points.iter().enumerate() {
            let result = tree.nearest(point).unwrap();
            assert_eq!(result.index, i);
            assert!(result.dist_sq < 1e-15);
        }
    }
}
