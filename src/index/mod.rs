pub mod builder;
pub mod store;

use crate::kdtree::KdTree;
use crate::quads::{DIMCODES, Quad};

/// Metadata for a star in the index.
#[derive(Debug, Clone)]
pub struct IndexStar {
    pub catalog_id: u64,
    pub ra: f64,
    pub dec: f64,
    pub mag: f64,
}

/// A complete index: stars + quads + KD-trees for fast search.
pub struct Index {
    /// KD-tree over star positions (3D unit vectors).
    pub star_tree: KdTree<3>,
    /// Star metadata.
    pub stars: Vec<IndexStar>,
    /// KD-tree over quad codes (4D).
    pub code_tree: KdTree<{ DIMCODES }>,
    /// Quad definitions.
    pub quads: Vec<Quad>,
    /// Scale range this index covers (radians).
    pub scale_lower: f64,
    pub scale_upper: f64,
}

impl std::fmt::Debug for Index {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Index")
            .field("stars", &self.stars.len())
            .field("quads", &self.quads.len())
            .field("scale_lower", &self.scale_lower)
            .field("scale_upper", &self.scale_upper)
            .finish()
    }
}
