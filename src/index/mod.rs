pub mod build_manifest;
pub mod builder;
pub mod cell_builder;
pub mod live;
pub mod multiband_cell_builder;
pub mod quads;
pub mod source;
pub mod store;

use std::sync::OnceLock;

use starfield::Equatorial;
use starfield::time::{Time, Timescale};

use crate::geom::ProperMotion;
use crate::kdtree::KdTree;
use crate::quads::{DIMCODES, Quad};

pub use live::{DropReport, EnsureReport, LiveIndex};
pub use source::{HealpixCell, IndexFragment, IndexSource, ZdclFile};
pub use store::load_bundle_bands;

/// Process-wide default `Timescale` with no leap-second or delta-T
/// tables loaded.
///
/// The only `Time` operation zodiacal performs on catalog epochs is
/// [`Time::j`](starfield::time::Time::j) — TT Julian decimal year —
/// which is pure arithmetic on the underlying JD and doesn't traverse
/// the UTC / TAI / UT1 chain. So an empty default `Timescale` is
/// sufficient for proper-motion propagation; loading the full leap /
/// delta-T tables would be wasted work. (If a future caller does need
/// leap-second-accurate UTC↔TT conversions for catalog epochs, build
/// a populated `Timescale` explicitly and feed it into the typed
/// surfaces — they all accept a `Timescale` parameter.)
///
/// Lazily initialised on first use so the heap allocation happens once
/// per process. All `Time` values constructed from this Timescale share
/// the same `Arc<TimescaleInner>` (see starfield#134 / 0.12.2).
pub fn default_timescale() -> &'static Timescale {
    static TS: OnceLock<Timescale> = OnceLock::new();
    TS.get_or_init(Timescale::default)
}

/// Metadata for a star in the index.
///
/// `position` is the catalog sky position at `ref_epoch`.
/// `proper_motion = None` indicates no PM data is available (Gaia
/// 2-parameter solution or legacy v1/v2 `.zdcl` load with no Gaia
/// sidecar); the solver and verifier then leave the position fixed at
/// `ref_epoch` rather than propagating.
#[derive(Debug, Clone)]
pub struct IndexStar {
    pub catalog_id: u64,
    pub position: Equatorial,
    pub mag: f64,
    pub proper_motion: Option<ProperMotion>,
    pub ref_epoch: Time,
}

impl IndexStar {
    /// Build an `IndexStar` from a legacy or test source with no proper
    /// motion data. `ref_epoch` defaults to TT 2016.0 (Gaia DR3),
    /// constructed from the process-wide [`default_timescale`].
    pub fn without_pm(catalog_id: u64, position: Equatorial, mag: f64) -> Self {
        Self {
            catalog_id,
            position,
            mag,
            proper_motion: None,
            ref_epoch: default_timescale().j(2016.0),
        }
    }
}

// Compile-time guard: `IndexStar` (and therefore `Index` and any
// `Arc<Index>` we hand out to server / realtime callers) must be
// `Send + Sync`. The non-obvious bit is `ref_epoch: Time` — `Time` is
// `Sync` from starfield 0.12.4 onward (OrbitalCommons/starfield#138 →
// `OnceLock<f64>` replacing the old `Cell<Option<f64>>` caches). If a
// future starfield bump regresses on that, this assertion will catch
// it before runtime.
const _: () = {
    const fn assert_sync<T: Sync>() {}
    assert_sync::<IndexStar>();
};

/// Build metadata stored in a v2 index file.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IndexMetadata {
    pub scale_lower_arcsec: f64,
    pub scale_upper_arcsec: f64,
    pub n_stars: usize,
    pub n_quads: usize,
    pub max_stars_per_cell: usize,
    pub uniformize_depth: u8,
    pub quad_depth: u8,
    pub passes: usize,
    pub max_reuse: usize,
    pub build_timestamp: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub catalog_path: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub band_index: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scale_factor: Option<f64>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub mag_range: Option<(f64, f64)>,
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
    /// Build metadata (present in v2+ index files).
    pub metadata: Option<IndexMetadata>,
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

impl From<IndexFragment> for Index {
    /// Build a runnable `Index` (with reconstructed KD-trees) from an
    /// `IndexFragment` returned by `IndexSource::load_full` or
    /// `IndexSource::load_cells`. Tree-reconstruction is O(N log N); for
    /// a typical regional load (~10 k stars) it costs a few milliseconds.
    fn from(frag: IndexFragment) -> Self {
        use crate::geom::sphere::radec_to_xyz;

        let star_points: Vec<[f64; 3]> = frag
            .stars
            .iter()
            .map(|s| radec_to_xyz(s.position.ra, s.position.dec))
            .collect();
        let star_indices: Vec<usize> = (0..frag.stars.len()).collect();
        let star_tree = KdTree::<3>::build(star_points, star_indices);

        let code_indices: Vec<usize> = (0..frag.codes.len()).collect();
        let code_tree = KdTree::<{ DIMCODES }>::build(frag.codes, code_indices);

        Self {
            star_tree,
            stars: frag.stars,
            code_tree,
            quads: frag.quads,
            scale_lower: frag.scale_lower,
            scale_upper: frag.scale_upper,
            metadata: frag.metadata,
        }
    }
}
