//! Blind astrometry plate-solving library.
//!
//! Zodiacal identifies the region of sky depicted in an astronomical image
//! by matching geometric star patterns against a reference catalog,
//! returning a WCS (World Coordinate System) solution.

#[cfg(feature = "gaia-shards")]
pub mod cli_build_from_shards;
pub mod extraction;
pub mod fitting;
pub mod geom;
pub mod index;
pub mod kdtree;
pub mod pointing;
pub mod quads;
pub mod realtime;
pub mod refinement;
pub mod solver;
pub mod tweak;
pub mod verify;
