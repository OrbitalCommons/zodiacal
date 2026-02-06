//! Blind astrometry plate-solving library.
//!
//! Zodiacal identifies the region of sky depicted in an astronomical image
//! by matching geometric star patterns against a reference catalog,
//! returning a WCS (World Coordinate System) solution.

pub mod extraction;
pub mod fitting;
pub mod geom;
pub mod kdtree;
pub mod quads;
