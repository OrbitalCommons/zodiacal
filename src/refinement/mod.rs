//! High-precision astrometric refinement.
//!
//! Takes a solved WCS and refines it to the ~10 mas absolute-astrometry level
//! by applying the full apparent-place transformation (proper motion, parallax,
//! gravitational light deflection, stellar aberration) to matched catalog stars.
//!
//! The underlying physics is implemented in `starfield`; this module adapts
//! zodiacal's catalog + observation types into starfield's `Star` / `Position`
//! pipeline and adds a weighted iterative WCS re-fit on top.

mod apparent;
mod refine;
mod sidecar;
mod types;

#[cfg(test)]
mod tests;

pub use apparent::apparent_radec;
pub use refine::refine_solution;
pub use sidecar::{DEFAULT_PIVOT_STRIDE, SidecarReader, SidecarRecord, write_sidecar};
pub use types::{
    GaiaAstrometry, ObservationContext, ObserverState, RefinedMatch, RefinedSolution,
    RefinementCatalog, RefinementConfig, RefinementError,
};
