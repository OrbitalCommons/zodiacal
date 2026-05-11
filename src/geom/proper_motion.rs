//! Stellar proper motion in the Gaia DR3 convention.
//!
//! TODO: drop this module once `starfield::ProperMotion` ships
//! (tracked at OrbitalCommons/starfield#136). The companion change in
//! starfield-datasources (OrbitalCommons/starfield-datasources#56) will
//! let the bundle reader pull `Option<ProperMotion>` straight off
//! `GaiaRecord` rather than constructing it locally.

/// Linear proper motion of a catalog source, in **mas/yr**.
///
/// Follows the Gaia DR3 convention: [`ProperMotion::pmra`] carries the
/// `cos(dec)` factor, so it is a true-sky rate (the rate at which the
/// source moves across the sky in the local east direction), not the
/// coordinate rate `dRA/dt`. [`ProperMotion::pmdec`] is a plain Dec
/// coordinate rate (Dec lines are great circles, so no factor is
/// needed).
///
/// To convert to a coordinate-rate for use in `(ra, dec)` arithmetic,
/// divide `pmra` by `cos(dec)`. See [`crate::geom::sphere::propagate_pm`]
/// for the application.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ProperMotion {
    /// RA proper motion, **mas/yr**, with `cos(dec)` already folded in.
    pub pmra: f64,
    /// Dec proper motion, **mas/yr**.
    pub pmdec: f64,
}

impl ProperMotion {
    /// A zero proper motion (the source is fixed at its catalog
    /// reference epoch).
    pub const ZERO: ProperMotion = ProperMotion {
        pmra: 0.0,
        pmdec: 0.0,
    };
}
