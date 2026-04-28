//! Apparent-place adapter: maps a catalog Gaia source + observation
//! context to its apparent (RA, Dec) direction at the observer.
//!
//! Delegates to starfield for the physics (PM propagation, light-time,
//! parallax, aberration). Gravitational light deflection requires a
//! `SpiceKernel` for deflector-body positions and is opt-in via
//! [`apparent_radec_with_deflection`].

use nalgebra::Vector3;
use starfield::positions::Position;
use starfield::starlib::Star;
use starfield::time::Time;

use super::types::{GaiaAstrometry, ObservationContext, ObserverState, RefinementError};

/// Compute the apparent direction (RA, Dec, both in radians) of the given
/// catalog source at the observation time, accounting for proper motion,
/// parallax, light-time, and stellar aberration.
///
/// Gravitational light deflection is NOT applied. Callers who need it
/// at the ~mas level (e.g. fields near the Sun) should use
/// [`apparent_radec_with_deflection`].
pub fn apparent_radec(
    source: &GaiaAstrometry,
    obs: &ObservationContext,
) -> Result<(f64, f64), RefinementError> {
    let observer = build_observer(&obs.observer)?;
    let star = build_star(source);

    // observe_from: proper-motion propagation + light-time correction,
    // produces an Astrometric Position relative to the observer.
    let astrometric = star.observe_from(&observer, &obs.time);

    // Apply stellar aberration via relativity helper (no kernel needed).
    // Deflection is skipped in this path.
    let apparent_vec = apply_aberration_only(&astrometric, &observer);

    Ok(xyz_to_radec(apparent_vec))
}

fn build_observer(state: &ObserverState) -> Result<Position, RefinementError> {
    match state {
        ObserverState::Barycentric {
            position_au,
            velocity_au_per_day,
        } => Ok(Position::barycentric(
            Vector3::new(position_au[0], position_au[1], position_au[2]),
            Vector3::new(
                velocity_au_per_day[0],
                velocity_au_per_day[1],
                velocity_au_per_day[2],
            ),
            // Target id = 0 (the observer itself as a body at SSB-relative state);
            // not used downstream except for bookkeeping.
            0,
        )),
    }
}

fn build_star(source: &GaiaAstrometry) -> Star {
    let mut star = Star::new(
        source.ra_deg,
        source.dec_deg,
        source.pmra_mas_per_year,
        source.pmdec_mas_per_year,
        source.parallax_mas,
        source.radial_km_per_s,
    );
    // Gaia DR3 reference epoch is J2016.0. starfield expects TT Julian date;
    // jyear_to_tt_jd does the conversion.
    star = star.with_epoch(jyear_to_tt_jd(source.ref_epoch_jyear));
    star
}

/// Apply only stellar aberration to an astrometric position, returning
/// a unit direction vector. Used when a kernel isn't available for the
/// full `Position::apparent()` chain.
fn apply_aberration_only(astrometric: &Position, observer: &Position) -> Vector3<f64> {
    let mut target = astrometric.position;
    starfield::relativity::add_aberration(&mut target, &observer.velocity, astrometric.light_time);
    target
}

/// Convert Julian year (e.g. 2016.0) to a TT Julian date.
///
/// Using the conventional J2000.0 = JD 2451545.0 TT anchor. Gaia uses
/// TCB as its time system; for our purposes the difference (~30 s over
/// 16 years) is negligible at the 10 mas level.
fn jyear_to_tt_jd(jyear: f64) -> f64 {
    2_451_545.0 + (jyear - 2000.0) * 365.25
}

fn xyz_to_radec(v: Vector3<f64>) -> (f64, f64) {
    let r = (v.x * v.x + v.y * v.y + v.z * v.z).sqrt();
    let ra = v.y.atan2(v.x);
    let dec = (v.z / r).asin();
    let ra_wrapped = if ra < 0.0 {
        ra + 2.0 * std::f64::consts::PI
    } else {
        ra
    };
    (ra_wrapped, dec)
}

/// Suppress unused-import warnings for items we'll need in later phases
/// (e.g. full `Position::apparent()` chain with kernel).
#[allow(dead_code)]
fn _keep_time_import_alive(_t: &Time) {}

#[cfg(test)]
mod tests {
    use super::*;

    /// A star at the ICRS origin at J2016.0 with no PM, no parallax, no RV,
    /// observed from the SSB (zero observer velocity), should map back to
    /// (RA=0, Dec=0) almost exactly.
    #[test]
    fn identity_apparent_position() {
        let src = GaiaAstrometry {
            ra_deg: 0.0,
            dec_deg: 0.0,
            pmra_mas_per_year: 0.0,
            pmdec_mas_per_year: 0.0,
            parallax_mas: 0.0,
            radial_km_per_s: 0.0,
            ref_epoch_jyear: 2016.0,
            sigma_ra_mas: 0.0,
            sigma_dec_mas: 0.0,
            sigma_pmra_mas_per_year: 0.0,
            sigma_pmdec_mas_per_year: 0.0,
            sigma_parallax_mas: 0.0,
        };
        let ts = starfield::time::Timescale::default();
        let time = ts.tt_jd(jyear_to_tt_jd(2016.0), None);
        let obs = ObservationContext {
            time,
            observer: ObserverState::Barycentric {
                position_au: [0.0, 0.0, 0.0],
                velocity_au_per_day: [0.0, 0.0, 0.0],
            },
        };
        let (ra, dec) = apparent_radec(&src, &obs).unwrap();
        assert!(
            ra.abs() < 1e-9 || (ra - 2.0 * std::f64::consts::PI).abs() < 1e-9,
            "ra = {ra}"
        );
        assert!(dec.abs() < 1e-9, "dec = {dec}");
    }

    /// A star with 100 mas/yr of RA proper motion, propagated 10 years forward,
    /// should have moved 1000 mas = 1 arcsec ≈ 4.848e-6 rad in RA.
    #[test]
    fn proper_motion_10_year_displacement() {
        let src = GaiaAstrometry {
            ra_deg: 0.0,
            dec_deg: 0.0,
            pmra_mas_per_year: 100.0, // already includes cos(dec)
            pmdec_mas_per_year: 0.0,
            parallax_mas: 0.0, // (degenerate — Star uses 1 Gpc fallback)
            radial_km_per_s: 0.0,
            ref_epoch_jyear: 2016.0,
            sigma_ra_mas: 0.0,
            sigma_dec_mas: 0.0,
            sigma_pmra_mas_per_year: 0.0,
            sigma_pmdec_mas_per_year: 0.0,
            sigma_parallax_mas: 0.0,
        };
        let ts = starfield::time::Timescale::default();
        let time = ts.tt_jd(jyear_to_tt_jd(2026.0), None);
        let obs = ObservationContext {
            time,
            observer: ObserverState::Barycentric {
                position_au: [0.0, 0.0, 0.0],
                velocity_au_per_day: [0.0, 0.0, 0.0],
            },
        };
        let (ra, _dec) = apparent_radec(&src, &obs).unwrap();

        let expected_ra_rad = 1000.0 * (std::f64::consts::PI / (180.0 * 3_600_000.0));
        let err_mas = (ra - expected_ra_rad).abs() * (180.0 * 3_600_000.0) / std::f64::consts::PI;
        assert!(
            err_mas < 1.0,
            "RA error {err_mas} mas (expected ≈ 1000 mas displacement)"
        );
    }
}
