use std::f64::consts::TAU;

use starfield::Equatorial;
use starfield::time::Time;

use crate::geom::ProperMotion;

/// Convert (RA, Dec) in radians to a unit vector `[x, y, z]`.
pub fn radec_to_xyz(ra: f64, dec: f64) -> [f64; 3] {
    let cos_dec = dec.cos();
    [cos_dec * ra.cos(), cos_dec * ra.sin(), dec.sin()]
}

/// Convert a unit vector to (RA, Dec) in radians.
/// RA is in `[0, 2*pi)`, Dec is in `[-pi/2, pi/2]`.
pub fn xyz_to_radec(xyz: [f64; 3]) -> (f64, f64) {
    let mut ra = f64::atan2(xyz[1], xyz[0]);
    if ra < 0.0 {
        ra += TAU;
    }
    let dec = xyz[2].asin();
    (ra, dec)
}

/// Great-circle angular distance between two unit vectors, in radians.
pub fn angular_distance(a: [f64; 3], b: [f64; 3]) -> f64 {
    let dot = a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
    dot.clamp(-1.0, 1.0).acos()
}

/// Midpoint of two points on the unit sphere (normalized).
pub fn star_midpoint(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    let mx = a[0] + b[0];
    let my = a[1] + b[1];
    let mz = a[2] + b[2];
    let norm = (mx * mx + my * my + mz * mz).sqrt();
    let inv = 1.0 / norm;
    [mx * inv, my * inv, mz * inv]
}

/// Gnomonic (TAN) projection of `point` onto the tangent plane at `reference`.
///
/// Returns `Some((x, y))` where x increases in the direction of increasing RA
/// and y increases toward the north pole (increasing Dec).
///
/// Returns `None` if the point is on the opposite hemisphere from the reference.
///
/// Matches astrometry.net's `star_coords()` from starutil.inc with `tangent=TRUE`.
pub fn star_coords(point: [f64; 3], reference: [f64; 3]) -> Option<(f64, f64)> {
    let s = point;
    let r = reference;

    let sdotr = s[0] * r[0] + s[1] * r[1] + s[2] * r[2];
    if sdotr <= 0.0 {
        return None;
    }

    let inv_sdotr = 1.0 / sdotr;

    if r[2] == 1.0 {
        let inv_s2 = 1.0 / s[2];
        return Some((s[0] * inv_s2, s[1] * inv_s2));
    } else if r[2] == -1.0 {
        let inv_s2 = 1.0 / s[2];
        return Some((-s[0] * inv_s2, s[1] * inv_s2));
    }

    // eta: perpendicular to r, in direction of increasing RA (eta_z = 0)
    let mut etax = -r[1];
    let mut etay = r[0];
    let eta_norm = etax.hypot(etay);
    let inv_en = 1.0 / eta_norm;
    etax *= inv_en;
    etay *= inv_en;

    // xi = r cross eta: vector pointing northward (increasing Dec)
    let xix = -r[2] * etay;
    let xiy = r[2] * etax;
    let xiz = r[0] * etay - r[1] * etax;

    let x = (s[0] * etax + s[1] * etay) * inv_sdotr;
    let y = (s[0] * xix + s[1] * xiy + s[2] * xiz) * inv_sdotr;

    Some((x, y))
}

/// Linear proper-motion extrapolation of a catalog position from
/// `ref_epoch` to `obs_epoch`.
///
/// Inputs:
/// - `position` — sky position at `ref_epoch`.
/// - `pm` — Gaia DR3 proper motion (`pmra` carries the cos(dec) factor).
/// - `ref_epoch`, `obs_epoch` — observed via [`starfield::time::Time::j`]
///   (TT Julian decimal year, so 2016.0 for Gaia DR3).
///
/// The math is the standard small-angle linear extrapolation. For
/// fast-moving stars (~5000 mas/yr) and decade-scale gaps this is
/// accurate to ~10 µas, far below any pixel tolerance we care about.
/// Parallax, light-time, and stellar aberration are NOT applied
/// here — those live in the apparent-place pipeline used by the
/// `crate::refinement` module.
pub fn propagate_pm(
    position: Equatorial,
    pm: ProperMotion,
    ref_epoch: &Time,
    obs_epoch: &Time,
) -> Equatorial {
    let dt_yr = obs_epoch.j() - ref_epoch.j();
    // 1 mas = pi / (180 * 3600 * 1000) rad
    const MAS_TO_RAD: f64 = std::f64::consts::PI / (180.0 * 3600.0 * 1000.0);
    let cos_dec = position.dec.cos();
    // pmra has cos(dec) baked in; divide it out to get the coordinate
    // rate dRA/dt. cos(dec) is bounded away from zero everywhere except
    // exactly at the poles; sources within a few arcsec of either pole
    // are vanishingly rare in Gaia.
    let dra = pm.pmra * dt_yr * MAS_TO_RAD / cos_dec;
    let ddec = pm.pmdec * dt_yr * MAS_TO_RAD;
    Equatorial::new(position.ra + dra, position.dec + ddec)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    const EPS: f64 = 1e-12;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {a} ~= {b} (diff = {})",
            (a - b).abs()
        );
    }

    fn assert_vec_close(a: [f64; 3], b: [f64; 3], tol: f64) {
        for i in 0..3 {
            assert_close(a[i], b[i], tol);
        }
    }

    #[test]
    fn roundtrip_radec_xyz() {
        let cases = [
            (0.0, 0.0),
            (PI, 0.0),
            (PI / 4.0, PI / 6.0),
            (3.0 * PI / 2.0, -PI / 4.0),
            (0.0, FRAC_PI_2),
            (0.0, -FRAC_PI_2),
            (1.234, 0.567),
        ];
        for (ra, dec) in cases {
            let xyz = radec_to_xyz(ra, dec);
            let (ra2, dec2) = xyz_to_radec(xyz);
            assert_close(dec, dec2, EPS);
            let dra = ((ra - ra2 + PI) % TAU + TAU) % TAU - PI;
            assert_close(dra, 0.0, EPS);
        }
    }

    #[test]
    fn known_positions() {
        assert_vec_close(radec_to_xyz(0.0, 0.0), [1.0, 0.0, 0.0], EPS);
        assert_vec_close(radec_to_xyz(FRAC_PI_2, 0.0), [0.0, 1.0, 0.0], EPS);
        assert_vec_close(radec_to_xyz(0.0, FRAC_PI_2), [0.0, 0.0, 1.0], EPS);
        assert_vec_close(radec_to_xyz(0.0, -FRAC_PI_2), [0.0, 0.0, -1.0], EPS);
    }

    #[test]
    fn xyz_to_radec_poles() {
        let (ra, dec) = xyz_to_radec([0.0, 0.0, 1.0]);
        assert_close(dec, FRAC_PI_2, EPS);
        assert_close(ra, 0.0, EPS);

        let (_, dec) = xyz_to_radec([0.0, 0.0, -1.0]);
        assert_close(dec, -FRAC_PI_2, EPS);
    }

    #[test]
    fn propagate_pm_zero_dt_is_identity() {
        let ts = starfield::time::Timescale::default();
        let pos = Equatorial::new(1.234, 0.567);
        let pm = ProperMotion {
            pmra: 12.34,
            pmdec: -5.67,
        };
        let t = ts.j(2016.0);
        let out = propagate_pm(pos, pm, &t, &t);
        assert_close(pos.ra, out.ra, EPS);
        assert_close(pos.dec, out.dec, EPS);
    }

    #[test]
    fn propagate_pm_zero_pm_is_identity() {
        let ts = starfield::time::Timescale::default();
        let pos = Equatorial::new(0.5, 0.3);
        let out = propagate_pm(pos, ProperMotion::ZERO, &ts.j(2016.0), &ts.j(2002.0));
        assert_close(pos.ra, out.ra, EPS);
        assert_close(pos.dec, out.dec, EPS);
    }

    #[test]
    fn propagate_pm_known_drift_m101_rank4() {
        // Gaia DR3 1609274703365302528 (the M101 rank-4 detection from
        // PR #126's investigation). Verified against the user-space
        // astropy calculation: 13.126 yr × pmra=-12.145 mas/yr (cos dec
        // included) → -159.5 mas in true sky, -3.3 mas in declination.
        let ts = starfield::time::Timescale::default();
        let pos = Equatorial::from_degrees(210.826611, 54.347952);
        let pm = ProperMotion {
            pmra: -12.14525611646663,
            pmdec: -0.25469925378605757,
        };
        // From Gaia 2016.0 backward to HST DATE-OBS 2002-11-15
        // midpoint = MJD 52593.98 = decimal year 2002.87392.
        let out = propagate_pm(pos, pm, &ts.j(2016.0), &ts.j(2002.87392));
        let dra_arcsec = (out.ra - pos.ra).to_degrees() * 3600.0 * pos.dec.cos();
        let ddec_arcsec = (out.dec - pos.dec).to_degrees() * 3600.0;
        // Expected: -12.145 * (2002.874 - 2016.0) = +159.4 mas RA·cos
        // (sign: -pmra × negative dt → +mas), +3.3 mas Dec.
        assert!(
            (dra_arcsec - 0.1594).abs() < 1e-3,
            "RA drift mismatch: got {dra_arcsec} arcsec, want ~0.1594"
        );
        assert!(
            (ddec_arcsec - 0.0033).abs() < 1e-3,
            "Dec drift mismatch: got {ddec_arcsec} arcsec, want ~0.0033"
        );
    }

    #[test]
    fn propagate_pm_sign_convention() {
        let ts = starfield::time::Timescale::default();
        let pos = Equatorial::new(1.0, 0.0);
        let pm_pos = ProperMotion {
            pmra: 100.0,
            pmdec: 100.0,
        };
        let pm_neg = ProperMotion {
            pmra: -100.0,
            pmdec: -100.0,
        };
        let out_pos = propagate_pm(pos, pm_pos, &ts.j(2000.0), &ts.j(2010.0));
        assert!(
            out_pos.ra > 1.0,
            "pmra>0, dt>0 should push ra up: got {}",
            out_pos.ra
        );
        assert!(
            out_pos.dec > 0.0,
            "pmdec>0, dt>0 should push dec up: got {}",
            out_pos.dec
        );
        let out_neg = propagate_pm(pos, pm_neg, &ts.j(2000.0), &ts.j(2010.0));
        assert!(
            out_neg.ra < 1.0,
            "pmra<0, dt>0 should pull ra down: got {}",
            out_neg.ra
        );
        assert!(
            out_neg.dec < 0.0,
            "pmdec<0, dt>0 should pull dec down: got {}",
            out_neg.dec
        );
    }

    #[test]
    fn propagate_pm_cos_dec_correction() {
        // At dec = 60°, cos(dec) = 0.5, so a given pmra (which has
        // cos(dec) baked in) should produce twice the RA-coordinate
        // change as the same pmra at dec = 0.
        let ts = starfield::time::Timescale::default();
        let pm = ProperMotion {
            pmra: 100.0,
            pmdec: 0.0,
        };
        let out_eq = propagate_pm(Equatorial::new(0.0, 0.0), pm, &ts.j(0.0), &ts.j(10.0));
        let out_60 = propagate_pm(
            Equatorial::new(0.0, 60.0_f64.to_radians()),
            pm,
            &ts.j(0.0),
            &ts.j(10.0),
        );
        let ratio = out_60.ra / out_eq.ra;
        assert!(
            (ratio - 2.0).abs() < 1e-9,
            "ra change at dec=60 should be 2x equator: ratio={ratio}"
        );
    }

    #[test]
    fn angular_distance_known() {
        let a = radec_to_xyz(0.0, 0.0);
        let b = radec_to_xyz(FRAC_PI_2, 0.0);
        assert_close(angular_distance(a, b), FRAC_PI_2, EPS);

        assert_close(angular_distance(a, a), 0.0, EPS);

        let c = radec_to_xyz(PI, 0.0);
        assert_close(angular_distance(a, c), PI, EPS);

        let np = radec_to_xyz(0.0, FRAC_PI_2);
        let sp = radec_to_xyz(0.0, -FRAC_PI_2);
        assert_close(angular_distance(np, sp), PI, EPS);
    }

    #[test]
    fn midpoint_basic() {
        let a = radec_to_xyz(0.0, 0.0);
        let b = radec_to_xyz(FRAC_PI_2, 0.0);
        let m = star_midpoint(a, b);
        let expected = radec_to_xyz(PI / 4.0, 0.0);
        assert_vec_close(m, expected, EPS);

        let norm = (m[0] * m[0] + m[1] * m[1] + m[2] * m[2]).sqrt();
        assert_close(norm, 1.0, EPS);
    }

    #[test]
    fn midpoint_symmetric() {
        let a = radec_to_xyz(0.5, 0.3);
        let b = radec_to_xyz(0.7, -0.1);
        let m1 = star_midpoint(a, b);
        let m2 = star_midpoint(b, a);
        assert_vec_close(m1, m2, EPS);
    }

    #[test]
    fn star_coords_at_reference() {
        let r = radec_to_xyz(1.0, 0.5);
        let (x, y) = star_coords(r, r).unwrap();
        assert_close(x, 0.0, EPS);
        assert_close(y, 0.0, EPS);
    }

    #[test]
    fn star_coords_opposite_hemisphere() {
        let r = radec_to_xyz(0.0, 0.0);
        let s = radec_to_xyz(PI, 0.0);
        assert!(star_coords(s, r).is_none());
    }

    #[test]
    fn star_coords_small_ra_offset() {
        let r = radec_to_xyz(0.0, 0.0);
        let delta = 1e-4;
        let s = radec_to_xyz(delta, 0.0);
        let (x, y) = star_coords(s, r).unwrap();
        assert_close(x, delta, 1e-8);
        assert_close(y, 0.0, 1e-8);
    }

    #[test]
    fn star_coords_small_dec_offset() {
        let r = radec_to_xyz(0.0, 0.0);
        let delta = 1e-4;
        let s = radec_to_xyz(0.0, delta);
        let (x, y) = star_coords(s, r).unwrap();
        assert_close(x, 0.0, 1e-8);
        assert_close(y, delta, 1e-8);
    }

    #[test]
    fn star_coords_north_pole_reference() {
        let r = [0.0, 0.0, 1.0];
        let s = radec_to_xyz(0.0, FRAC_PI_2 - 0.01);
        let result = star_coords(s, r);
        assert!(result.is_some());
    }
}
