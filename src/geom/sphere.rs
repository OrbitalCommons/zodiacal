use std::f64::consts::TAU;

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
