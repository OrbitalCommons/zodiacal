use super::sphere;
use super::tan::TanWcs;

/// SIP (Simple Imaging Polynomial) WCS distortion model.
///
/// Extends the standard TAN projection with polynomial distortion terms
/// that model optical aberrations such as barrel/pincushion distortion.
///
/// Forward direction (pixel -> sky):
///   u, v = pixel offset from crpix
///   U = u + sum(A[p][q] * u^p * v^q)  for p+q >= 2
///   V = v + sum(B[p][q] * u^p * v^q)  for p+q >= 2
///   Then CD * (U, V) -> intermediate world coordinates -> sphere
///
/// Inverse direction (sky -> pixel):
///   Get (U, V) from TAN inverse projection
///   u = U + sum(AP[p][q] * U^p * V^q)  for p+q >= 2
///   v = V + sum(BP[p][q] * U^p * V^q)  for p+q >= 2
///   pixel = (u + crpix[0], v + crpix[1])
#[derive(Debug, Clone)]
pub struct SipWcs {
    /// Underlying TAN WCS.
    pub tan: TanWcs,
    /// Forward distortion coefficients for U.
    pub a: Vec<Vec<f64>>,
    /// Forward distortion coefficients for V.
    pub b: Vec<Vec<f64>>,
    /// Order of forward A polynomial.
    pub a_order: usize,
    /// Order of forward B polynomial.
    pub b_order: usize,
    /// Inverse distortion coefficients for u.
    pub ap: Vec<Vec<f64>>,
    /// Inverse distortion coefficients for v.
    pub bp: Vec<Vec<f64>>,
    /// Order of inverse AP polynomial.
    pub ap_order: usize,
    /// Order of inverse BP polynomial.
    pub bp_order: usize,
}

/// Evaluate a SIP polynomial: sum(coeffs[p][q] * u^p * v^q) for p+q >= 2.
fn eval_polynomial(coeffs: &[Vec<f64>], order: usize, u: f64, v: f64) -> f64 {
    let mut result = 0.0;
    let mut u_pow = 1.0; // u^p
    for p in 0..=order {
        let mut v_pow = 1.0; // v^q
        for q in 0..=order {
            if p + q >= 2 && p < coeffs.len() && q < coeffs[p].len() {
                result += coeffs[p][q] * u_pow * v_pow;
            }
            v_pow *= v;
        }
        u_pow *= u;
    }
    result
}

/// Create a zero-filled 2D coefficient array of size (order+1) x (order+1).
pub fn zero_coeffs(order: usize) -> Vec<Vec<f64>> {
    vec![vec![0.0; order + 1]; order + 1]
}

impl SipWcs {
    /// Create an identity SIP WCS (zero distortion coefficients) from a TAN WCS.
    pub fn from_tan(tan: TanWcs, order: usize) -> Self {
        SipWcs {
            a: zero_coeffs(order),
            b: zero_coeffs(order),
            a_order: order,
            b_order: order,
            ap: zero_coeffs(order),
            bp: zero_coeffs(order),
            ap_order: order,
            bp_order: order,
            tan,
        }
    }

    /// Convert pixel coordinates to a unit vector on the celestial sphere.
    pub fn pixel_to_xyz(&self, px: f64, py: f64) -> [f64; 3] {
        let u = px - self.tan.crpix[0];
        let v = py - self.tan.crpix[1];

        // Apply forward SIP distortion.
        let u_dist = u + eval_polynomial(&self.a, self.a_order, u, v);
        let v_dist = v + eval_polynomial(&self.b, self.b_order, u, v);

        // Apply CD matrix to get intermediate world coordinates.
        let x = self.tan.cd[0][0] * u_dist + self.tan.cd[0][1] * v_dist;
        let y = self.tan.cd[1][0] * u_dist + self.tan.cd[1][1] * v_dist;

        self.tan.iwc_to_xyz(x, y)
    }

    /// Convert a unit vector on the celestial sphere to pixel coordinates.
    ///
    /// Returns `None` if the point is behind the tangent plane.
    pub fn xyz_to_pixel(&self, xyz: [f64; 3]) -> Option<(f64, f64)> {
        let reference = sphere::radec_to_xyz(self.tan.crval[0], self.tan.crval[1]);
        let (x, y) = sphere::star_coords(xyz, reference)?;

        // Invert CD matrix to get undistorted pixel offsets (U, V).
        let det = self.tan.cd[0][0] * self.tan.cd[1][1] - self.tan.cd[0][1] * self.tan.cd[1][0];
        let inv_det = 1.0 / det;
        let u_big = inv_det * (self.tan.cd[1][1] * x - self.tan.cd[0][1] * y);
        let v_big = inv_det * (-self.tan.cd[1][0] * x + self.tan.cd[0][0] * y);

        // Apply inverse SIP polynomial to recover original pixel offsets.
        let u = u_big + eval_polynomial(&self.ap, self.ap_order, u_big, v_big);
        let v = v_big + eval_polynomial(&self.bp, self.bp_order, u_big, v_big);

        Some((u + self.tan.crpix[0], v + self.tan.crpix[1]))
    }

    /// Convert pixel coordinates to (RA, Dec) in radians.
    pub fn pixel_to_radec(&self, px: f64, py: f64) -> (f64, f64) {
        sphere::xyz_to_radec(self.pixel_to_xyz(px, py))
    }

    /// Convert (RA, Dec) in radians to pixel coordinates.
    ///
    /// Returns `None` if the position is behind the tangent plane.
    pub fn radec_to_pixel(&self, ra: f64, dec: f64) -> Option<(f64, f64)> {
        self.xyz_to_pixel(sphere::radec_to_xyz(ra, dec))
    }

    /// Approximate pixel scale in degrees per pixel from the CD matrix determinant.
    pub fn pixel_scale(&self) -> f64 {
        self.tan.pixel_scale()
    }

    /// RA, Dec (radians) of the image center pixel.
    pub fn field_center(&self) -> (f64, f64) {
        self.pixel_to_radec(self.tan.image_size[0] / 2.0, self.tan.image_size[1] / 2.0)
    }

    /// Angular radius (radians) of the smallest circle centered on the image
    /// center that encloses all four corners.
    pub fn field_radius(&self) -> f64 {
        let (cx, cy) = (self.tan.image_size[0] / 2.0, self.tan.image_size[1] / 2.0);
        let center = self.pixel_to_xyz(cx, cy);
        let w = self.tan.image_size[0];
        let h = self.tan.image_size[1];

        [
            self.pixel_to_xyz(0.0, 0.0),
            self.pixel_to_xyz(w, 0.0),
            self.pixel_to_xyz(0.0, h),
            self.pixel_to_xyz(w, h),
        ]
        .iter()
        .map(|c| sphere::angular_distance(center, *c))
        .fold(0.0_f64, f64::max)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    const EPS: f64 = 1e-10;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {a} ~= {b} (diff = {})",
            (a - b).abs()
        );
    }

    fn test_tan_wcs() -> TanWcs {
        let arcsec_rad = (1.0_f64 / 3600.0).to_radians();
        TanWcs {
            crval: [PI, 0.25],
            crpix: [512.0, 512.0],
            cd: [[arcsec_rad, 0.0], [0.0, arcsec_rad]],
            image_size: [1024.0, 1024.0],
        }
    }

    #[test]
    fn identity_sip_matches_tan() {
        let tan = test_tan_wcs();
        let sip = SipWcs::from_tan(tan.clone(), 3);

        for &(px, py) in &[
            (512.0, 512.0),
            (0.0, 0.0),
            (1024.0, 1024.0),
            (256.0, 768.0),
            (100.0, 900.0),
        ] {
            let tan_xyz = tan.pixel_to_xyz(px, py);
            let sip_xyz = sip.pixel_to_xyz(px, py);
            for i in 0..3 {
                assert_close(tan_xyz[i], sip_xyz[i], EPS);
            }

            let (tan_ra, tan_dec) = tan.pixel_to_radec(px, py);
            let (sip_ra, sip_dec) = sip.pixel_to_radec(px, py);
            assert_close(tan_ra, sip_ra, EPS);
            assert_close(tan_dec, sip_dec, EPS);

            let (tan_px, tan_py) = tan.xyz_to_pixel(tan_xyz).unwrap();
            let (sip_px, sip_py) = sip.xyz_to_pixel(tan_xyz).unwrap();
            assert_close(tan_px, sip_px, 1e-6);
            assert_close(tan_py, sip_py, 1e-6);
        }
    }

    #[test]
    fn barrel_distortion_round_trip() {
        let tan = test_tan_wcs();
        let mut sip = SipWcs::from_tan(tan, 3);

        // Apply known barrel distortion: A[2][0] and B[0][2] coefficients.
        let k = 1e-6;
        sip.a[2][0] = k;
        sip.b[0][2] = k;

        // Compute matching inverse coefficients.
        // For small distortion, AP ~ -A and BP ~ -B to first order.
        sip.ap[2][0] = -k;
        sip.bp[0][2] = -k;

        // Test round-trip at various positions.
        for &(px, py) in &[
            (512.0, 512.0),
            (256.0, 256.0),
            (768.0, 768.0),
            (100.0, 512.0),
            (512.0, 100.0),
            (200.0, 800.0),
        ] {
            let (ra, dec) = sip.pixel_to_radec(px, py);
            let (px2, py2) = sip.radec_to_pixel(ra, dec).unwrap();
            assert_close(px, px2, 0.1);
            assert_close(py, py2, 0.1);
        }
    }

    #[test]
    fn pixel_scale_at_center_matches_tan() {
        let tan = test_tan_wcs();
        let sip = SipWcs::from_tan(tan.clone(), 3);

        assert_close(sip.pixel_scale(), tan.pixel_scale(), 1e-15);
    }

    #[test]
    fn distortion_moves_edge_pixels() {
        let tan = test_tan_wcs();
        let mut sip = SipWcs::from_tan(tan.clone(), 3);

        let k = 1e-6;
        sip.a[2][0] = k;
        sip.b[0][2] = k;

        // At crpix, distortion should be zero since u=v=0.
        let sip_center = sip.pixel_to_xyz(512.0, 512.0);
        let tan_center = tan.pixel_to_xyz(512.0, 512.0);
        for i in 0..3 {
            assert_close(sip_center[i], tan_center[i], EPS);
        }

        // At corner, distortion should shift the result.
        let sip_corner = sip.pixel_to_xyz(0.0, 0.0);
        let tan_corner = tan.pixel_to_xyz(0.0, 0.0);
        let dist = sphere::angular_distance(sip_corner, tan_corner);
        assert!(
            dist > 1e-8,
            "expected distortion to move corner, got dist = {dist}"
        );
    }

    #[test]
    fn eval_polynomial_zero_coeffs() {
        let coeffs = zero_coeffs(3);
        let val = eval_polynomial(&coeffs, 3, 100.0, 200.0);
        assert_close(val, 0.0, EPS);
    }

    #[test]
    fn eval_polynomial_known_value() {
        // Set A[2][0] = 1.0 => result should be u^2.
        let mut coeffs = zero_coeffs(3);
        coeffs[2][0] = 1.0;
        let val = eval_polynomial(&coeffs, 3, 5.0, 3.0);
        assert_close(val, 25.0, EPS);

        // Set A[1][1] = 2.0 additionally => result = u^2 + 2*u*v.
        coeffs[1][1] = 2.0;
        let val = eval_polynomial(&coeffs, 3, 5.0, 3.0);
        assert_close(val, 25.0 + 30.0, EPS);
    }

    #[test]
    fn field_center_and_radius() {
        let tan = test_tan_wcs();
        let sip = SipWcs::from_tan(tan.clone(), 3);

        let (sip_ra, sip_dec) = sip.field_center();
        let (tan_ra, tan_dec) = tan.field_center();
        assert_close(sip_ra, tan_ra, EPS);
        assert_close(sip_dec, tan_dec, EPS);

        let sip_r = sip.field_radius();
        let tan_r = tan.field_radius();
        assert_close(sip_r, tan_r, 1e-8);
    }
}
