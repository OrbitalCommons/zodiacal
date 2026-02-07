use super::sphere;

/// TAN (gnomonic) WCS projection.
///
/// Maps between pixel coordinates and celestial positions using the
/// standard FITS TAN projection.
#[derive(Debug, Clone)]
pub struct TanWcs {
    /// Reference point on sky (RA, Dec) in radians.
    pub crval: [f64; 2],
    /// Reference point in pixel coordinates.
    pub crpix: [f64; 2],
    /// CD matrix mapping pixel offsets to intermediate world coordinates (radians).
    /// `cd[0] = [cd1_1, cd1_2]`, `cd[1] = [cd2_1, cd2_2]`.
    pub cd: [[f64; 2]; 2],
    /// Image dimensions `(width, height)` in pixels.
    pub image_size: [f64; 2],
}

impl TanWcs {
    /// Convert pixel coordinates to a unit vector on the celestial sphere.
    pub fn pixel_to_xyz(&self, px: f64, py: f64) -> [f64; 3] {
        let u = px - self.crpix[0];
        let v = py - self.crpix[1];
        let x = self.cd[0][0] * u + self.cd[0][1] * v;
        let y = self.cd[1][0] * u + self.cd[1][1] * v;
        self.iwc_to_xyz(x, y)
    }

    /// Convert a unit vector on the celestial sphere to pixel coordinates.
    ///
    /// Returns `None` if the point is behind the tangent plane.
    pub fn xyz_to_pixel(&self, xyz: [f64; 3]) -> Option<(f64, f64)> {
        let reference = sphere::radec_to_xyz(self.crval[0], self.crval[1]);
        let (x, y) = sphere::star_coords(xyz, reference)?;

        let det = self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0];
        let inv_det = 1.0 / det;

        let u = inv_det * (self.cd[1][1] * x - self.cd[0][1] * y);
        let v = inv_det * (-self.cd[1][0] * x + self.cd[0][0] * y);

        Some((u + self.crpix[0], v + self.crpix[1]))
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
        let det = self.cd[0][0] * self.cd[1][1] - self.cd[0][1] * self.cd[1][0];
        det.abs().sqrt().to_degrees()
    }

    /// RA, Dec (radians) of the image center pixel.
    pub fn field_center(&self) -> (f64, f64) {
        self.pixel_to_radec(self.image_size[0] / 2.0, self.image_size[1] / 2.0)
    }

    /// Angular radius (radians) of the smallest circle centered on the image
    /// center that encloses all four corners.
    pub fn field_radius(&self) -> f64 {
        let (cx, cy) = (self.image_size[0] / 2.0, self.image_size[1] / 2.0);
        let center = self.pixel_to_xyz(cx, cy);
        let w = self.image_size[0];
        let h = self.image_size[1];

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

    /// Deproject intermediate world coordinates (radians) from the tangent plane
    /// to a unit vector on the sphere.
    ///
    /// Follows the astrometry.net `tan_iwc2xyzarr` algorithm.
    fn iwc_to_xyz(&self, x: f64, y: f64) -> [f64; 3] {
        let x = -x;

        let r = sphere::radec_to_xyz(self.crval[0], self.crval[1]);
        let (rx, ry, rz) = (r[0], r[1], r[2]);

        let (ix, iy) = if rz == 1.0 || rz == -1.0 {
            (-1.0, 0.0)
        } else {
            let ix = ry;
            let iy = -rx;
            let norm = ix.hypot(iy);
            (ix / norm, iy / norm)
        };

        let mut jx = iy * rz;
        let mut jy = -ix * rz;
        let mut jz = ix * ry - iy * rx;
        let jnorm = (jx * jx + jy * jy + jz * jz).sqrt();
        jx /= jnorm;
        jy /= jnorm;
        jz /= jnorm;

        let px = ix * x + jx * y + rx;
        let py = iy * x + jy * y + ry;
        let pz = jz * y + rz;
        let norm = (px * px + py * py + pz * pz).sqrt();

        [px / norm, py / norm, pz / norm]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

    const EPS: f64 = 1e-10;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {a} ~= {b} (diff = {})",
            (a - b).abs()
        );
    }

    fn test_wcs() -> TanWcs {
        let arcsec_rad = (1.0_f64 / 3600.0).to_radians();
        TanWcs {
            crval: [PI, 0.25],
            crpix: [512.0, 512.0],
            cd: [[arcsec_rad, 0.0], [0.0, arcsec_rad]],
            image_size: [1024.0, 1024.0],
        }
    }

    #[test]
    fn roundtrip_pixel_radec() {
        let wcs = test_wcs();
        for &(px, py) in &[
            (512.0, 512.0),
            (0.0, 0.0),
            (1024.0, 1024.0),
            (256.0, 768.0),
            (100.0, 900.0),
        ] {
            let (ra, dec) = wcs.pixel_to_radec(px, py);
            let (px2, py2) = wcs.radec_to_pixel(ra, dec).unwrap();
            assert_close(px, px2, 1e-6);
            assert_close(py, py2, 1e-6);
        }
    }

    #[test]
    fn roundtrip_pixel_xyz() {
        let wcs = test_wcs();
        for px in (0..=1024).step_by(128) {
            for py in (0..=1024).step_by(128) {
                let xyz = wcs.pixel_to_xyz(px as f64, py as f64);
                let (px2, py2) = wcs.xyz_to_pixel(xyz).unwrap();
                assert_close(px as f64, px2, 1e-6);
                assert_close(py as f64, py2, 1e-6);
            }
        }
    }

    #[test]
    fn crpix_maps_to_crval() {
        let wcs = test_wcs();
        let (ra, dec) = wcs.pixel_to_radec(wcs.crpix[0], wcs.crpix[1]);
        assert_close(ra, wcs.crval[0], EPS);
        assert_close(dec, wcs.crval[1], EPS);
    }

    #[test]
    fn pixel_scale_sanity() {
        let wcs = test_wcs();
        assert_close(wcs.pixel_scale(), 1.0 / 3600.0, 1e-12);
    }

    #[test]
    fn field_center_near_crval() {
        let wcs = test_wcs();
        let (ra, dec) = wcs.field_center();
        assert_close(ra, wcs.crval[0], EPS);
        assert_close(dec, wcs.crval[1], EPS);
    }

    #[test]
    fn field_radius_positive() {
        let wcs = test_wcs();
        let radius = wcs.field_radius();
        assert!(radius > 0.0);
        let expected_approx = (512.0 * 2.0_f64.sqrt() / 3600.0).to_radians();
        assert_close(radius, expected_approx, expected_approx * 0.01);
    }

    #[test]
    fn xyz_to_pixel_behind_tangent_plane() {
        let wcs = test_wcs();
        let antipodal = sphere::radec_to_xyz(wcs.crval[0] + PI, -wcs.crval[1]);
        assert!(wcs.xyz_to_pixel(antipodal).is_none());
    }

    #[test]
    fn rotated_cd_matrix() {
        let arcsec_rad = (1.0_f64 / 3600.0).to_radians();
        let angle = PI / 4.0;
        let c = angle.cos() * arcsec_rad;
        let s = angle.sin() * arcsec_rad;
        let wcs = TanWcs {
            crval: [1.0, 0.5],
            crpix: [500.0, 500.0],
            cd: [[c, -s], [s, c]],
            image_size: [1000.0, 1000.0],
        };

        for px in (0..=1000).step_by(200) {
            for py in (0..=1000).step_by(200) {
                let (ra, dec) = wcs.pixel_to_radec(px as f64, py as f64);
                let (px2, py2) = wcs.radec_to_pixel(ra, dec).unwrap();
                assert_close(px as f64, px2, 1e-5);
                assert_close(py as f64, py2, 1e-5);
            }
        }

        assert_close(wcs.pixel_scale(), 1.0 / 3600.0, 1e-12);
    }

    #[test]
    fn wcs_near_pole() {
        let arcsec_rad = (1.0_f64 / 3600.0).to_radians();
        let wcs = TanWcs {
            crval: [0.0, FRAC_PI_2 - 0.01],
            crpix: [256.0, 256.0],
            cd: [[arcsec_rad, 0.0], [0.0, arcsec_rad]],
            image_size: [512.0, 512.0],
        };

        let (ra, dec) = wcs.pixel_to_radec(256.0, 256.0);
        let (px2, py2) = wcs.radec_to_pixel(ra, dec).unwrap();
        assert_close(256.0, px2, 1e-6);
        assert_close(256.0, py2, 1e-6);
    }

    #[test]
    fn wcs_equator_zero_ra() {
        let arcsec_rad = (2.0_f64 / 3600.0).to_radians();
        let wcs = TanWcs {
            crval: [0.0, 0.0],
            crpix: [100.0, 100.0],
            cd: [[arcsec_rad, 0.0], [0.0, arcsec_rad]],
            image_size: [200.0, 200.0],
        };

        let (ra, dec) = wcs.pixel_to_radec(100.0, 100.0);
        assert_close(ra, 0.0, EPS);
        assert_close(dec, 0.0, EPS);

        assert_close(wcs.pixel_scale(), 2.0 / 3600.0, 1e-12);

        for px in (0..=200).step_by(50) {
            for py in (0..=200).step_by(50) {
                let (ra, dec) = wcs.pixel_to_radec(px as f64, py as f64);
                let (px2, py2) = wcs.radec_to_pixel(ra, dec).unwrap();
                assert_close(px as f64, px2, 1e-5);
                assert_close(py as f64, py2, 1e-5);
            }
        }
    }

    #[test]
    fn wcs_test_image_roundtrip() {
        let ps = (0.1296_f64 / 3600.0).to_radians(); // 0.1296 arcsec/pixel in radians
        let crval_ra = 12.3634_f64.to_radians();
        let crval_dec = (-9.7928_f64).to_radians();

        // Negative CD matrix (typical for real FITS images: RA decreases with pixel x)
        let wcs_neg = TanWcs {
            crval: [crval_ra, crval_dec],
            crpix: [3190.0, 4784.0],
            cd: [[-ps, 0.0], [0.0, -ps]],
            image_size: [6380.0, 9568.0],
        };

        // Positive CD matrix (non-standard sign)
        let wcs_pos = TanWcs {
            crval: [crval_ra, crval_dec],
            crpix: [3190.0, 4784.0],
            cd: [[ps, 0.0], [0.0, ps]],
            image_size: [6380.0, 9568.0],
        };

        let test_pixels: &[(f64, f64)] = &[
            (3190.0, 4784.0), // center (crpix)
            (0.0, 0.0),       // top-left corner
            (6380.0, 9568.0), // bottom-right corner
            (6380.0, 0.0),    // top-right corner
            (0.0, 9568.0),    // bottom-left corner
            (1000.0, 2000.0), // arbitrary interior point
            (5000.0, 7000.0), // another interior point
        ];

        println!(
            "=== Pixel scale: {:.6e} rad/px = {:.4} arcsec/px ===",
            ps,
            ps.to_degrees() * 3600.0
        );
        println!(
            "=== CRVAL: RA={:.4} deg, Dec={:.4} deg ===",
            crval_ra.to_degrees(),
            crval_dec.to_degrees()
        );
        println!();

        // Test round-trip for negative CD matrix
        println!("--- Negative CD matrix: cd = [[-ps, 0], [0, -ps]] ---");
        for &(px, py) in test_pixels {
            let (ra, dec) = wcs_neg.pixel_to_radec(px, py);
            let (px2, py2) = wcs_neg.radec_to_pixel(ra, dec).unwrap();
            let err_px = (px - px2).abs();
            let err_py = (py - py2).abs();
            println!(
                "  pixel ({:7.1}, {:7.1}) -> RA={:10.6} Dec={:10.6} deg -> pixel ({:11.6}, {:11.6})  err=({:.2e}, {:.2e})",
                px,
                py,
                ra.to_degrees(),
                dec.to_degrees(),
                px2,
                py2,
                err_px,
                err_py,
            );
            assert_close(px, px2, 1e-4);
            assert_close(py, py2, 1e-4);
        }
        println!();

        // Test round-trip for positive CD matrix
        println!("--- Positive CD matrix: cd = [[ps, 0], [0, ps]] ---");
        for &(px, py) in test_pixels {
            let (ra, dec) = wcs_pos.pixel_to_radec(px, py);
            let (px2, py2) = wcs_pos.radec_to_pixel(ra, dec).unwrap();
            let err_px = (px - px2).abs();
            let err_py = (py - py2).abs();
            println!(
                "  pixel ({:7.1}, {:7.1}) -> RA={:10.6} Dec={:10.6} deg -> pixel ({:11.6}, {:11.6})  err=({:.2e}, {:.2e})",
                px,
                py,
                ra.to_degrees(),
                dec.to_degrees(),
                px2,
                py2,
                err_px,
                err_py,
            );
            assert_close(px, px2, 1e-4);
            assert_close(py, py2, 1e-4);
        }
        println!();

        // Compare RA/Dec at the same pixel positions under both conventions
        println!("--- RA/Dec comparison: negative vs positive CD at same pixel positions ---");
        println!(
            "  {:>15}  {:>18} {:>18}  {:>18} {:>18}",
            "pixel", "RA_neg (deg)", "Dec_neg (deg)", "RA_pos (deg)", "Dec_pos (deg)"
        );
        for &(px, py) in test_pixels {
            let (ra_neg, dec_neg) = wcs_neg.pixel_to_radec(px, py);
            let (ra_pos, dec_pos) = wcs_pos.pixel_to_radec(px, py);
            println!(
                "  ({:7.1},{:6.1})  {:18.10} {:18.10}  {:18.10} {:18.10}",
                px,
                py,
                ra_neg.to_degrees(),
                dec_neg.to_degrees(),
                ra_pos.to_degrees(),
                dec_pos.to_degrees(),
            );

            // At crpix both should map to crval
            if (px - 3190.0).abs() < 0.1 && (py - 4784.0).abs() < 0.1 {
                println!("    ^ center pixel: both should equal CRVAL");
                assert_close(ra_neg, crval_ra, 1e-10);
                assert_close(dec_neg, crval_dec, 1e-10);
                assert_close(ra_pos, crval_ra, 1e-10);
                assert_close(dec_pos, crval_dec, 1e-10);
            }
        }
        println!();

        // Verify that negative CD flips the direction relative to positive CD
        // For pixel offset (+dx, 0) from crpix:
        //   negative CD -> RA should decrease (move west)
        //   positive CD -> RA should increase (move east)
        let dx = 500.0;
        let (ra_neg_right, _) = wcs_neg.pixel_to_radec(3190.0 + dx, 4784.0);
        let (ra_pos_right, _) = wcs_pos.pixel_to_radec(3190.0 + dx, 4784.0);
        println!(
            "--- Direction check: pixel offset +{} in x from crpix ---",
            dx
        );
        println!(
            "  Negative CD: RA = {:.10} deg (delta from crval: {:.6e} deg)",
            ra_neg_right.to_degrees(),
            (ra_neg_right - crval_ra).to_degrees(),
        );
        println!(
            "  Positive CD: RA = {:.10} deg (delta from crval: {:.6e} deg)",
            ra_pos_right.to_degrees(),
            (ra_pos_right - crval_ra).to_degrees(),
        );

        // With negative cd[0][0], moving +x in pixel space should decrease RA
        // (because IWC x = cd[0][0]*u = -ps*u, and then iwc_to_xyz negates x again,
        //  so the net effect on the sphere depends on the full chain)
        let ra_delta_neg = ra_neg_right - crval_ra;
        let ra_delta_pos = ra_pos_right - crval_ra;
        println!("  Neg CD: RA delta = {:.6e} rad", ra_delta_neg);
        println!("  Pos CD: RA delta = {:.6e} rad", ra_delta_pos);
        println!(
            "  These should have opposite signs: neg={:.6e}, pos={:.6e}",
            ra_delta_neg, ra_delta_pos
        );

        // The two deltas should be opposite in sign (the CD sign flips the mapping)
        assert!(
            ra_delta_neg * ra_delta_pos < 0.0,
            "Expected opposite RA directions for opposite CD signs, got neg={}, pos={}",
            ra_delta_neg,
            ra_delta_pos,
        );

        println!();
        println!("All round-trip and sign-convention tests passed.");
    }
}
