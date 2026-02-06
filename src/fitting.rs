use crate::geom::sphere::{star_coords, xyz_to_radec};
use crate::geom::tan::TanWcs;

#[derive(Debug)]
pub enum FitError {
    TooFewCorrespondences,
    SingularMatrix,
    ProjectionFailed,
}

/// Fit a TAN WCS from corresponding sky positions and pixel positions.
///
/// Given N matched pairs of (3D unit vector on sky, pixel coordinate),
/// compute the TanWcs that best maps pixels to sky.
///
/// Requires at least 3 correspondences.
pub fn fit_tan_wcs(
    star_xyz: &[[f64; 3]],
    field_xy: &[(f64, f64)],
    image_size: (f64, f64),
) -> Result<TanWcs, FitError> {
    let n = star_xyz.len();
    if n < 3 || field_xy.len() < 3 {
        return Err(FitError::TooFewCorrespondences);
    }
    let n = n.min(field_xy.len());

    // 1. Compute tangent point as centroid of unit vectors (averaged and normalized).
    let mut cx = 0.0;
    let mut cy = 0.0;
    let mut cz = 0.0;
    for xyz in &star_xyz[..n] {
        cx += xyz[0];
        cy += xyz[1];
        cz += xyz[2];
    }
    let norm = (cx * cx + cy * cy + cz * cz).sqrt();
    let crval_xyz = [cx / norm, cy / norm, cz / norm];
    let (crval_ra, crval_dec) = xyz_to_radec(crval_xyz);

    // 2. Set crpix to image center.
    let crpix = [image_size.0 / 2.0, image_size.1 / 2.0];

    // 3. Project each star onto the tangent plane.
    let mut xi = Vec::with_capacity(n);
    let mut eta = Vec::with_capacity(n);
    let mut u = Vec::with_capacity(n);
    let mut v = Vec::with_capacity(n);

    for i in 0..n {
        let (x, y) = star_coords(star_xyz[i], crval_xyz).ok_or(FitError::ProjectionFailed)?;
        xi.push(x);
        eta.push(y);
        u.push(field_xy[i].0 - crpix[0]);
        v.push(field_xy[i].1 - crpix[1]);
    }

    // 4. Solve least-squares for the CD matrix via normal equations.
    //    xi_i  = cd[0][0] * u_i + cd[0][1] * v_i
    //    eta_i = cd[1][0] * u_i + cd[1][1] * v_i
    //
    //    A^T A = [[sum(u*u), sum(u*v)],
    //             [sum(u*v), sum(v*v)]]
    //    A^T b_xi  = [sum(u*xi),  sum(v*xi)]
    //    A^T b_eta = [sum(u*eta), sum(v*eta)]

    let mut sum_uu = 0.0;
    let mut sum_uv = 0.0;
    let mut sum_vv = 0.0;
    let mut sum_u_xi = 0.0;
    let mut sum_v_xi = 0.0;
    let mut sum_u_eta = 0.0;
    let mut sum_v_eta = 0.0;

    for i in 0..n {
        sum_uu += u[i] * u[i];
        sum_uv += u[i] * v[i];
        sum_vv += v[i] * v[i];
        sum_u_xi += u[i] * xi[i];
        sum_v_xi += v[i] * xi[i];
        sum_u_eta += u[i] * eta[i];
        sum_v_eta += v[i] * eta[i];
    }

    // 5. Invert the 2x2 normal matrix: [[sum_uu, sum_uv], [sum_uv, sum_vv]]
    let det = sum_uu * sum_vv - sum_uv * sum_uv;
    if det.abs() < 1e-30 {
        return Err(FitError::SingularMatrix);
    }
    let inv_det = 1.0 / det;

    let cd00 = inv_det * (sum_vv * sum_u_xi - sum_uv * sum_v_xi);
    let cd01 = inv_det * (-sum_uv * sum_u_xi + sum_uu * sum_v_xi);
    let cd10 = inv_det * (sum_vv * sum_u_eta - sum_uv * sum_v_eta);
    let cd11 = inv_det * (-sum_uv * sum_u_eta + sum_uu * sum_v_eta);

    Ok(TanWcs {
        crval: [crval_ra, crval_dec],
        crpix,
        cd: [[cd00, cd01], [cd10, cd11]],
        image_size: [image_size.0, image_size.1],
    })
}

/// Fit TAN WCS from exactly 4 matched quad stars.
/// This is the fast path used during solving when a quad match is found.
pub fn fit_tan_wcs_from_quad(
    star_xyz: &[[f64; 3]; 4],
    field_xy: &[(f64, f64); 4],
    image_size: (f64, f64),
) -> Result<TanWcs, FitError> {
    fit_tan_wcs(star_xyz.as_slice(), field_xy.as_slice(), image_size)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::sphere::radec_to_xyz;
    use std::f64::consts::PI;

    fn assert_close(a: f64, b: f64, tol: f64) {
        assert!(
            (a - b).abs() < tol,
            "expected {a} ~= {b} (diff = {})",
            (a - b).abs()
        );
    }

    fn make_test_wcs(
        crval: [f64; 2],
        pixel_scale_arcsec: f64,
        rotation_rad: f64,
        image_size: (f64, f64),
    ) -> TanWcs {
        let scale = (pixel_scale_arcsec / 3600.0).to_radians();
        let c = rotation_rad.cos() * scale;
        let s = rotation_rad.sin() * scale;
        TanWcs {
            crval,
            crpix: [image_size.0 / 2.0, image_size.1 / 2.0],
            cd: [[c, -s], [s, c]],
            image_size: [image_size.0, image_size.1],
        }
    }

    fn generate_correspondences(wcs: &TanWcs, n: usize) -> (Vec<[f64; 3]>, Vec<(f64, f64)>) {
        let mut star_xyz = Vec::with_capacity(n);
        let mut field_xy = Vec::with_capacity(n);

        let w = wcs.image_size[0];
        let h = wcs.image_size[1];

        let side = (n as f64).sqrt().ceil() as usize;
        let mut count = 0;
        for iy in 0..side {
            for ix in 0..side {
                if count >= n {
                    break;
                }
                let px = w * 0.1 + w * 0.8 * (ix as f64) / (side as f64 - 1.0).max(1.0);
                let py = h * 0.1 + h * 0.8 * (iy as f64) / (side as f64 - 1.0).max(1.0);
                let xyz = wcs.pixel_to_xyz(px, py);
                star_xyz.push(xyz);
                field_xy.push((px, py));
                count += 1;
            }
        }

        (star_xyz, field_xy)
    }

    fn verify_round_trip(
        fitted: &TanWcs,
        star_xyz: &[[f64; 3]],
        field_xy: &[(f64, f64)],
        tol_pixels: f64,
    ) {
        for i in 0..star_xyz.len() {
            let (px, py) = fitted.xyz_to_pixel(star_xyz[i]).unwrap();
            let residual_x = (px - field_xy[i].0).abs();
            let residual_y = (py - field_xy[i].1).abs();
            assert!(
                residual_x < tol_pixels,
                "x residual {residual_x} at star {i} exceeds {tol_pixels} pixels"
            );
            assert!(
                residual_y < tol_pixels,
                "y residual {residual_y} at star {i} exceeds {tol_pixels} pixels"
            );
        }
    }

    #[test]
    fn identity_like_wcs() {
        // Small field so the centroid closely matches the true crval.
        let original = make_test_wcs([PI, 0.25], 1.0, 0.0, (100.0, 100.0));
        let (star_xyz, field_xy) = generate_correspondences(&original, 16);

        let fitted = fit_tan_wcs(&star_xyz, &field_xy, (100.0, 100.0)).unwrap();

        assert_close(fitted.cd[0][0], original.cd[0][0], 1e-10);
        assert_close(fitted.cd[0][1], original.cd[0][1], 1e-10);
        assert_close(fitted.cd[1][0], original.cd[1][0], 1e-10);
        assert_close(fitted.cd[1][1], original.cd[1][1], 1e-10);
    }

    #[test]
    fn rotated_wcs() {
        let rotation = PI / 4.0;
        let original = make_test_wcs([1.0, 0.5], 1.0, rotation, (100.0, 100.0));
        let (star_xyz, field_xy) = generate_correspondences(&original, 16);

        let fitted = fit_tan_wcs(&star_xyz, &field_xy, (100.0, 100.0)).unwrap();

        assert_close(fitted.cd[0][0], original.cd[0][0], 1e-10);
        assert_close(fitted.cd[0][1], original.cd[0][1], 1e-10);
        assert_close(fitted.cd[1][0], original.cd[1][0], 1e-10);
        assert_close(fitted.cd[1][1], original.cd[1][1], 1e-10);
    }

    #[test]
    fn overdetermined_25_stars() {
        // 25 stars in a perfect 5x5 grid for symmetric centroid.
        let original = make_test_wcs([2.0, -0.3], 1.0, 0.0, (100.0, 100.0));
        let (star_xyz, field_xy) = generate_correspondences(&original, 25);

        let fitted = fit_tan_wcs(&star_xyz, &field_xy, (100.0, 100.0)).unwrap();

        assert_close(fitted.cd[0][0], original.cd[0][0], 1e-10);
        assert_close(fitted.cd[0][1], original.cd[0][1], 1e-10);
        assert_close(fitted.cd[1][0], original.cd[1][0], 1e-10);
        assert_close(fitted.cd[1][1], original.cd[1][1], 1e-10);
    }

    #[test]
    fn minimum_three_stars() {
        // Three non-collinear stars whose pixel centroid is (50, 50),
        // so the 3D centroid closely matches the true crval.
        let original = make_test_wcs([0.5, 0.8], 1.0, 0.0, (100.0, 100.0));
        let pixels = [(50.0, 20.0), (20.0, 65.0), (80.0, 65.0)];
        let star_xyz: Vec<[f64; 3]> = pixels
            .iter()
            .map(|&(px, py)| original.pixel_to_xyz(px, py))
            .collect();
        let field_xy: Vec<(f64, f64)> = pixels.to_vec();

        let fitted = fit_tan_wcs(&star_xyz, &field_xy, (100.0, 100.0)).unwrap();

        verify_round_trip(&fitted, &star_xyz, &field_xy, 1e-6);
    }

    #[test]
    fn too_few_correspondences() {
        let xyz = radec_to_xyz(0.0, 0.0);
        assert!(matches!(
            fit_tan_wcs(
                &[xyz, xyz],
                &[(100.0, 100.0), (200.0, 200.0)],
                (512.0, 512.0)
            ),
            Err(FitError::TooFewCorrespondences)
        ));
        assert!(matches!(
            fit_tan_wcs(&[xyz], &[(100.0, 100.0)], (512.0, 512.0)),
            Err(FitError::TooFewCorrespondences)
        ));
        assert!(matches!(
            fit_tan_wcs(&[], &[], (512.0, 512.0)),
            Err(FitError::TooFewCorrespondences)
        ));
    }

    #[test]
    fn round_trip_consistency() {
        let original = make_test_wcs([PI / 3.0, 0.2], 1.0, 0.3, (1024.0, 1024.0));
        let (star_xyz, field_xy) = generate_correspondences(&original, 16);

        let fitted = fit_tan_wcs(&star_xyz, &field_xy, (1024.0, 1024.0)).unwrap();

        verify_round_trip(&fitted, &star_xyz, &field_xy, 0.01);
    }

    #[test]
    fn quad_fit() {
        let original = make_test_wcs([1.5, 0.4], 1.0, 0.0, (100.0, 100.0));
        let (star_xyz, field_xy) = generate_correspondences(&original, 4);

        let star_arr: [[f64; 3]; 4] = [star_xyz[0], star_xyz[1], star_xyz[2], star_xyz[3]];
        let field_arr: [(f64, f64); 4] = [field_xy[0], field_xy[1], field_xy[2], field_xy[3]];

        let fitted = fit_tan_wcs_from_quad(&star_arr, &field_arr, (100.0, 100.0)).unwrap();

        verify_round_trip(&fitted, &star_xyz, &field_xy, 1e-6);
    }

    #[test]
    fn pixel_scale_1_arcsec() {
        let original = make_test_wcs([0.0, 0.0], 1.0, 0.0, (100.0, 100.0));
        let (star_xyz, field_xy) = generate_correspondences(&original, 16);

        let fitted = fit_tan_wcs(&star_xyz, &field_xy, (100.0, 100.0)).unwrap();

        assert_close(fitted.cd[0][0], original.cd[0][0], 1e-10);
        assert_close(fitted.cd[0][1], original.cd[0][1], 1e-10);
        assert_close(fitted.cd[1][0], original.cd[1][0], 1e-10);
        assert_close(fitted.cd[1][1], original.cd[1][1], 1e-10);

        let expected_scale = 1.0 / 3600.0;
        assert_close(fitted.pixel_scale(), expected_scale, 1e-12);
    }

    #[test]
    fn pixel_scale_10_arcsec() {
        let original = make_test_wcs([3.0, -0.5], 10.0, 0.0, (50.0, 50.0));
        let (star_xyz, field_xy) = generate_correspondences(&original, 16);

        let fitted = fit_tan_wcs(&star_xyz, &field_xy, (50.0, 50.0)).unwrap();

        assert_close(fitted.cd[0][0], original.cd[0][0], 1e-10);
        assert_close(fitted.cd[0][1], original.cd[0][1], 1e-10);
        assert_close(fitted.cd[1][0], original.cd[1][0], 1e-10);
        assert_close(fitted.cd[1][1], original.cd[1][1], 1e-10);

        let expected_scale = 10.0 / 3600.0;
        assert_close(fitted.pixel_scale(), expected_scale, 1e-10);
    }
}
