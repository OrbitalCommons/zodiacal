//! Iterative SIP distortion refinement.
//!
//! Given an initial TAN WCS and matched field/index sources, fits SIP
//! polynomial distortion coefficients to reduce systematic residuals.

use crate::extraction::DetectedSource;
use crate::fitting::FitError;
use crate::geom::sip::{SipWcs, zero_coeffs};
use crate::geom::sphere::radec_to_xyz;
use crate::geom::tan::TanWcs;
use crate::index::Index;
use crate::kdtree::KdTree;

/// Refine an initial TAN WCS by fitting SIP distortion polynomials.
///
/// Algorithm per iteration:
/// 1. Project index stars in the field of view to pixels using the current WCS
/// 2. Match field sources to nearest projected reference star (using KdTree<2>)
/// 3. Compute residuals between TAN-predicted pixel and actual field pixel
/// 4. Build a Vandermonde matrix with u^p * v^q terms (p+q >= 2, p+q <= order)
/// 5. Solve normal equations for A, B coefficients
/// 6. Compute inverse AP, BP by fitting the reverse direction
/// 7. Update SipWcs and repeat
pub fn tweak_solution(
    initial_wcs: &TanWcs,
    field_sources: &[DetectedSource],
    index: &Index,
    sip_order: usize,
    iterations: usize,
) -> Result<SipWcs, FitError> {
    if field_sources.len() < 3 {
        return Err(FitError::TooFewCorrespondences);
    }

    let mut sip = SipWcs::from_tan(initial_wcs.clone(), sip_order);

    // Build a 2D KD-tree of field source positions for matching.
    let field_points: Vec<[f64; 2]> = field_sources.iter().map(|s| [s.x, s.y]).collect();
    let field_indices: Vec<usize> = (0..field_sources.len()).collect();
    let field_tree = KdTree::<2>::build(field_points, field_indices);

    // Enumerate the (p, q) terms where p+q >= 2 and p+q <= order.
    let terms = sip_terms(sip_order);
    let n_terms = terms.len();

    if n_terms == 0 {
        return Ok(sip);
    }

    for _iter in 0..iterations {
        // Step 1: Find reference stars in field of view.
        let (center_ra, center_dec) = sip.field_center();
        let center_xyz = radec_to_xyz(center_ra, center_dec);
        let field_radius = sip.field_radius();
        let radius_sq = 2.0 * (1.0 - field_radius.cos());
        let nearby = index.star_tree.range_search(&center_xyz, radius_sq);

        // Step 2: Project reference stars to pixels, match to field sources.
        let match_radius_sq = 25.0; // 5 pixel match radius
        let mut matched_ref_px = Vec::new();
        let mut matched_field_px = Vec::new();

        for result in &nearby {
            let star = &index.stars[result.index];
            let xyz = radec_to_xyz(star.ra, star.dec);

            // Use TAN projection to get undistorted predicted pixel position.
            let tan_pixel = sip.tan.xyz_to_pixel(xyz);
            let (tan_px, tan_py) = match tan_pixel {
                Some(p) => p,
                None => continue,
            };

            // Skip if outside image bounds with margin.
            let margin = 10.0;
            if tan_px < -margin
                || tan_px > sip.tan.image_size[0] + margin
                || tan_py < -margin
                || tan_py > sip.tan.image_size[1] + margin
            {
                continue;
            }

            // Find nearest field source.
            let query = [tan_px, tan_py];
            if let Some(nearest) = field_tree.nearest(&query)
                && nearest.dist_sq <= match_radius_sq
            {
                let fs = &field_sources[nearest.index];
                matched_ref_px.push((tan_px, tan_py));
                matched_field_px.push((fs.x, fs.y));
            }
        }

        let n_matches = matched_ref_px.len();
        if n_matches < n_terms + 1 {
            break;
        }

        // Step 3-5: Compute residuals and fit A, B polynomials.
        // Residuals: the TAN projection predicts tan_px, but the actual field source
        // is at field_px. The SIP forward distortion maps:
        //   U = u + SIP_A(u,v)
        // where u = field_px - crpix (the actual pixel offset), and U is what
        // the CD matrix converts to IWC. The TAN inverse gives us U = tan_px - crpix.
        // So: delta_u = (tan_px - crpix) - (field_px - crpix) = tan_px - field_px
        // and we want: SIP_A(u,v) = delta_u, i.e., the correction to add to u
        // so that the distorted position matches the TAN predicted position.

        let crpix = sip.tan.crpix;

        // For the forward coefficients A, B:
        // u, v = field pixel offset from crpix.
        // We want A(u,v) such that u + A(u,v) = U_tan = tan_px - crpix.
        // So A(u,v) = (tan_px - crpix) - u = tan_px - field_px.
        let mut delta_u_fwd = Vec::with_capacity(n_matches);
        let mut delta_v_fwd = Vec::with_capacity(n_matches);
        let mut u_fwd = Vec::with_capacity(n_matches);
        let mut v_fwd = Vec::with_capacity(n_matches);

        for i in 0..n_matches {
            let u = matched_field_px[i].0 - crpix[0];
            let v = matched_field_px[i].1 - crpix[1];
            let du = matched_ref_px[i].0 - matched_field_px[i].0;
            let dv = matched_ref_px[i].1 - matched_field_px[i].1;
            u_fwd.push(u);
            v_fwd.push(v);
            delta_u_fwd.push(du);
            delta_v_fwd.push(dv);
        }

        // Build Vandermonde matrix and solve.
        let vandermonde_fwd = build_vandermonde(&u_fwd, &v_fwd, &terms);

        let a_coeffs = solve_normal_equations(&vandermonde_fwd, &delta_u_fwd, n_terms);
        let b_coeffs = solve_normal_equations(&vandermonde_fwd, &delta_v_fwd, n_terms);

        if let Some(a_c) = a_coeffs {
            sip.a = unpack_coefficients(&a_c, &terms, sip_order);
        }
        if let Some(b_c) = b_coeffs {
            sip.b = unpack_coefficients(&b_c, &terms, sip_order);
        }

        // Step 6: Compute inverse AP, BP.
        // For inverse: U, V = TAN-predicted pixel offset from crpix.
        // We want AP(U,V) such that u = U + AP(U,V), i.e.,
        // AP(U,V) = field_px - crpix - (tan_px - crpix) = field_px - tan_px = -delta_u.
        let mut u_inv = Vec::with_capacity(n_matches);
        let mut v_inv = Vec::with_capacity(n_matches);
        let mut delta_u_inv = Vec::with_capacity(n_matches);
        let mut delta_v_inv = Vec::with_capacity(n_matches);

        for i in 0..n_matches {
            let u_big = matched_ref_px[i].0 - crpix[0];
            let v_big = matched_ref_px[i].1 - crpix[1];
            u_inv.push(u_big);
            v_inv.push(v_big);
            delta_u_inv.push(-delta_u_fwd[i]);
            delta_v_inv.push(-delta_v_fwd[i]);
        }

        let vandermonde_inv = build_vandermonde(&u_inv, &v_inv, &terms);

        let ap_coeffs = solve_normal_equations(&vandermonde_inv, &delta_u_inv, n_terms);
        let bp_coeffs = solve_normal_equations(&vandermonde_inv, &delta_v_inv, n_terms);

        if let Some(ap_c) = ap_coeffs {
            sip.ap = unpack_coefficients(&ap_c, &terms, sip_order);
        }
        if let Some(bp_c) = bp_coeffs {
            sip.bp = unpack_coefficients(&bp_c, &terms, sip_order);
        }
    }

    Ok(sip)
}

/// Enumerate (p, q) pairs where p+q >= 2 and p+q <= order.
fn sip_terms(order: usize) -> Vec<(usize, usize)> {
    let mut terms = Vec::new();
    for total in 2..=order {
        for p in 0..=total {
            let q = total - p;
            terms.push((p, q));
        }
    }
    terms
}

/// Build a Vandermonde matrix for the SIP polynomial terms.
/// Returns a flat row-major matrix of size n_points x n_terms.
fn build_vandermonde(u: &[f64], v: &[f64], terms: &[(usize, usize)]) -> Vec<f64> {
    let n = u.len();
    let n_terms = terms.len();
    let mut matrix = vec![0.0; n * n_terms];

    for i in 0..n {
        for (j, &(p, q)) in terms.iter().enumerate() {
            matrix[i * n_terms + j] = u[i].powi(p as i32) * v[i].powi(q as i32);
        }
    }

    matrix
}

/// Solve M^T * M * x = M^T * b via Gaussian elimination with partial pivoting.
///
/// M is n_rows x n_cols stored row-major. b is length n_rows.
/// Returns the solution vector of length n_cols, or None if singular.
fn solve_normal_equations(m: &[f64], b: &[f64], n_cols: usize) -> Option<Vec<f64>> {
    let n_rows = b.len();

    // Compute M^T * M (n_cols x n_cols, symmetric).
    let mut mtm = vec![0.0; n_cols * n_cols];
    for i in 0..n_cols {
        for j in i..n_cols {
            let mut s = 0.0;
            for k in 0..n_rows {
                s += m[k * n_cols + i] * m[k * n_cols + j];
            }
            mtm[i * n_cols + j] = s;
            mtm[j * n_cols + i] = s;
        }
    }

    // Compute M^T * b (length n_cols).
    let mut mtb = vec![0.0; n_cols];
    for i in 0..n_cols {
        let mut s = 0.0;
        for k in 0..n_rows {
            s += m[k * n_cols + i] * b[k];
        }
        mtb[i] = s;
    }

    // Solve mtm * x = mtb via Gaussian elimination with partial pivoting.
    solve_linear_system(&mut mtm, &mut mtb, n_cols)
}

/// Solve A * x = b in place using Gaussian elimination with partial pivoting.
/// A is n x n stored row-major. b is length n.
/// Returns the solution vector or None if singular.
fn solve_linear_system(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot.
        let mut max_val = a[col * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let v = a[row * n + col].abs();
            if v > max_val {
                max_val = v;
                max_row = row;
            }
        }

        if max_val < 1e-30 {
            return None;
        }

        // Swap rows.
        if max_row != col {
            for j in 0..n {
                a.swap(col * n + j, max_row * n + j);
            }
            b.swap(col, max_row);
        }

        // Eliminate below.
        let pivot = a[col * n + col];
        for row in (col + 1)..n {
            let factor = a[row * n + col] / pivot;
            for j in col..n {
                a[row * n + j] -= factor * a[col * n + j];
            }
            b[row] -= factor * b[col];
        }
    }

    // Back substitution.
    let mut x = vec![0.0; n];
    for col in (0..n).rev() {
        let mut sum = b[col];
        for j in (col + 1)..n {
            sum -= a[col * n + j] * x[j];
        }
        x[col] = sum / a[col * n + col];
    }

    Some(x)
}

/// Unpack a flat coefficient vector into a 2D array indexed by (p, q).
fn unpack_coefficients(coeffs: &[f64], terms: &[(usize, usize)], order: usize) -> Vec<Vec<f64>> {
    let mut result = zero_coeffs(order);
    for (i, &(p, q)) in terms.iter().enumerate() {
        if p <= order && q <= order {
            result[p][q] = coeffs[i];
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::extraction::DetectedSource;
    use crate::geom::sphere::radec_to_xyz;
    use crate::geom::tan::TanWcs;
    use crate::index::{Index, IndexStar};
    use crate::kdtree::KdTree;
    use crate::quads::{DIMCODES, Quad};

    fn make_test_wcs() -> TanWcs {
        let arcsec_rad = (2.0_f64 / 3600.0).to_radians();
        TanWcs {
            crval: [1.0, 0.5],
            crpix: [256.0, 256.0],
            cd: [[arcsec_rad, 0.0], [0.0, arcsec_rad]],
            image_size: [512.0, 512.0],
        }
    }

    fn make_test_index(wcs: &TanWcs, n_stars: usize) -> (Vec<DetectedSource>, Index) {
        let mut stars = Vec::new();
        let mut sources = Vec::new();

        let side = (n_stars as f64).sqrt().ceil() as usize;
        let w = wcs.image_size[0];
        let h = wcs.image_size[1];

        let mut count = 0;
        for iy in 0..side {
            for ix in 0..side {
                if count >= n_stars {
                    break;
                }
                let px = w * 0.1 + w * 0.8 * (ix as f64) / (side as f64 - 1.0).max(1.0);
                let py = h * 0.1 + h * 0.8 * (iy as f64) / (side as f64 - 1.0).max(1.0);
                let (ra, dec) = wcs.pixel_to_radec(px, py);

                stars.push(IndexStar {
                    catalog_id: count as u64,
                    ra,
                    dec,
                    mag: count as f64,
                });
                sources.push(DetectedSource {
                    x: px,
                    y: py,
                    flux: 1000.0 - count as f64,
                });
                count += 1;
            }
        }

        let points: Vec<[f64; 3]> = stars.iter().map(|s| radec_to_xyz(s.ra, s.dec)).collect();
        let indices: Vec<usize> = (0..stars.len()).collect();
        let star_tree = KdTree::<3>::build(points, indices);
        let code_tree = KdTree::<{ DIMCODES }>::build(vec![], vec![]);
        let quads: Vec<Quad> = vec![];

        let index = Index {
            star_tree,
            stars,
            code_tree,
            quads,
            scale_lower: 0.0,
            scale_upper: 1.0,
            metadata: None,
        };

        (sources, index)
    }

    #[test]
    fn zero_distortion_gives_near_zero_coefficients() {
        let wcs = make_test_wcs();
        let (sources, index) = make_test_index(&wcs, 36);

        let result = tweak_solution(&wcs, &sources, &index, 3, 3);
        assert!(result.is_ok());

        let sip = result.unwrap();

        // All coefficients should be near zero since there's no distortion.
        for p in 0..=sip.a_order {
            for q in 0..=sip.a_order {
                if p + q >= 2 {
                    assert!(
                        sip.a[p][q].abs() < 1e-10,
                        "A[{p}][{q}] = {} should be ~0",
                        sip.a[p][q]
                    );
                    assert!(
                        sip.b[p][q].abs() < 1e-10,
                        "B[{p}][{q}] = {} should be ~0",
                        sip.b[p][q]
                    );
                }
            }
        }
    }

    #[test]
    fn recovers_known_distortion() {
        let wcs = make_test_wcs();
        let k = 1e-7;

        // Generate field sources with a known distortion applied.
        // The "true" pixel positions are distorted from the TAN positions.
        // u_distorted = u + k * u^2 => the field sources are at distorted positions.
        let side = 8;
        let w = wcs.image_size[0];
        let h = wcs.image_size[1];
        let mut stars = Vec::new();
        let mut sources = Vec::new();

        for iy in 0..side {
            for ix in 0..side {
                let px = w * 0.1 + w * 0.8 * (ix as f64) / (side as f64 - 1.0);
                let py = h * 0.1 + h * 0.8 * (iy as f64) / (side as f64 - 1.0);
                let (ra, dec) = wcs.pixel_to_radec(px, py);

                stars.push(IndexStar {
                    catalog_id: (iy * side + ix) as u64,
                    ra,
                    dec,
                    mag: 10.0,
                });

                // The field source is at a distorted position.
                // If the TAN WCS predicts pixel (px, py) for this star,
                // and the actual source is displaced by the distortion,
                // then the SIP should recover: SIP_A(u,v) = tan_px - field_px.
                // With u = field_px - crpix, we want SIP_A ~ k*u^2 where
                // the displacement = k*u^2.
                let u = px - wcs.crpix[0];
                let v = py - wcs.crpix[1];
                let dx = k * u * u;
                let dy = k * v * v;
                sources.push(DetectedSource {
                    x: px - dx,
                    y: py - dy,
                    flux: 1000.0,
                });
            }
        }

        let points: Vec<[f64; 3]> = stars.iter().map(|s| radec_to_xyz(s.ra, s.dec)).collect();
        let indices: Vec<usize> = (0..stars.len()).collect();
        let star_tree = KdTree::<3>::build(points, indices);
        let code_tree = KdTree::<{ DIMCODES }>::build(vec![], vec![]);

        let index = Index {
            star_tree,
            stars,
            code_tree,
            quads: vec![],
            scale_lower: 0.0,
            scale_upper: 1.0,
            metadata: None,
        };

        let result = tweak_solution(&wcs, &sources, &index, 3, 5);
        assert!(result.is_ok());

        let sip = result.unwrap();

        // The A[2][0] coefficient should be close to k.
        assert!(
            (sip.a[2][0] - k).abs() < k * 0.5,
            "A[2][0] = {} expected ~{k}",
            sip.a[2][0]
        );
        // The B[0][2] coefficient should be close to k.
        assert!(
            (sip.b[0][2] - k).abs() < k * 0.5,
            "B[0][2] = {} expected ~{k}",
            sip.b[0][2]
        );
    }

    #[test]
    fn too_few_sources_returns_error() {
        let wcs = make_test_wcs();
        let sources = vec![
            DetectedSource {
                x: 100.0,
                y: 100.0,
                flux: 1.0,
            },
            DetectedSource {
                x: 200.0,
                y: 200.0,
                flux: 1.0,
            },
        ];

        let (_, index) = make_test_index(&wcs, 10);
        let result = tweak_solution(&wcs, &sources, &index, 3, 3);
        assert!(result.is_err());
    }

    #[test]
    fn sip_terms_enumeration() {
        let terms = sip_terms(2);
        assert_eq!(terms, vec![(0, 2), (1, 1), (2, 0)]);

        let terms = sip_terms(3);
        assert_eq!(
            terms,
            vec![(0, 2), (1, 1), (2, 0), (0, 3), (1, 2), (2, 1), (3, 0)]
        );
    }

    #[test]
    fn linear_solver_identity() {
        // Solve I * x = b => x = b.
        let mut a = vec![1.0, 0.0, 0.0, 1.0];
        let mut b = vec![3.0, 7.0];
        let x = solve_linear_system(&mut a, &mut b, 2).unwrap();
        assert!((x[0] - 3.0).abs() < 1e-12);
        assert!((x[1] - 7.0).abs() < 1e-12);
    }

    #[test]
    fn linear_solver_2x2() {
        // [2 1] [x]   [5]
        // [1 3] [y] = [7]
        // x=8/5, y=9/5
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![5.0, 7.0];
        let x = solve_linear_system(&mut a, &mut b, 2).unwrap();
        assert!((x[0] - 8.0 / 5.0).abs() < 1e-12);
        assert!((x[1] - 9.0 / 5.0).abs() < 1e-12);
    }

    #[test]
    fn linear_solver_singular() {
        let mut a = vec![1.0, 2.0, 2.0, 4.0];
        let mut b = vec![3.0, 6.0];
        let result = solve_linear_system(&mut a, &mut b, 2);
        assert!(result.is_none());
    }
}
