//! Quad codec: geometric hashing of 4-star patterns for blind astrometry.
//!
//! Computes a 4-dimensional code from 4 stars that is invariant to rotation
//! and scale on the sky. With canonicalization, equivalent geometric
//! configurations produce the same code regardless of star labeling.

use crate::geom::sphere::{star_coords, star_midpoint};

/// Number of stars in a quad.
pub const DIMQUADS: usize = 4;

/// Dimensionality of the code (2 coordinates per non-backbone star).
pub const DIMCODES: usize = 2 * (DIMQUADS - 2);

/// A geometric hash code: 4 floating-point values encoding the relative
/// positions of stars C and D within the reference frame defined by stars A and B.
pub type Code = [f64; DIMCODES];

/// A quad: 4 star indices referencing positions in some external catalog.
#[derive(Debug, Clone)]
pub struct Quad {
    pub star_ids: [usize; DIMQUADS],
}

/// Compute the geometric hash code for 4 stars given their 3D unit-vector positions.
///
/// Algorithm (from astrometry.net `quad-utils.c:quad_compute_star_code`):
/// 1. Stars A (index 0) and B (index 1) form the "backbone"
/// 2. Compute midpoint M of A and B on the sphere
/// 3. Project all stars onto the tangent plane at M via gnomonic projection
/// 4. Compute rotation and scale that normalize the AB baseline
/// 5. For stars C (index 2) and D (index 3), express their positions
///    relative to A in the rotated, scaled coordinate system
///
/// Note: The reference implementation swaps the x,y outputs of `star_coords`
/// when assigning to its internal variables. We replicate that exactly.
pub fn compute_code(star_xyz: &[[f64; 3]; DIMQUADS]) -> Code {
    let mid_ab = star_midpoint(star_xyz[0], star_xyz[1]);

    // The C code calls: star_coords(sA, midAB, TRUE, &Ay, &Ax)
    // star_coords returns (x=RA_dir, y=Dec_dir), assigned as x->Ay, y->Ax.
    let (a_y, a_x) = star_coords(star_xyz[0], mid_ab)
        .expect("Star A failed to project onto tangent plane at midpoint");
    let (b_y, b_x) = star_coords(star_xyz[1], mid_ab)
        .expect("Star B failed to project onto tangent plane at midpoint");

    let ab_x = b_x - a_x;
    let ab_y = b_y - a_y;
    let scale = ab_x * ab_x + ab_y * ab_y;
    let invscale = 1.0 / scale;
    let costheta = (ab_y + ab_x) * invscale;
    let sintheta = (ab_y - ab_x) * invscale;

    let mut code = [0.0f64; DIMCODES];

    for i in 2..DIMQUADS {
        let (d_y, d_x) = star_coords(star_xyz[i], mid_ab)
            .expect("Star failed to project onto tangent plane at midpoint");
        let ad_x = d_x - a_x;
        let ad_y = d_y - a_y;
        let x = ad_x * costheta + ad_y * sintheta;
        let y = -ad_x * sintheta + ad_y * costheta;
        code[2 * (i - 2)] = x;
        code[2 * (i - 2) + 1] = y;
    }

    code
}

/// Enforce ordering invariants so equivalent geometric configurations
/// produce the same canonical code.
///
/// Two invariants are enforced (from astrometry.net `quad-utils.c:quad_enforce_invariants`):
///
/// 1. **Mean-x constraint**: The mean of x-coordinates (code[0], code[2], ...)
///    must be <= 0.5. If not, swap stars A and B (flip the backbone) and
///    transform the code as `code[i] = 1.0 - code[i]`.
///
/// 2. **cx <= dx ordering**: The x-coordinates of non-backbone stars must be
///    in non-decreasing order. If not, swap the offending stars and their
///    code entries.
///
/// Returns `(canonical_code, reordered_star_ids, parity_flipped)`.
pub fn enforce_invariants(
    code: Code,
    star_ids: [usize; DIMQUADS],
) -> (Code, [usize; DIMQUADS], bool) {
    let mut code = code;
    let mut ids = star_ids;
    let mut parity_flipped = false;

    let n_extra = DIMQUADS - 2;

    // Invariant 1: mean of x-coords <= 0.5
    let mut sum_x = 0.0;
    for i in 0..n_extra {
        sum_x += code[2 * i];
    }
    let mean_x = sum_x / n_extra as f64;

    if mean_x > 0.5 {
        ids.swap(0, 1);
        for v in &mut code {
            *v = 1.0 - *v;
        }
        parity_flipped = true;
    }

    // Invariant 2: cx <= dx <= ... (selection sort by x-coordinate)
    for i in 0..n_extra {
        let mut j_smallest = None;
        let mut smallest = code[2 * i];
        for j in (i + 1)..n_extra {
            let x2 = code[2 * j];
            if x2 < smallest {
                smallest = x2;
                j_smallest = Some(j);
            }
        }
        if let Some(j) = j_smallest {
            ids.swap(i + 2, j + 2);
            code.swap(2 * i, 2 * j);
            code.swap(2 * i + 1, 2 * j + 1);
        }
    }

    (code, ids, parity_flipped)
}

/// Compute the geometric hash code and enforce invariants in one step.
///
/// Returns `(canonical_code, reordered_star_ids, parity_flipped)`.
pub fn compute_canonical_code(
    star_xyz: &[[f64; 3]; DIMQUADS],
    star_ids: [usize; DIMQUADS],
) -> (Code, [usize; DIMQUADS], bool) {
    let code = compute_code(star_xyz);
    enforce_invariants(code, star_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::sphere::radec_to_xyz;
    use std::f64::consts::PI;

    const CODE_EPS: f64 = 1e-10;

    fn assert_code_close(a: &Code, b: &Code, tol: f64) {
        for i in 0..DIMCODES {
            assert!(
                (a[i] - b[i]).abs() <= tol,
                "code[{i}]: {} vs {} (diff = {})",
                a[i],
                b[i],
                (a[i] - b[i]).abs()
            );
        }
    }

    fn rotate(m: &[[f64; 3]; 3], v: [f64; 3]) -> [f64; 3] {
        [
            m[0][0] * v[0] + m[0][1] * v[1] + m[0][2] * v[2],
            m[1][0] * v[0] + m[1][1] * v[1] + m[1][2] * v[2],
            m[2][0] * v[0] + m[2][1] * v[1] + m[2][2] * v[2],
        ]
    }

    fn rot_z(theta: f64) -> [[f64; 3]; 3] {
        let c = theta.cos();
        let s = theta.sin();
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    }

    fn rot_x(theta: f64) -> [[f64; 3]; 3] {
        let c = theta.cos();
        let s = theta.sin();
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]
    }

    fn rot_y(theta: f64) -> [[f64; 3]; 3] {
        let c = theta.cos();
        let s = theta.sin();
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]]
    }

    fn mat_mul(a: &[[f64; 3]; 3], b: &[[f64; 3]; 3]) -> [[f64; 3]; 3] {
        let mut out = [[0.0; 3]; 3];
        for i in 0..3 {
            for j in 0..3 {
                out[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] + a[i][2] * b[2][j];
            }
        }
        out
    }

    fn test_stars() -> [[f64; 3]; DIMQUADS] {
        [
            radec_to_xyz(0.10, 0.20),
            radec_to_xyz(0.12, 0.21),
            radec_to_xyz(0.11, 0.205),
            radec_to_xyz(0.105, 0.195),
        ]
    }

    #[test]
    fn compute_code_deterministic() {
        let stars = test_stars();
        let c1 = compute_code(&stars);
        let c2 = compute_code(&stars);
        assert_code_close(&c1, &c2, 0.0);
    }

    #[test]
    fn compute_code_finite() {
        let stars = test_stars();
        let code = compute_code(&stars);
        for &v in &code {
            assert!(v.is_finite());
        }
    }

    #[test]
    fn rotation_invariance() {
        let stars = test_stars();
        let (code_orig, _, _) = compute_canonical_code(&stars, [0, 1, 2, 3]);

        let r = mat_mul(&rot_z(0.7), &mat_mul(&rot_x(1.3), &rot_y(0.5)));
        let rotated: [[f64; 3]; DIMQUADS] = std::array::from_fn(|i| rotate(&r, stars[i]));
        let (code_rot, _, _) = compute_canonical_code(&rotated, [0, 1, 2, 3]);

        assert_code_close(&code_orig, &code_rot, CODE_EPS);
    }

    #[test]
    fn rotation_invariance_multiple_angles() {
        let stars = test_stars();
        let (code_orig, _, _) = compute_canonical_code(&stars, [0, 1, 2, 3]);

        for angle in [0.1, 0.5, 1.0, 2.0, 3.0, 5.0] {
            let r = mat_mul(
                &rot_z(angle),
                &mat_mul(&rot_x(angle * 0.7), &rot_y(angle * 1.3)),
            );
            let rotated: [[f64; 3]; DIMQUADS] = std::array::from_fn(|i| rotate(&r, stars[i]));
            let (code_rot, _, _) = compute_canonical_code(&rotated, [0, 1, 2, 3]);
            assert_code_close(&code_orig, &code_rot, CODE_EPS);
        }
    }

    #[test]
    fn scale_invariance_small_angles() {
        // Scale invariance is approximate on the sphere. At small angular
        // scales the gnomonic projection is nearly linear, so codes for
        // geometrically similar quads at different scales should be close.
        let center_ra = 1.0;
        let center_dec = 0.5;

        let scale1 = 1e-4;
        let stars1 = [
            radec_to_xyz(center_ra - scale1, center_dec),
            radec_to_xyz(center_ra + scale1, center_dec),
            radec_to_xyz(center_ra + scale1 * 0.3, center_dec + scale1 * 0.7),
            radec_to_xyz(center_ra - scale1 * 0.2, center_dec - scale1 * 0.5),
        ];

        let scale2 = 2e-4;
        let stars2 = [
            radec_to_xyz(center_ra - scale2, center_dec),
            radec_to_xyz(center_ra + scale2, center_dec),
            radec_to_xyz(center_ra + scale2 * 0.3, center_dec + scale2 * 0.7),
            radec_to_xyz(center_ra - scale2 * 0.2, center_dec - scale2 * 0.5),
        ];

        let (code1, _, _) = compute_canonical_code(&stars1, [0, 1, 2, 3]);
        let (code2, _, _) = compute_canonical_code(&stars2, [0, 1, 2, 3]);

        assert_code_close(&code1, &code2, 1e-4);
    }

    #[test]
    fn scale_invariance_moderate_angles() {
        // Larger angular separations have more curvature, so use looser tolerance.
        let center_ra = 1.0;
        let center_dec = 0.5;

        let scale1 = 0.005;
        let stars1 = [
            radec_to_xyz(center_ra - scale1, center_dec),
            radec_to_xyz(center_ra + scale1, center_dec),
            radec_to_xyz(center_ra, center_dec + scale1 * 0.6),
            radec_to_xyz(center_ra + scale1 * 0.3, center_dec - scale1 * 0.4),
        ];

        let scale2 = 0.01;
        let stars2 = [
            radec_to_xyz(center_ra - scale2, center_dec),
            radec_to_xyz(center_ra + scale2, center_dec),
            radec_to_xyz(center_ra, center_dec + scale2 * 0.6),
            radec_to_xyz(center_ra + scale2 * 0.3, center_dec - scale2 * 0.4),
        ];

        let (code1, _, _) = compute_canonical_code(&stars1, [0, 1, 2, 3]);
        let (code2, _, _) = compute_canonical_code(&stars2, [0, 1, 2, 3]);

        assert_code_close(&code1, &code2, 5e-3);
    }

    #[test]
    fn invariant_cx_le_dx() {
        let stars = test_stars();
        let (code, _, _) = enforce_invariants(compute_code(&stars), [0, 1, 2, 3]);
        assert!(
            code[0] <= code[2] + 1e-15,
            "cx ({}) > dx ({})",
            code[0],
            code[2]
        );
    }

    #[test]
    fn invariant_mean_x_le_half() {
        let stars = test_stars();
        let (code, _, _) = enforce_invariants(compute_code(&stars), [0, 1, 2, 3]);
        let n = (DIMQUADS - 2) as f64;
        let mean_x = (code[0] + code[2]) / n;
        assert!(mean_x <= 0.5 + 1e-15, "mean_x ({mean_x}) > 0.5");
    }

    #[test]
    fn enforce_invariants_cx_swap() {
        let code: Code = [0.8, 0.3, 0.2, 0.7];
        let ids = [10, 20, 30, 40];
        let (new_code, new_ids, _) = enforce_invariants(code, ids);

        assert!(new_code[0] <= new_code[2]);
        assert_eq!(new_ids[2], 40);
        assert_eq!(new_ids[3], 30);
    }

    #[test]
    fn enforce_invariants_mean_x_flip() {
        let code: Code = [0.9, 0.1, 0.8, 0.2];
        let ids = [10, 20, 30, 40];
        let (new_code, new_ids, flipped) = enforce_invariants(code, ids);

        assert!(flipped);
        assert_eq!(new_ids[0], 20);
        assert_eq!(new_ids[1], 10);

        let n = (DIMQUADS - 2) as f64;
        let mean_x = (new_code[0] + new_code[2]) / n;
        assert!(mean_x <= 0.5 + 1e-15);
        assert!(new_code[0] <= new_code[2] + 1e-15);
    }

    #[test]
    fn known_configuration() {
        // Place 4 stars in a simple pattern on the equator.
        let a = radec_to_xyz(0.0, 0.0);
        let b = radec_to_xyz(0.02, 0.0);
        let c = radec_to_xyz(0.01, 0.005);
        let d = radec_to_xyz(0.01, -0.005);

        let stars = [a, b, c, d];
        let code = compute_code(&stars);

        for &v in &code {
            assert!(v.is_finite(), "code value not finite: {v}");
        }

        // Due to the coordinate convention (swapped x/y from star_coords),
        // C and D (which differ only in Dec) end up with their x-coords
        // summing to ~1.0 (they are symmetric about 0.5 in the rotated frame).
        assert!(
            (code[0] + code[2] - 1.0).abs() < 0.01,
            "cx ({}) + dx ({}) should be near 1.0",
            code[0],
            code[2]
        );
    }

    #[test]
    fn degenerate_very_close_stars() {
        let eps = 1e-8;
        let a = radec_to_xyz(1.0, 0.5);
        let b = radec_to_xyz(1.0 + eps, 0.5);
        let c = radec_to_xyz(1.0 + eps * 0.3, 0.5 + eps * 0.7);
        let d = radec_to_xyz(1.0 - eps * 0.2, 0.5 - eps * 0.4);
        let stars = [a, b, c, d];
        let code = compute_code(&stars);
        for &v in &code {
            assert!(v.is_finite(), "code contains non-finite value: {v}");
        }
    }

    #[test]
    fn degenerate_stars_near_pole() {
        let dec = PI / 2.0 - 0.01;
        let a = radec_to_xyz(0.0, dec);
        let b = radec_to_xyz(0.5, dec);
        let c = radec_to_xyz(0.25, dec + 0.002);
        let d = radec_to_xyz(0.75, dec - 0.002);
        let stars = [a, b, c, d];
        let code = compute_code(&stars);
        for &v in &code {
            assert!(
                v.is_finite(),
                "code contains non-finite value near pole: {v}"
            );
        }
    }

    #[test]
    fn canonical_code_idempotent() {
        let stars = test_stars();
        let code = compute_code(&stars);
        let (c1, ids1, _) = enforce_invariants(code, [0, 1, 2, 3]);
        let (c2, ids2, _) = enforce_invariants(c1, ids1);
        assert_code_close(&c1, &c2, 0.0);
        assert_eq!(ids1, ids2);
    }

    #[test]
    fn swapping_cd_gives_same_canonical() {
        let stars = test_stars();
        let swapped = [stars[0], stars[1], stars[3], stars[2]];
        let (code1, _, _) = compute_canonical_code(&stars, [0, 1, 2, 3]);
        let (code2, _, _) = compute_canonical_code(&swapped, [0, 1, 3, 2]);
        assert_code_close(&code1, &code2, CODE_EPS);
    }

    #[test]
    fn swapping_ab_gives_same_canonical() {
        let stars = test_stars();
        let swapped = [stars[1], stars[0], stars[2], stars[3]];
        let (code1, _, _) = compute_canonical_code(&stars, [0, 1, 2, 3]);
        let (code2, _, _) = compute_canonical_code(&swapped, [1, 0, 2, 3]);
        assert_code_close(&code1, &code2, CODE_EPS);
    }

    #[test]
    fn all_permutations_same_canonical() {
        // Any permutation of the 4 stars should yield the same canonical code
        // when we correctly identify the backbone vs non-backbone stars.
        // Here we test that swapping AB and swapping CD both independently
        // produce the same canonical result.
        let stars = test_stars();
        let (code_base, _, _) = compute_canonical_code(&stars, [0, 1, 2, 3]);

        // Swap A<->B
        let ab_swap = [stars[1], stars[0], stars[2], stars[3]];
        let (code_ab, _, _) = compute_canonical_code(&ab_swap, [1, 0, 2, 3]);
        assert_code_close(&code_base, &code_ab, CODE_EPS);

        // Swap C<->D
        let cd_swap = [stars[0], stars[1], stars[3], stars[2]];
        let (code_cd, _, _) = compute_canonical_code(&cd_swap, [0, 1, 3, 2]);
        assert_code_close(&code_base, &code_cd, CODE_EPS);

        // Both swapped
        let both_swap = [stars[1], stars[0], stars[3], stars[2]];
        let (code_both, _, _) = compute_canonical_code(&both_swap, [1, 0, 3, 2]);
        assert_code_close(&code_base, &code_both, CODE_EPS);
    }

    #[test]
    fn invariants_hold_for_many_configurations() {
        // Test invariants across many star configurations
        let configs: Vec<[[f64; 3]; DIMQUADS]> = vec![
            test_stars(),
            [
                radec_to_xyz(0.5, 0.3),
                radec_to_xyz(0.52, 0.31),
                radec_to_xyz(0.51, 0.305),
                radec_to_xyz(0.505, 0.295),
            ],
            [
                radec_to_xyz(3.0, -0.5),
                radec_to_xyz(3.01, -0.49),
                radec_to_xyz(3.005, -0.495),
                radec_to_xyz(3.008, -0.502),
            ],
        ];

        for stars in &configs {
            let (code, _, _) = compute_canonical_code(stars, [0, 1, 2, 3]);

            // cx <= dx
            assert!(
                code[0] <= code[2] + 1e-15,
                "cx ({}) > dx ({})",
                code[0],
                code[2]
            );

            // mean_x <= 0.5
            let mean_x = (code[0] + code[2]) / 2.0;
            assert!(mean_x <= 0.5 + 1e-15, "mean_x ({mean_x}) > 0.5");

            // all values finite
            for &v in &code {
                assert!(v.is_finite());
            }
        }
    }
}
