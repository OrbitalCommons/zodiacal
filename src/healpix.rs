//! HEALPix utilities that complement the `cdshealpix` crate.

use std::f64::consts::PI;

/// Compute an appropriate HEALPix depth for a given angular scale (radians).
///
/// Returns the depth where each pixel is approximately the same angular size
/// as the given scale.
pub fn depth_for_scale(scale_rad: f64) -> u8 {
    let nside_f = (PI / 3.0).sqrt() / scale_rad;
    (nside_f.log2().ceil() as u8).min(29)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn depth_for_scale_reasonable() {
        // 1 degree ~ 0.0175 rad → depth ~6
        let d = depth_for_scale(0.0175);
        assert!(d >= 4 && d <= 8, "depth_for_scale(1deg) = {d}, expected ~6");

        // 1 arcmin ~ 0.000291 rad → depth ~12
        let d2 = depth_for_scale(0.000291);
        assert!(
            d2 >= 10 && d2 <= 14,
            "depth_for_scale(1arcmin) = {d2}, expected ~12"
        );
    }
}
