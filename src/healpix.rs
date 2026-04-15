//! HEALPix (Hierarchical Equal Area isoLatitude Pixelisation) utilities.
//!
//! Thin wrapper around the `cdshealpix` crate, providing a stable API
//! for zodiacal's index builder and solver.

use std::f64::consts::PI;

/// Nside for a given depth: 2^depth.
pub fn nside(depth: u8) -> u64 {
    1u64 << depth
}

/// Total number of pixels at a given depth: 12 * nside^2.
pub fn npix(depth: u8) -> u64 {
    cdshealpix::nested::n_hash(depth)
}

/// Solid angle (steradians) of a single pixel at the given depth.
pub fn pixel_area(depth: u8) -> f64 {
    4.0 * PI / npix(depth) as f64
}

/// Compute an appropriate HEALPix depth for a given angular scale (radians).
///
/// Returns the depth where each pixel is approximately the same angular size
/// as the given scale. Uses: pixel_side ≈ sqrt(4π / (12 * Nside²))
pub fn depth_for_scale(scale_rad: f64) -> u8 {
    let nside_f = (PI / 3.0).sqrt() / scale_rad;
    (nside_f.log2().ceil() as u8).min(29)
}

/// Convert (lon, lat) in radians to a nested HEALPix pixel index.
///
/// `lon` is right ascension (or longitude) in [0, 2π).
/// `lat` is declination (or latitude) in [-π/2, π/2].
pub fn lon_lat_to_nested(lon: f64, lat: f64, depth: u8) -> u64 {
    cdshealpix::nested::hash(depth, lon, lat)
}

/// Convert a nested HEALPix pixel index to the (lon, lat) of its center.
///
/// Returns (lon, lat) in radians.
pub fn nested_to_center(hash: u64, depth: u8) -> (f64, f64) {
    cdshealpix::nested::center(depth, hash)
}

/// Return the (up to 8) neighbouring pixel indices in nested scheme.
pub fn neighbours(hash: u64, depth: u8) -> Vec<u64> {
    let mut result = Vec::with_capacity(9);
    cdshealpix::nested::append_bulk_neighbours(depth, hash, &mut result);
    result
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
