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
    use std::f64::consts::{FRAC_PI_2, TAU};

    const EPS: f64 = 1e-8;

    #[test]
    fn nside_and_npix() {
        assert_eq!(nside(0), 1);
        assert_eq!(nside(1), 2);
        assert_eq!(nside(3), 8);

        assert_eq!(npix(0), 12);
        assert_eq!(npix(1), 48);
        assert_eq!(npix(2), 192);
    }

    #[test]
    fn pixel_area_sum() {
        for depth in 0..5 {
            let total = pixel_area(depth) * npix(depth) as f64;
            assert!(
                (total - 4.0 * PI).abs() < EPS,
                "depth {depth}: total={total}"
            );
        }
    }

    #[test]
    fn roundtrip_known_positions() {
        let positions = [
            (0.0, 0.0),
            (PI, 0.0),
            (FRAC_PI_2, std::f64::consts::FRAC_PI_4),
            (0.0, 1.3),
            (PI, -1.3),
            (1.0, 0.5),
            (5.0, -0.3),
        ];

        for depth in 1..8 {
            for &(lon, lat) in &positions {
                let hash = lon_lat_to_nested(lon, lat, depth);
                assert!(
                    hash < npix(depth),
                    "hash {hash} >= npix {} at depth {depth}",
                    npix(depth)
                );

                let (clon, clat) = nested_to_center(hash, depth);

                let pixel_rad = pixel_area(depth).sqrt();
                let dlon = (clon - lon).abs().min(TAU - (clon - lon).abs());
                let dlat = (clat - lat).abs();
                assert!(
                    dlon < pixel_rad * 3.0 && dlat < pixel_rad * 3.0,
                    "depth {depth}, ({lon}, {lat}) -> hash {hash} -> ({clon}, {clat}), dlon={dlon}, dlat={dlat}"
                );
            }
        }
    }

    #[test]
    fn all_pixels_covered() {
        for depth in 0..4 {
            let mut seen = vec![false; npix(depth) as usize];

            let n = 500;
            for i in 0..n {
                let lon = TAU * i as f64 / n as f64;
                for j in 0..n {
                    let lat = -FRAC_PI_2 + PI * j as f64 / (n - 1) as f64;
                    let hash = lon_lat_to_nested(lon, lat, depth);
                    seen[hash as usize] = true;
                }
            }

            let covered = seen.iter().filter(|&&v| v).count();
            assert_eq!(
                covered,
                npix(depth) as usize,
                "depth {depth}: only {covered}/{} pixels covered",
                npix(depth)
            );
        }
    }

    #[test]
    fn neighbours_count() {
        for depth in 2..6 {
            let hash = npix(depth) / 2;
            let nbrs = neighbours(hash, depth);
            assert!(
                nbrs.len() >= 7 && nbrs.len() <= 8,
                "pixel {hash} at depth {depth} has {} neighbours",
                nbrs.len()
            );
        }
    }

    #[test]
    fn neighbours_symmetric() {
        for depth in 1..5 {
            let np = npix(depth);
            for hash in 0..np {
                let nbrs = neighbours(hash, depth);
                for &n in &nbrs {
                    let n_nbrs = neighbours(n, depth);
                    assert!(
                        n_nbrs.contains(&hash),
                        "depth {depth}: pixel {hash} has neighbour {n}, but {n} does not list {hash}"
                    );
                }
            }
        }
    }

    #[test]
    fn neighbours_valid_range() {
        for depth in 0..5 {
            let np = npix(depth);
            for hash in 0..np {
                let nbrs = neighbours(hash, depth);
                for &n in &nbrs {
                    assert!(
                        n < np,
                        "depth {depth}, hash {hash}: neighbour {n} >= npix {np}"
                    );
                }
                assert!(
                    !nbrs.contains(&hash),
                    "depth {depth}, hash {hash}: self-loop"
                );
            }
        }
    }

    #[test]
    fn depth_for_scale_reasonable() {
        let d = depth_for_scale(0.0175);
        assert!(d >= 4 && d <= 8, "depth_for_scale(1deg) = {d}, expected ~6");

        let d2 = depth_for_scale(0.000291);
        assert!(
            d2 >= 10 && d2 <= 14,
            "depth_for_scale(1arcmin) = {d2}, expected ~12"
        );
    }

    #[test]
    fn north_pole() {
        for depth in 1..8 {
            let hash = lon_lat_to_nested(0.0, FRAC_PI_2, depth);
            assert!(hash < npix(depth));
            let (_, lat) = nested_to_center(hash, depth);
            assert!(lat > 1.0, "north pole center lat = {lat}");
        }
    }

    #[test]
    fn south_pole() {
        for depth in 1..8 {
            let hash = lon_lat_to_nested(0.0, -FRAC_PI_2, depth);
            assert!(hash < npix(depth));
            let (_, lat) = nested_to_center(hash, depth);
            assert!(lat < -1.0, "south pole center lat = {lat}");
        }
    }
}
