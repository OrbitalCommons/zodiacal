//! HEALPix (Hierarchical Equal Area isoLatitude Pixelisation) implementation.
//!
//! Implements the nested indexing scheme for HEALPix, ported from the
//! astrometry.net C implementation (`util/healpix.c`).
//!
//! The 12 base healpixes are laid out as:
//! - 0–3: north polar cap
//! - 4–7: equatorial belt
//! - 8–11: south polar cap
//!
//! Within each base healpix, `x` increases northeast and `y` increases northwest.

use std::f64::consts::{FRAC_PI_2, FRAC_PI_4, PI, TAU};

/// Nside for a given depth: 2^depth.
pub fn nside(depth: u8) -> u64 {
    1u64 << depth
}

/// Total number of pixels at a given depth: 12 * nside^2.
pub fn npix(depth: u8) -> u64 {
    12 * nside(depth) * nside(depth)
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
    // pixel_area = 4*pi / (12*Nside^2) = pi / (3 * Nside^2)
    // We want pixel_side ~ scale_rad, where pixel_side ~ sqrt(pixel_area)
    // sqrt(pi / (3 * Nside^2)) ~ scale_rad
    // Nside^2 ~ pi / (3 * scale_rad^2)
    // Nside ~ sqrt(pi/3) / scale_rad
    let nside_f = (PI / 3.0).sqrt() / scale_rad;
    (nside_f.log2().ceil() as u8).min(29)
}

/// Convert (lon, lat) in radians to a nested HEALPix pixel index.
///
/// `lon` is right ascension (or longitude) in [0, 2π).
/// `lat` is declination (or latitude) in [-π/2, π/2].
pub fn lon_lat_to_nested(lon: f64, lat: f64, depth: u8) -> u64 {
    let (base, x, y) = lon_lat_to_base_xy(lon, lat, nside(depth) as f64);
    compose_nested(base, x, y, depth)
}

/// Convert a nested HEALPix pixel index to the (lon, lat) of its center.
///
/// Returns (lon, lat) in radians.
pub fn nested_to_center(hash: u64, depth: u8) -> (f64, f64) {
    let (base, x, y) = decompose_nested(hash, depth);
    base_xy_to_lon_lat(base, x as f64 + 0.5, y as f64 + 0.5, nside(depth) as f64)
}

/// Return the (up to 8) neighbouring pixel indices in nested scheme.
pub fn neighbours(hash: u64, depth: u8) -> Vec<u64> {
    let ns = nside(depth) as i64;
    let (base, x, y) = decompose_nested(hash, depth);
    let x = x as i64;
    let y = y as i64;

    // 8 directions: E, NE, N, NW, W, SW, S, SE
    let dirs: [(i64, i64); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];

    let mut result = Vec::with_capacity(8);

    for (dx, dy) in dirs {
        let nx = x + dx;
        let ny = y + dy;

        if nx >= 0 && nx < ns && ny >= 0 && ny < ns {
            // Still within the same base healpix
            result.push(compose_nested(base, nx as u64, ny as u64, depth));
            continue;
        }

        // Crossed a boundary — find the neighbor base healpix
        let cross_x = nx < 0 || nx >= ns;
        let cross_y = ny < 0 || ny >= ns;

        let neighbour_base = if cross_x && cross_y {
            base_neighbour(base, dx.signum(), dy.signum())
        } else if cross_x {
            base_neighbour(base, dx.signum(), 0)
        } else {
            base_neighbour(base, 0, dy.signum())
        };

        let Some(nb) = neighbour_base else {
            continue;
        };

        // Compute coordinates in the neighbor base healpix.
        // When crossing between different "rows" of the base healpix grid,
        // coordinates may need to be transformed.
        let (fnx, fny) = transform_across_boundary(base, nb, nx, ny, ns);

        if fnx >= 0 && fnx < ns && fny >= 0 && fny < ns {
            result.push(compose_nested(nb, fnx as u64, fny as u64, depth));
        }
    }

    result
}

// ---------------------------------------------------------------------------
// Internal: base healpix classification
// ---------------------------------------------------------------------------

fn is_north(base: u64) -> bool {
    base <= 3
}

fn is_south(base: u64) -> bool {
    base >= 8
}

// ---------------------------------------------------------------------------
// Internal: coordinate ↔ (base, x, y)
// ---------------------------------------------------------------------------

/// Convert (lon, lat) to (base_hp, x, y) in the XY scheme with continuous coords.
fn lon_lat_to_base_xy(lon: f64, lat: f64, ns: f64) -> (u64, u64, u64) {
    let z = lat.sin();
    let mut phi = lon;
    if phi < 0.0 {
        phi += TAU;
    }
    if phi >= TAU {
        phi -= TAU;
    }

    let phi_t = phi % FRAC_PI_2;

    // Determine quadrant column
    let column = ((phi / FRAC_PI_2).floor() as i64).rem_euclid(4) as u64;

    if z.abs() >= 2.0 / 3.0 {
        // Polar cap
        let north = z >= 0.0;
        let zfactor = if north { 1.0 } else { -1.0 };

        // Solve eqns 19/20 from the HEALPix paper for kx = Ns - xx, ky = Ns - yy
        let root_x = (1.0 - z * zfactor) * 3.0 * (ns * (2.0 * phi_t - PI) / PI).powi(2);
        let kx = if root_x <= 0.0 { 0.0 } else { root_x.sqrt() };

        let root_y = (1.0 - z * zfactor) * 3.0 * (ns * 2.0 * phi_t / PI).powi(2);
        let ky = if root_y <= 0.0 { 0.0 } else { root_y.sqrt() };

        let (xx, yy) = if north { (ns - kx, ns - ky) } else { (ky, kx) };

        let x = (xx.floor() as u64).min(ns as u64 - 1);
        let y = (yy.floor() as u64).min(ns as u64 - 1);

        let base = if north { column } else { 8 + column };
        (base, x, y)
    } else {
        // Equatorial region
        let zunits = (z + 2.0 / 3.0) / (4.0 / 3.0);
        let phiunits = phi_t / FRAC_PI_2;

        let u1 = zunits + phiunits;
        let u2 = zunits - phiunits + 1.0;

        let mut xx = u1 * ns;
        let mut yy = u2 * ns;

        let base = if xx >= ns {
            xx -= ns;
            if yy >= ns {
                yy -= ns;
                column // north polar
            } else {
                ((column + 1) % 4) + 4 // right equatorial
            }
        } else if yy >= ns {
            yy -= ns;
            column + 4 // left equatorial
        } else {
            8 + column // south polar
        };

        let x = (xx.floor() as u64).min(ns as u64 - 1);
        let y = (yy.floor() as u64).min(ns as u64 - 1);

        (base, x, y)
    }
}

/// Convert (base_hp, x, y) continuous coords back to (lon, lat).
fn base_xy_to_lon_lat(base: u64, x: f64, y: f64, ns: f64) -> (f64, f64) {
    let x_norm = x / ns;
    let y_norm = y / ns;

    // Check if this pixel is in the polar or equatorial regime
    let is_polar_region = if is_north(base) {
        (x_norm + y_norm) > 1.0
    } else if is_south(base) {
        (x_norm + y_norm) < 1.0
    } else {
        false
    };

    if !is_polar_region {
        // Equatorial computation
        let (phi_off, z_off, chp) = if base <= 3 {
            (1.0, 0.0, base)
        } else if base <= 7 {
            (0.0, -1.0, base - 4)
        } else {
            (1.0, -2.0, base - 8)
        };

        let z = (2.0 / 3.0) * (x_norm + y_norm + z_off);
        let phi = FRAC_PI_4 * (x_norm - y_norm + phi_off + 2.0 * chp as f64);

        let lat = z.clamp(-1.0, 1.0).asin();
        let mut lon = phi;
        if lon < 0.0 {
            lon += TAU;
        }
        if lon >= TAU {
            lon -= TAU;
        }
        (lon, lat)
    } else {
        // Polar computation — inverse of eqns 19/20 from HEALPix paper
        let north = is_north(base);
        let zfactor = if north { 1.0 } else { -1.0 };

        // For south polar, swap and flip to work in north-polar convention
        let (px, py) = if north { (x, y) } else { (ns - y, ns - x) };

        let kx = ns - px;
        let ky = ns - py;

        // phi_t = pi * (Ns - y) / (2 * ((Ns - x) + (Ns - y)))
        let phi_t = if kx + ky == 0.0 {
            0.0
        } else {
            PI * ky / (2.0 * (kx + ky))
        };

        // Recover z, using two branches to avoid division-by-zero
        let z = if phi_t < FRAC_PI_4 {
            let denom = (2.0 * phi_t - PI) * ns;
            if denom.abs() < 1e-15 {
                zfactor
            } else {
                let val = PI * kx / denom;
                (1.0 - val * val / 3.0) * zfactor
            }
        } else {
            let denom = 2.0 * phi_t * ns;
            if denom.abs() < 1e-15 {
                zfactor
            } else {
                let val = PI * ky / denom;
                (1.0 - val * val / 3.0) * zfactor
            }
        };

        let base_col = if is_south(base) { base - 8 } else { base };
        let phi = FRAC_PI_2 * base_col as f64 + phi_t;

        let lat = z.clamp(-1.0, 1.0).asin();
        let mut lon = phi;
        if lon < 0.0 {
            lon += TAU;
        }
        if lon >= TAU {
            lon -= TAU;
        }
        (lon, lat)
    }
}

// ---------------------------------------------------------------------------
// Internal: XY ↔ nested bit-interleaving
// ---------------------------------------------------------------------------

/// Compose a nested index from (base, x, y).
fn compose_nested(base: u64, x: u64, y: u64, depth: u8) -> u64 {
    let ns2 = nside(depth) * nside(depth);
    let sub = xy_to_nested_sub(x, y);
    base * ns2 + sub
}

/// Decompose a nested index into (base, x, y).
fn decompose_nested(hash: u64, depth: u8) -> (u64, u64, u64) {
    let ns2 = nside(depth) * nside(depth);
    let base = hash / ns2;
    let sub = hash % ns2;
    let (x, y) = nested_sub_to_xy(sub);
    (base, x, y)
}

/// Bit-interleave (x, y) → sub-index. x provides even bits, y provides odd bits.
fn xy_to_nested_sub(x: u64, y: u64) -> u64 {
    let mut result = 0u64;
    let mut xx = x;
    let mut yy = y;
    let mut bit = 0;
    while xx > 0 || yy > 0 {
        result |= (xx & 1) << bit;
        bit += 1;
        result |= (yy & 1) << bit;
        bit += 1;
        xx >>= 1;
        yy >>= 1;
    }
    result
}

/// De-interleave sub-index → (x, y).
fn nested_sub_to_xy(sub: u64) -> (u64, u64) {
    let mut x = 0u64;
    let mut y = 0u64;
    let mut s = sub;
    let mut bit = 0;
    while s > 0 {
        x |= (s & 1) << bit;
        s >>= 1;
        y |= (s & 1) << bit;
        s >>= 1;
        bit += 1;
    }
    (x, y)
}

// ---------------------------------------------------------------------------
// Internal: base healpix adjacency
// ---------------------------------------------------------------------------

/// Return the neighboring base healpix in direction (dx, dy), where each is -1, 0, or +1.
/// Returns None if no such neighbor exists.
fn base_neighbour(base: u64, dx: i64, dy: i64) -> Option<u64> {
    let hp = base as i64;

    if dx == 0 && dy == 0 {
        return Some(base);
    }

    if is_north(base) {
        // North polar: base 0..3
        let col = hp; // 0..3
        match (dx, dy) {
            (1, 0) => Some(((col + 1) % 4) as u64),
            (0, 1) => Some(((col + 3) % 4) as u64),
            (1, 1) => Some(((col + 2) % 4) as u64),
            (-1, 0) => Some((col + 4) as u64),
            (0, -1) => Some((4 + (col + 1) % 4) as u64),
            (-1, -1) => Some((col + 8) as u64),
            _ => None,
        }
    } else if is_south(base) {
        // South polar: base 8..11
        let col = hp - 8; // 0..3
        match (dx, dy) {
            (1, 0) => Some((4 + (col + 1) % 4) as u64),
            (0, 1) => Some((col + 4) as u64),
            (1, 1) => Some(col as u64), // to north polar
            (-1, 0) => Some((8 + (col + 3) % 4) as u64),
            (0, -1) => Some((8 + (col + 1) % 4) as u64),
            (-1, -1) => Some((8 + (col + 2) % 4) as u64),
            _ => None,
        }
    } else {
        // Equatorial: base 4..7
        let col = hp - 4; // 0..3
        match (dx, dy) {
            (1, 0) => Some(col as u64),                                   // to north
            (0, 1) => Some(((col + 3) % 4) as u64),                       // to north
            (-1, 0) => Some((8 + (col + 3) % 4) as u64),                  // to south
            (0, -1) => Some((col + 8) as u64),                            // to south
            (1, -1) => Some((4 + (col + 1) % 4) as u64),                  // to equatorial right
            (-1, 1) => Some(((4 + (col + 3) % 4).rem_euclid(12)) as u64), // to equatorial left
            _ => None,
        }
    }
}

/// Transform coordinates when crossing from one base healpix to another.
///
/// Given the original (nx, ny) that fell outside [0, ns) in `from_base`,
/// compute the valid coordinates in `to_base`.
fn transform_across_boundary(
    from_base: u64,
    to_base: u64,
    nx: i64,
    ny: i64,
    ns: i64,
) -> (i64, i64) {
    let from_row = base_row(from_base);
    let to_row = base_row(to_base);

    // Wrap coordinates into [0, ns) as a starting point
    let mut fnx = nx.rem_euclid(ns);
    let mut fny = ny.rem_euclid(ns);

    // When crossing between different rows (polar/equatorial), coordinates
    // may need to be swapped and/or reflected.
    let crossed_x = nx < 0 || nx >= ns;
    let crossed_y = ny < 0 || ny >= ns;

    match (from_row, to_row) {
        // North polar to north polar: swap coords
        (0, 0) => {
            if crossed_x && !crossed_y {
                // Crossed x boundary: swap and set x to edge
                fnx = ny;
                fny = ns - 1;
            } else if crossed_y && !crossed_x {
                // Crossed y boundary: swap and set y to edge
                fny = nx;
                fnx = ns - 1;
            } else {
                // Corner: diagonal neighbor
                fnx = ns - 1;
                fny = ns - 1;
            }
        }
        // South polar to south polar: swap coords (mirror of north-north)
        (2, 2) => {
            if crossed_x && !crossed_y {
                fnx = ny.rem_euclid(ns);
                fny = 0;
            } else if crossed_y && !crossed_x {
                fny = nx.rem_euclid(ns);
                fnx = 0;
            } else {
                fnx = 0;
                fny = 0;
            }
        }
        // Same row, just wrap
        (a, b) if a == b => {
            // Equatorial-to-equatorial or same-row wrapping
        }
        // North polar to equatorial: just wrap
        (0, 1) | (1, 0) | (1, 2) | (2, 1) => {
            // Standard wrapping is sufficient for these transitions
        }
        _ => {}
    }

    (fnx, fny)
}

/// Return the row of a base healpix: 0=north, 1=equatorial, 2=south.
fn base_row(base: u64) -> u8 {
    if base <= 3 {
        0
    } else if base <= 7 {
        1
    } else {
        2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::{FRAC_PI_2, PI};

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
        // Sum of all pixel areas should be 4π
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
            (0.0, 0.0),             // on equator
            (PI, 0.0),              // equator, opposite side
            (FRAC_PI_2, FRAC_PI_4), // mid-latitude
            (0.0, 1.3),             // near north pole
            (PI, -1.3),             // near south pole
            (1.0, 0.5),             // generic
            (5.0, -0.3),            // another generic
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

                // Center should be within roughly one pixel of the input
                let pixel_rad = pixel_area(depth).sqrt();
                let dlon = (clon - lon).abs().min(TAU - (clon - lon).abs());
                let dlat = (clat - lat).abs();
                assert!(
                    dlon < pixel_rad * 3.0 && dlat < pixel_rad * 3.0,
                    "depth {depth}, ({lon}, {lat}) -> hash {hash} -> ({clon}, {clat}), dlon={dlon}, dlat={dlat}, pixel_rad={pixel_rad}"
                );
            }
        }
    }

    #[test]
    fn all_pixels_covered() {
        // At low depth, every pixel should be reachable
        for depth in 0..4 {
            let mut seen = vec![false; npix(depth) as usize];

            // Sample a dense grid of sky positions
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
    fn bit_interleave_roundtrip() {
        for x in 0..32 {
            for y in 0..32 {
                let sub = xy_to_nested_sub(x, y);
                let (rx, ry) = nested_sub_to_xy(sub);
                assert_eq!((x, y), (rx, ry), "roundtrip failed for ({x}, {y})");
            }
        }
    }

    #[test]
    fn neighbours_count() {
        // Interior pixels should have 8 neighbours
        for depth in 2..6 {
            let ns = nside(depth);
            // Pick an interior pixel (base=4, x=ns/2, y=ns/2)
            let hash = compose_nested(4, ns / 2, ns / 2, depth);
            let nbrs = neighbours(hash, depth);
            assert_eq!(
                nbrs.len(),
                8,
                "interior pixel at depth {depth} should have 8 neighbours"
            );
        }
    }

    #[test]
    fn neighbours_symmetric() {
        // If A is a neighbour of B, then B should be a neighbour of A
        for depth in 1..5 {
            let np = npix(depth);
            for hash in 0..np {
                let nbrs = neighbours(hash, depth);
                for &n in &nbrs {
                    let n_nbrs = neighbours(n, depth);
                    assert!(
                        n_nbrs.contains(&hash),
                        "depth {depth}: pixel {hash} has neighbour {n}, but {n} does not list {hash}. \
                         {hash}'s nbrs: {nbrs:?}, {n}'s nbrs: {n_nbrs:?}"
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
                // No self-loops
                assert!(
                    !nbrs.contains(&hash),
                    "depth {depth}, hash {hash}: self-loop"
                );
            }
        }
    }

    #[test]
    fn depth_for_scale_reasonable() {
        // 1 degree ~ 0.0175 rad
        let d = depth_for_scale(0.0175);
        // At depth d, pixel side ~ 0.0175 rad
        // nside(d) ~ sqrt(pi/3) / 0.0175 ~ 58.5, so depth ~ 6
        assert!(d >= 4 && d <= 8, "depth_for_scale(1deg) = {d}, expected ~6");

        // 1 arcmin ~ 0.000291 rad
        let d2 = depth_for_scale(0.000291);
        // nside ~ 3531, depth ~ 12
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
