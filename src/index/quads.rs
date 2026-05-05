//! Per-cell quad construction shared across the single-band and
//! multi-band cell-driven builders.
//!
//! `build_quads_for_cell` was originally a private helper inside
//! `cell_builder.rs`. PR3 hoists it (plus the local
//! `canonical_quad_order` helper) into its own module so the
//! multi-band orchestrator can call the same primitive once per band
//! without re-implementing it. The single-band path keeps its existing
//! shape: `cell_builder::build_index_cell_driven` invokes
//! [`build_quads_for_cell`] verbatim.
//!
//! Behaviour is byte-identical to the prior in-file definition; the
//! signatures are unchanged on purpose so existing callers re-import
//! and keep working.

use std::collections::HashSet;

use crate::geom::sphere::{angular_distance, radec_to_xyz, star_midpoint};
use crate::quads::{Code, DIMQUADS, Quad, compute_canonical_code};

use super::cell_builder::{CellBuildConfig, CellStar};
use super::multiband_cell_builder::ScaleBand;

/// Build quads for one cell using only the cell's own stars. Stars are
/// sorted by magnitude (brightest first); quad indices in the returned
/// `Quad` values are positions in the *input* `stars` slice (i.e.
/// pre-sort indices), so callers can look up `catalog_id`/etc. directly.
///
/// This is the original `cell_builder::build_quads_for_cell`, hoisted
/// to a shared module in PR3 so the multi-band orchestrator can reuse
/// it. Behaviour is byte-identical to the prior implementation.
pub fn build_quads_for_cell(
    stars: &[CellStar],
    config: &CellBuildConfig,
) -> (Vec<Quad>, Vec<Code>) {
    if stars.len() < DIMQUADS {
        return (Vec::new(), Vec::new());
    }

    // Sort indices by magnitude (brightest first) so use_count limits
    // bias toward the brightest stars.
    let mut order: Vec<usize> = (0..stars.len()).collect();
    order.sort_by(|&a, &b| {
        stars[a]
            .mag
            .partial_cmp(&stars[b].mag)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let xyzs: Vec<[f64; 3]> = order
        .iter()
        .map(|&i| radec_to_xyz(stars[i].ra_rad, stars[i].dec_rad))
        .collect();

    let mut quads: Vec<Quad> = Vec::new();
    let mut codes: Vec<Code> = Vec::new();
    let mut seen: HashSet<[usize; DIMQUADS]> = HashSet::new();
    let mut use_count: Vec<usize> = vec![0; xyzs.len()];

    let chord_sq_upper = 2.0 * (1.0 - config.scale_upper.cos());

    'outer: for a_idx in 0..xyzs.len() {
        if quads.len() >= config.quads_per_cell {
            break;
        }
        if use_count[a_idx] >= config.max_reuse {
            continue;
        }

        let a_xyz = xyzs[a_idx];
        for b_idx in (a_idx + 1)..xyzs.len() {
            if use_count[b_idx] >= config.max_reuse {
                continue;
            }
            let b_xyz = xyzs[b_idx];
            let ab_dist = angular_distance(a_xyz, b_xyz);
            if ab_dist < config.scale_lower || ab_dist > config.scale_upper {
                continue;
            }

            let mid = star_midpoint(a_xyz, b_xyz);
            let cd_radius_sq = 2.0 * (1.0 - ab_dist.cos());

            // Linear scan for candidates near the midpoint. At
            // per-cell granularity n_stars is small (tens to a few
            // hundred typically), so an O(n) scan beats a KD-tree
            // construction. The chord_sq_upper outer guard avoids
            // computing midpoints on too-distant pairs.
            let _ = chord_sq_upper;

            let mut candidates: Vec<usize> = Vec::new();
            for (c_idx, c) in xyzs.iter().enumerate() {
                if c_idx == a_idx || c_idx == b_idx {
                    continue;
                }
                let dx = c[0] - mid[0];
                let dy = c[1] - mid[1];
                let dz = c[2] - mid[2];
                if dx * dx + dy * dy + dz * dz < cd_radius_sq {
                    candidates.push(c_idx);
                }
            }

            for ci in 0..candidates.len() {
                if quads.len() >= config.quads_per_cell {
                    break 'outer;
                }
                for di in (ci + 1)..candidates.len() {
                    let c_idx = candidates[ci];
                    let d_idx = candidates[di];

                    let mut key = [a_idx, b_idx, c_idx, d_idx];
                    key.sort();
                    if !seen.insert(key) {
                        continue;
                    }

                    if use_count[c_idx] >= config.max_reuse || use_count[d_idx] >= config.max_reuse
                    {
                        continue;
                    }

                    let raw_xyz = [a_xyz, b_xyz, xyzs[c_idx], xyzs[d_idx]];
                    let raw_ids = [a_idx, b_idx, c_idx, d_idx];
                    let (ordered_xyz, ordered_ids) = canonical_quad_order(&raw_xyz, raw_ids);
                    let (code, canonical_ids, _) =
                        compute_canonical_code(&ordered_xyz, ordered_ids);

                    for &idx in &canonical_ids {
                        use_count[idx] += 1;
                    }

                    // Translate canonical_ids (positions in the sorted
                    // local list) back to positions in the *input*
                    // `stars` slice so the caller can look up
                    // `catalog_id` directly.
                    let star_ids_input: [usize; DIMQUADS] =
                        std::array::from_fn(|i| order[canonical_ids[i]]);
                    quads.push(Quad {
                        star_ids: star_ids_input,
                    });
                    codes.push(code);

                    if quads.len() >= config.quads_per_cell {
                        break 'outer;
                    }
                }
            }
        }
    }

    (quads, codes)
}

/// Canonical quad ordering: place the longest-backbone pair at indices
/// `[0]` and `[1]`. This is a near-duplicate of
/// `builder::canonical_quad_order` kept here to avoid widening that
/// helper's visibility.
pub(crate) fn canonical_quad_order(
    star_xyz: &[[f64; 3]; DIMQUADS],
    star_ids: [usize; DIMQUADS],
) -> ([[f64; 3]; DIMQUADS], [usize; DIMQUADS]) {
    let mut best_pair = (0, 1);
    let mut best_dist = 0.0f64;
    for i in 0..DIMQUADS {
        for j in (i + 1)..DIMQUADS {
            let d = angular_distance(star_xyz[i], star_xyz[j]);
            if d > best_dist {
                best_dist = d;
                best_pair = (i, j);
            }
        }
    }
    let (ai, bi) = best_pair;
    let mut others: Vec<usize> = (0..DIMQUADS).filter(|&i| i != ai && i != bi).collect();
    others.sort_by_key(|&i| star_ids[i]);
    let order = [ai, bi, others[0], others[1]];
    let new_xyz: [[f64; 3]; DIMQUADS] = std::array::from_fn(|i| star_xyz[order[i]]);
    let new_ids: [usize; DIMQUADS] = std::array::from_fn(|i| star_ids[order[i]]);
    (new_xyz, new_ids)
}

/// Build per-band quads for one cell over the same `stars` buffer.
///
/// Calls [`build_quads_for_cell`] once per band, with a synthetic
/// [`CellBuildConfig`] derived from each [`ScaleBand`]. Returns one
/// `(band_idx, quads, codes)` entry per band, in the input order of
/// `bands`.
pub fn build_quads_for_cell_multiband(
    stars: &[CellStar],
    bands: &[ScaleBand],
) -> Vec<(u32, Vec<Quad>, Vec<Code>)> {
    bands
        .iter()
        .map(|band| {
            let cfg = CellBuildConfig {
                scale_lower: arcsec_to_rad(band.scale_lower_arcsec),
                scale_upper: arcsec_to_rad(band.scale_upper_arcsec),
                quads_per_cell: band.quads_per_cell,
                max_reuse: band.max_reuse,
                // Unused by build_quads_for_cell — only the four fields
                // above matter to the inner loop. Fill with sane
                // defaults so the struct is well-formed.
                final_cell_depth: 0,
                pivot_stride: 1,
            };
            let (quads, codes) = build_quads_for_cell(stars, &cfg);
            (band.band_idx, quads, codes)
        })
        .collect()
}

#[inline]
fn arcsec_to_rad(arcsec: f64) -> f64 {
    arcsec * std::f64::consts::PI / (180.0 * 3600.0)
}
