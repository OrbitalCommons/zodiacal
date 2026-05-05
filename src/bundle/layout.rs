//! Shared filename + path helpers for the bundle on-disk layout.
//!
//! Bundle per-cell shards live under `<root>/quads/cell_NNNNN.zqd` and
//! `<root>/gaia/cell_NNNNN.zga`. The width `N` is a function of the
//! bundle's HEALPix `cell_depth` — wider cell counts need wider names.
//! These helpers centralise that computation so the builder, tidy
//! phase, and reader all agree on the format.
//!
//! The recipe (matches `docs/bundle-format.md`):
//!
//! - At depth `d` there are `12 * 4^d` cells.
//! - Decimal width of the maximum cell id is `ceil(log10(n_cells))`.
//! - We pad with one extra digit ("safety margin"), and never emit
//!   fewer than two digits so even depth 0 yields a readable name.

use std::path::{Path, PathBuf};

/// Number of HEALPix cells at the given `cell_depth` (always
/// `12 * 4^depth`).
#[inline]
pub fn n_cells_at_depth(depth: u8) -> u64 {
    12u64 << (2 * depth as u64)
}

/// Filename width for `cell_NNNN…` shard names at the given depth.
///
/// Width is `max(2, ceil(log10(n_cells)) + 1)` — the `+ 1` is a safety
/// margin so id boundaries don't accidentally squish to the rim of the
/// allocated digit count, and the `max(2, …)` floor keeps depth-0
/// names readable.
#[inline]
pub fn cell_filename_width(depth: u8) -> usize {
    let n = n_cells_at_depth(depth);
    let max_id = n.saturating_sub(1);
    // ceil(log10(max_id + 1)) — use the trivial digit count of the
    // largest id rather than playing games with log10.
    let mut digits = 1usize;
    let mut v = max_id;
    while v >= 10 {
        v /= 10;
        digits += 1;
    }
    let with_safety = digits + 1;
    with_safety.max(2)
}

/// Build a path to a per-cell shard file inside the bundle root or
/// work_dir.
///
/// `subdir` is the per-extension directory (`"quads"` or `"gaia"`).
/// `ext` is the file extension *without* the leading dot
/// (`"zqd"`/`"zga"`). `cell_id` is the HEALPix index at `depth`.
pub fn cell_shard_path(
    work_dir: &Path,
    subdir: &str,
    ext: &str,
    depth: u8,
    cell_id: u32,
) -> PathBuf {
    let width = cell_filename_width(depth);
    let name = format!("cell_{cell_id:0width$}.{ext}");
    work_dir.join(subdir).join(name)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cell_filename_width_depth_0() {
        // 12 cells, max id 11 (2 digits) + 1 safety = 3.
        assert_eq!(cell_filename_width(0), 3);
    }

    #[test]
    fn cell_filename_width_depth_5() {
        // 12,288 cells, max id 12,287 (5 digits) + 1 safety = 6.
        assert_eq!(cell_filename_width(5), 6);
    }

    #[test]
    fn cell_filename_width_depth_8() {
        // 786,432 cells, max id 786,431 (6 digits) + 1 safety = 7.
        assert_eq!(cell_filename_width(8), 7);
    }

    #[test]
    fn cell_shard_path_pads() {
        let p = cell_shard_path(Path::new("/tmp/build"), "quads", "zqd", 5, 42);
        let s = p.to_string_lossy();
        assert!(s.ends_with("/quads/cell_000042.zqd"), "unexpected path {s}");
        assert!(s.starts_with("/tmp/build/quads/"));
    }

    #[test]
    fn cell_shard_path_uses_subdir_and_ext() {
        let q = cell_shard_path(Path::new("/x"), "quads", "zqd", 5, 7);
        let g = cell_shard_path(Path::new("/x"), "gaia", "zga", 5, 7);
        assert!(q.to_string_lossy().contains("/quads/"));
        assert!(q.to_string_lossy().ends_with(".zqd"));
        assert!(g.to_string_lossy().contains("/gaia/"));
        assert!(g.to_string_lossy().ends_with(".zga"));
    }

    #[test]
    fn n_cells_matches_healpix_formula() {
        assert_eq!(n_cells_at_depth(0), 12);
        assert_eq!(n_cells_at_depth(1), 48);
        assert_eq!(n_cells_at_depth(5), 12_288);
        assert_eq!(n_cells_at_depth(8), 786_432);
    }
}
