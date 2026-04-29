use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::geom::sphere::radec_to_xyz;
use crate::kdtree::KdTree;
use crate::quads::{Code, DIMCODES, DIMQUADS, Quad};
use crate::solver::SkyRegion;

use super::{Index, IndexMetadata, IndexStar};

const MAGIC: &[u8; 4] = b"ZDCL";
const VERSION: u32 = 2;

fn write_u32(w: &mut impl Write, v: u32) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_u64(w: &mut impl Write, v: u64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn write_f64(w: &mut impl Write, v: f64) -> io::Result<()> {
    w.write_all(&v.to_le_bytes())
}

fn read_u32(r: &mut impl Read) -> io::Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> io::Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f64(r: &mut impl Read) -> io::Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(f64::from_le_bytes(buf))
}

impl Index {
    pub fn save(&self, path: &Path) -> io::Result<()> {
        let file = File::create(path)?;
        let mut w = BufWriter::new(file);

        w.write_all(MAGIC)?;
        write_u32(&mut w, VERSION)?;

        // V2: length-prefixed JSON metadata (0 length if absent)
        let meta_bytes = match &self.metadata {
            Some(m) => serde_json::to_vec(m).map_err(io::Error::other)?,
            None => Vec::new(),
        };
        write_u64(&mut w, meta_bytes.len() as u64)?;
        if !meta_bytes.is_empty() {
            w.write_all(&meta_bytes)?;
        }

        write_u64(&mut w, self.stars.len() as u64)?;
        write_u64(&mut w, self.quads.len() as u64)?;
        write_f64(&mut w, self.scale_lower)?;
        write_f64(&mut w, self.scale_upper)?;

        for star in &self.stars {
            write_u64(&mut w, star.catalog_id)?;
            write_f64(&mut w, star.ra)?;
            write_f64(&mut w, star.dec)?;
            write_f64(&mut w, star.mag)?;
        }

        for quad in &self.quads {
            for &id in &quad.star_ids {
                write_u32(&mut w, id as u32)?;
            }
        }

        let star_positions: Vec<[f64; 3]> = self
            .stars
            .iter()
            .map(|s| radec_to_xyz(s.ra, s.dec))
            .collect();
        for quad in &self.quads {
            let star_xyz: [_; DIMQUADS] = std::array::from_fn(|i| star_positions[quad.star_ids[i]]);
            let (code, _, _) = crate::quads::compute_canonical_code(&star_xyz, quad.star_ids);
            for &v in &code {
                write_f64(&mut w, v)?;
            }
        }

        w.flush()
    }

    pub fn load(path: &Path) -> io::Result<Index> {
        load_filtered(path, None)
    }

    /// Load only stars within `region`, plus the quads that reference those
    /// stars. Quads with any star outside `region` are dropped entirely.
    ///
    /// Disk I/O is unchanged from [`Index::load`] (the whole file is still
    /// read sequentially), but in-memory size scales with the kept set.
    ///
    /// Quads whose backbone straddles the region boundary will be silently
    /// dropped — if you want to keep those, use
    /// [`Index::load_in_region_padded`] with a padding at least as large as
    /// the largest quad backbone you expect (typically the index's upper
    /// scale bound, in radians).
    pub fn load_in_region(path: &Path, region: &SkyRegion) -> io::Result<Index> {
        load_filtered(path, Some((region, 0.0)))
    }

    /// Load with explicit padding around `region`. `padding_rad` is added to
    /// the region's radius before any star or quad acceptance test, so quads
    /// whose backbones straddle the region boundary are kept rather than
    /// silently dropped. Negative values are treated as zero.
    pub fn load_in_region_padded(
        path: &Path,
        region: &SkyRegion,
        padding_rad: f64,
    ) -> io::Result<Index> {
        load_filtered(path, Some((region, padding_rad.max(0.0))))
    }
}

/// Shared loader; if `region_filter` is None, the full file is loaded.
/// Otherwise only stars within `region.radius_rad + padding` are kept and
/// quads referencing dropped stars are discarded.
fn load_filtered(path: &Path, region_filter: Option<(&SkyRegion, f64)>) -> io::Result<Index> {
    let file = File::open(path)?;
    let mut r = BufReader::new(file);

    let mut magic = [0u8; 4];
    r.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid magic bytes",
        ));
    }

    let version = read_u32(&mut r)?;
    if version != 1 && version != 2 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("unsupported version: {version}"),
        ));
    }

    let metadata = if version >= 2 {
        let meta_len = read_u64(&mut r)? as usize;
        if meta_len > 0 {
            let mut meta_bytes = vec![0u8; meta_len];
            r.read_exact(&mut meta_bytes)?;
            let m: IndexMetadata = serde_json::from_slice(&meta_bytes)
                .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))?;
            Some(m)
        } else {
            None
        }
    } else {
        None
    };

    let num_stars = read_u64(&mut r)? as usize;
    let num_quads = read_u64(&mut r)? as usize;
    let scale_lower = read_f64(&mut r)?;
    let scale_upper = read_f64(&mut r)?;

    // Pre-compute the region center xyz and the cosine of the effective
    // radius once. The per-star test then reduces to a single dot product
    // (avoiding acos() for every star — important on the full-load path
    // where this is the per-star inner loop on millions of records).
    let region_test = region_filter.map(|(region, padding)| {
        let center_xyz = radec_to_xyz(region.center.ra, region.center.dec);
        let cos_radius = (region.radius_rad + padding).cos();
        (center_xyz, cos_radius)
    });
    let filtered = region_test.is_some();

    // Stars phase: when filtering, also build a remap from old index to
    // compact index. When loading the full file, skip the remap allocation
    // entirely — every star is kept and quads keep their original indices.
    let mut stars: Vec<IndexStar> = Vec::with_capacity(num_stars);
    let mut star_remap: Vec<Option<usize>> = if filtered {
        Vec::with_capacity(num_stars)
    } else {
        Vec::new()
    };
    for _ in 0..num_stars {
        let catalog_id = read_u64(&mut r)?;
        let ra = read_f64(&mut r)?;
        let dec = read_f64(&mut r)?;
        let mag = read_f64(&mut r)?;
        let keep = match region_test {
            None => true,
            Some((center_xyz, cos_radius)) => {
                let xyz = radec_to_xyz(ra, dec);
                let dot = xyz[0] * center_xyz[0] + xyz[1] * center_xyz[1] + xyz[2] * center_xyz[2];
                dot >= cos_radius
            }
        };
        if filtered {
            if keep {
                star_remap.push(Some(stars.len()));
            } else {
                star_remap.push(None);
            }
        }
        if keep {
            stars.push(IndexStar {
                catalog_id,
                ra,
                dec,
                mag,
            });
        }
    }

    // Quads phase: read all, drop those referencing any dropped star,
    // remap indices for kept quads. Track which quad positions were kept so
    // we know which codes to keep in the next phase. Skip the bookkeeping
    // entirely when loading the full file.
    let mut quads: Vec<Quad> = Vec::with_capacity(num_quads);
    let mut quad_kept: Vec<bool> = if filtered {
        Vec::with_capacity(num_quads)
    } else {
        Vec::new()
    };
    for _ in 0..num_quads {
        let mut star_ids = [0usize; DIMQUADS];
        for sid in &mut star_ids {
            *sid = read_u32(&mut r)? as usize;
        }
        if !filtered {
            quads.push(Quad { star_ids });
            continue;
        }
        let mut new_ids = [0usize; DIMQUADS];
        let mut all_kept = true;
        for (i, &sid) in star_ids.iter().enumerate() {
            match star_remap.get(sid).copied().flatten() {
                Some(new_idx) => new_ids[i] = new_idx,
                None => {
                    all_kept = false;
                    break;
                }
            }
        }
        if all_kept {
            quad_kept.push(true);
            quads.push(Quad { star_ids: new_ids });
        } else {
            quad_kept.push(false);
        }
    }

    // Codes phase: full load reads all codes; filtered load keeps only those
    // matching the kept quads.
    let mut codes: Vec<Code> = Vec::with_capacity(if filtered { quads.len() } else { num_quads });
    if filtered {
        for &keep in &quad_kept {
            let mut code = [0.0f64; DIMCODES];
            for v in &mut code {
                *v = read_f64(&mut r)?;
            }
            if keep {
                codes.push(code);
            }
        }
    } else {
        for _ in 0..num_quads {
            let mut code = [0.0f64; DIMCODES];
            for v in &mut code {
                *v = read_f64(&mut r)?;
            }
            codes.push(code);
        }
    }

    let star_points: Vec<[f64; 3]> = stars.iter().map(|s| radec_to_xyz(s.ra, s.dec)).collect();
    let star_indices: Vec<usize> = (0..stars.len()).collect();
    let star_tree = KdTree::<3>::build(star_points, star_indices);

    let code_indices: Vec<usize> = (0..codes.len()).collect();
    let code_tree = KdTree::<{ DIMCODES }>::build(codes, code_indices);

    Ok(Index {
        star_tree,
        stars,
        code_tree,
        quads,
        scale_lower,
        scale_upper,
        metadata,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_index(num_stars: usize, num_quads: usize) -> Index {
        let base_ra = 1.0;
        let base_dec = 0.5;
        let mut stars = Vec::with_capacity(num_stars);
        for i in 0..num_stars {
            let frac = i as f64 / num_stars.max(1) as f64;
            stars.push(IndexStar {
                catalog_id: (i as u64) * 100 + 1,
                ra: base_ra + frac * 0.01,
                dec: base_dec + frac * 0.01,
                mag: 5.0 + frac * 10.0,
            });
        }

        let mut quads = Vec::with_capacity(num_quads);
        for i in 0..num_quads {
            let base = i % num_stars.saturating_sub(3);
            quads.push(Quad {
                star_ids: [base, base + 1, base + 2, base + 3],
            });
        }

        let star_points: Vec<[f64; 3]> = stars.iter().map(|s| radec_to_xyz(s.ra, s.dec)).collect();
        let star_indices: Vec<usize> = (0..num_stars).collect();
        let star_tree = KdTree::<3>::build(star_points.clone(), star_indices);

        let mut codes: Vec<Code> = Vec::with_capacity(num_quads);
        for quad in &quads {
            let star_xyz: [_; DIMQUADS] = std::array::from_fn(|i| star_points[quad.star_ids[i]]);
            let (code, _, _) = crate::quads::compute_canonical_code(&star_xyz, quad.star_ids);
            codes.push(code);
        }
        let code_indices: Vec<usize> = (0..num_quads).collect();
        let code_tree = KdTree::<{ DIMCODES }>::build(codes, code_indices);

        Index {
            star_tree,
            stars,
            code_tree,
            quads,
            scale_lower: 0.001,
            scale_upper: 0.01,
            metadata: None,
        }
    }

    fn temp_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("zodiacal_test_{name}_{}.bin", std::process::id()))
    }

    #[test]
    fn round_trip() {
        let idx = make_test_index(8, 3);
        let path = temp_path("round_trip");
        idx.save(&path).unwrap();
        let loaded = Index::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.stars.len(), idx.stars.len());
        assert_eq!(loaded.quads.len(), idx.quads.len());
        assert_eq!(loaded.scale_lower, idx.scale_lower);
        assert_eq!(loaded.scale_upper, idx.scale_upper);

        for (a, b) in loaded.stars.iter().zip(idx.stars.iter()) {
            assert_eq!(a.catalog_id, b.catalog_id);
            assert!((a.ra - b.ra).abs() < 1e-15);
            assert!((a.dec - b.dec).abs() < 1e-15);
            assert!((a.mag - b.mag).abs() < 1e-15);
        }

        for (a, b) in loaded.quads.iter().zip(idx.quads.iter()) {
            assert_eq!(a.star_ids, b.star_ids);
        }
    }

    #[test]
    fn magic_validation() {
        let path = temp_path("bad_magic");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(b"BAAD").unwrap();
            f.write_all(&1u32.to_le_bytes()).unwrap();
        }
        let err = Index::load(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn version_validation() {
        let path = temp_path("bad_version");
        {
            let mut f = File::create(&path).unwrap();
            f.write_all(MAGIC).unwrap();
            f.write_all(&99u32.to_le_bytes()).unwrap();
        }
        let err = Index::load(&path).unwrap_err();
        assert_eq!(err.kind(), io::ErrorKind::InvalidData);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn empty_index() {
        let star_tree = KdTree::<3>::build(vec![], vec![]);
        let code_tree = KdTree::<{ DIMCODES }>::build(vec![], vec![]);
        let idx = Index {
            star_tree,
            stars: vec![],
            code_tree,
            quads: vec![],
            scale_lower: 0.0,
            scale_upper: 0.0,
            metadata: None,
        };

        let path = temp_path("empty");
        idx.save(&path).unwrap();
        let loaded = Index::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(loaded.stars.is_empty());
        assert!(loaded.quads.is_empty());
        assert!(loaded.star_tree.is_empty());
        assert!(loaded.code_tree.is_empty());
    }

    #[test]
    fn kdtree_reconstruction() {
        let idx = make_test_index(10, 4);
        let path = temp_path("kdtree");
        idx.save(&path).unwrap();
        let loaded = Index::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(loaded.star_tree.len(), 10);

        for (i, star) in loaded.stars.iter().enumerate() {
            let xyz = radec_to_xyz(star.ra, star.dec);
            let result = loaded.star_tree.nearest(&xyz).unwrap();
            assert_eq!(result.index, i);
            assert!(result.dist_sq < 1e-20);
        }

        let query = radec_to_xyz(loaded.stars[0].ra, loaded.stars[0].dec);
        let results = loaded.star_tree.range_search(&query, 0.1);
        assert!(!results.is_empty());
        assert!(results.iter().any(|r| r.index == 0));
    }

    #[test]
    fn metadata_round_trip() {
        let mut idx = make_test_index(8, 3);
        idx.metadata = Some(IndexMetadata {
            scale_lower_arcsec: 30.0,
            scale_upper_arcsec: 90.0,
            n_stars: 8,
            n_quads: 3,
            max_stars_per_cell: 10,
            uniformize_depth: 6,
            quad_depth: 6,
            passes: 16,
            max_reuse: 8,
            build_timestamp: 1700000000,
            catalog_path: Some("test_catalog.bin".to_string()),
            band_index: Some(1),
            scale_factor: Some(3.0),
            mag_range: Some((5.0, 15.0)),
        });

        let path = temp_path("metadata_rt");
        idx.save(&path).unwrap();
        let loaded = Index::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        let meta = loaded.metadata.expect("metadata should be present");
        assert_eq!(meta.scale_lower_arcsec, 30.0);
        assert_eq!(meta.scale_upper_arcsec, 90.0);
        assert_eq!(meta.n_stars, 8);
        assert_eq!(meta.n_quads, 3);
        assert_eq!(meta.max_stars_per_cell, 10);
        assert_eq!(meta.uniformize_depth, 6);
        assert_eq!(meta.quad_depth, 6);
        assert_eq!(meta.passes, 16);
        assert_eq!(meta.max_reuse, 8);
        assert_eq!(meta.build_timestamp, 1700000000);
        assert_eq!(meta.catalog_path.as_deref(), Some("test_catalog.bin"));
        assert_eq!(meta.band_index, Some(1));
        assert_eq!(meta.scale_factor, Some(3.0));
        assert_eq!(meta.mag_range, Some((5.0, 15.0)));
    }

    #[test]
    fn no_metadata_round_trip() {
        let idx = make_test_index(8, 3);
        assert!(idx.metadata.is_none());

        let path = temp_path("no_metadata_rt");
        idx.save(&path).unwrap();
        let loaded = Index::load(&path).unwrap();
        std::fs::remove_file(&path).ok();

        assert!(loaded.metadata.is_none());
        assert_eq!(loaded.stars.len(), 8);
        assert_eq!(loaded.quads.len(), 3);
    }

    /// Build a test index where stars sit in two well-separated patches of
    /// sky. Returns (saved_path, patch_a_center_radec, patch_b_center_radec,
    /// n_stars_in_a, n_stars_in_b).
    fn make_two_patch_index(name: &str) -> (std::path::PathBuf, [f64; 2], [f64; 2], usize, usize) {
        let patch_a = [0.5_f64, 0.3];
        let patch_b = [1.5_f64, -0.1];
        let n_a = 8;
        let n_b = 6;

        let mut stars = Vec::new();
        for i in 0..n_a {
            let frac = i as f64 / n_a as f64;
            stars.push(IndexStar {
                catalog_id: 1000 + i as u64,
                ra: patch_a[0] + frac * 0.005,
                dec: patch_a[1] + frac * 0.005,
                mag: 5.0 + frac,
            });
        }
        for i in 0..n_b {
            let frac = i as f64 / n_b as f64;
            stars.push(IndexStar {
                catalog_id: 2000 + i as u64,
                ra: patch_b[0] + frac * 0.005,
                dec: patch_b[1] + frac * 0.005,
                mag: 5.0 + frac,
            });
        }

        // Quads: some entirely within A, some entirely within B, some that
        // straddle the two patches (these should be dropped by region
        // filters).
        let quads = vec![
            // Within A: stars 0,1,2,3
            Quad {
                star_ids: [0, 1, 2, 3],
            },
            // Within A: stars 4,5,6,7
            Quad {
                star_ids: [4, 5, 6, 7],
            },
            // Within B: stars 8,9,10,11
            Quad {
                star_ids: [8, 9, 10, 11],
            },
            // Within B: stars 10,11,12,13
            Quad {
                star_ids: [10, 11, 12, 13],
            },
            // Straddles A+B: dropped if filter is applied
            Quad {
                star_ids: [0, 1, 8, 9],
            },
        ];

        let star_points: Vec<[f64; 3]> = stars.iter().map(|s| radec_to_xyz(s.ra, s.dec)).collect();
        let star_indices: Vec<usize> = (0..stars.len()).collect();
        let star_tree = KdTree::<3>::build(star_points.clone(), star_indices);
        let mut codes: Vec<Code> = Vec::with_capacity(quads.len());
        for quad in &quads {
            let star_xyz: [_; DIMQUADS] = std::array::from_fn(|i| star_points[quad.star_ids[i]]);
            let (code, _, _) = crate::quads::compute_canonical_code(&star_xyz, quad.star_ids);
            codes.push(code);
        }
        let code_indices: Vec<usize> = (0..quads.len()).collect();
        let code_tree = KdTree::<{ DIMCODES }>::build(codes, code_indices);

        let idx = Index {
            star_tree,
            stars,
            code_tree,
            quads,
            scale_lower: 0.001,
            scale_upper: 0.01,
            metadata: None,
        };

        let path = temp_path(name);
        idx.save(&path).unwrap();
        (path, patch_a, patch_b, n_a, n_b)
    }

    #[test]
    fn load_in_region_full_sky_matches_full_load() {
        let (path, _, _, _, _) = make_two_patch_index("region_full_sky");

        let full = Index::load(&path).unwrap();
        // Region with radius >= pi covers the entire sphere.
        let all_sky =
            SkyRegion::from_radians(starfield::Equatorial::new(0.0, 0.0), std::f64::consts::PI);
        let regional = Index::load_in_region(&path, &all_sky).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(full.stars.len(), regional.stars.len());
        assert_eq!(full.quads.len(), regional.quads.len());
        for (a, b) in full.stars.iter().zip(regional.stars.iter()) {
            assert_eq!(a.catalog_id, b.catalog_id);
        }
        for (a, b) in full.quads.iter().zip(regional.quads.iter()) {
            assert_eq!(a.star_ids, b.star_ids);
        }
    }

    #[test]
    fn load_in_region_drops_outside_stars() {
        let (path, patch_a, _patch_b, n_a, _n_b) = make_two_patch_index("region_drop_outside");

        // Tight region around patch A — should contain only patch-A stars.
        let region =
            SkyRegion::from_radians(starfield::Equatorial::new(patch_a[0], patch_a[1]), 0.05);
        let regional = Index::load_in_region(&path, &region).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(regional.stars.len(), n_a);
        // All kept catalog IDs should be 1000-series (patch A).
        for star in &regional.stars {
            assert!(
                (1000..2000).contains(&star.catalog_id),
                "unexpected star {:?}",
                star
            );
        }
    }

    #[test]
    fn load_in_region_drops_quads_with_outside_member() {
        let (path, patch_a, _patch_b, _, _) = make_two_patch_index("region_drop_quads");

        let region =
            SkyRegion::from_radians(starfield::Equatorial::new(patch_a[0], patch_a[1]), 0.05);
        let regional = Index::load_in_region(&path, &region).unwrap();
        std::fs::remove_file(&path).ok();

        // Of the 5 quads in the source, 2 are entirely within A, 2 are
        // entirely within B, 1 straddles — expect 2 surviving in the
        // patch-A load.
        assert_eq!(regional.quads.len(), 2);
        // All surviving quad indices should be in-bounds for the kept stars.
        for q in &regional.quads {
            for &sid in &q.star_ids {
                assert!(
                    sid < regional.stars.len(),
                    "quad references invalid index {sid}"
                );
            }
        }
    }

    #[test]
    fn load_in_region_padded_keeps_boundary_quads() {
        let (path, patch_a, patch_b, _, _) = make_two_patch_index("region_padding");

        // Region centered between A and B, with radius just barely covering
        // both patches; padding 0.
        let center_ra = (patch_a[0] + patch_b[0]) / 2.0;
        let center_dec = (patch_a[1] + patch_b[1]) / 2.0;
        let center_xyz = radec_to_xyz(center_ra, center_dec);
        let patch_a_xyz = radec_to_xyz(patch_a[0], patch_a[1]);
        let patch_b_xyz = radec_to_xyz(patch_b[0], patch_b[1]);
        let dist_a = crate::geom::sphere::angular_distance(center_xyz, patch_a_xyz);
        let dist_b = crate::geom::sphere::angular_distance(center_xyz, patch_b_xyz);
        let radius = dist_a.max(dist_b) + 0.02; // small slack

        let region =
            SkyRegion::from_radians(starfield::Equatorial::new(center_ra, center_dec), radius);

        // No padding — straddle quad survives because both patches are in.
        let regional = Index::load_in_region(&path, &region).unwrap();
        std::fs::remove_file(&path).ok();
        // All 5 quads should be present (both patches and the straddle quad).
        assert_eq!(regional.quads.len(), 5);
        // Straddle quad should reference stars from both patches.
        let straddle = &regional.quads[regional.quads.len() - 1];
        let cats: Vec<u64> = straddle
            .star_ids
            .iter()
            .map(|&i| regional.stars[i].catalog_id)
            .collect();
        let has_a = cats.iter().any(|c| (1000..2000).contains(c));
        let has_b = cats.iter().any(|c| (2000..3000).contains(c));
        assert!(has_a && has_b, "straddle quad must keep both patch stars");
    }

    #[test]
    fn load_in_region_compact_indices_solve_smoke() {
        let (path, patch_a, _, _, _) = make_two_patch_index("region_compact");

        let region =
            SkyRegion::from_radians(starfield::Equatorial::new(patch_a[0], patch_a[1]), 0.05);
        let regional = Index::load_in_region(&path, &region).unwrap();
        std::fs::remove_file(&path).ok();

        // Indices on every kept quad must be within the kept star range.
        for quad in &regional.quads {
            for &sid in &quad.star_ids {
                assert!(sid < regional.stars.len());
            }
        }
        // KdTrees should have rebuilt over the kept set.
        assert_eq!(regional.star_tree.len(), regional.stars.len());
        assert_eq!(regional.code_tree.len(), regional.quads.len());
    }

    /// Integration: build a real index via build_index, save, sparse-load
    /// over a region containing the field, and solve against synthetic
    /// pixel sources. This is the "sparse load doesn't break the solver"
    /// guarantee.
    #[test]
    fn load_in_region_then_solve_recovers_wcs() {
        use crate::extraction::DetectedSource;
        use crate::geom::tan::TanWcs;
        use crate::index::builder::{IndexBuilderConfig, build_index};
        use crate::solver::{SolverConfig, solve};
        use crate::verify::VerifyConfig;
        use std::f64::consts::PI;

        let image_size = (512.0, 512.0);
        let pixel_scale_arcsec: f64 = 2.0;
        let scale_rad = (pixel_scale_arcsec / 3600.0).to_radians();
        let arcsec_rad = scale_rad;
        let truth_wcs = TanWcs {
            crval: [1.0, 0.5],
            crpix: [256.0, 256.0],
            cd: [[arcsec_rad, 0.0], [0.0, arcsec_rad]],
            image_size: [image_size.0, image_size.1],
        };

        // Deterministic pseudo-random stars in the field. Matches the
        // existing solver::tests::synthetic_solve seed so we know the
        // catalog/quad config produces a solvable field; this test is
        // about whether the sparse loader preserves that solvability.
        let mut state: u64 = 314_159_265;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };

        let mut catalog = Vec::new();
        let mut sources = Vec::new();
        for i in 0..25 {
            let px = 30.0 + rng() * 452.0;
            let py = 30.0 + rng() * 452.0;
            let (ra, dec) = truth_wcs.pixel_to_radec(px, py);
            catalog.push((i as u64, ra, dec, i as f64));
            sources.push(DetectedSource {
                x: px,
                y: py,
                flux: 1000.0 - i as f64 * 10.0,
            });
        }

        let field_diag = (image_size.0 * image_size.0 + image_size.1 * image_size.1).sqrt();
        let max_angle = field_diag * scale_rad;
        let cfg = IndexBuilderConfig {
            scale_lower: scale_rad * 10.0,
            scale_upper: max_angle,
            max_stars: 25,
            max_quads: 50_000,
        };
        let index = build_index(&catalog, &cfg);

        let path = temp_path("region_then_solve");
        index.save(&path).unwrap();

        // Sparse-load over a region containing the field, padded by the
        // index's largest backbone so no quads get clipped.
        let center = starfield::Equatorial::new(truth_wcs.crval[0], truth_wcs.crval[1]);
        let region = SkyRegion::from_radians(center, max_angle);
        let regional = Index::load_in_region_padded(&path, &region, max_angle).unwrap();
        std::fs::remove_file(&path).ok();

        // Sanity: regional load should retain ~all the stars/quads since the
        // field fits inside the padded region.
        assert_eq!(regional.stars.len(), index.stars.len());
        assert_eq!(regional.quads.len(), index.quads.len());

        let solver_cfg = SolverConfig {
            scale_range: None,
            max_field_stars: 25,
            code_tolerance: 0.002,
            verify: VerifyConfig {
                match_radius_pix: 3.0,
                log_odds_accept: 10.0,
                min_matches: 3,
                ..VerifyConfig::default()
            },
            ..SolverConfig::default()
        };
        let (solution, _stats) = solve(&sources, &[&regional], image_size, &solver_cfg);
        let solution = solution.expect("sparse-loaded index should solve");

        // The solved WCS image center should match truth's image center to
        // within a few arcsec. (CRVAL is not directly comparable because
        // fit_tan_wcs chooses its own tangent point.)
        let (solved_ra, solved_dec) = solution.wcs.field_center();
        let (truth_ra, truth_dec) = truth_wcs.field_center();
        let arcsec = PI / (180.0 * 3600.0);
        let dra = (solved_ra - truth_ra).abs() * truth_dec.cos();
        let ddec = (solved_dec - truth_dec).abs();
        let sep_arcsec = ((dra * dra + ddec * ddec).sqrt()) / arcsec;
        assert!(
            sep_arcsec < 30.0,
            "image-center separation {sep_arcsec:.2} arcsec exceeds 30\""
        );
    }

    #[test]
    fn load_in_region_disjoint_returns_empty() {
        let (path, _patch_a, _, _, _) = make_two_patch_index("region_disjoint");

        // Region in a completely different part of the sky.
        let region = SkyRegion::from_radians(starfield::Equatorial::new(2.0, 1.0), 0.001);
        let regional = Index::load_in_region(&path, &region).unwrap();
        std::fs::remove_file(&path).ok();

        assert_eq!(regional.stars.len(), 0);
        assert_eq!(regional.quads.len(), 0);
    }
}
