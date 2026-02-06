use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Write};
use std::path::Path;

use crate::geom::sphere::radec_to_xyz;
use crate::kdtree::KdTree;
use crate::quads::{Code, DIMCODES, DIMQUADS, Quad};

use super::{Index, IndexStar};

const MAGIC: &[u8; 4] = b"ZDCL";
const VERSION: u32 = 1;

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
        if version != VERSION {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("unsupported version: {version}"),
            ));
        }

        let num_stars = read_u64(&mut r)? as usize;
        let num_quads = read_u64(&mut r)? as usize;
        let scale_lower = read_f64(&mut r)?;
        let scale_upper = read_f64(&mut r)?;

        let mut stars = Vec::with_capacity(num_stars);
        for _ in 0..num_stars {
            let catalog_id = read_u64(&mut r)?;
            let ra = read_f64(&mut r)?;
            let dec = read_f64(&mut r)?;
            let mag = read_f64(&mut r)?;
            stars.push(IndexStar {
                catalog_id,
                ra,
                dec,
                mag,
            });
        }

        let mut quads = Vec::with_capacity(num_quads);
        for _ in 0..num_quads {
            let mut star_ids = [0usize; DIMQUADS];
            for sid in &mut star_ids {
                *sid = read_u32(&mut r)? as usize;
            }
            quads.push(Quad { star_ids });
        }

        let mut codes: Vec<Code> = Vec::with_capacity(num_quads);
        for _ in 0..num_quads {
            let mut code = [0.0f64; DIMCODES];
            for v in &mut code {
                *v = read_f64(&mut r)?;
            }
            codes.push(code);
        }

        let star_points: Vec<[f64; 3]> = stars.iter().map(|s| radec_to_xyz(s.ra, s.dec)).collect();
        let star_indices: Vec<usize> = (0..num_stars).collect();
        let star_tree = KdTree::<3>::build(star_points, star_indices);

        let code_indices: Vec<usize> = (0..num_quads).collect();
        let code_tree = KdTree::<{ DIMCODES }>::build(codes, code_indices);

        Ok(Index {
            star_tree,
            stars,
            code_tree,
            quads,
            scale_lower,
            scale_upper,
        })
    }
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
}
