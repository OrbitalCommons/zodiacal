use std::path::{Path, PathBuf};
use std::process;

use clap::{Parser, Subcommand};
use ndarray::Array2;

use zodiacal::extraction::ExtractionConfig;
use zodiacal::index::Index;
use zodiacal::index::builder::{CatalogBuilderConfig, build_index_from_catalog};
use zodiacal::solver::{SolverConfig, solve_image};

#[derive(Parser)]
#[command(name = "zodiacal", about = "Blind astrometry plate solver")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Solve a PNG image against prebuilt star indexes.
    Solve {
        /// Path to the image file (PNG, JPEG, etc.)
        image: PathBuf,

        /// Path to index file(s). Can be repeated.
        #[arg(short, long, required = true)]
        index: Vec<PathBuf>,

        /// Max sources to extract from image.
        #[arg(long, default_value = "200")]
        max_sources: usize,

        /// Detection threshold in sigma units.
        #[arg(long, default_value = "5.0")]
        threshold_sigma: f64,

        /// Pixel scale range hint in arcsec/pixel (e.g. "0.5,5.0").
        #[arg(long)]
        scale_range: Option<String>,

        /// Code matching tolerance (squared L2 in code space).
        #[arg(long, default_value = "0.01")]
        code_tolerance: f64,
    },

    /// Build a star index from a starfield binary catalog.
    BuildIndex {
        /// Path to a starfield binary catalog file.
        #[arg(short, long)]
        catalog: PathBuf,

        /// Output path for the zodiacal index file.
        #[arg(short, long)]
        output: PathBuf,

        /// Minimum quad scale in arcseconds.
        #[arg(long, default_value = "30.0")]
        scale_lower: f64,

        /// Maximum quad scale in arcseconds.
        #[arg(long, default_value = "1800.0")]
        scale_upper: f64,

        /// Maximum brightest stars per HEALPix cell.
        #[arg(long, default_value = "10")]
        max_stars_per_cell: usize,

        /// Maximum number of quads to generate.
        #[arg(long, default_value = "100000")]
        max_quads: usize,

        /// HEALPix depth for star uniformization (auto if omitted).
        #[arg(long)]
        uniformize_depth: Option<u8>,

        /// Number of quad-building passes per cell.
        #[arg(long, default_value = "16")]
        passes: usize,

        /// Maximum times a star can appear in quads.
        #[arg(long, default_value = "8")]
        max_reuse: usize,
    },

    /// Diagnose extraction by comparing detected sources to index stars.
    Diagnose {
        /// Path to the image file
        image: PathBuf,

        /// Path to the index file
        #[arg(short, long)]
        index: PathBuf,

        /// Known RA of field center (degrees)
        #[arg(long)]
        ra: f64,

        /// Known Dec of field center (degrees)
        #[arg(long)]
        dec: f64,

        /// Known pixel scale (arcsec/pixel)
        #[arg(long)]
        scale: f64,

        /// Detection threshold in sigma units
        #[arg(long, default_value = "5.0")]
        threshold_sigma: f64,

        /// Max sources to extract
        #[arg(long, default_value = "200")]
        max_sources: usize,
    },
}

#[cfg(feature = "fits")]
fn load_fits(path: &Path) -> Array2<f32> {
    use fitsrs::Fits;
    use fitsrs::card::Value;
    use fitsrs::hdu::HDU;
    use fitsrs::hdu::data::image::Pixels;
    use std::fs::File;
    use std::io::BufReader;

    let f = File::open(path).unwrap_or_else(|e| {
        eprintln!("Failed to open FITS file {}: {e}", path.display());
        process::exit(1);
    });
    let reader = BufReader::new(f);
    let mut hdu_list = Fits::from_reader(reader);

    // Find the first image HDU with actual data (NAXIS >= 2).
    // The primary HDU may be empty (NAXIS=0) with the image in an extension.
    let hdu = loop {
        match hdu_list.next() {
            Some(Ok(HDU::Primary(hdu))) | Some(Ok(HDU::XImage(hdu))) => {
                let naxis = hdu.get_header().get_xtension().get_naxis();
                if naxis.len() >= 2 {
                    break hdu;
                }
            }
            Some(Ok(_)) => continue,
            Some(Err(e)) => {
                eprintln!("Failed to read FITS HDU: {e}");
                process::exit(1);
            }
            None => {
                eprintln!("No image HDU with data found in FITS file");
                process::exit(1);
            }
        }
    };

    let xtension = hdu.get_header().get_xtension();
    let naxis = xtension.get_naxis();
    let naxis1 = naxis[0] as usize;
    let naxis2 = naxis[1] as usize;

    let header = hdu.get_header();
    let bzero: f64 = match header.get("BZERO") {
        Some(Value::Float { value, .. }) => *value,
        Some(Value::Integer { value, .. }) => *value as f64,
        _ => 0.0,
    };
    let bscale: f64 = match header.get("BSCALE") {
        Some(Value::Float { value, .. }) => *value,
        Some(Value::Integer { value, .. }) => *value as f64,
        _ => 1.0,
    };

    eprintln!(
        "FITS: {}x{} BITPIX={:?} BZERO={} BSCALE={}",
        naxis1,
        naxis2,
        xtension.get_bitpix(),
        bzero,
        bscale
    );

    let image_data = hdu_list.get_data(&hdu);
    let pixels = image_data.pixels();

    let raw: Vec<f32> = match pixels {
        Pixels::U8(it) => it.map(|v| (v as f64 * bscale + bzero) as f32).collect(),
        Pixels::I16(it) => it.map(|v| (v as f64 * bscale + bzero) as f32).collect(),
        Pixels::I32(it) => it.map(|v| (v as f64 * bscale + bzero) as f32).collect(),
        Pixels::I64(it) => it.map(|v| (v as f64 * bscale + bzero) as f32).collect(),
        Pixels::F32(it) => it.map(|v| (v as f64 * bscale + bzero) as f32).collect(),
        Pixels::F64(it) => it.map(|v| (v * bscale + bzero) as f32).collect(),
    };

    let expected = naxis1 * naxis2;
    if raw.len() != expected {
        eprintln!(
            "FITS pixel count mismatch: expected {} ({}x{}), got {}",
            expected,
            naxis1,
            naxis2,
            raw.len()
        );
        process::exit(1);
    }

    // The test FITS files are written by meter-sim's fitsio which passes
    // ndarray dimensions as [rows, cols] to FITS, mapping them to
    // NAXIS1=rows and NAXIS2=cols (non-standard). The data is also flipped
    // vertically on write (FITS origin is bottom-left). We reshape as
    // (naxis1, naxis2) = (rows, cols) and flip vertically to undo both.
    let arr = Array2::from_shape_vec((naxis1, naxis2), raw).unwrap_or_else(|e| {
        eprintln!("Failed to reshape FITS data: {e}");
        process::exit(1);
    });
    arr.slice(ndarray::s![..;-1, ..]).to_owned()
}

fn load_png(path: &Path) -> Array2<f32> {
    let img = image::open(path).unwrap_or_else(|e| {
        eprintln!("Failed to load image {}: {e}", path.display());
        process::exit(1);
    });
    let gray = img.to_luma32f();
    let (width, height) = gray.dimensions();
    let raw: Vec<f32> = gray.into_raw();
    Array2::from_shape_vec((height as usize, width as usize), raw).unwrap_or_else(|e| {
        eprintln!("Failed to reshape image data: {e}");
        process::exit(1);
    })
}

fn load_image(path: &Path) -> Array2<f32> {
    match path.extension().and_then(|e| e.to_str()) {
        #[cfg(feature = "fits")]
        Some("fits" | "fit" | "fts") => load_fits(path),
        #[cfg(not(feature = "fits"))]
        Some("fits" | "fit" | "fts") => {
            eprintln!("FITS support not enabled. Build with --features fits");
            process::exit(1);
        }
        _ => load_png(path),
    }
}

fn parse_scale_range(s: &str) -> (f64, f64) {
    let parts: Vec<&str> = s.split(',').collect();
    if parts.len() != 2 {
        eprintln!("scale-range must be two comma-separated values (e.g. \"0.5,5.0\")");
        process::exit(1);
    }
    let lo: f64 = parts[0].trim().parse().unwrap_or_else(|_| {
        eprintln!("Invalid scale-range lower bound: {}", parts[0]);
        process::exit(1);
    });
    let hi: f64 = parts[1].trim().parse().unwrap_or_else(|_| {
        eprintln!("Invalid scale-range upper bound: {}", parts[1]);
        process::exit(1);
    });
    (lo, hi)
}

fn cmd_solve(
    image_path: &Path,
    index_paths: &[PathBuf],
    max_sources: usize,
    threshold_sigma: f64,
    scale_range: Option<(f64, f64)>,
    code_tolerance: f64,
) {
    let array = load_image(image_path);
    let (h, w) = array.dim();
    eprintln!("Loaded image: {w} x {h} pixels");

    let indexes: Vec<Index> = index_paths
        .iter()
        .map(|p| {
            Index::load(p).unwrap_or_else(|e| {
                eprintln!("Failed to load index {}: {e}", p.display());
                process::exit(1);
            })
        })
        .collect();
    let index_refs: Vec<&Index> = indexes.iter().collect();
    eprintln!("Loaded {} index(es)", indexes.len());

    let extraction_config = ExtractionConfig {
        threshold_sigma,
        max_sources,
        ..ExtractionConfig::default()
    };
    let solver_config = SolverConfig {
        scale_range,
        code_tolerance,
        ..SolverConfig::default()
    };

    match solve_image(&array, &index_refs, &extraction_config, &solver_config) {
        Some(solution) => {
            let (ra, dec) = solution.wcs.field_center();
            let scale_arcsec = solution.wcs.pixel_scale() * 3600.0;
            let rotation = f64::atan2(solution.wcs.cd[1][0], solution.wcs.cd[0][0]).to_degrees();

            println!("Solved!");
            println!(
                "  Field center: RA = {:.4} deg, Dec = {:+.4} deg",
                ra.to_degrees(),
                dec.to_degrees()
            );
            println!("  Pixel scale: {:.3} arcsec/pixel", scale_arcsec);
            println!("  Rotation: {:.1} deg", rotation);
            println!(
                "  Matched: {}, Distractors: {}, Log-odds: {:.1}",
                solution.verify_result.n_matched,
                solution.verify_result.n_distractor,
                solution.verify_result.log_odds
            );
        }
        None => {
            eprintln!("No solution found.");
            process::exit(1);
        }
    }
}

fn cmd_build_index(catalog_path: &Path, output_path: &Path, config: &CatalogBuilderConfig) {
    use starfield::catalogs::MinimalCatalog;

    let catalog = MinimalCatalog::load(catalog_path).unwrap_or_else(|e| {
        eprintln!("Failed to load catalog {}: {e}", catalog_path.display());
        process::exit(1);
    });
    eprintln!("Loaded catalog: {} stars", catalog.len());

    let index = build_index_from_catalog(&catalog, config);
    eprintln!(
        "Built index: {} stars, {} quads",
        index.stars.len(),
        index.quads.len()
    );

    index.save(output_path).unwrap_or_else(|e| {
        eprintln!("Failed to save index {}: {e}", output_path.display());
        process::exit(1);
    });
    eprintln!("Saved index to {}", output_path.display());
}

fn cmd_diagnose(
    image_path: &Path,
    index_path: &Path,
    ra_deg: f64,
    dec_deg: f64,
    scale_arcsec: f64,
    threshold_sigma: f64,
    max_sources: usize,
) {
    use zodiacal::extraction::extract_sources;
    use zodiacal::geom::sphere::{angular_distance, radec_to_xyz};
    use zodiacal::geom::tan::TanWcs;

    let array = load_image(image_path);
    let (h, w) = array.dim();
    eprintln!("Image: {w} x {h} pixels");

    let index = Index::load(index_path).unwrap_or_else(|e| {
        eprintln!("Failed to load index: {e}");
        process::exit(1);
    });
    eprintln!(
        "Index: {} stars, {} quads",
        index.stars.len(),
        index.quads.len()
    );

    // Extract sources.
    let config = ExtractionConfig {
        threshold_sigma,
        max_sources,
        ..ExtractionConfig::default()
    };
    let sources = extract_sources(&array, &config);
    eprintln!("Extracted sources: {}", sources.len());
    for (i, src) in sources.iter().take(10).enumerate() {
        eprintln!(
            "  src[{i:3}] ({:7.1}, {:7.1}) flux={:.0}",
            src.x, src.y, src.flux
        );
    }

    // Try all 8 CD matrix sign/axis combinations to find the correct WCS.
    let scale_deg = scale_arcsec / 3600.0;
    let scale_rad = scale_deg.to_radians();
    let cd_combos: Vec<([[f64; 2]; 2], &str)> = vec![
        ([[scale_rad, 0.0], [0.0, scale_rad]], "RA+ Dec+"),
        ([[scale_rad, 0.0], [0.0, -scale_rad]], "RA+ Dec-"),
        ([[-scale_rad, 0.0], [0.0, scale_rad]], "RA- Dec+"),
        ([[-scale_rad, 0.0], [0.0, -scale_rad]], "RA- Dec-"),
        ([[0.0, scale_rad], [scale_rad, 0.0]], "swapped RA+ Dec+"),
        ([[0.0, scale_rad], [-scale_rad, 0.0]], "swapped RA+ Dec-"),
        ([[0.0, -scale_rad], [scale_rad, 0.0]], "swapped RA- Dec+"),
        ([[0.0, -scale_rad], [-scale_rad, 0.0]], "swapped RA- Dec-"),
    ];

    eprintln!("\nCD matrix matching (50px radius, top 50 sources):");
    for (cd, name) in &cd_combos {
        let wcs = TanWcs {
            crval: [ra_deg.to_radians(), dec_deg.to_radians()],
            crpix: [w as f64 / 2.0, h as f64 / 2.0],
            cd: *cd,
            image_size: [w as f64, h as f64],
        };

        let (center_ra, center_dec) = wcs.field_center();
        let center_xyz = radec_to_xyz(center_ra, center_dec);
        let field_radius = wcs.field_radius();
        let radius_sq = 2.0 * (1.0 - field_radius.cos());

        let nearby = index.star_tree.range_search(&center_xyz, radius_sq);
        let mut ref_pixels: Vec<(f64, f64)> = Vec::new();
        for result in &nearby {
            let star = &index.stars[result.index];
            if let Some((px, py)) = wcs.radec_to_pixel(star.ra, star.dec)
                && px >= 0.0
                && px <= w as f64
                && py >= 0.0
                && py <= h as f64
            {
                ref_pixels.push((px, py));
            }
        }

        let match_radius = 50.0;
        let mut n_matched = 0;
        let mut total_dist = 0.0;
        for src in sources.iter().take(50) {
            let mut best_dist = f64::MAX;
            for &(rx, ry) in &ref_pixels {
                let d = ((src.x - rx).powi(2) + (src.y - ry).powi(2)).sqrt();
                if d < best_dist {
                    best_dist = d;
                }
            }
            if best_dist < match_radius {
                n_matched += 1;
                total_dist += best_dist;
            }
        }

        let avg = if n_matched > 0 {
            total_dist / n_matched as f64
        } else {
            0.0
        };
        eprintln!(
            "  {name:20}: {n_matched:2}/50 matched ({} refs), avg dist {avg:.1}px",
            ref_pixels.len()
        );
    }

    // Reverse match: try all 8 CD combos and y-flip, projecting sources to sky
    // and searching for nearest index stars.
    let search_radius_rad: f64 = 0.0003; // ~1 arcmin
    let search_radius_sq = 2.0 * (1.0 - search_radius_rad.cos());
    let n_check = sources.len().min(30);

    // Try different axis conventions: (x,y), (x,h-1-y), (y,x), (y,w-1-x)
    // with different CD sign combos
    struct WcsCombo {
        cd: [[f64; 2]; 2],
        swap_xy: bool,
        flip_y: bool,
        name: &'static str,
    }
    let ps = scale_rad;
    let wcs_combos: Vec<WcsCombo> = vec![
        // Standard (x,y) with various CD signs
        WcsCombo {
            cd: [[-ps, 0.0], [0.0, ps]],
            swap_xy: false,
            flip_y: false,
            name: "cd[-,+] xy",
        },
        WcsCombo {
            cd: [[-ps, 0.0], [0.0, -ps]],
            swap_xy: false,
            flip_y: false,
            name: "cd[-,-] xy",
        },
        WcsCombo {
            cd: [[ps, 0.0], [0.0, ps]],
            swap_xy: false,
            flip_y: false,
            name: "cd[+,+] xy",
        },
        WcsCombo {
            cd: [[ps, 0.0], [0.0, -ps]],
            swap_xy: false,
            flip_y: false,
            name: "cd[+,-] xy",
        },
        // Swapped axes (y,x)
        WcsCombo {
            cd: [[-ps, 0.0], [0.0, ps]],
            swap_xy: true,
            flip_y: false,
            name: "cd[-,+] yx",
        },
        WcsCombo {
            cd: [[-ps, 0.0], [0.0, -ps]],
            swap_xy: true,
            flip_y: false,
            name: "cd[-,-] yx",
        },
        WcsCombo {
            cd: [[ps, 0.0], [0.0, ps]],
            swap_xy: true,
            flip_y: false,
            name: "cd[+,+] yx",
        },
        WcsCombo {
            cd: [[ps, 0.0], [0.0, -ps]],
            swap_xy: true,
            flip_y: false,
            name: "cd[+,-] yx",
        },
        // y-flipped
        WcsCombo {
            cd: [[-ps, 0.0], [0.0, ps]],
            swap_xy: false,
            flip_y: true,
            name: "cd[-,+] xy yf",
        },
        WcsCombo {
            cd: [[-ps, 0.0], [0.0, -ps]],
            swap_xy: false,
            flip_y: true,
            name: "cd[-,-] xy yf",
        },
        // Swapped + y-flipped
        WcsCombo {
            cd: [[-ps, 0.0], [0.0, ps]],
            swap_xy: true,
            flip_y: true,
            name: "cd[-,+] yx yf",
        },
        WcsCombo {
            cd: [[-ps, 0.0], [0.0, -ps]],
            swap_xy: true,
            flip_y: true,
            name: "cd[-,-] yx yf",
        },
        WcsCombo {
            cd: [[ps, 0.0], [0.0, ps]],
            swap_xy: true,
            flip_y: true,
            name: "cd[+,+] yx yf",
        },
        WcsCombo {
            cd: [[ps, 0.0], [0.0, -ps]],
            swap_xy: true,
            flip_y: true,
            name: "cd[+,-] yx yf",
        },
    ];

    eprintln!(
        "\nReverse match: source pixel -> sky -> nearest index star (search={:.1} arcmin)",
        search_radius_rad.to_degrees() * 60.0
    );
    for combo in &wcs_combos {
        // For swapped axes, crpix and image_size also swap
        let (crpx, crpy, iw, ih) = if combo.swap_xy {
            (h as f64 / 2.0, w as f64 / 2.0, h as f64, w as f64)
        } else {
            (w as f64 / 2.0, h as f64 / 2.0, w as f64, h as f64)
        };
        let wcs = TanWcs {
            crval: [ra_deg.to_radians(), dec_deg.to_radians()],
            crpix: [crpx, crpy],
            cd: combo.cd,
            image_size: [iw, ih],
        };

        let mut n_with_neighbor = 0;
        let mut total_sep = 0.0;
        for src in sources.iter().take(n_check) {
            let (px, py) = if combo.swap_xy {
                let y = if combo.flip_y {
                    (h as f64 - 1.0) - src.y
                } else {
                    src.y
                };
                (y, src.x)
            } else {
                let y = if combo.flip_y {
                    (h as f64 - 1.0) - src.y
                } else {
                    src.y
                };
                (src.x, y)
            };
            let (src_ra, src_dec) = wcs.pixel_to_radec(px, py);
            let src_xyz = radec_to_xyz(src_ra, src_dec);
            let nearby = index.star_tree.range_search(&src_xyz, search_radius_sq);
            if !nearby.is_empty() {
                let best = nearby
                    .iter()
                    .min_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap())
                    .unwrap();
                let star = &index.stars[best.index];
                let sep_rad = angular_distance(src_xyz, radec_to_xyz(star.ra, star.dec));
                total_sep += sep_rad.to_degrees() * 3600.0;
                n_with_neighbor += 1;
            }
        }
        let avg_sep = if n_with_neighbor > 0 {
            total_sep / n_with_neighbor as f64
        } else {
            0.0
        };
        eprintln!(
            "  {:20}: {:2}/{n_check} matched, avg sep {avg_sep:.1}\"",
            combo.name, n_with_neighbor
        );
    }

    // Print detailed results for cd[-,+] xy (standard, no swap/flip)
    eprintln!("\nDetailed reverse match (cd=[[-ps,0],[0,+ps]] xy):");
    let wcs_best = TanWcs {
        crval: [ra_deg.to_radians(), dec_deg.to_radians()],
        crpix: [w as f64 / 2.0, h as f64 / 2.0],
        cd: [[-scale_rad, 0.0], [0.0, scale_rad]],
        image_size: [w as f64, h as f64],
    };
    for (i, src) in sources.iter().take(n_check).enumerate() {
        let (px, py) = (src.x, src.y);
        let (src_ra, src_dec) = wcs_best.pixel_to_radec(px, py);
        let src_xyz = radec_to_xyz(src_ra, src_dec);
        let nearby = index.star_tree.range_search(&src_xyz, search_radius_sq);
        if nearby.is_empty() {
            eprintln!(
                "  {:3}  ({:7.1},{:7.1})  RA={:9.4} Dec={:+9.4}  no match",
                i,
                src.x,
                src.y,
                src_ra.to_degrees(),
                src_dec.to_degrees()
            );
        } else {
            let best = nearby
                .iter()
                .min_by(|a, b| a.dist_sq.partial_cmp(&b.dist_sq).unwrap())
                .unwrap();
            let star = &index.stars[best.index];
            let sep_rad = angular_distance(src_xyz, radec_to_xyz(star.ra, star.dec));
            let sep_arcsec = sep_rad.to_degrees() * 3600.0;
            eprintln!(
                "  {:3}  ({:7.1},{:7.1})  RA={:9.4} Dec={:+9.4}  idx RA={:9.4} Dec={:+9.4}  sep={:.1}\"",
                i,
                src.x,
                src.y,
                src_ra.to_degrees(),
                src_dec.to_degrees(),
                star.ra.to_degrees(),
                star.dec.to_degrees(),
                sep_arcsec
            );
        }
    }
}

fn main() {
    let cli = Cli::parse();

    match &cli.command {
        Commands::Solve {
            image,
            index,
            max_sources,
            threshold_sigma,
            scale_range,
            code_tolerance,
        } => {
            let sr = scale_range.as_ref().map(|s| parse_scale_range(s));
            cmd_solve(
                image,
                index,
                *max_sources,
                *threshold_sigma,
                sr,
                *code_tolerance,
            );
        }
        Commands::BuildIndex {
            catalog,
            output,
            scale_lower,
            scale_upper,
            max_stars_per_cell,
            max_quads,
            uniformize_depth,
            passes,
            max_reuse,
        } => {
            let config = CatalogBuilderConfig {
                scale_lower: (*scale_lower / 3600.0).to_radians(),
                scale_upper: (*scale_upper / 3600.0).to_radians(),
                max_stars_per_cell: *max_stars_per_cell,
                max_quads: *max_quads,
                uniformize_depth: *uniformize_depth,
                quad_depth: None,
                passes: *passes,
                max_reuse: *max_reuse,
            };
            cmd_build_index(catalog, output, &config);
        }
        Commands::Diagnose {
            image,
            index,
            ra,
            dec,
            scale,
            threshold_sigma,
            max_sources,
        } => {
            cmd_diagnose(
                image,
                index,
                *ra,
                *dec,
                *scale,
                *threshold_sigma,
                *max_sources,
            );
        }
    }
}
