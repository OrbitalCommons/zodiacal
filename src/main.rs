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

    // FITS data: row-major, NAXIS1=columns (width), NAXIS2=rows (height)
    Array2::from_shape_vec((naxis2, naxis1), raw).unwrap_or_else(|e| {
        eprintln!("Failed to reshape FITS data: {e}");
        process::exit(1);
    })
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
    }
}
