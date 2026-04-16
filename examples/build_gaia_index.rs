//! Build a zodiacal index from the local Gaia DR1 catalog.
//!
//! Processes all Gaia CSV files in parallel using rayon for fast index building.
//!
//! Usage:
//!   cargo run --example build_gaia_index --release -- [options]
//!
//! Options:
//!   --output PATH       Output index file (default: gaia_index.zdcl)
//!   --mag-limit MAG     Magnitude limit (default: 18.0)
//!   --scale-lo ARCSEC   Min quad scale in arcsec (default: 10.0)
//!   --scale-hi ARCSEC   Max quad scale in arcsec (default: 1800.0)
//!   --stars-per-cell N  Max stars per HEALPix cell (default: 20)
//!   --passes N          Quad building passes per cell (default: 16)

use std::path::PathBuf;
use std::time::Instant;

use zodiacal::index::builder::CatalogBuilderConfig;
use zodiacal::index::gaia::{GaiaSpatialCatalog, build_index_from_gaia};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut output = PathBuf::from("gaia_index.zdcl");
    let mut mag_limit = 18.0_f64;
    let mut scale_lo = 10.0_f64;
    let mut scale_hi = 1800.0_f64;
    let mut stars_per_cell = 20_usize;
    let mut passes = 16_usize;
    let mut uni_depth: Option<u8> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--output" => {
                i += 1;
                output = PathBuf::from(&args[i]);
            }
            "--mag-limit" => {
                i += 1;
                mag_limit = args[i].parse().expect("invalid --mag-limit");
            }
            "--scale-lo" => {
                i += 1;
                scale_lo = args[i].parse().expect("invalid --scale-lo");
            }
            "--scale-hi" => {
                i += 1;
                scale_hi = args[i].parse().expect("invalid --scale-hi");
            }
            "--stars-per-cell" => {
                i += 1;
                stars_per_cell = args[i].parse().expect("invalid --stars-per-cell");
            }
            "--passes" => {
                i += 1;
                passes = args[i].parse().expect("invalid --passes");
            }
            "--depth" => {
                i += 1;
                uni_depth = Some(args[i].parse().expect("invalid --depth"));
            }
            _ => {
                eprintln!("Unknown arg: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    eprintln!("=== Gaia Index Builder (parallel) ===");
    eprintln!("  mag_limit:      {mag_limit}");
    eprintln!("  scale range:    {scale_lo}\" - {scale_hi}\"");
    eprintln!("  stars/cell:     {stars_per_cell}");
    eprintln!("  passes:         {passes}");
    eprintln!("  output:         {}", output.display());
    eprintln!();

    let t0 = Instant::now();

    let gaia_dir = GaiaSpatialCatalog::default_dir().unwrap_or_else(|| {
        eprintln!("No cache directory found");
        std::process::exit(1);
    });

    let config = CatalogBuilderConfig {
        scale_lower: (scale_lo / 3600.0_f64).to_radians(),
        scale_upper: (scale_hi / 3600.0_f64).to_radians(),
        max_stars_per_cell: stars_per_cell,
        passes,
        uniformize_depth: uni_depth,
        ..CatalogBuilderConfig::default()
    };

    let index = build_index_from_gaia(&gaia_dir, mag_limit, &config).unwrap_or_else(|e| {
        eprintln!("Failed to build index: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "\nIndex built in {:.1}s: {} stars, {} quads",
        t0.elapsed().as_secs_f64(),
        index.stars.len(),
        index.quads.len(),
    );

    index.save(&output).unwrap_or_else(|e| {
        eprintln!("Failed to save index: {e}");
        std::process::exit(1);
    });

    eprintln!(
        "Saved to {} ({:.1}MB)",
        output.display(),
        std::fs::metadata(&output).map(|m| m.len()).unwrap_or(0) as f64 / 1e6
    );
    eprintln!("Total time: {:.1}s", t0.elapsed().as_secs_f64());
}
