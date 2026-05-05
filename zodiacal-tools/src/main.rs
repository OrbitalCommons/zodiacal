//! `zodiacal-tools` — operational binaries for building and inspecting
//! zodiacal indexes (`.zdcl`) and refinement sidecars (`.zdcl.gaia`).
//!
//! These tools depend on `starfield-gaia` (currently a git dep until it
//! publishes to crates.io) and are deliberately split out of the
//! `zodiacal` library crate so that crate stays publishable without
//! pulling unpublished deps into its feature graph.

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand};

use zodiacal::index::source::DEFAULT_CELL_DEPTH;
use zodiacal::refinement::DEFAULT_PIVOT_STRIDE;

mod build_from_excerpt;
mod build_from_shards;

use build_from_excerpt::{BuildFromExcerptConfig, run as run_build_from_excerpt};
use build_from_shards::{BuildFromShardsConfig, run as run_build_from_shards};

#[derive(Parser)]
#[command(
    name = "zodiacal-tools",
    about = "Operational tools for zodiacal indexes and sidecars"
)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Build a `.zdcl` index plus a `.zdcl.gaia` refinement sidecar
    /// from a directory of Gaia DR3 `GaiaSource_*.csv.gz` (or
    /// excerpt-cache `shard_*.csv.gz`) shards in one pass.
    ///
    /// Reads every shard once, applies the magnitude limit, then builds
    /// the quad index (v3 layout) and writes the sidecar (sorted by
    /// source_id).
    ///
    /// Memory note: the v1 sidecar writer holds all records in RAM.
    /// Each star is ~120 bytes, so G≤16 (~83 M stars) is the practical
    /// ceiling on a 64 GB box; G>17 is rejected up front.
    BuildFromShards {
        /// Directory containing `*.csv(.gz)` shards. If omitted, uses
        /// the starfield-managed cache (`Downloader::<Dr3>::list_cached()`).
        #[arg(long)]
        shards_dir: Option<PathBuf>,

        /// Output prefix; emits `<prefix>.zdcl` and `<prefix>.zdcl.gaia`.
        #[arg(long)]
        output_prefix: PathBuf,

        /// G-band magnitude cutoff (must be <= 17.0; deeper won't fit in RAM).
        #[arg(long, default_value = "16.0")]
        mag_limit: f64,

        /// Minimum quad scale in arcseconds.
        #[arg(long, default_value = "30.0")]
        scale_lower: f64,

        /// Maximum quad scale in arcseconds.
        #[arg(long, default_value = "1800.0")]
        scale_upper: f64,

        /// Maximum number of quads to generate. Required: there is no
        /// default because the right value depends on the catalog
        /// depth, FOV target, and host RAM — picking one for the user
        /// hides the trade-off and (for the brightness-first builder)
        /// concentrates the budget in a few dense sky regions.
        #[arg(long)]
        max_quads: usize,

        /// HEALPix depth used to group stars in the v3 `.zdcl` file.
        #[arg(long, default_value_t = DEFAULT_CELL_DEPTH)]
        cell_depth: u8,

        /// Rayon worker thread count for parallel shard reads
        /// (default: rayon's default = all logical cores).
        #[arg(long)]
        threads: Option<usize>,
    },

    /// Build a `.zdcl` index plus a `.zdcl.gaia` refinement sidecar
    /// from a HEALPix-sharded `starfield-gaia` excerpt directory,
    /// cell-by-cell.
    ///
    /// Drives `LazyLoadingCatalog::<Dr3>::entries_in_cell` so memory
    /// is bounded by the largest cell's row count, not the whole
    /// catalog. Crash-safe: the work directory holds a manifest +
    /// per-cell artifacts that a re-run resumes from.
    ///
    /// Per-cell brightness truncation is applied at adapter time via
    /// `--max-stars-per-cell` so deeper mag limits (G ≤ 19/20 on the
    /// bycell excerpt) don't overwhelm the per-cell quad builder.
    BuildFromExcerpt {
        /// HEALPix-sharded excerpt directory (one shard file per
        /// cell at the level recorded in its manifest).
        #[arg(long)]
        excerpt_dir: PathBuf,

        /// Output prefix; emits `<prefix>.zdcl` and `<prefix>.zdcl.gaia`.
        #[arg(long)]
        output_prefix: PathBuf,

        /// Working directory for the manifest + per-cell artifacts +
        /// sidecar chunks. A re-run with the same path resumes.
        #[arg(long)]
        work_dir: PathBuf,

        /// G-band magnitude cutoff applied at the source.
        #[arg(long)]
        mag_limit: f64,

        /// Brightest-N stars to keep per HEALPix cell after the mag
        /// cut. Bounds peak per-cell RAM and per-cell quad-build cost.
        #[arg(long)]
        max_stars_per_cell: usize,

        /// Quads to emit per HEALPix cell. The cell-driven builder
        /// targets this value per cell; cells with insufficient
        /// scale-valid stars produce fewer.
        #[arg(long)]
        quads_per_cell: usize,

        /// Cap on how many quads any one star may participate in
        /// within a cell. Mirrors `CatalogBuilderConfig.max_reuse`.
        #[arg(long, default_value_t = 8)]
        max_reuse: usize,

        /// Minimum quad scale in arcseconds.
        #[arg(long, default_value = "30.0")]
        scale_lower: f64,

        /// Maximum quad scale in arcseconds.
        #[arg(long, default_value = "1800.0")]
        scale_upper: f64,

        /// HEALPix depth used to group stars in the final v3 `.zdcl`
        /// file. Independent of the excerpt's read-time depth.
        #[arg(long, default_value_t = DEFAULT_CELL_DEPTH)]
        cell_depth: u8,

        /// Sidecar pivot table stride.
        #[arg(long, default_value_t = DEFAULT_PIVOT_STRIDE)]
        pivot_stride: u32,

        /// Rayon worker thread count.
        #[arg(long)]
        threads: Option<usize>,
    },
}

fn main() {
    let cli = Cli::parse();
    match &cli.command {
        Commands::BuildFromShards {
            shards_dir,
            output_prefix,
            mag_limit,
            scale_lower,
            scale_upper,
            max_quads,
            cell_depth,
            threads,
        } => {
            let cfg = match BuildFromShardsConfig::builder()
                .shards_dir(shards_dir.clone())
                .output_prefix(output_prefix.clone())
                .mag_limit(*mag_limit)
                .scale_lower_arcsec(*scale_lower)
                .scale_upper_arcsec(*scale_upper)
                .max_quads(*max_quads)
                .cell_depth(*cell_depth)
                .threads(*threads)
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("build-from-shards: invalid config: {e}");
                    process::exit(1);
                }
            };
            if let Err(e) = run_build_from_shards(&cfg) {
                eprintln!("build-from-shards failed: {e}");
                process::exit(1);
            }
        }
        Commands::BuildFromExcerpt {
            excerpt_dir,
            output_prefix,
            work_dir,
            mag_limit,
            max_stars_per_cell,
            quads_per_cell,
            max_reuse,
            scale_lower,
            scale_upper,
            cell_depth,
            pivot_stride,
            threads,
        } => {
            let cfg = match BuildFromExcerptConfig::builder()
                .excerpt_dir(excerpt_dir.clone())
                .output_prefix(output_prefix.clone())
                .work_dir(work_dir.clone())
                .mag_limit(*mag_limit)
                .max_stars_per_cell(*max_stars_per_cell)
                .quads_per_cell(*quads_per_cell)
                .max_reuse(*max_reuse)
                .scale_lower_arcsec(*scale_lower)
                .scale_upper_arcsec(*scale_upper)
                .final_cell_depth(*cell_depth)
                .pivot_stride(*pivot_stride)
                .threads(*threads)
                .build()
            {
                Ok(c) => c,
                Err(e) => {
                    eprintln!("build-from-excerpt: invalid config: {e}");
                    process::exit(1);
                }
            };
            if let Err(e) = run_build_from_excerpt(&cfg) {
                eprintln!("build-from-excerpt failed: {e}");
                process::exit(1);
            }
        }
    }
}
