//! `zodiacal-tools` — operational binaries for building and inspecting
//! zodiacal indexes (`.zdcl`) and refinement sidecars (`.zdcl.gaia`).
//!
//! These tools depend on `starfield-gaia` (currently a git dep until it
//! publishes to crates.io) and are deliberately split out of the
//! `zodiacal` library crate so that crate stays publishable without
//! pulling unpublished deps into its feature graph.

use std::path::PathBuf;
use std::process;

use clap::{Parser, Subcommand, ValueEnum};

use zodiacal::index::source::DEFAULT_CELL_DEPTH;

mod bench_bundle;
mod bench_indexes;
mod bench_triage;
mod build_from_excerpt_series;
mod build_from_shards;

use bench_bundle::{BenchBundleConfig, run as run_bench_bundle};
use bench_indexes::{BenchIndexesConfig, run as run_bench_indexes};
use bench_triage::{BenchTriageConfig, run as run_bench_triage};
use build_from_excerpt_series::{
    BuildFromExcerptSeriesConfig, OutputFormat, run as run_build_from_excerpt_series,
};
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

/// Output-form selector for `build-from-excerpt-series`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, ValueEnum)]
enum OutputFormatArg {
    /// Folder bundle (`<prefix>.zdcl.bundle/`).
    Dir,
    /// Zip-archive bundle (`<prefix>.zdcl.bundle.zip`).
    Zip,
    /// Both forms.
    Both,
}

impl From<OutputFormatArg> for OutputFormat {
    fn from(v: OutputFormatArg) -> Self {
        match v {
            OutputFormatArg::Dir => OutputFormat::Dir,
            OutputFormatArg::Zip => OutputFormat::Zip,
            OutputFormatArg::Both => OutputFormat::Both,
        }
    }
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

    /// Sweep the 1000-case test corpus against a list of pre-built
    /// single-band `.zdcl` indexes (the old multi-file layout, e.g.
    /// `gaia_jbt_NN.zdcl`). Mirrors `bench-bundle`'s output schema.
    BenchIndexes {
        /// One or more `.zdcl` files to load and pass to the solver
        /// as `&[&Index]`.
        #[arg(long, num_args = 1.., required = true)]
        index_file: Vec<PathBuf>,

        #[arg(long, default_value = "test_cases")]
        test_cases_dir: PathBuf,

        #[arg(long)]
        limit: Option<usize>,

        #[arg(long)]
        scale_hint: bool,

        #[arg(long, default_value_t = 0)]
        timeout_secs: u64,
    },

    /// Sweep the 1000-case test corpus against a built bundle. For
    /// each `NNNN.json` test case, opens a region of the bundle around
    /// the truth center, builds one Index per band, runs `solve()`,
    /// and emits a CSV row per case to stdout plus a summary on
    /// stderr.
    BenchBundle {
        /// Bundle to query (folder or `.zip`).
        #[arg(long)]
        bundle_path: PathBuf,

        /// Directory of `NNNN.json` test cases.
        #[arg(long, default_value = "../zodiacal-test-cases/set1-legacy")]
        test_cases_dir: PathBuf,

        /// Region radius in degrees. For a 2°×2° square use 1.4142.
        #[arg(long, default_value = "1.4142")]
        radius_deg: f64,

        /// Stop after this many cases (default: all).
        #[arg(long)]
        limit: Option<usize>,

        /// Hint the solver with truth pixel scale ±5%.
        #[arg(long)]
        scale_hint: bool,

        /// Per-case solve timeout in seconds. 0 disables.
        #[arg(long, default_value_t = 0)]
        timeout_secs: u64,

        /// Optional directory to write per-case trace JSON sidecars to.
        /// On a successful solve, emits `<case>.trace.json` containing
        /// the full WCS, the matched quad's 4 (field, catalog) pairs,
        /// and the verification matched-pair list (for plotting hits/
        /// misses against the image).
        #[arg(long)]
        trace_out: Option<PathBuf>,

        /// Observation epoch (Julian decimal year, e.g. 2002.874)
        /// used to propagate catalog proper motions before the WCS fit
        /// and verification step. When omitted, HST cases auto-fill
        /// from `hst.t_min_mjd`/`t_max_mjd` and synthetic Gaia-epoch
        /// cases default to no propagation.
        #[arg(long)]
        obs_epoch: Option<f64>,
    },

    /// Triage why specific bench cases failed. Projects bundle catalog
    /// stars and quads through the truth WCS into image pixels, then
    /// reports catalog-vs-detection coverage and a count of "findable"
    /// quads (4 cataloged stars projecting onto detected sources).
    BenchTriage {
        /// Bundle to query (folder or `.zip`).
        #[arg(long)]
        bundle_path: PathBuf,

        /// Directory of `NNNN.json` test cases.
        #[arg(long, default_value = "../zodiacal-test-cases/set2-dr3-mag19")]
        test_cases_dir: PathBuf,

        /// Path to a prior `bench-bundle` CSV. Every row with `solved=0`
        /// is triaged. Mutually exclusive with `--case-ids`.
        #[arg(long)]
        csv: Option<PathBuf>,

        /// Explicit comma-separated case IDs (e.g. `0042,0123`). Takes
        /// precedence over `--csv` when both are given.
        #[arg(long, value_delimiter = ',')]
        case_ids: Vec<String>,

        /// Region radius in degrees. Should match the bench's radius.
        #[arg(long, default_value = "1.4142")]
        radius_deg: f64,

        /// Pixel-radius for catalog-vs-detection match.
        #[arg(long, default_value = "3.0")]
        match_radius_pix: f64,

        /// Cap on bright field stars considered (mirror solver's
        /// `max_field_stars`).
        #[arg(long, default_value_t = 50)]
        max_field_stars: usize,

        /// Stop after this many cases (default: all).
        #[arg(long)]
        limit: Option<usize>,

        /// If set, sweep the 8 standard CD orientations on each case
        /// and print which one matches the most detections. Used to
        /// figure out the renderer's WCS convention.
        #[arg(long, default_value_t = false)]
        probe_cd: bool,
    },

    /// Build a multi-band `.zdcl.bundle` from a starfield-gaia
    /// "bycell" excerpt directory (a tree of `shard_NNNN.csv.gz`
    /// files, one per HEALPix cell at the excerpt's depth).
    ///
    /// Drives PR3's parallel cell-driven build phase plus PR4's
    /// single-threaded tidy phase to produce a folder or zip bundle
    /// (or both). The work directory survives across runs to support
    /// resume from crashes; pass `--prune-work-dir` to opt into
    /// destructive cleanup once the bundle is committed.
    ///
    /// PR6 v1 only supports the **fast path** where the bundle's
    /// `--cell-depth` equals the excerpt's depth. Cross-depth
    /// resharding is rejected with a clear error.
    BuildFromExcerptSeries {
        /// Directory containing `shard_NNNN.csv.gz` files. Omit to use
        /// the starfield-gaia cache directory.
        #[arg(long)]
        excerpt_dir: Option<PathBuf>,

        /// Build scratch directory. Required; survives across runs to
        /// support resume.
        #[arg(long)]
        work_dir: PathBuf,

        /// Output prefix; final artifact is `<prefix>.zdcl.bundle/`
        /// or `<prefix>.zdcl.bundle.zip` (or both).
        #[arg(long)]
        output_prefix: PathBuf,

        /// Output form to emit.
        #[arg(long, value_enum, default_value_t = OutputFormatArg::Dir)]
        output_format: OutputFormatArg,

        /// G-band magnitude cutoff applied to source rows. Capped at
        /// 17.0; deeper requires the streaming sidecar work.
        #[arg(long, default_value = "16.0")]
        mag_limit: f64,

        /// Per-cell brightness-truncation cap (applied after
        /// per-cell load, before quad emission).
        #[arg(long, default_value_t = 10_000)]
        max_stars_per_cell: usize,

        /// HEALPix nested-scheme depth for cell sharding. Must equal
        /// the excerpt's depth (PR6 v1 limitation).
        #[arg(long, default_value_t = DEFAULT_CELL_DEPTH)]
        cell_depth: u8,

        /// Number of scale bands. Band `i`'s scale range is
        /// `[scale_lower * f^i, scale_lower * f^(i+1)]`, where
        /// `f = (scale_upper / scale_lower)^(1 / bands)`.
        #[arg(long, default_value_t = 12)]
        bands: usize,

        /// Lower edge of band 0, in arcseconds.
        #[arg(long, default_value = "10.0")]
        scale_lower: f64,

        /// Upper edge of band `bands - 1`, in arcseconds.
        #[arg(long, default_value = "600.0")]
        scale_upper: f64,

        /// Per-band quad-count cap.
        #[arg(long, default_value_t = 100)]
        quads_per_cell: usize,

        /// Per-cell, per-band cap on how many times a single star
        /// may be referenced across emitted quads.
        #[arg(long, default_value_t = 8)]
        max_reuse: u32,

        /// Free-text experiment label embedded in the manifest.
        #[arg(long, default_value = "")]
        experiment: String,

        /// Rayon worker thread count.
        #[arg(long)]
        threads: Option<usize>,

        /// Remove the work directory after a successful tidy.
        /// Default: keep, so a follow-up tidy can produce the
        /// alternate output form without rebuilding.
        #[arg(long, default_value_t = false)]
        prune_work_dir: bool,

        /// Wall-clock seconds between BuildManifest snapshots to
        /// disk. The orchestrator runs a dedicated actor thread
        /// that owns the manifest; workers send completion events
        /// to it via a channel and never block on a manifest mutex.
        /// Resume granularity is "at most this many seconds of
        /// progress redone after a crash". Pass 0 to save only on
        /// shutdown (no periodic snapshots — fastest, but a crash
        /// loses everything since the last clean exit).
        #[arg(long, default_value_t = 30)]
        manifest_save_interval_secs: u64,
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
            let cfg = BuildFromShardsConfig {
                shards_dir: shards_dir.clone(),
                output_prefix: output_prefix.clone(),
                mag_limit: *mag_limit,
                scale_lower_arcsec: *scale_lower,
                scale_upper_arcsec: *scale_upper,
                max_quads: *max_quads,
                cell_depth: *cell_depth,
                threads: *threads,
            };
            if let Err(e) = run_build_from_shards(&cfg) {
                eprintln!("build-from-shards failed: {e}");
                process::exit(1);
            }
        }
        Commands::BenchIndexes {
            index_file,
            test_cases_dir,
            limit,
            scale_hint,
            timeout_secs,
        } => {
            let cfg = BenchIndexesConfig {
                index_files: index_file.clone(),
                test_cases_dir: test_cases_dir.clone(),
                limit: *limit,
                scale_hint: *scale_hint,
                timeout_secs: *timeout_secs,
            };
            if let Err(e) = run_bench_indexes(&cfg) {
                eprintln!("bench-indexes failed: {e}");
                process::exit(1);
            }
        }
        Commands::BenchBundle {
            bundle_path,
            test_cases_dir,
            radius_deg,
            limit,
            scale_hint,
            timeout_secs,
            trace_out,
            obs_epoch,
        } => {
            // Build a `Time` for the user-supplied --obs-epoch flag.
            // We don't need leap-second / delta-T tables here — Time is
            // only consumed by `propagate_pm` via `Time::j()` (pure TT
            // arithmetic).
            let cli_obs_epoch = obs_epoch.map(|year| starfield::time::Timescale::default().j(year));
            let cfg = BenchBundleConfig {
                bundle_path: bundle_path.clone(),
                test_cases_dir: test_cases_dir.clone(),
                radius_deg: *radius_deg,
                limit: *limit,
                scale_hint: *scale_hint,
                timeout_secs: *timeout_secs,
                trace_out: trace_out.clone(),
                obs_epoch: cli_obs_epoch,
            };
            if let Err(e) = run_bench_bundle(&cfg) {
                eprintln!("bench-bundle failed: {e}");
                process::exit(1);
            }
        }
        Commands::BenchTriage {
            bundle_path,
            test_cases_dir,
            csv,
            case_ids,
            radius_deg,
            match_radius_pix,
            max_field_stars,
            limit,
            probe_cd,
        } => {
            let cfg = BenchTriageConfig {
                bundle_path: bundle_path.clone(),
                test_cases_dir: test_cases_dir.clone(),
                csv: csv.clone(),
                case_ids: case_ids.clone(),
                radius_deg: *radius_deg,
                match_radius_pix: *match_radius_pix,
                max_field_stars: *max_field_stars,
                limit: *limit,
                probe_cd: *probe_cd,
            };
            if let Err(e) = run_bench_triage(&cfg) {
                eprintln!("bench-triage failed: {e}");
                process::exit(1);
            }
        }
        Commands::BuildFromExcerptSeries {
            excerpt_dir,
            work_dir,
            output_prefix,
            output_format,
            mag_limit,
            max_stars_per_cell,
            cell_depth,
            bands,
            scale_lower,
            scale_upper,
            quads_per_cell,
            max_reuse,
            experiment,
            threads,
            prune_work_dir,
            manifest_save_interval_secs,
        } => {
            let cfg = BuildFromExcerptSeriesConfig {
                excerpt_dir: excerpt_dir.clone(),
                work_dir: work_dir.clone(),
                output_prefix: output_prefix.clone(),
                output_format: (*output_format).into(),
                mag_limit: *mag_limit,
                max_stars_per_cell: *max_stars_per_cell,
                cell_depth: *cell_depth,
                bands: *bands,
                scale_lower_arcsec: *scale_lower,
                scale_upper_arcsec: *scale_upper,
                quads_per_cell: *quads_per_cell,
                max_reuse: *max_reuse,
                experiment: experiment.clone(),
                threads: *threads,
                prune_work_dir: *prune_work_dir,
                manifest_save_interval_secs: *manifest_save_interval_secs,
            };
            if let Err(e) = run_build_from_excerpt_series(&cfg) {
                eprintln!("build-from-excerpt-series failed: {e}");
                process::exit(1);
            }
        }
    }
}
