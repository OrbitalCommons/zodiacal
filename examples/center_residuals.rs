//! Compare image-center conventions against frozen renderer boresights.
//! Usage: cargo run --release --example center_residuals -- CASE_ROOT LIMIT INDEX...
use starfield::Equatorial;
use std::{error::Error, fs, path::PathBuf, time::Duration};
use zodiacal::solver::SkyRegion;
use zodiacal::{
    extraction::SourcesJson,
    index::Index,
    solver::{SolveOutcome, SolverConfig, solve_checked},
    verify::VerifyConfig,
};
fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args().skip(1);
    let root = PathBuf::from(args.next().ok_or("CASE_ROOT required")?);
    let limit: usize = args.next().ok_or("LIMIT required")?.parse()?;
    let indexes = args
        .map(|p| {
            eprintln!("Loading {p}");
            Index::load(std::path::Path::new(&p))
        })
        .collect::<Result<Vec<_>, _>>()?;
    if indexes.is_empty() {
        return Err("at least one INDEX required".into());
    }
    let refs: Vec<_> = indexes.iter().collect();
    println!(
        "set,case,status,truth_x_offset,truth_y_offset,legacy_ra_arcsec,legacy_dec_arcsec,geometric_ra_arcsec,geometric_dec_arcsec,flipped_ra_arcsec,flipped_dec_arcsec"
    );
    for set in [
        "set1-legacy",
        "set2-dr3-mag19",
        "set3-dr3-mag19-galaxies-10k",
    ] {
        let mut paths = fs::read_dir(root.join(set))?
            .map(|p| p.map(|p| p.path()))
            .collect::<Result<Vec<_>, _>>()?;
        paths.retain(|p| p.extension().is_some_and(|e| e == "json"));
        paths.sort();
        for path in paths.into_iter().take(limit) {
            let case: SourcesJson = serde_json::from_reader(fs::File::open(&path)?)?;
            let (ra, dec) = (
                case.ra_deg.ok_or("missing RA")?.to_radians(),
                case.dec_deg.ok_or("missing Dec")?.to_radians(),
            );
            let scale = case.plate_scale_arcsec.ok_or("missing scale")?;
            let config = SolverConfig {
                scale_range: Some((scale * 0.75, scale * 1.25)),
                within: Some(SkyRegion::try_from_radians(
                    Equatorial { ra, dec },
                    0.5_f64.to_radians(),
                )?),
                timeout: Some(Duration::from_secs(3)),
                verify: VerifyConfig {
                    match_radius_pix: 3.,
                    log_odds_accept: 100.,
                    ..Default::default()
                },
                ..Default::default()
            };
            let report = solve_checked(
                &case.sources,
                &refs,
                (case.image_width, case.image_height),
                &config,
            )?;
            let name = path.file_stem().unwrap().to_string_lossy();
            if let SolveOutcome::Solved(solution) = report.outcome {
                let wcs = &solution.wcs;
                let (cx, cy) = (case.image_width / 2., case.image_height / 2.);
                let (tx, ty) = wcs
                    .radec_to_pixel(ra, dec)
                    .ok_or("truth outside projection")?;
                print!("{set},{name},solved,{:.8},{:.8}", tx - cx, ty - cy);
                for (x, y) in [(cx, cy), (cx - 0.5, cy - 0.5), (cx, cy - 1.)] {
                    let (got_ra, got_dec) = wcs.pixel_to_radec(x, y);
                    let dra = (got_ra - ra + std::f64::consts::PI)
                        .rem_euclid(std::f64::consts::TAU)
                        - std::f64::consts::PI;
                    print!(
                        ",{:.8},{:.8}",
                        dra.to_degrees() * dec.cos() * 3600.,
                        (got_dec - dec).to_degrees() * 3600.
                    );
                }
                println!();
            } else {
                println!("{set},{name},unsolved,,,,,,,,");
            }
            eprintln!("{set}/{name}");
        }
    }
    Ok(())
}
