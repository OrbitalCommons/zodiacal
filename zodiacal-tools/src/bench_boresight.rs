//! Explicit comparison pixel for benchmark truth recorded at a renderer boresight.
use clap::Args;
use zodiacal::geom::tan::TanWcs;

#[derive(Args, Clone, Copy, Debug, Default)]
pub struct BoresightOffset {
    /// Comparison-pixel X offset from width/2 (pixels). Does not change solving.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true, value_parser = finite_offset)]
    boresight_offset_x: f64,
    /// Comparison-pixel Y offset from height/2. Use -1 for set1-legacy, 0 for set2/set3.
    #[arg(long, default_value_t = 0.0, allow_hyphen_values = true, value_parser = finite_offset)]
    boresight_offset_y: f64,
}

fn finite_offset(value: &str) -> Result<f64, String> {
    let value: f64 = value
        .parse()
        .map_err(|e| format!("invalid pixel offset: {e}"))?;
    if value.is_finite() {
        Ok(value)
    } else {
        Err("pixel offset must be finite".into())
    }
}

impl BoresightOffset {
    pub fn pixel(self, wcs: &TanWcs) -> [f64; 2] {
        [
            wcs.image_size[0] / 2.0 + self.boresight_offset_x,
            wcs.image_size[1] / 2.0 + self.boresight_offset_y,
        ]
    }
    pub fn radec(self, wcs: &TanWcs) -> (f64, f64) {
        let [x, y] = self.pixel(wcs);
        wcs.pixel_to_radec(x, y)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::Parser;
    #[derive(Parser)]
    struct Cli {
        #[command(flatten)]
        offset: BoresightOffset,
    }

    #[test]
    fn comparison_pixel_tracks_flipped_boresight() {
        let wcs = TanWcs {
            crval: [1.0, 0.5],
            crpix: [50.0, 39.0],
            cd: [[1e-5, 0.0], [0.0, -1e-5]],
            image_size: [100.0, 80.0],
        };
        let default = Cli::try_parse_from(["bench"]).unwrap().offset;
        assert_eq!(default.radec(&wcs), wcs.field_center());
        let flipped = Cli::try_parse_from(["bench", "--boresight-offset-y", "-1"])
            .unwrap()
            .offset;
        assert_eq!(flipped.pixel(&wcs), wcs.crpix);
        let (ra, dec) = flipped.radec(&wcs);
        assert!((ra - wcs.crval[0]).abs() < 1e-12);
        assert!((dec - wcs.crval[1]).abs() < 1e-12);
        for bad in ["NaN", "inf", "-inf"] {
            assert!(Cli::try_parse_from(["bench", "--boresight-offset-y", bad]).is_err());
        }
    }
}
