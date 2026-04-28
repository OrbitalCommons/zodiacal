use std::collections::HashMap;
use std::path::Path;

use crate::geom::sip::SipWcs;

use super::sidecar::{SidecarReader, SidecarRecord};

/// Observation time + observer state, used to evaluate the apparent
/// direction of a catalog star at the moment of exposure.
pub struct ObservationContext {
    pub time: starfield::time::Time,
    pub observer: ObserverState,
}

/// Where the observer is in the solar system at the observation time.
///
/// For v1 only `Barycentric` is supported. Terrestrial observations must
/// be pre-reduced to a BCRS state vector by the caller.
pub enum ObserverState {
    /// BCRS position and velocity of the observer at `ObservationContext::time`.
    /// Units: AU and AU/day.
    Barycentric {
        position_au: [f64; 3],
        velocity_au_per_day: [f64; 3],
    },
}

/// Full Gaia 6-parameter astrometric solution for one source, plus 1-sigma
/// uncertainties. Correlations are intentionally omitted from v1; a full
/// 5x5 covariance block is a future addition (see PLAN.md §11.4).
#[derive(Debug, Clone, Copy)]
pub struct GaiaAstrometry {
    pub ra_deg: f64,
    pub dec_deg: f64,
    /// Proper motion in RA, already including cos(dec).
    pub pmra_mas_per_year: f64,
    pub pmdec_mas_per_year: f64,
    pub parallax_mas: f64,
    pub radial_km_per_s: f64,
    /// Reference epoch as a Julian year (e.g. 2016.0 for Gaia DR3).
    pub ref_epoch_jyear: f64,

    pub sigma_ra_mas: f64,
    pub sigma_dec_mas: f64,
    pub sigma_pmra_mas_per_year: f64,
    pub sigma_pmdec_mas_per_year: f64,
    pub sigma_parallax_mas: f64,
}

/// In-memory catalog of Gaia astrometry keyed by the same `catalog_id`
/// stored on each `IndexStar`. A persistent sidecar format (parquet) is
/// deferred until `starfield-datasources` exposes a canonical ingest path.
#[derive(Debug, Default, Clone)]
pub struct RefinementCatalog {
    pub sources: HashMap<u64, GaiaAstrometry>,
}

impl RefinementCatalog {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn insert(&mut self, catalog_id: u64, astrometry: GaiaAstrometry) {
        self.sources.insert(catalog_id, astrometry);
    }

    pub fn get(&self, catalog_id: u64) -> Option<&GaiaAstrometry> {
        self.sources.get(&catalog_id)
    }

    pub fn len(&self) -> usize {
        self.sources.len()
    }

    pub fn is_empty(&self) -> bool {
        self.sources.is_empty()
    }

    /// Load a subset of the sidecar at `path` filtered to the given
    /// `source_ids`. Missing ids are silently skipped. The caller owns
    /// the list (typically the `catalog_id`s from the in-memory `Index`).
    pub fn load_sidecar_filtered(
        path: &Path,
        source_ids: &[u64],
    ) -> Result<Self, RefinementError> {
        let reader = SidecarReader::open(path).map_err(|e| {
            RefinementError::Starfield(format!("sidecar open failed: {e}"))
        })?;
        let results = reader.get_many(source_ids);

        let mut sources = HashMap::with_capacity(results.len());
        for record in results.into_iter().flatten() {
            sources.insert(record.source_id, sidecar_record_to_astrometry(&record));
        }
        Ok(Self { sources })
    }
}

/// Convert a sidecar on-disk record into the in-memory `GaiaAstrometry`
/// used by the refinement pipeline.
///
/// NaN astrometry fields (unpublished parallax, missing RV) become zero,
/// matching `starfield::starlib::Star`'s fallback behavior.
fn sidecar_record_to_astrometry(r: &SidecarRecord) -> GaiaAstrometry {
    fn nan_to_zero(v: f64) -> f64 {
        if v.is_nan() { 0.0 } else { v }
    }
    GaiaAstrometry {
        ra_deg: r.ra,
        dec_deg: r.dec,
        pmra_mas_per_year: nan_to_zero(r.pmra),
        pmdec_mas_per_year: nan_to_zero(r.pmdec),
        parallax_mas: nan_to_zero(r.parallax),
        radial_km_per_s: nan_to_zero(r.radial_velocity),
        ref_epoch_jyear: r.ref_epoch,
        sigma_ra_mas: r.sigma_ra as f64,
        sigma_dec_mas: r.sigma_dec as f64,
        sigma_pmra_mas_per_year: r.sigma_pmra as f64,
        sigma_pmdec_mas_per_year: r.sigma_pmdec as f64,
        sigma_parallax_mas: r.sigma_parallax as f64,
    }
}

#[derive(Debug, Clone)]
pub struct RefinementConfig {
    pub match_radius_pix: f64,
    pub max_iterations: usize,
    pub convergence_pix: f64,
    pub min_matches: usize,
}

impl Default for RefinementConfig {
    fn default() -> Self {
        Self {
            match_radius_pix: 3.0,
            max_iterations: 5,
            convergence_pix: 0.05,
            min_matches: 10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct RefinedSolution {
    pub wcs: SipWcs,
    pub n_iterations: usize,
    pub residual_rms_mas: f64,
    pub residual_rms_pix: f64,
    pub matched: Vec<RefinedMatch>,
}

#[derive(Debug, Clone, Copy)]
pub struct RefinedMatch {
    pub catalog_id: u64,
    pub field_source_idx: usize,
    pub apparent_ra_deg: f64,
    pub apparent_dec_deg: f64,
    pub residual_mas: f64,
    pub weight: f64,
}

#[derive(Debug)]
pub enum RefinementError {
    InsufficientMatches { required: usize, actual: usize },
    DidNotConverge(usize),
    TerrestrialNotSupported,
    Starfield(String),
}

impl std::fmt::Display for RefinementError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::InsufficientMatches { required, actual } => {
                write!(
                    f,
                    "fewer than {required} field↔catalog matches found ({actual})"
                )
            }
            Self::DidNotConverge(n) => write!(f, "refinement did not converge in {n} iterations"),
            Self::TerrestrialNotSupported => {
                write!(
                    f,
                    "terrestrial observer not yet supported; provide Barycentric state"
                )
            }
            Self::Starfield(msg) => write!(f, "starfield error: {msg}"),
        }
    }
}

impl std::error::Error for RefinementError {}
