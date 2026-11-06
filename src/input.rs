//! Typed validation failures for solver inputs.
use crate::extraction::SourceError;

/// A configuration constraint required by the solver algorithms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ConfigError {
    ScaleRange,
    FieldStarCount,
    CodeTolerance,
    ObservationEpoch,
    RegionCenter,
    RegionRadius,
    MatchRadius,
    DistractorFraction,
    LogOddsThresholds,
    MinimumMatches,
}
impl std::fmt::Display for ConfigError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let message = match self {
            Self::ScaleRange => "scale range must be finite, positive and ordered",
            Self::FieldStarCount => "max_field_stars must be at least four",
            Self::CodeTolerance => "code tolerance must be finite and nonnegative",
            Self::ObservationEpoch => "observation epoch must be finite",
            Self::RegionCenter => {
                "region center must have finite RA and declination in [-pi/2, pi/2]"
            }
            Self::RegionRadius => "region radius must be finite and in [0, pi]",
            Self::MatchRadius => "match radius and its square must be finite and positive",
            Self::DistractorFraction => "distractor fraction must be strictly between zero and one",
            Self::LogOddsThresholds => "log-odds thresholds must be ordered and not NaN",
            Self::MinimumMatches => "minimum matches must be positive",
        };
        f.write_str(message)
    }
}
impl std::error::Error for ConfigError {}

/// Checked solve failed before searching: configuration or source geometry is invalid.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SolveInputError {
    Sources(SourceError),
    Config(ConfigError),
}
impl std::fmt::Display for SolveInputError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Sources(e) => e.fmt(f),
            Self::Config(e) => e.fmt(f),
        }
    }
}
impl std::error::Error for SolveInputError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        Some(match self {
            Self::Sources(e) => e,
            Self::Config(e) => e,
        })
    }
}
impl From<SourceError> for SolveInputError {
    fn from(e: SourceError) -> Self {
        Self::Sources(e)
    }
}
impl From<ConfigError> for SolveInputError {
    fn from(e: ConfigError) -> Self {
        Self::Config(e)
    }
}
