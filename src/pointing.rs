//! `PointingSource` — abstracts "where is this platform looking right now?"
//!
//! Plan 4 of the deployment-mode roadmap (`plans/04-pointing-sources.md`).
//! Two concrete implementations:
//!
//! - [`GroundStation`] — `(lat, lon, height, time) → zenith region`
//! - [`SpacecraftBoresight`] — `(ephemeris, attitude) → boresight region`
//!
//! Both feed into [`crate::index::LiveIndex::set_region`] so the
//! orchestrator (plan 5) can keep only the cells currently observable.

use std::f64::consts::PI;

use nalgebra::UnitQuaternion;
use starfield::Equatorial;
use starfield::time::Time;

use crate::geom::sphere::xyz_to_radec;
use crate::refinement::ObserverState;
use crate::solver::SkyRegion;

/// "Where is this platform looking right now?" The returned `SkyRegion`
/// drives [`crate::index::LiveIndex`] cell membership; the optional
/// observer state, if present, is consumed by the refinement pipeline
/// for parallax / aberration corrections.
pub trait PointingSource: Send {
    fn current_region(&self, t: &Time) -> SkyRegion;
    /// Default: no observer state (refinement falls back to "no
    /// observer" mode — no parallax, no aberration). Implementors
    /// that have spacecraft / station geometry override this.
    fn observer_state(&self, _t: &Time) -> Option<ObserverState> {
        None
    }
}

// ----- Spacecraft path ---------------------------------------------------

/// BCRS state vector — position (AU) + velocity (AU/day) — supplied by
/// the caller (typically from a SPICE kernel via `anise`, a TLE via
/// `starfield::sgp4lib`, or a custom Kalman filter).
#[derive(Debug, Clone, Copy)]
pub struct BcrsState {
    pub position_au: [f64; 3],
    pub velocity_au_per_day: [f64; 3],
}

/// Source of spacecraft barycentric state at a given time.
pub trait EphemerisSource: Send {
    fn state_at(&self, t: &Time) -> Option<BcrsState>;
}

/// Source of body-to-inertial attitude quaternion at a given time. The
/// convention is "body frame → inertial frame" (multiplying a body-frame
/// vector by the quaternion gives its inertial-frame coordinates).
pub trait AttitudeSource: Send {
    fn quaternion_at(&self, t: &Time) -> Option<UnitQuaternion<f64>>;
}

/// `PointingSource` for a spacecraft with known ephemeris and
/// attitude. The boresight is the body-frame unit vector
/// `boresight_body` rotated into the inertial frame by the current
/// attitude quaternion.
pub struct SpacecraftBoresight<E: EphemerisSource, A: AttitudeSource> {
    pub ephemeris: E,
    pub attitude: A,
    /// Detector half-angle (radians).
    pub detector_half_angle_rad: f64,
    /// Extra padding beyond the detector FOV (radians). Typically
    /// 5-10° to keep cells loaded for short-term slews.
    pub fov_padding_rad: f64,
    /// Body-frame unit vector that defines "boresight". Default
    /// recommendation: `[0.0, 0.0, 1.0]` (+Z).
    pub boresight_body: [f64; 3],
}

impl<E: EphemerisSource, A: AttitudeSource> SpacecraftBoresight<E, A> {
    pub fn new(
        ephemeris: E,
        attitude: A,
        detector_half_angle_rad: f64,
        fov_padding_rad: f64,
    ) -> Self {
        Self {
            ephemeris,
            attitude,
            detector_half_angle_rad,
            fov_padding_rad,
            boresight_body: [0.0, 0.0, 1.0],
        }
    }
}

impl<E: EphemerisSource, A: AttitudeSource> PointingSource for SpacecraftBoresight<E, A> {
    fn current_region(&self, t: &Time) -> SkyRegion {
        // If attitude is unavailable, fall back to a full-sky region so
        // the orchestrator falls back to blind solving rather than
        // refusing entirely.
        let q = match self.attitude.quaternion_at(t) {
            Some(q) => q,
            None => {
                return SkyRegion::from_radians(Equatorial::new(0.0, 0.0), PI);
            }
        };
        let body = nalgebra::Vector3::new(
            self.boresight_body[0],
            self.boresight_body[1],
            self.boresight_body[2],
        );
        let inertial = q * body;
        let (ra, dec) = xyz_to_radec([inertial.x, inertial.y, inertial.z]);
        SkyRegion::from_radians(
            Equatorial::new(ra, dec),
            self.detector_half_angle_rad + self.fov_padding_rad,
        )
    }

    fn observer_state(&self, t: &Time) -> Option<ObserverState> {
        let s = self.ephemeris.state_at(t)?;
        Some(ObserverState::Barycentric {
            position_au: s.position_au,
            velocity_au_per_day: s.velocity_au_per_day,
        })
    }
}

// ----- Ground station path -----------------------------------------------

/// `PointingSource` for a fixed point on the rotating Earth. Returns a
/// region centered on the local zenith with radius `90° - min_altitude`
/// (i.e., the visible cap above the user-chosen minimum altitude).
///
/// **`observer_state` is `None` in v1** (issue #44 tracks the
/// terrestrial → BCRS state pipeline). Until that lands, ground-mode
/// refinement falls back to no parallax/aberration corrections — fine
/// for arcsec-scale work, suboptimal for the 10 mas target.
pub struct GroundStation {
    /// Geodetic latitude in radians (positive north).
    pub latitude_rad: f64,
    /// Geodetic longitude in radians (positive east).
    pub longitude_rad: f64,
    /// Stars below this altitude (in radians) are excluded from the
    /// returned region. Typical: 20-30°.
    pub min_altitude_rad: f64,
}

impl GroundStation {
    pub fn from_degrees(latitude_deg: f64, longitude_deg: f64, min_altitude_deg: f64) -> Self {
        Self {
            latitude_rad: latitude_deg.to_radians(),
            longitude_rad: longitude_deg.to_radians(),
            min_altitude_rad: min_altitude_deg.to_radians(),
        }
    }

    /// Zenith (RA, Dec) in radians at the given time.
    ///
    /// Dec at zenith == observer geodetic latitude (close enough — at
    /// the cm level you'd want geodetic-vs-geocentric latitude
    /// distinction; not relevant for cell-loading purposes).
    ///
    /// RA at zenith == Local Apparent Sidereal Time, expressed in
    /// radians.
    pub fn zenith(&self, time: &Time) -> (f64, f64) {
        // GAST is in hours; LST = GAST + longitude(in hours).
        let gast_hours = time.gast();
        let lst_hours = (gast_hours + self.longitude_rad * 12.0 / PI).rem_euclid(24.0);
        let zenith_ra = lst_hours * PI / 12.0;
        let zenith_dec = self.latitude_rad;
        (zenith_ra, zenith_dec)
    }
}

impl PointingSource for GroundStation {
    fn current_region(&self, t: &Time) -> SkyRegion {
        let (ra, dec) = self.zenith(t);
        let radius = (PI / 2.0) - self.min_altitude_rad;
        SkyRegion::from_radians(Equatorial::new(ra, dec), radius)
    }
}

// ----- Trivial / test-fixture sources ------------------------------------

/// A pre-fabricated region. Useful for tests, replays, and any caller
/// that already knows where it's looking from external context.
pub struct StaticRegion(pub SkyRegion);

impl PointingSource for StaticRegion {
    fn current_region(&self, _t: &Time) -> SkyRegion {
        self.0.clone()
    }
}

/// An attitude source that always returns the same quaternion.
pub struct StaticAttitude(pub UnitQuaternion<f64>);

impl AttitudeSource for StaticAttitude {
    fn quaternion_at(&self, _t: &Time) -> Option<UnitQuaternion<f64>> {
        Some(self.0)
    }
}

/// An ephemeris source that always returns the same BCRS state.
pub struct StaticEphemeris(pub BcrsState);

impl EphemerisSource for StaticEphemeris {
    fn state_at(&self, _t: &Time) -> Option<BcrsState> {
        Some(self.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Vector3;
    use starfield::time::Timescale;

    fn now() -> Time {
        let ts = Timescale::default();
        ts.tt_jd(2_460_000.5, None)
    }

    #[test]
    fn static_region_returns_supplied_region() {
        let sky = SkyRegion::from_radians(Equatorial::new(1.5, -0.3), 0.1);
        let src = StaticRegion(sky.clone());
        let returned = src.current_region(&now());
        assert!((returned.center.ra - sky.center.ra).abs() < 1e-15);
        assert!((returned.center.dec - sky.center.dec).abs() < 1e-15);
        assert!((returned.radius_rad - sky.radius_rad).abs() < 1e-15);
    }

    #[test]
    fn static_region_has_no_observer_state() {
        let src = StaticRegion(SkyRegion::from_radians(Equatorial::new(0.0, 0.0), 0.1));
        assert!(src.observer_state(&now()).is_none());
    }

    #[test]
    fn ground_station_zenith_dec_equals_latitude() {
        let station = GroundStation::from_degrees(34.0, -118.0, 20.0);
        let (_ra, dec) = station.zenith(&now());
        assert!((dec - 34.0_f64.to_radians()).abs() < 1e-12);
    }

    #[test]
    fn ground_station_radius_matches_min_altitude() {
        let station = GroundStation::from_degrees(0.0, 0.0, 30.0);
        let region = station.current_region(&now());
        let expected = (PI / 2.0) - 30.0_f64.to_radians();
        assert!((region.radius_rad - expected).abs() < 1e-12);
    }

    #[test]
    fn ground_station_lst_advances_with_time() {
        // Across a sidereal day, the zenith RA should sweep through
        // ~2π. Take two times one sidereal day apart and verify the
        // zenith RA wraps back close to its starting value.
        let ts = Timescale::default();
        let t0 = ts.tt_jd(2_460_000.5, None);
        let t1 = ts.tt_jd(2_460_001.5 - 0.0027378, None); // ~1 sidereal day later
        let station = GroundStation::from_degrees(0.0, 0.0, 30.0);
        let (ra0, _) = station.zenith(&t0);
        let (ra1, _) = station.zenith(&t1);
        // Expect <few-arcmin difference (sidereal-day approx is rough).
        let diff = (ra0 - ra1).abs().min((ra0 - ra1 + 2.0 * PI).abs());
        assert!(diff < 0.05, "zenith RA wrap mismatch: {diff} rad");
    }

    #[test]
    fn ground_station_observer_state_is_none_v1() {
        // Issue #44 will replace this; for now the trait default kicks in.
        let station = GroundStation::from_degrees(0.0, 0.0, 30.0);
        assert!(station.observer_state(&now()).is_none());
    }

    #[test]
    fn spacecraft_identity_quaternion_points_at_north_pole() {
        let attitude = StaticAttitude(UnitQuaternion::identity());
        let ephem = StaticEphemeris(BcrsState {
            position_au: [1.0, 0.0, 0.0],
            velocity_au_per_day: [0.0, 0.017, 0.0],
        });
        let sat = SpacecraftBoresight::new(ephem, attitude, 0.05, 0.05);
        let region = sat.current_region(&now());
        // Default boresight = +Z body. Identity quaternion → +Z inertial = north pole.
        assert!((region.center.dec - PI / 2.0).abs() < 1e-12);
        // Region radius = detector_half + padding = 0.10 rad.
        assert!((region.radius_rad - 0.10).abs() < 1e-12);
    }

    #[test]
    fn spacecraft_quaternion_rotates_boresight() {
        // Rotate the +Z boresight 90° around Y in the body→inertial
        // direction so the boresight points along +X inertial.
        let q = UnitQuaternion::from_axis_angle(&Vector3::y_axis(), PI / 2.0);
        let attitude = StaticAttitude(q);
        let ephem = StaticEphemeris(BcrsState {
            position_au: [0.0; 3],
            velocity_au_per_day: [0.0; 3],
        });
        let sat = SpacecraftBoresight::new(ephem, attitude, 0.01, 0.0);
        let region = sat.current_region(&now());
        // +X inertial = (RA=0, Dec=0).
        assert!(region.center.ra.abs() < 1e-12 || (region.center.ra - 2.0 * PI).abs() < 1e-12);
        assert!(region.center.dec.abs() < 1e-12);
    }

    #[test]
    fn spacecraft_observer_state_passes_through_ephemeris() {
        let state = BcrsState {
            position_au: [1.5, -0.2, 0.7],
            velocity_au_per_day: [0.001, 0.017, -0.003],
        };
        let attitude = StaticAttitude(UnitQuaternion::identity());
        let sat = SpacecraftBoresight::new(StaticEphemeris(state), attitude, 0.05, 0.05);
        match sat.observer_state(&now()) {
            Some(ObserverState::Barycentric {
                position_au,
                velocity_au_per_day,
            }) => {
                assert_eq!(position_au, state.position_au);
                assert_eq!(velocity_au_per_day, state.velocity_au_per_day);
            }
            other => panic!("expected Barycentric state, got {other:?}"),
        }
    }

    /// AttitudeSource that always returns None — simulates "estimator
    /// hasn't initialized yet."
    struct NoneAttitude;
    impl AttitudeSource for NoneAttitude {
        fn quaternion_at(&self, _t: &Time) -> Option<UnitQuaternion<f64>> {
            None
        }
    }

    #[test]
    fn spacecraft_no_attitude_falls_back_to_full_sky() {
        let ephem = StaticEphemeris(BcrsState {
            position_au: [0.0; 3],
            velocity_au_per_day: [0.0; 3],
        });
        let sat = SpacecraftBoresight::new(ephem, NoneAttitude, 0.01, 0.01);
        let region = sat.current_region(&now());
        // Full-sky fallback: radius >= π.
        assert!(region.radius_rad >= PI - 1e-12);
    }
}
