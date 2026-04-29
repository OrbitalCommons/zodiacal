# Plan 4: PointingSource (where am I looking right now?)

## Goal

Abstract "what region of sky should be loaded" so the same `RealtimeSolver`
orchestrator can drive a ground telescope, a star tracker, or any future
deployment without forking the code path. Two concrete implementations:

- `GroundStation` — `(lat, lon, height, time) → zenith RA/Dec, visible region`
- `SpacecraftBoresight` — `(ephemeris(t), attitude(t)) → boresight RA/Dec, FOV region`

## Non-goals

- No ephemeris reading itself. We define `EphemerisSource`/`AttitudeSource`
  traits and let the caller bring their own (anise, SPICE, TLE, custom Kalman).
- No precision time-keeping. We accept `starfield::time::Time` and call its
  TT/TDB accessors. Leap seconds, EOPs, etc. are someone else's problem.
- No history/extrapolation. Each call gets a fresh `current_region(t)`. If
  callers want hysteresis (don't drop a cell that just left), the orchestrator
  in plan 5 handles it.

## API surface

```rust
// src/pointing.rs (new module)

pub trait PointingSource: Send {
    /// The sky region currently observable (or otherwise interesting) at
    /// time `t`. Used to drive `LiveIndex::set_region`.
    fn current_region(&self, t: Time) -> SkyRegion;

    /// Optional: BCRS observer state for refinement's apparent-place chain.
    /// If `None`, refinement falls back to no-observer mode (no parallax /
    /// aberration corrections).
    fn observer_state(&self, t: Time) -> Option<ObserverState> {
        None
    }
}
```

### Ephemeris + attitude trait surface

```rust
// src/pointing/sources.rs

pub trait EphemerisSource: Send {
    /// BCRS state vector: position (AU) and velocity (AU/day).
    fn state_at(&self, t: Time) -> Option<BcrsState>;
}

pub trait AttitudeSource: Send {
    /// Body-to-inertial quaternion at time `t`. Convention: this rotates
    /// the spacecraft body frame into the inertial (J2000 / ICRS) frame.
    fn quaternion_at(&self, t: Time) -> Option<UnitQuaternion<f64>>;
}

#[derive(Debug, Clone, Copy)]
pub struct BcrsState {
    pub position_au: [f64; 3],
    pub velocity_au_per_day: [f64; 3],
}
```

Trait split because real systems often plug different libraries for each:
ephemeris from anise/SPICE/TLE, attitude from a Kalman filter / sun-sensor
fusion / static initial guess.

### `GroundStation`

```rust
pub struct GroundStation {
    pub location: GeographicPosition,    // lat, lon, height above ellipsoid
    /// Stars below this altitude are excluded. Typical: 20-30°.
    pub min_altitude_deg: f64,
}

impl GroundStation {
    pub fn new(latitude_deg: f64, longitude_deg: f64, height_m: f64, min_altitude_deg: f64) -> Self;
}

impl PointingSource for GroundStation {
    fn current_region(&self, t: Time) -> SkyRegion {
        // 1. Compute zenith RA/Dec from (lat, lon, time) using starfield::toposlib.
        // 2. Region center = zenith.
        // 3. Region radius = 90° - min_altitude_deg.
        ...
    }

    fn observer_state(&self, t: Time) -> Option<ObserverState> {
        // BCRS state of the station: ITRS xyz → GCRS via Earth rotation
        // → BCRS via Earth ephemeris. Depends on issue #44 landing.
        ...
    }
}
```

### `SpacecraftBoresight`

```rust
pub struct SpacecraftBoresight<E: EphemerisSource, A: AttitudeSource> {
    pub ephemeris: E,
    pub attitude: A,
    /// Detector half-angle (radians). Region radius = this + fov_padding.
    pub detector_half_angle_rad: f64,
    /// Extra padding beyond the detector FOV (typically 5-10°).
    pub fov_padding_rad: f64,
    /// Body-frame unit vector that defines "boresight" (typically +Z).
    /// Default: [0.0, 0.0, 1.0].
    pub boresight_body: [f64; 3],
}

impl<E, A> PointingSource for SpacecraftBoresight<E, A>
where E: EphemerisSource, A: AttitudeSource {
    fn current_region(&self, t: Time) -> SkyRegion {
        let q = self.attitude.quaternion_at(t).unwrap_or_default();
        let boresight_inertial = q * Vector3::from(self.boresight_body);
        let (ra, dec) = xyz_to_radec(boresight_inertial.into());
        SkyRegion::from_radians(
            Equatorial::new(ra, dec),
            self.detector_half_angle_rad + self.fov_padding_rad,
        )
    }

    fn observer_state(&self, t: Time) -> Option<ObserverState> {
        let s = self.ephemeris.state_at(t)?;
        Some(ObserverState::Barycentric {
            position_au: s.position_au,
            velocity_au_per_day: s.velocity_au_per_day,
        })
    }
}
```

Notably: `SpacecraftBoresight` has zero starfield dependency — it's pure
quaternion → unit vector → spherical coords. Ephemeris and attitude are user-
supplied via the traits. This is the most reusable mode.

### Adapters (lives elsewhere — separate crate or zodiacal-realtime)

For convenience, ship optional thin adapters for common sources. These are
NOT part of the core pointing module — they bring transitive deps:

- `AniseEphemerisAdapter` — wraps an `anise::Almanac` for `EphemerisSource`.
- `Sgp4EphemerisAdapter` — wraps a TLE via `starfield::sgp4lib`.
- `StaticAttitude` — for testing; returns a fixed quaternion.
- `SimpleKalmanAttitude` — a placeholder for real attitude estimation.

Recommend: gate behind a `pointing-adapters` feature flag. Or — better —
move them to a separate `zodiacal-pointing-adapters` crate so the core
stays clean.

## Algorithm details

### `GroundStation::current_region`

1. Construct a `starfield::toposlib::Topos` from `(lat, lon, height)`.
2. Get `Topos::at(t).altaz_origin()` or equivalent — this is the zenith
   direction in the inertial (ICRS) frame at time `t`.
3. Convert zenith xyz → (RA, Dec).
4. Region radius = `(90.0 - min_altitude_deg).to_radians()`.

Verify the exact starfield API; the `toposlib::Topos` API is what we want
but the method names may differ. (Worst case, implement the LST-based zenith
calculation inline — it's ~10 lines.)

### `GroundStation::observer_state`

The full chain:

1. ITRS xyz of the station: `Geoid::geodetic_to_xyz(lat, lon, h)` (uses WGS84).
2. ITRS → GCRS: rotate by Earth's rotation matrix at time `t`. Requires:
   - Earth Rotation Angle (ERA) at `t` (via starfield::time::era_at).
   - Polar motion (xp, yp) at `t` (requires IERS EOP table — see issue #44).
   - TIO locator s' at `t` (negligible at mas level; can ignore).
3. GCRS → BCRS: add Earth's BCRS state at `t` (from a JPL ephemeris).

This is exactly the work tracked in issue #44 and explains why ground-mode
refinement is gated on it. Ground pointing without observer_state still works
(no refinement parallax/aberration), so we ship the `current_region` half
first and stub `observer_state` to `None` until #44 lands.

### `SpacecraftBoresight::current_region`

Pure quaternion math. No EOPs, no time scales, no ephemeris involvement
(except via the trait at `observer_state` time). The boresight is independent
of the spacecraft's position — it's purely an attitude question.

Edge case: `attitude.quaternion_at(t)` returns `None` (e.g., extrapolation
beyond a known interval). Pick a sensible default — recommend returning a
full-sky region (radius π) so the caller falls back to blind solving rather
than refusing entirely.

## Backwards compatibility

Brand-new module. No impact on existing code.

The `ObserverState` enum returned from `observer_state` already exists in the
refinement module as of PR #49. Lifting it to a more central location may
make sense — see "Open questions".

## Tests

Unit tests in `src/pointing/mod.rs::tests`:

- **`ground_station_zenith_at_known_time`**: configure a ground station at
  a known location, query at a known time, verify zenith RA/Dec matches an
  external reference (astropy). Tolerance: 1 arcmin (we don't need atomic
  precision for cell loading).
- **`ground_station_visible_region_radius`**: with `min_altitude_deg = 30`,
  verify region radius is 60° regardless of location/time.
- **`spacecraft_boresight_identity_quaternion`**: identity quaternion +
  body boresight = +Z → region centered at (RA=0, Dec=π/2). Wait, no — +Z
  in the inertial frame is the celestial pole, so dec = π/2. Yes.
- **`spacecraft_boresight_known_rotation`**: 90° rotation about ICRS x-axis
  → boresight points to dec=0, ra=π/2 (or similar — pick the convention).
  Verify with hand computation.
- **`spacecraft_boresight_observer_state_passthrough`**: mock ephemeris
  returns a known BCRS state; verify `observer_state` returns it unchanged.
- **`pointing_source_default_observer_state_is_none`**: a custom impl that
  doesn't override `observer_state` returns `None`.

Integration tests:

- **`ground_station_drives_live_index`**: `GroundStation` + `LiveIndex` over
  a synthetic source. Tick across several hours of simulated time, verify
  loaded cells track the rotating zenith.
- **`spacecraft_drives_live_index_with_quaternion_sweep`**: mock attitude
  source rotates the boresight in a known sweep, verify loaded cells follow.

## Effort estimate

| Step | Effort |
|---|---|
| `PointingSource` trait + module skeleton | 0.5 day |
| `GroundStation::current_region` | 1 day (depends on starfield API discovery) |
| `GroundStation::observer_state` | (deferred — gated on #44) |
| `EphemerisSource` + `AttitudeSource` traits | 0.5 day |
| `SpacecraftBoresight::current_region` + `observer_state` | 1 day |
| Unit tests (6) | 1 day |
| Integration tests (2) | 1 day |
| Optional adapters (Anise, SGP4, Static) | 1 day (out of scope for v1) |
| Docs + examples | 0.5 day |
| **Total v1 (no terrestrial observer)** | **~5-6 days** |

Add 1-2 days when issue #44 lands and we wire up `GroundStation::observer_state`.

## Dependencies

- **Issue #44** (terrestrial observer state) for `GroundStation::observer_state`.
  The `current_region` half is independent and can ship first.
- **Plan 1, 2, 3** are independent of plan 4 — pointing sources don't care how
  the index is loaded.
- Plan 5 (RealtimeSolver) consumes this trait.

## Open questions

- **Where does `ObserverState` live?** Currently in `src/refinement/types.rs`
  because that's where it was first needed. Now plan 4 wants it too. Two
  options:
  1. Re-export from `pointing` and keep the home in `refinement` (avoids
     breaking existing public API).
  2. Move it to a new top-level module like `src/observer.rs` and re-export
     from both `refinement` and `pointing`.
  Recommendation: **option 1** for v1 (less churn), revisit if a third
  consumer emerges.
- **Quaternion type.** The trait signature uses `nalgebra::UnitQuaternion<f64>`.
  Acceptable since `nalgebra` is already a dep (PR #49). Alternative: define
  a local `Quaternion { w, x, y, z }` to avoid the bound. Recommendation:
  use `nalgebra` — it's already there, has the math we'd otherwise reinvent.
- **Time accuracy.** `current_region` only needs to be accurate enough to
  load the right cells (~degree-scale tolerance). Sub-second time accuracy
  is overkill. But `observer_state` for refinement DOES need sub-second
  accuracy. Document this distinction so users don't over-spec their time
  inputs for the loading path.
- **Boresight body convention.** Pinning "+Z is boresight" may not match all
  spacecraft. Make it configurable (already in the API as `boresight_body`).
  Default to +Z but document that real flight software typically configures
  per-instrument.
