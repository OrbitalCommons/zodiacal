//! `RealtimeSolver` — orchestrator that ties [`LiveIndex`] +
//! [`PointingSource`] + the existing solver/refinement pipelines into
//! one realtime-friendly `tick()` / `solve()` interface.
//!
//! Plan 5 of the deployment-mode roadmap (`plans/05-realtime-solver.md`).
//! This is the front door for the two realtime modes (telescope and
//! star tracker). The server mode (mode 1) doesn't need this — it's a
//! thin convenience over the existing [`crate::index::Index`] /
//! [`crate::solver::solve`] API.

use std::io;
use std::time::{Duration, Instant};

use starfield::time::Time;

use crate::extraction::DetectedSource;
use crate::geom::sphere::angular_distance;
use crate::geom::sphere::radec_to_xyz;
use crate::index::{EnsureReport, IndexSource, LiveIndex};
use crate::pointing::PointingSource;
use crate::solver::{SkyRegion, Solution, SolverConfig, solve};

/// When the orchestrator considers re-running [`PointingSource::current_region`]
/// and re-syncing the [`LiveIndex`] cell membership.
#[derive(Debug, Clone, Copy)]
pub enum RefreshPolicy {
    /// Refresh on every solve. Simplest; fine for slow cadence (< 1 Hz).
    EveryTick,
    /// Refresh if the new region center is more than `angular_threshold_rad`
    /// away from the previous refresh's center, OR if more than `max_age`
    /// has elapsed since the last refresh.
    OnPointingDelta {
        angular_threshold_rad: f64,
        max_age: Duration,
    },
    /// Refresh only at fixed intervals.
    OnInterval { period: Duration },
}

/// Snapshot of the orchestrator's last refresh — used by
/// `RefreshPolicy::OnPointingDelta` to detect when the pointing has
/// drifted enough to warrant a re-load.
#[derive(Debug, Clone)]
pub struct RefreshSnapshot {
    pub when: Instant,
    pub region: SkyRegion,
    pub cells_loaded: usize,
    pub stars_loaded: usize,
}

#[derive(Debug)]
pub struct RealtimeOutput {
    pub solution: Option<Solution>,
    pub refresh: Option<EnsureReport>,
    pub solve_elapsed: Duration,
    pub refresh_elapsed: Duration,
    pub build_generation: u64,
}

/// Plan-5 orchestrator. Owns the `LiveIndex` and the `PointingSource`;
/// each `solve()` call ticks the index, runs the solver against the
/// current loaded set, and returns diagnostics.
///
/// Refinement integration is deliberately **not** wired in v1 — the
/// existing `refine_solution` API takes an `Index` (concrete) plus a
/// `RefinementCatalog` and an `ObservationContext`, and the catalog
/// piece is the part that's still in flight (issue #45 sidecar
/// integration). When refinement is wired, it'll add another stage
/// after the existing tweak path; the `RealtimeOutput` shape supports
/// it via additional fields.
pub struct RealtimeSolver<S: IndexSource, P: PointingSource> {
    live: LiveIndex<S>,
    pointing: P,
    solver_config: SolverConfig,
    refresh_policy: RefreshPolicy,
    last_refresh: Option<RefreshSnapshot>,
    region_padding_rad: f64,
}

impl<S: IndexSource, P: PointingSource> RealtimeSolver<S, P> {
    /// Construct with default config: `EveryTick` refresh, no padding,
    /// `SolverConfig::default()`. Use the methods below to override.
    pub fn new(source: S, pointing: P) -> Self {
        Self {
            live: LiveIndex::open(source),
            pointing,
            solver_config: SolverConfig::default(),
            refresh_policy: RefreshPolicy::EveryTick,
            last_refresh: None,
            region_padding_rad: 0.0,
        }
    }

    pub fn with_solver_config(mut self, c: SolverConfig) -> Self {
        self.solver_config = c;
        self
    }

    pub fn with_refresh_policy(mut self, p: RefreshPolicy) -> Self {
        self.refresh_policy = p;
        self
    }

    pub fn with_region_padding_rad(mut self, padding: f64) -> Self {
        self.region_padding_rad = padding.max(0.0);
        self
    }

    pub fn loaded_cell_count(&self) -> usize {
        self.live.loaded_cell_count()
    }

    pub fn loaded_star_count(&self) -> usize {
        self.live.loaded_star_count()
    }

    pub fn last_refresh(&self) -> Option<&RefreshSnapshot> {
        self.last_refresh.as_ref()
    }

    pub fn live_index(&self) -> &LiveIndex<S> {
        &self.live
    }

    pub fn pointing(&self) -> &P {
        &self.pointing
    }

    /// Refresh loaded cells per the configured `RefreshPolicy`. Returns
    /// `Some(report)` if a refresh actually ran, `None` if the policy
    /// decided to skip.
    pub fn tick(&mut self, t: &Time) -> io::Result<Option<EnsureReport>> {
        let candidate = self.candidate_region(t);
        if !self.policy_should_refresh(&candidate) {
            return Ok(None);
        }
        let report = self.live.set_region(&candidate)?;
        self.last_refresh = Some(RefreshSnapshot {
            when: Instant::now(),
            region: candidate,
            cells_loaded: self.live.loaded_cell_count(),
            stars_loaded: self.live.loaded_star_count(),
        });
        Ok(Some(report))
    }

    fn candidate_region(&self, t: &Time) -> SkyRegion {
        let raw = self.pointing.current_region(t);
        if self.region_padding_rad > 0.0 {
            SkyRegion::from_radians(raw.center, raw.radius_rad + self.region_padding_rad)
        } else {
            raw
        }
    }

    fn policy_should_refresh(&self, candidate: &SkyRegion) -> bool {
        match (self.refresh_policy, &self.last_refresh) {
            (RefreshPolicy::EveryTick, _) => true,
            (_, None) => true,
            (
                RefreshPolicy::OnPointingDelta {
                    angular_threshold_rad,
                    max_age,
                },
                Some(prev),
            ) => {
                if prev.when.elapsed() >= max_age {
                    return true;
                }
                let prev_xyz = radec_to_xyz(prev.region.center.ra, prev.region.center.dec);
                let new_xyz = radec_to_xyz(candidate.center.ra, candidate.center.dec);
                angular_distance(prev_xyz, new_xyz) >= angular_threshold_rad
            }
            (RefreshPolicy::OnInterval { period }, Some(prev)) => prev.when.elapsed() >= period,
        }
    }

    /// Run a full solve at time `t`. Calls `tick(t)` first, then solves
    /// against the current loaded set. Returns the solution + diagnostics.
    pub fn solve(
        &mut self,
        sources: &[DetectedSource],
        image_size: (f64, f64),
        t: &Time,
    ) -> io::Result<RealtimeOutput> {
        let refresh_start = Instant::now();
        let refresh = self.tick(t)?;
        let refresh_elapsed = refresh_start.elapsed();

        let snapshot = self.live.as_index();
        let solve_start = Instant::now();
        let (solution, _stats) = solve(sources, &[&snapshot], image_size, &self.solver_config);
        let solve_elapsed = solve_start.elapsed();
        Ok(RealtimeOutput {
            solution,
            refresh,
            solve_elapsed,
            refresh_elapsed,
            build_generation: self.live.build_generation(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::index::{HealpixCell, IndexFragment, IndexMetadata};
    use crate::pointing::StaticRegion;
    use crate::quads::{Code, DIMQUADS};
    use crate::{extraction::DetectedSource, geom::tan::TanWcs};
    use crate::{
        index::IndexSource,
        index::builder::{IndexBuilderConfig, build_index},
        verify::VerifyConfig,
    };
    use starfield::Equatorial;
    use starfield::time::Timescale;
    use std::sync::Mutex;

    fn time_for_test() -> Time {
        Timescale::default().tt_jd(2_460_000.5, None)
    }

    /// Mock IndexSource that wraps a flat catalog as a single virtual
    /// cell; reuses the existing build_index pipeline so the solver
    /// actually has matchable quads.
    struct ScenarioSource {
        index: crate::index::Index,
        load_count: Mutex<usize>,
    }
    impl IndexSource for ScenarioSource {
        fn cells_intersecting(&self, _region: &SkyRegion) -> Vec<HealpixCell> {
            vec![HealpixCell { depth: 0, id: 0 }]
        }
        fn load_cells(&self, _cells: &[HealpixCell]) -> io::Result<IndexFragment> {
            *self.load_count.lock().unwrap() += 1;
            // Recompute codes from the index's quads.
            let star_xyz: Vec<[f64; 3]> = self
                .index
                .stars
                .iter()
                .map(|s| radec_to_xyz(s.ra, s.dec))
                .collect();
            let mut codes: Vec<Code> = Vec::with_capacity(self.index.quads.len());
            for q in &self.index.quads {
                let xyz: [_; DIMQUADS] = std::array::from_fn(|i| star_xyz[q.star_ids[i]]);
                let (code, _, _) = crate::quads::compute_canonical_code(&xyz, q.star_ids);
                codes.push(code);
            }
            Ok(IndexFragment {
                stars: self.index.stars.clone(),
                quads: self.index.quads.clone(),
                codes,
                scale_lower: self.index.scale_lower,
                scale_upper: self.index.scale_upper,
                metadata: None,
            })
        }
        fn cell_depth(&self) -> u8 {
            0
        }
        fn metadata(&self) -> Option<&IndexMetadata> {
            None
        }
        fn star_count(&self) -> usize {
            self.index.stars.len()
        }
        fn quad_count(&self) -> usize {
            self.index.quads.len()
        }
        fn scale_range(&self) -> (f64, f64) {
            (self.index.scale_lower, self.index.scale_upper)
        }
    }

    fn make_synthetic_scenario() -> (Vec<DetectedSource>, ScenarioSource, TanWcs) {
        let image_size = (512.0, 512.0);
        let pixel_scale_arcsec: f64 = 2.0;
        let scale_rad = (pixel_scale_arcsec / 3600.0).to_radians();
        let truth = TanWcs {
            crval: [1.0, 0.5],
            crpix: [256.0, 256.0],
            cd: [[scale_rad, 0.0], [0.0, scale_rad]],
            image_size: [image_size.0, image_size.1],
        };
        let mut state: u64 = 314_159_265;
        let mut rng = || -> f64 {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state as f64) / (u64::MAX as f64)
        };
        let mut catalog = Vec::new();
        let mut sources = Vec::new();
        for i in 0..25 {
            let px = 30.0 + rng() * 452.0;
            let py = 30.0 + rng() * 452.0;
            let (ra, dec) = truth.pixel_to_radec(px, py);
            catalog.push((i as u64, ra, dec, i as f64));
            sources.push(DetectedSource {
                x: px,
                y: py,
                flux: 1000.0 - i as f64 * 10.0,
            });
        }
        let field_diag = (image_size.0 * image_size.0 + image_size.1 * image_size.1).sqrt();
        let max_angle = field_diag * scale_rad;
        let cfg = IndexBuilderConfig {
            scale_lower: scale_rad * 10.0,
            scale_upper: max_angle,
            max_stars: 25,
            max_quads: 50_000,
        };
        let index = build_index(&catalog, &cfg);
        (
            sources,
            ScenarioSource {
                index,
                load_count: Mutex::new(0),
            },
            truth,
        )
    }

    #[test]
    fn tick_no_refresh_when_policy_skips() {
        let (_sources, source, truth) = make_synthetic_scenario();
        let region = SkyRegion::from_radians(Equatorial::new(truth.crval[0], truth.crval[1]), 0.05);
        let pointing = StaticRegion(region);
        let mut rt = RealtimeSolver::new(source, pointing).with_refresh_policy(
            RefreshPolicy::OnPointingDelta {
                angular_threshold_rad: 0.5,
                max_age: Duration::from_secs(3600),
            },
        );
        let t = time_for_test();
        let r1 = rt.tick(&t).unwrap();
        // First tick always refreshes (no prior snapshot).
        assert!(r1.is_some());
        let r2 = rt.tick(&t).unwrap();
        // Same pointing, recent snapshot → policy skips.
        assert!(r2.is_none());
    }

    #[test]
    fn tick_every_tick_always_refreshes() {
        let (_sources, source, truth) = make_synthetic_scenario();
        let region = SkyRegion::from_radians(Equatorial::new(truth.crval[0], truth.crval[1]), 0.05);
        let pointing = StaticRegion(region);
        let mut rt =
            RealtimeSolver::new(source, pointing).with_refresh_policy(RefreshPolicy::EveryTick);
        let t = time_for_test();
        assert!(rt.tick(&t).unwrap().is_some());
        assert!(rt.tick(&t).unwrap().is_some());
        assert!(rt.tick(&t).unwrap().is_some());
    }

    #[test]
    fn solve_recovers_known_wcs_through_orchestrator() {
        let (sources, source, truth) = make_synthetic_scenario();
        let region = SkyRegion::from_radians(Equatorial::new(truth.crval[0], truth.crval[1]), 0.05);
        let pointing = StaticRegion(region);
        let solver_cfg = SolverConfig {
            scale_range: None,
            max_field_stars: 25,
            code_tolerance: 0.002,
            verify: VerifyConfig {
                match_radius_pix: 3.0,
                log_odds_accept: 10.0,
                min_matches: 3,
                ..VerifyConfig::default()
            },
            ..SolverConfig::default()
        };
        let mut rt = RealtimeSolver::new(source, pointing).with_solver_config(solver_cfg);
        let t = time_for_test();
        let out = rt.solve(&sources, (512.0, 512.0), &t).unwrap();
        let solution = out.solution.expect("orchestrator should solve");
        // Image-center separation < 30".
        let (solved_ra, solved_dec) = solution.wcs.field_center();
        let (truth_ra, truth_dec) = truth.field_center();
        let arcsec = std::f64::consts::PI / (180.0 * 3600.0);
        let dra = (solved_ra - truth_ra).abs() * truth_dec.cos();
        let ddec = (solved_dec - truth_dec).abs();
        let sep = ((dra * dra + ddec * ddec).sqrt()) / arcsec;
        assert!(sep < 30.0, "image-center sep {sep:.2} arcsec");
        // build_generation bumped from initial load.
        assert!(out.build_generation >= 1);
    }

    #[test]
    fn solve_returns_none_when_no_match() {
        let (_sources, source, truth) = make_synthetic_scenario();
        let region = SkyRegion::from_radians(Equatorial::new(truth.crval[0], truth.crval[1]), 0.05);
        let pointing = StaticRegion(region);
        let mut rt = RealtimeSolver::new(source, pointing);
        // Empty source list — solver should bail without crashing.
        let t = time_for_test();
        let out = rt.solve(&[], (512.0, 512.0), &t).unwrap();
        assert!(out.solution.is_none());
    }

    #[test]
    fn region_padding_inflates_loaded_set() {
        let (_sources, source, truth) = make_synthetic_scenario();
        let region = SkyRegion::from_radians(Equatorial::new(truth.crval[0], truth.crval[1]), 0.05);
        let pointing = StaticRegion(region.clone());
        let mut rt = RealtimeSolver::new(source, pointing).with_region_padding_rad(0.1);
        let t = time_for_test();
        rt.tick(&t).unwrap();
        let snap = rt.last_refresh().unwrap();
        // Padded region radius = 0.05 + 0.1 = 0.15.
        assert!((snap.region.radius_rad - 0.15).abs() < 1e-12);
    }

    #[test]
    fn first_tick_always_refreshes_regardless_of_policy() {
        // OnInterval policy with a very long period; first tick should
        // still refresh because last_refresh is None.
        let (_sources, source, truth) = make_synthetic_scenario();
        let region = SkyRegion::from_radians(Equatorial::new(truth.crval[0], truth.crval[1]), 0.05);
        let pointing = StaticRegion(region);
        let mut rt =
            RealtimeSolver::new(source, pointing).with_refresh_policy(RefreshPolicy::OnInterval {
                period: Duration::from_secs(3600),
            });
        let t = time_for_test();
        assert!(rt.tick(&t).unwrap().is_some());
    }
}
