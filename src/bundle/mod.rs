//! `.zdcl.bundle` — multi-band, HEALPix-sharded index format.
//!
//! See `docs/bundle-format.md` for the full specification.
//!
//! This module currently provides the storage abstraction
//! (`accessor::SubfileAccessor`) used by the bundle reader; the per-cell
//! shard formats and the higher-level `ZdclBundle` reader are added in
//! follow-up PRs.

pub mod accessor;
