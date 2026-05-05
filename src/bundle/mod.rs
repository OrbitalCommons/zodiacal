//! `.zdcl.bundle` — multi-band, HEALPix-sharded index format.
//!
//! See `docs/bundle-format.md` for the full specification.
//!
//! This module currently provides the storage abstraction
//! (`accessor::SubfileAccessor`) used by the bundle reader and the
//! per-cell shard formats; the higher-level `ZdclBundle` reader is
//! added in follow-up PRs.

pub mod accessor;
pub mod gaia_shard;
pub mod quad_shard;
