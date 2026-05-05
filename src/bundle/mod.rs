//! `.zdcl.bundle` — multi-band, HEALPix-sharded index format.
//!
//! See `docs/bundle-format.md` for the full specification.
//!
//! This module currently provides the storage abstraction
//! (`accessor::SubfileAccessor`) used by the bundle reader, the
//! per-cell shard formats, and the manifest schema; the higher-level
//! `ZdclBundle` reader is added in follow-up PRs.

pub mod accessor;
pub mod gaia_shard;
pub mod layout;
pub mod manifest;
pub mod quad_shard;
