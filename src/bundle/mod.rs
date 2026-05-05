//! `.zdcl.bundle` — multi-band, HEALPix-sharded index format.
//!
//! See `docs/bundle-format.md` for the full specification.
//!
//! This module provides the storage abstraction
//! (`accessor::SubfileAccessor`) used by the bundle reader, the
//! per-cell shard formats, the manifest schema, the tidy phase that
//! finalizes a work directory into a folder/zip bundle, and the
//! consumption-side [`reader::ZdclBundle`].

pub mod accessor;
pub mod gaia_shard;
pub mod layout;
pub mod manifest;
pub mod quad_shard;
pub mod reader;
pub mod tidy;

pub use reader::{MultiBandFragment, VerifyError, VerifyErrorKind, VerifyReport, ZdclBundle};
