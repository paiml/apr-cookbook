//! Model bundling utilities for zero-dependency deployment.
//!
//! This module provides utilities for embedding ML models directly into
//! Rust binaries using `include_bytes!()`, enabling single-file deployment
//! without external dependencies.
//!
//! # Philosophy (Muda Elimination)
//!
//! Traditional ML deployment requires:
//! - Model files (shipped separately)
//! - Runtime dependencies (Python, CUDA)
//! - Container images (often gigabytes)
//!
//! With APR bundling:
//! - Single static binary
//! - Zero runtime dependencies
//! - Kilobytes, not gigabytes
//!
//! # Example
//!
//! ```ignore
//! use apr_cookbook::bundle::BundledModel;
//!
//! const MODEL_BYTES: &[u8] = include_bytes!("../models/sentiment.apr");
//!
//! fn main() -> apr_cookbook::Result<()> {
//!     let model = BundledModel::from_bytes(MODEL_BYTES)?;
//!     println!("Model: {}", model.name());
//!     Ok(())
//! }
//! ```

mod v1;
mod v2;

pub use self::v1::*;
pub use self::v2::*;
