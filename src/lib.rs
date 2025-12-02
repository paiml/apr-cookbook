//! # APR Cookbook
//!
//! Idiomatic Rust examples demonstrating the power of tiny ML models
//! using the `.apr` format. Guided by Toyota Way principles:
//!
//! - **Muda** (waste elimination): Zero-dependency binaries
//! - **Jidoka** (built-in quality): Rust type system + strict validation
//! - **Genchi Genbutsu** (go and see): Edge/WASM deployment
//!
//! ## Example Categories
//!
//! - **Bundling**: Embed models into binaries with `include_bytes!()`
//! - **Conversion**: `SafeTensors` ↔ `.apr` ↔ GGUF format conversion
//! - **Acceleration**: SIMD/GPU with automatic fallback
//! - **CLI**: Production-ready command-line tools

pub mod aprender_integration;
pub mod bundle;
pub mod convert;
pub mod error;

pub use error::{CookbookError, Result};

// Re-export aprender types for convenience
pub use aprender::format::{ModelType, SaveOptions};

/// Re-exports for convenient access
pub mod prelude {
    pub use crate::aprender_integration::{
        load_model, load_model_from_bytes, save_model, AprModelInfo, SimpleModel,
    };
    pub use crate::bundle::{BundledModel, ModelBundle};
    pub use crate::convert::{AprConverter, ConversionFormat};
    pub use crate::error::{CookbookError, Result};
    pub use crate::{ModelType, SaveOptions};
}
