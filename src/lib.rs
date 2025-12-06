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
//! - **Model Creation**: Create `.apr` models from scratch
//! - **Bundling**: Embed models into binaries with `include_bytes!()`
//! - **Conversion**: `SafeTensors` ↔ `.apr` ↔ GGUF format conversion
//! - **Continuous Training**: Update models with streaming data
//! - **Registry**: Manage models with Pacha integration
//! - **API**: Model serving via Realizar
//! - **Serverless**: AWS Lambda deployment
//! - **WASM**: Browser deployment via Presentar
//! - **Acceleration**: SIMD/GPU with automatic fallback
//! - **Distillation**: Model compression from HuggingFace
//! - **CLI**: Production-ready command-line tools
//!
//! ## IIUR Principles
//!
//! All recipes follow the IIUR principles:
//! - **Isolated**: No shared state, temp directories with auto-cleanup
//! - **Idempotent**: Deterministic RNG, same input produces same output
//! - **Useful**: Solves real production problems
//! - **Reproducible**: Cross-platform, CI-verified

pub mod aprender_integration;
pub mod bundle;
pub mod convert;
pub mod error;
pub mod recipe;

pub use error::{CookbookError, Result};

// Re-export aprender types for convenience
pub use aprender::format::{ModelType, SaveOptions};

/// Re-exports for convenient access
pub mod prelude {
    pub use crate::aprender_integration::{
        load_model, load_model_from_bytes, save_model, AprModelInfo, SimpleModel,
    };
    pub use crate::bundle::{BundledModel, ModelBundle, ModelMetadata};
    pub use crate::convert::{
        AprConverter, ConversionFormat, ConversionMetadata, DataType, TensorData,
    };
    pub use crate::error::{CookbookError, Result};
    pub use crate::recipe::{
        generate_model_payload, generate_test_data, hash_name_to_seed, MetricValue, RecipeContext,
        RecipeMetadata,
    };
    pub use crate::{ModelType, SaveOptions};
}
