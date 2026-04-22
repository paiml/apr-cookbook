//! Recipe infrastructure for isolated, idempotent, and reproducible examples.
//!
//! This module provides the `RecipeContext` utility that ensures all recipes
//! follow the IIUR principles:
//!
//! - **Isolated**: Uses temp directories, no shared state
//! - **Idempotent**: Deterministic RNG seeded by recipe name
//! - **Useful**: Standardized reporting and metrics
//! - **Reproducible**: Cross-platform, CI-verified
//!
//! # Philosophy (Toyota Way)
//!
//! - **Jidoka**: Built-in quality via type-safe context
//! - **Muda**: Automatic cleanup eliminates resource waste
//! - **Heijunka**: Consistent recipe structure

mod context;
mod testdata;

pub use context::*;
pub use testdata::*;

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
