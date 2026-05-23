//! Recipe execution context types: `RecipeContext`, `RecipeMetadata`, `MetricValue`.

use super::testdata::hash_name_to_seed;
use crate::error::{CookbookError, Result};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::{Duration, Instant};
use tempfile::TempDir;

/// Context for recipe execution providing isolation and reproducibility.
///
/// Each recipe creates a `RecipeContext` which provides:
/// - Isolated temporary directory (auto-cleanup on drop)
/// - Deterministic RNG seeded by recipe name
/// - Timing and reporting utilities
///
/// # Example
///
/// ```
/// use apr_cookbook::recipe::RecipeContext;
///
/// fn main() -> apr_cookbook::Result<()> {
///     let mut ctx = RecipeContext::new("my_recipe")?;
///     let model_path = ctx.path("model.apr");
///     // ... do work in isolated temp directory
///     ctx.record_metric("size_bytes", 1024);
///     ctx.report()?;
///     Ok(())  // temp directory automatically cleaned up
/// }
/// ```
#[derive(Debug)]
pub struct RecipeContext {
    /// Recipe name for identification and seed generation
    name: String,
    /// Isolated temporary directory (auto-cleanup on drop)
    temp_dir: TempDir,
    /// Deterministic RNG seeded by recipe name hash
    rng: StdRng,
    /// Start time for duration tracking
    start_time: Instant,
    /// Collected metrics for reporting
    metrics: HashMap<String, MetricValue>,
    /// Recipe metadata
    metadata: RecipeMetadata,
}

/// Metadata about a recipe.
#[derive(Debug, Clone, Default)]
pub struct RecipeMetadata {
    /// Recipe name
    pub name: String,
    /// Category (e.g., "bundling", "conversion")
    pub category: Option<String>,
    /// Learning objective
    pub objective: Option<String>,
    /// Required features
    pub features: Vec<String>,
}

/// A metric value that can be recorded.
#[derive(Debug, Clone)]
pub enum MetricValue {
    /// Integer metric (e.g., byte count)
    Int(i64),
    /// Float metric (e.g., throughput)
    Float(f64),
    /// Duration metric
    Duration(Duration),
    /// String metric
    String(String),
}

impl RecipeContext {
    /// Create a new recipe context with isolated environment.
    ///
    /// The RNG is seeded deterministically from the recipe name,
    /// ensuring reproducible results across runs.
    ///
    /// # Errors
    ///
    /// Returns an error if the temporary directory cannot be created.
    pub fn new(name: &str) -> Result<Self> {
        let seed = hash_name_to_seed(name);
        let temp_dir = tempfile::tempdir().map_err(CookbookError::from)?;

        Ok(Self {
            name: name.to_string(),
            temp_dir,
            rng: StdRng::seed_from_u64(seed),
            start_time: Instant::now(),
            metrics: HashMap::new(),
            metadata: RecipeMetadata {
                name: name.to_string(),
                ..Default::default()
            },
        })
    }

    /// Create a context with custom metadata.
    ///
    /// # Errors
    ///
    /// Returns an error if the temporary directory cannot be created.
    pub fn with_metadata(name: &str, metadata: RecipeMetadata) -> Result<Self> {
        let mut ctx = Self::new(name)?;
        ctx.metadata = metadata;
        Ok(ctx)
    }

    /// Get a path within the isolated temp directory.
    ///
    /// All file operations should use paths from this method to ensure
    /// isolation and automatic cleanup.
    #[must_use]
    pub fn path(&self, filename: &str) -> PathBuf {
        self.temp_dir.path().join(filename)
    }

    /// Get the temp directory path.
    #[must_use]
    pub fn temp_dir(&self) -> &std::path::Path {
        self.temp_dir.path()
    }

    /// Get mutable access to the deterministic RNG.
    ///
    /// This RNG is seeded from the recipe name, so the same recipe
    /// will always produce the same sequence of random numbers.
    #[must_use]
    pub fn rng(&mut self) -> &mut StdRng {
        &mut self.rng
    }

    /// Get the recipe name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Get the recipe metadata.
    #[must_use]
    pub fn metadata(&self) -> &RecipeMetadata {
        &self.metadata
    }

    /// Get elapsed time since context creation.
    #[must_use]
    pub fn elapsed(&self) -> Duration {
        self.start_time.elapsed()
    }

    /// Record an integer metric.
    pub fn record_metric(&mut self, name: &str, value: i64) {
        self.metrics
            .insert(name.to_string(), MetricValue::Int(value));
    }

    /// Record a float metric.
    pub fn record_float_metric(&mut self, name: &str, value: f64) {
        self.metrics
            .insert(name.to_string(), MetricValue::Float(value));
    }

    /// Record a duration metric.
    pub fn record_duration(&mut self, name: &str, duration: Duration) {
        self.metrics
            .insert(name.to_string(), MetricValue::Duration(duration));
    }

    /// Record a string metric.
    pub fn record_string_metric(&mut self, name: &str, value: impl Into<String>) {
        self.metrics
            .insert(name.to_string(), MetricValue::String(value.into()));
    }

    /// Get a recorded metric.
    #[must_use]
    pub fn get_metric(&self, name: &str) -> Option<&MetricValue> {
        self.metrics.get(name)
    }

    /// Print a standardized report of recipe execution.
    ///
    /// # Errors
    ///
    /// Returns an error if writing to stdout fails (rare).
    pub fn report(&self) -> Result<()> {
        println!("=== Recipe: {} ===", self.name);
        println!("Duration: {:.2}ms", self.elapsed().as_secs_f64() * 1000.0);

        if !self.metrics.is_empty() {
            println!("Metrics:");
            for (name, value) in &self.metrics {
                match value {
                    MetricValue::Int(v) => println!("  {}: {}", name, v),
                    MetricValue::Float(v) => println!("  {}: {:.4}", name, v),
                    MetricValue::Duration(d) => {
                        println!("  {}: {:.2}ms", name, d.as_secs_f64() * 1000.0);
                    }
                    MetricValue::String(s) => println!("  {}: {}", name, s),
                }
            }
        }

        Ok(())
    }

    /// Verify that running the recipe twice produces the same output.
    ///
    /// This is a test helper for verifying idempotency.
    #[must_use]
    pub fn verify_idempotency<F, T>(&mut self, f: F) -> bool
    where
        F: Fn(&mut Self) -> T,
        T: PartialEq,
    {
        // Reset RNG to initial state
        let seed = hash_name_to_seed(&self.name);
        self.rng = StdRng::seed_from_u64(seed);
        let result1 = f(self);

        // Reset again and run
        self.rng = StdRng::seed_from_u64(seed);
        let result2 = f(self);

        result1 == result2
    }
}

impl RecipeMetadata {
    /// Create metadata from just a name.
    #[must_use]
    pub fn from_name(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            ..Default::default()
        }
    }

    /// Set the category.
    #[must_use]
    pub fn with_category(mut self, category: impl Into<String>) -> Self {
        self.category = Some(category.into());
        self
    }

    /// Set the learning objective.
    #[must_use]
    pub fn with_objective(mut self, objective: impl Into<String>) -> Self {
        self.objective = Some(objective.into());
        self
    }

    /// Add a required feature.
    #[must_use]
    pub fn with_feature(mut self, feature: impl Into<String>) -> Self {
        self.features.push(feature.into());
        self
    }
}
