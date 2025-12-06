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

/// Hash a recipe name to a deterministic u64 seed.
///
/// Uses BLAKE3 for consistent cross-platform hashing.
#[must_use]
pub fn hash_name_to_seed(name: &str) -> u64 {
    let hash = blake3::hash(name.as_bytes());
    let bytes = hash.as_bytes();
    u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ])
}

/// Generate deterministic test data for a given seed.
///
/// Useful for creating reproducible test fixtures.
#[must_use]
pub fn generate_test_data(seed: u64, size: usize) -> Vec<f32> {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    (0..size).map(|_| rng.gen_range(-1.0..1.0)).collect()
}

/// Generate a deterministic model payload for testing.
///
/// Creates fake "model weights" that are reproducible.
#[must_use]
pub fn generate_model_payload(seed: u64, n_params: usize) -> Vec<u8> {
    use rand::Rng;
    let mut rng = StdRng::seed_from_u64(seed);
    let weights: Vec<f32> = (0..n_params)
        .map(|_| rng.gen_range(-1.0f32..1.0f32))
        .collect();

    // Serialize as raw f32 bytes
    weights.iter().flat_map(|f| f.to_le_bytes()).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recipe_context_creation() {
        let ctx = RecipeContext::new("test_recipe").unwrap();
        assert_eq!(ctx.name(), "test_recipe");
        assert!(ctx.temp_dir().exists());
    }

    #[test]
    fn test_recipe_context_path() {
        let ctx = RecipeContext::new("test_recipe").unwrap();
        let path = ctx.path("model.apr");
        assert!(path.starts_with(ctx.temp_dir()));
        assert!(path.ends_with("model.apr"));
    }

    #[test]
    fn test_deterministic_rng() {
        // Same recipe name should produce same RNG sequence
        let mut ctx1 = RecipeContext::new("deterministic_test").unwrap();
        let mut ctx2 = RecipeContext::new("deterministic_test").unwrap();

        use rand::Rng;
        let seq1: Vec<u64> = (0..10).map(|_| ctx1.rng().gen()).collect();
        let seq2: Vec<u64> = (0..10).map(|_| ctx2.rng().gen()).collect();

        assert_eq!(
            seq1, seq2,
            "Same recipe name should produce same RNG sequence"
        );
    }

    #[test]
    fn test_different_recipes_different_rng() {
        let mut ctx1 = RecipeContext::new("recipe_a").unwrap();
        let mut ctx2 = RecipeContext::new("recipe_b").unwrap();

        use rand::Rng;
        let val1: u64 = ctx1.rng().gen();
        let val2: u64 = ctx2.rng().gen();

        assert_ne!(
            val1, val2,
            "Different recipe names should produce different RNG"
        );
    }

    #[test]
    fn test_temp_dir_isolation() {
        let ctx1 = RecipeContext::new("isolation_test_1").unwrap();
        let ctx2 = RecipeContext::new("isolation_test_2").unwrap();

        assert_ne!(
            ctx1.temp_dir(),
            ctx2.temp_dir(),
            "Each context should have its own temp directory"
        );
    }

    #[test]
    fn test_metrics_recording() {
        let mut ctx = RecipeContext::new("metrics_test").unwrap();

        ctx.record_metric("byte_count", 1024);
        ctx.record_float_metric("throughput", 123.456);
        ctx.record_duration("inference_time", Duration::from_millis(42));
        ctx.record_string_metric("model_name", "test-model");

        match ctx.get_metric("byte_count") {
            Some(MetricValue::Int(v)) => assert_eq!(*v, 1024),
            _ => panic!("Expected Int metric"),
        }

        match ctx.get_metric("throughput") {
            Some(MetricValue::Float(v)) => assert!((v - 123.456).abs() < 0.001),
            _ => panic!("Expected Float metric"),
        }
    }

    #[test]
    fn test_hash_name_to_seed_deterministic() {
        let seed1 = hash_name_to_seed("my_recipe");
        let seed2 = hash_name_to_seed("my_recipe");
        assert_eq!(seed1, seed2);
    }

    #[test]
    fn test_hash_name_to_seed_different_names() {
        let seed1 = hash_name_to_seed("recipe_a");
        let seed2 = hash_name_to_seed("recipe_b");
        assert_ne!(seed1, seed2);
    }

    #[test]
    fn test_generate_test_data_deterministic() {
        let data1 = generate_test_data(42, 100);
        let data2 = generate_test_data(42, 100);
        assert_eq!(data1, data2);
    }

    #[test]
    fn test_generate_test_data_different_seeds() {
        let data1 = generate_test_data(42, 100);
        let data2 = generate_test_data(43, 100);
        assert_ne!(data1, data2);
    }

    #[test]
    fn test_generate_model_payload_deterministic() {
        let payload1 = generate_model_payload(42, 256);
        let payload2 = generate_model_payload(42, 256);
        assert_eq!(payload1, payload2);
    }

    #[test]
    fn test_generate_model_payload_size() {
        let payload = generate_model_payload(42, 256);
        // 256 f32 values * 4 bytes each = 1024 bytes
        assert_eq!(payload.len(), 256 * 4);
    }

    #[test]
    fn test_recipe_metadata_builder() {
        let metadata = RecipeMetadata::from_name("test")
            .with_category("bundling")
            .with_objective("Learn model embedding")
            .with_feature("encryption");

        assert_eq!(metadata.name, "test");
        assert_eq!(metadata.category, Some("bundling".to_string()));
        assert_eq!(
            metadata.objective,
            Some("Learn model embedding".to_string())
        );
        assert_eq!(metadata.features, vec!["encryption"]);
    }

    #[test]
    fn test_verify_idempotency() {
        let mut ctx = RecipeContext::new("idempotency_test").unwrap();

        let is_idempotent = ctx.verify_idempotency(|ctx| {
            use rand::Rng;
            ctx.rng().gen::<u64>()
        });

        assert!(is_idempotent, "Same RNG operations should be idempotent");
    }

    #[test]
    fn test_temp_dir_cleanup() {
        let path = {
            let ctx = RecipeContext::new("cleanup_test").unwrap();
            ctx.temp_dir().to_path_buf()
        };
        // After ctx is dropped, temp dir should be cleaned up
        assert!(
            !path.exists(),
            "Temp directory should be cleaned up on drop"
        );
    }

    #[test]
    fn test_elapsed_time() {
        let ctx = RecipeContext::new("elapsed_test").unwrap();
        std::thread::sleep(Duration::from_millis(10));
        let elapsed = ctx.elapsed();
        assert!(elapsed >= Duration::from_millis(10));
    }
}
