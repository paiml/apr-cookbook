//! Integration with aprender format module.
//!
//! This module provides a thin wrapper around aprender's format
//! functionality, adapted for cookbook examples.

use crate::error::{CookbookError, Result};
use aprender::format::{self, ModelInfo, ModelType, SaveOptions};
use serde::{de::DeserializeOwned, Serialize};
use std::path::Path;

/// Wrapper for aprender's model inspection functionality.
#[derive(Debug, Clone)]
pub struct AprModelInfo {
    /// Raw model info from aprender
    inner: ModelInfo,
}

impl AprModelInfo {
    /// Inspect an APR file without loading the full model.
    ///
    /// # Errors
    ///
    /// Returns an error if the file cannot be read or parsed.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self> {
        let info = format::inspect(path).map_err(|e| CookbookError::Aprender(e.to_string()))?;
        Ok(Self { inner: info })
    }

    /// Inspect APR data from bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if the bytes cannot be parsed.
    pub fn from_bytes(data: &[u8]) -> Result<Self> {
        let info =
            format::inspect_bytes(data).map_err(|e| CookbookError::Aprender(e.to_string()))?;
        Ok(Self { inner: info })
    }

    /// Get the model type.
    #[must_use]
    pub fn model_type(&self) -> ModelType {
        self.inner.model_type
    }

    /// Get the format version.
    #[must_use]
    pub fn version(&self) -> (u8, u8) {
        self.inner.format_version
    }

    /// Check if the model is compressed.
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.inner.payload_size < self.inner.uncompressed_size
    }

    /// Check if the model is encrypted.
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.inner.encrypted
    }

    /// Check if the model is signed.
    #[must_use]
    pub fn is_signed(&self) -> bool {
        self.inner.signed
    }

    /// Get the model name if available.
    #[must_use]
    pub fn name(&self) -> Option<&str> {
        self.inner.metadata.model_name.as_deref()
    }

    /// Get the model description if available.
    #[must_use]
    pub fn description(&self) -> Option<&str> {
        self.inner.metadata.description.as_deref()
    }

    /// Get the payload size.
    #[must_use]
    pub fn payload_size(&self) -> usize {
        self.inner.payload_size
    }

    /// Get access to the underlying aprender ModelInfo.
    #[must_use]
    pub fn inner(&self) -> &ModelInfo {
        &self.inner
    }
}

/// Save a model to APR format.
///
/// # Errors
///
/// Returns an error if serialization or file writing fails.
pub fn save_model<M: Serialize>(
    model: &M,
    model_type: ModelType,
    path: impl AsRef<Path>,
    options: SaveOptions,
) -> Result<()> {
    format::save(model, model_type, path, options)
        .map_err(|e| CookbookError::Aprender(e.to_string()))
}

/// Load a model from APR format.
///
/// # Errors
///
/// Returns an error if the file cannot be read or deserialized.
pub fn load_model<M: DeserializeOwned>(
    path: impl AsRef<Path>,
    expected_type: ModelType,
) -> Result<M> {
    format::load(path, expected_type).map_err(|e| CookbookError::Aprender(e.to_string()))
}

/// Load a model from APR bytes.
///
/// # Errors
///
/// Returns an error if deserialization fails.
pub fn load_model_from_bytes<M: DeserializeOwned>(
    data: &[u8],
    expected_type: ModelType,
) -> Result<M> {
    format::load_from_bytes(data, expected_type).map_err(|e| CookbookError::Aprender(e.to_string()))
}

/// A simple serializable model for testing.
#[derive(Debug, Clone, Serialize, serde::Deserialize, PartialEq)]
pub struct SimpleModel {
    /// Model weights as a flat vector
    pub weights: Vec<f32>,
    /// Input dimension
    pub input_dim: usize,
    /// Output dimension
    pub output_dim: usize,
}

impl SimpleModel {
    /// Create a new simple model.
    #[must_use]
    pub fn new(input_dim: usize, output_dim: usize) -> Self {
        Self {
            weights: vec![0.0; input_dim * output_dim],
            input_dim,
            output_dim,
        }
    }

    /// Create a model with random weights.
    #[must_use]
    pub fn random(input_dim: usize, output_dim: usize) -> Self {
        // Simple PRNG for reproducibility
        let mut weights = vec![0.0; input_dim * output_dim];
        let mut seed: u64 = 42;
        for w in &mut weights {
            seed = seed.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            *w = ((seed >> 33) as f32) / (u32::MAX as f32) - 0.5;
        }
        Self {
            weights,
            input_dim,
            output_dim,
        }
    }

    /// Get the number of parameters.
    #[must_use]
    pub fn num_params(&self) -> usize {
        self.weights.len()
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;
    use aprender::format::Compression;
    use tempfile::tempdir;

    #[test]
    fn test_simple_model_creation() {
        let model = SimpleModel::new(10, 5);
        assert_eq!(model.input_dim, 10);
        assert_eq!(model.output_dim, 5);
        assert_eq!(model.num_params(), 50);
        assert!(model.weights.iter().all(|&w| w == 0.0));
    }

    #[test]
    fn test_simple_model_random() {
        let model = SimpleModel::random(10, 5);
        assert_eq!(model.num_params(), 50);
        // Should have non-zero values
        assert!(model.weights.iter().any(|&w| w != 0.0));
    }

    #[test]
    fn test_save_and_load_roundtrip() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test_model.apr");

        let original = SimpleModel::random(100, 10);

        // Save
        let options = SaveOptions::default().with_name("test-model");
        save_model(&original, ModelType::Custom, &path, options).unwrap();

        // Load
        let loaded: SimpleModel = load_model(&path, ModelType::Custom).unwrap();

        assert_eq!(original, loaded);
    }

    #[test]
    fn test_save_and_load_with_compression() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("compressed_model.apr");

        let original = SimpleModel::random(1000, 100);

        // Save with compression
        let options = SaveOptions::default()
            .with_name("compressed-model")
            .with_compression(Compression::ZstdDefault);
        save_model(&original, ModelType::Custom, &path, options).unwrap();

        // Inspect to verify compression
        let info = AprModelInfo::from_path(&path).unwrap();
        assert!(info.is_compressed());

        // Load and verify
        let loaded: SimpleModel = load_model(&path, ModelType::Custom).unwrap();
        assert_eq!(original, loaded);
    }

    #[test]
    fn test_inspect_model_info() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("inspect_test.apr");

        let model = SimpleModel::random(50, 10);
        let options = SaveOptions::default()
            .with_name("inspection-test")
            .with_description("A test model for inspection");
        save_model(&model, ModelType::Custom, &path, options).unwrap();

        let info = AprModelInfo::from_path(&path).unwrap();
        assert_eq!(info.model_type(), ModelType::Custom);
        assert_eq!(info.name(), Some("inspection-test"));
        assert_eq!(info.description(), Some("A test model for inspection"));
        assert!(!info.is_encrypted());
        assert!(!info.is_signed());
    }

    #[test]
    fn test_load_from_bytes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bytes_test.apr");

        let original = SimpleModel::random(20, 5);
        let options = SaveOptions::default();
        save_model(&original, ModelType::Custom, &path, options).unwrap();

        // Read as bytes
        let bytes = std::fs::read(&path).unwrap();

        // Load from bytes
        let loaded: SimpleModel = load_model_from_bytes(&bytes, ModelType::Custom).unwrap();
        assert_eq!(original, loaded);
    }

    #[test]
    fn test_inspect_bytes() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("bytes_inspect.apr");

        let model = SimpleModel::new(10, 10);
        let options = SaveOptions::default().with_name("bytes-model");
        save_model(&model, ModelType::Custom, &path, options).unwrap();

        let bytes = std::fs::read(&path).unwrap();
        let info = AprModelInfo::from_bytes(&bytes).unwrap();

        assert_eq!(info.name(), Some("bytes-model"));
        assert_eq!(info.version(), (1, 0));
    }

    #[test]
    fn test_model_type_preserved() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("type_test.apr");

        let model = SimpleModel::new(5, 5);
        let options = SaveOptions::default();

        // Save as LinearRegression type
        save_model(&model, ModelType::LinearRegression, &path, options).unwrap();

        // Inspect should show LinearRegression
        let info = AprModelInfo::from_path(&path).unwrap();
        assert_eq!(info.model_type(), ModelType::LinearRegression);
    }
}
