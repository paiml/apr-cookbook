//! Error types for APR Cookbook operations.
//!
//! Following Jidoka principle: errors are explicit and actionable.

use std::path::PathBuf;
use thiserror::Error;

/// Result type alias for cookbook operations.
pub type Result<T> = std::result::Result<T, CookbookError>;

/// Errors that can occur during cookbook operations.
#[derive(Debug, Error)]
pub enum CookbookError {
    /// Model file not found at specified path.
    #[error("model not found: {path}")]
    ModelNotFound { path: PathBuf },

    /// Invalid model format or corrupted data.
    #[error("invalid model format: {reason}")]
    InvalidFormat { reason: String },

    /// Conversion between formats failed.
    #[error("conversion failed from {from} to {to}: {reason}")]
    ConversionFailed {
        from: String,
        to: String,
        reason: String,
    },

    /// I/O error during file operations.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),

    /// Serialization/deserialization error.
    #[error("serialization error: {0}")]
    Serialization(String),

    /// Dimension mismatch in tensor operations.
    #[error("dimension mismatch: expected {expected}, got {actual}")]
    DimensionMismatch { expected: String, actual: String },

    /// Feature not available (e.g., GPU on WASM).
    #[error("feature not available: {feature} - {reason}")]
    FeatureNotAvailable { feature: String, reason: String },

    /// Underlying aprender error.
    #[error("aprender error: {0}")]
    Aprender(String),
}

impl CookbookError {
    /// Create a new invalid format error.
    #[must_use]
    pub fn invalid_format(reason: impl Into<String>) -> Self {
        Self::InvalidFormat {
            reason: reason.into(),
        }
    }

    /// Create a new conversion failed error.
    #[must_use]
    pub fn conversion_failed(
        from: impl Into<String>,
        to: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        Self::ConversionFailed {
            from: from.into(),
            to: to.into(),
            reason: reason.into(),
        }
    }

    /// Create a feature not available error.
    #[must_use]
    pub fn feature_not_available(feature: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::FeatureNotAvailable {
            feature: feature.into(),
            reason: reason.into(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_error_display_model_not_found() {
        let err = CookbookError::ModelNotFound {
            path: PathBuf::from("/path/to/model.apr"),
        };
        assert_eq!(err.to_string(), "model not found: /path/to/model.apr");
    }

    #[test]
    fn test_error_display_invalid_format() {
        let err = CookbookError::invalid_format("magic bytes mismatch");
        assert_eq!(
            err.to_string(),
            "invalid model format: magic bytes mismatch"
        );
    }

    #[test]
    fn test_error_display_conversion_failed() {
        let err = CookbookError::conversion_failed("safetensors", "apr", "unsupported dtype");
        assert_eq!(
            err.to_string(),
            "conversion failed from safetensors to apr: unsupported dtype"
        );
    }

    #[test]
    fn test_error_display_dimension_mismatch() {
        let err = CookbookError::DimensionMismatch {
            expected: "784x128".to_string(),
            actual: "784x256".to_string(),
        };
        assert_eq!(
            err.to_string(),
            "dimension mismatch: expected 784x128, got 784x256"
        );
    }

    #[test]
    fn test_error_display_feature_not_available() {
        let err = CookbookError::feature_not_available("GPU", "not supported on WASM");
        assert_eq!(
            err.to_string(),
            "feature not available: GPU - not supported on WASM"
        );
    }

    #[test]
    fn test_io_error_conversion() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let err: CookbookError = io_err.into();
        assert!(err.to_string().contains("io error"));
    }
}
