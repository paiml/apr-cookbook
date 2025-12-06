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

use crate::error::{CookbookError, Result};

/// A model bundled from static bytes.
///
/// This struct wraps model data that has been embedded into the binary
/// at compile time using `include_bytes!()`.
#[derive(Debug, Clone)]
pub struct BundledModel {
    /// Raw model bytes
    bytes: Vec<u8>,
    /// Model metadata
    metadata: ModelMetadata,
}

/// Metadata extracted from a bundled model.
#[derive(Debug, Clone, Default)]
pub struct ModelMetadata {
    /// Model name (from header)
    pub name: Option<String>,
    /// Model description
    pub description: Option<String>,
    /// Format version
    pub version: (u8, u8),
    /// Whether the model is compressed
    pub compressed: bool,
    /// Whether the model is encrypted
    pub encrypted: bool,
    /// Whether the model is signed
    pub signed: bool,
    /// Number of parameters (if known)
    pub n_parameters: Option<usize>,
}

/// APR format magic bytes
const APR_MAGIC: &[u8; 4] = b"APRN";

/// Minimum valid APR header size
const MIN_HEADER_SIZE: usize = 32;

impl BundledModel {
    /// Create a bundled model from raw bytes.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The bytes are too short to contain a valid header
    /// - The magic bytes don't match the APR format
    /// - The header is malformed
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        // Validate minimum size
        if bytes.len() < MIN_HEADER_SIZE {
            return Err(CookbookError::invalid_format(format!(
                "data too short: {} bytes, minimum {} required",
                bytes.len(),
                MIN_HEADER_SIZE
            )));
        }

        // Validate magic bytes
        if &bytes[0..4] != APR_MAGIC {
            return Err(CookbookError::invalid_format(format!(
                "invalid magic bytes: expected APRN, got {:?}",
                &bytes[0..4]
            )));
        }

        // Parse header
        let metadata = Self::parse_header(bytes);

        Ok(Self {
            bytes: bytes.to_vec(),
            metadata,
        })
    }

    /// Parse the APR header to extract metadata.
    fn parse_header(bytes: &[u8]) -> ModelMetadata {
        // APR Header format (32 bytes):
        // [0-3]: Magic "APRN"
        // [4-5]: Version (major, minor)
        // [6]: Flags (compression, encryption, signing, etc.)
        // [7]: Reserved
        // [8-11]: Uncompressed size (u32 LE)
        // [12-31]: Reserved/metadata offset

        let version = (bytes[4], bytes[5]);
        let flags = bytes[6];

        let compressed = (flags & 0x01) != 0;
        let encrypted = (flags & 0x02) != 0;
        let signed = (flags & 0x04) != 0;

        ModelMetadata {
            name: None, // Would be parsed from metadata section
            description: None,
            version,
            compressed,
            encrypted,
            signed,
            n_parameters: None,
        }
    }

    /// Get the model name.
    #[must_use]
    pub fn name(&self) -> &str {
        self.metadata.name.as_deref().unwrap_or("unnamed")
    }

    /// Get the model metadata.
    #[must_use]
    pub fn metadata(&self) -> &ModelMetadata {
        &self.metadata
    }

    /// Get the raw bytes.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.bytes
    }

    /// Get the size in bytes.
    #[must_use]
    pub fn size(&self) -> usize {
        self.bytes.len()
    }

    /// Check if the model is compressed.
    #[must_use]
    pub fn is_compressed(&self) -> bool {
        self.metadata.compressed
    }

    /// Check if the model is encrypted.
    #[must_use]
    pub fn is_encrypted(&self) -> bool {
        self.metadata.encrypted
    }

    /// Check if the model is signed.
    #[must_use]
    pub fn is_signed(&self) -> bool {
        self.metadata.signed
    }

    /// Get the format version.
    #[must_use]
    pub fn version(&self) -> (u8, u8) {
        self.metadata.version
    }
}

/// Builder for creating model bundles.
///
/// Used primarily for testing and creating sample models.
#[derive(Debug, Default)]
pub struct ModelBundle {
    name: Option<String>,
    description: Option<String>,
    compressed: bool,
    encrypted: bool,
    signed: bool,
    payload: Vec<u8>,
}

impl ModelBundle {
    /// Create a new model bundle builder.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the model name.
    #[must_use]
    pub fn with_name(mut self, name: impl Into<String>) -> Self {
        self.name = Some(name.into());
        self
    }

    /// Set the model description.
    #[must_use]
    pub fn with_description(mut self, description: impl Into<String>) -> Self {
        self.description = Some(description.into());
        self
    }

    /// Enable compression.
    #[must_use]
    pub fn with_compression(mut self, enabled: bool) -> Self {
        self.compressed = enabled;
        self
    }

    /// Set the payload data.
    #[must_use]
    pub fn with_payload(mut self, payload: Vec<u8>) -> Self {
        self.payload = payload;
        self
    }

    /// Build the model bundle into raw bytes.
    #[must_use]
    pub fn build(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(MIN_HEADER_SIZE + self.payload.len());

        // Magic bytes
        bytes.extend_from_slice(APR_MAGIC);

        // Version (1.0)
        bytes.push(1);
        bytes.push(0);

        // Flags
        let mut flags: u8 = 0;
        if self.compressed {
            flags |= 0x01;
        }
        if self.encrypted {
            flags |= 0x02;
        }
        if self.signed {
            flags |= 0x04;
        }
        bytes.push(flags);

        // Reserved
        bytes.push(0);

        // Uncompressed size (u32 LE)
        let size = self.payload.len() as u32;
        bytes.extend_from_slice(&size.to_le_bytes());

        // Reserved bytes to reach MIN_HEADER_SIZE
        bytes.resize(MIN_HEADER_SIZE, 0);

        // Payload
        bytes.extend_from_slice(&self.payload);

        bytes
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    #[test]
    fn test_bundled_model_from_valid_bytes() {
        let bundle = ModelBundle::new()
            .with_name("test-model")
            .with_payload(vec![1, 2, 3, 4])
            .build();

        let model = BundledModel::from_bytes(&bundle).unwrap();
        assert_eq!(model.version(), (1, 0));
        assert!(!model.is_compressed());
        assert!(!model.is_encrypted());
        assert!(!model.is_signed());
    }

    #[test]
    fn test_bundled_model_from_compressed_bytes() {
        let bundle = ModelBundle::new()
            .with_compression(true)
            .with_payload(vec![1, 2, 3, 4])
            .build();

        let model = BundledModel::from_bytes(&bundle).unwrap();
        assert!(model.is_compressed());
    }

    #[test]
    fn test_bundled_model_rejects_short_data() {
        let short_data = vec![0u8; 10];
        let result = BundledModel::from_bytes(&short_data);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("data too short"));
    }

    #[test]
    fn test_bundled_model_rejects_invalid_magic() {
        let mut bad_data = vec![0u8; MIN_HEADER_SIZE];
        bad_data[0..4].copy_from_slice(b"XXXX");

        let result = BundledModel::from_bytes(&bad_data);
        assert!(result.is_err());

        let err = result.unwrap_err();
        assert!(err.to_string().contains("invalid magic bytes"));
    }

    #[test]
    fn test_model_bundle_builder() {
        let bundle = ModelBundle::new()
            .with_name("sentiment-classifier")
            .with_description("Classifies text sentiment")
            .with_compression(true)
            .with_payload(vec![42; 100])
            .build();

        assert!(bundle.len() >= MIN_HEADER_SIZE + 100);
        assert_eq!(&bundle[0..4], APR_MAGIC);
    }

    #[test]
    fn test_bundled_model_size() {
        let payload = vec![0u8; 1000];
        let bundle = ModelBundle::new().with_payload(payload).build();

        let model = BundledModel::from_bytes(&bundle).unwrap();
        assert_eq!(model.size(), bundle.len());
    }

    #[test]
    fn test_bundled_model_name_default() {
        let bundle = ModelBundle::new().build();
        let model = BundledModel::from_bytes(&bundle).unwrap();
        assert_eq!(model.name(), "unnamed");
    }

    #[test]
    fn test_bundled_model_as_bytes_roundtrip() {
        let original_payload = vec![1, 2, 3, 4, 5];
        let bundle = ModelBundle::new()
            .with_payload(original_payload.clone())
            .build();

        let model = BundledModel::from_bytes(&bundle).unwrap();
        let recovered = model.as_bytes();

        // The recovered bytes should match the original bundle
        assert_eq!(recovered, bundle.as_slice());
    }

    #[test]
    fn test_flags_parsing() {
        // Test all flag combinations
        for compressed in [false, true] {
            for encrypted in [false, true] {
                for signed in [false, true] {
                    let mut bundle = ModelBundle::new().with_compression(compressed);
                    bundle.encrypted = encrypted;
                    bundle.signed = signed;
                    let bytes = bundle.build();

                    let model = BundledModel::from_bytes(&bytes).unwrap();
                    assert_eq!(model.is_compressed(), compressed);
                    assert_eq!(model.is_encrypted(), encrypted);
                    assert_eq!(model.is_signed(), signed);
                }
            }
        }
    }
}
