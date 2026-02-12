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
        let magic = bytes.get(0..4).unwrap_or_default();
        if magic != APR_MAGIC {
            return Err(CookbookError::invalid_format(format!(
                "invalid magic bytes: expected APRN, got {magic:?}",
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

// ============================================================================
// APR v2 Format Types
// ============================================================================

/// Compression algorithm for APR v2 format.
///
/// APR v2 supports multiple compression algorithms with different tradeoffs:
/// - LZ4: Fast decompression (≥3 GB/s), moderate ratio
/// - ZSTD: Better ratio, good decompression speed
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Compression {
    /// No compression
    #[default]
    None,
    /// LZ4 compression - fast decompression
    Lz4,
    /// ZSTD compression - better ratio
    Zstd,
}

/// Quantization type for APR v2 format.
///
/// Quantization reduces model size with minimal accuracy loss:
/// - FP32: Full precision (no quantization)
/// - FP16: Half precision (2x smaller)
/// - Int8: 8-bit integers (4x smaller, <1% loss)
/// - Int4: 4-bit integers (8x smaller, <2% loss)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Quantization {
    /// Full precision 32-bit floats
    #[default]
    FP32,
    /// Half precision 16-bit floats
    FP16,
    /// 8-bit integer quantization
    Int8,
    /// 4-bit integer quantization
    Int4,
}

/// Ed25519 signature for model verification.
#[derive(Debug, Clone)]
pub struct ModelSignature {
    /// 64-byte Ed25519 signature
    pub signature: [u8; 64],
    /// 32-byte public key
    pub public_key: [u8; 32],
}

/// APR v2 model bundle with enhanced features.
///
/// New in v2:
/// - Explicit compression type (LZ4/ZSTD)
/// - Quantization support (Int4/Int8/FP16)
/// - Ed25519 signature verification
/// - Binary tensor index for O(1) lookup
#[derive(Debug, Default)]
pub struct ModelBundleV2 {
    name: Option<String>,
    description: Option<String>,
    compression: Compression,
    quantization: Quantization,
    signature: Option<ModelSignature>,
    tensors: Vec<TensorEntry>,
    payload: Vec<u8>,
}

/// A tensor entry in the binary index.
#[derive(Debug, Clone)]
pub struct TensorEntry {
    /// Tensor name
    pub name: String,
    /// Data type (from quantization)
    pub dtype: Quantization,
    /// Shape dimensions
    pub shape: Vec<usize>,
    /// Byte offset in payload
    pub offset: usize,
    /// Byte length
    pub length: usize,
}

impl ModelBundleV2 {
    /// Create a new APR v2 model bundle builder.
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

    /// Set the compression type.
    #[must_use]
    pub fn with_compression(mut self, compression: Compression) -> Self {
        self.compression = compression;
        self
    }

    /// Set the quantization type.
    #[must_use]
    pub fn with_quantization(mut self, quantization: Quantization) -> Self {
        self.quantization = quantization;
        self
    }

    /// Add a tensor to the index.
    #[must_use]
    pub fn add_tensor(mut self, name: impl Into<String>, shape: Vec<usize>, data: Vec<u8>) -> Self {
        let offset = self.payload.len();
        let length = data.len();
        self.tensors.push(TensorEntry {
            name: name.into(),
            dtype: self.quantization,
            shape,
            offset,
            length,
        });
        self.payload.extend(data);
        self
    }

    /// Sign the model with Ed25519 key pair.
    #[must_use]
    pub fn sign(mut self, signing_key: &[u8; 32]) -> Self {
        use ed25519_dalek::{Signer, SigningKey};

        let key = SigningKey::from_bytes(signing_key);
        let message = self.compute_hash();
        let signature = key.sign(&message);

        self.signature = Some(ModelSignature {
            signature: signature.to_bytes(),
            public_key: key.verifying_key().to_bytes(),
        });
        self
    }

    /// Compute hash of payload for signing.
    fn compute_hash(&self) -> [u8; 32] {
        use blake3::Hasher;
        let mut hasher = Hasher::new();
        hasher.update(&self.payload);
        *hasher.finalize().as_bytes()
    }

    /// Build the APR v2 bundle.
    #[must_use]
    pub fn build(&self) -> Vec<u8> {
        // APR v2 Header format (64 bytes):
        // [0-3]: Magic "APR2"
        // [4-5]: Version (2, 0)
        // [6]: Compression type (0=None, 1=LZ4, 2=ZSTD)
        // [7]: Quantization type (0=FP32, 1=FP16, 2=Int8, 3=Int4)
        // [8-11]: Tensor count (u32 LE)
        // [12-15]: Index offset (u32 LE)
        // [16-19]: Payload offset (u32 LE)
        // [20-23]: Payload size (u32 LE)
        // [24]: Has signature (0/1)
        // [25-63]: Reserved

        let mut bytes = Vec::new();

        // Magic bytes for v2
        bytes.extend_from_slice(b"APR2");

        // Version 2.0
        bytes.push(2);
        bytes.push(0);

        // Compression type
        bytes.push(match self.compression {
            Compression::None => 0,
            Compression::Lz4 => 1,
            Compression::Zstd => 2,
        });

        // Quantization type
        bytes.push(match self.quantization {
            Quantization::FP32 => 0,
            Quantization::FP16 => 1,
            Quantization::Int8 => 2,
            Quantization::Int4 => 3,
        });

        // Tensor count
        let tensor_count = self.tensors.len() as u32;
        bytes.extend_from_slice(&tensor_count.to_le_bytes());

        // Index offset (after 64-byte header + optional 96-byte signature)
        let index_offset: u32 = if self.signature.is_some() {
            64 + 96
        } else {
            64
        };
        bytes.extend_from_slice(&index_offset.to_le_bytes());

        // Payload offset (after index)
        // Each tensor entry: 4 (name_len) + name + 4 (shape_len) + shape + 8 (offset) + 8 (length) + 1 (dtype)
        let index_size: u32 = self
            .tensors
            .iter()
            .map(|t| 4 + t.name.len() + 4 + t.shape.len() * 8 + 8 + 8 + 1)
            .sum::<usize>() as u32;
        let payload_offset = index_offset + index_size;
        bytes.extend_from_slice(&payload_offset.to_le_bytes());

        // Payload size
        let payload_size = self.payload.len() as u32;
        bytes.extend_from_slice(&payload_size.to_le_bytes());

        // Has signature
        bytes.push(u8::from(self.signature.is_some()));

        // Reserved (pad to 64 bytes)
        bytes.resize(64, 0);

        // Optional signature block (96 bytes: 64 sig + 32 pubkey)
        if let Some(ref sig) = self.signature {
            bytes.extend_from_slice(&sig.signature);
            bytes.extend_from_slice(&sig.public_key);
        }

        // Tensor index
        for tensor in &self.tensors {
            // Name length + name
            bytes.extend_from_slice(&(tensor.name.len() as u32).to_le_bytes());
            bytes.extend_from_slice(tensor.name.as_bytes());

            // Shape
            bytes.extend_from_slice(&(tensor.shape.len() as u32).to_le_bytes());
            for &dim in &tensor.shape {
                bytes.extend_from_slice(&(dim as u64).to_le_bytes());
            }

            // Offset and length
            bytes.extend_from_slice(&(tensor.offset as u64).to_le_bytes());
            bytes.extend_from_slice(&(tensor.length as u64).to_le_bytes());

            // Data type
            bytes.push(match tensor.dtype {
                Quantization::FP32 => 0,
                Quantization::FP16 => 1,
                Quantization::Int8 => 2,
                Quantization::Int4 => 3,
            });
        }

        // Compress payload if needed
        let compressed_payload = match self.compression {
            Compression::None => self.payload.clone(),
            Compression::Lz4 => lz4_flex::compress_prepend_size(&self.payload),
            Compression::Zstd => {
                zstd::encode_all(&self.payload[..], 3).unwrap_or_else(|_| self.payload.clone())
            }
        };

        bytes.extend_from_slice(&compressed_payload);
        bytes
    }

    /// Get the compression type.
    #[must_use]
    pub fn compression(&self) -> Compression {
        self.compression
    }

    /// Get the quantization type.
    #[must_use]
    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Get tensor entries.
    #[must_use]
    pub fn tensors(&self) -> &[TensorEntry] {
        &self.tensors
    }
}

/// Load and verify an APR v2 bundle.
pub struct BundledModelV2 {
    bytes: Vec<u8>,
    compression: Compression,
    quantization: Quantization,
    tensor_count: u32,
    signature_valid: Option<bool>,
}

impl BundledModelV2 {
    /// Load an APR v2 bundle from bytes.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 64 {
            return Err(CookbookError::invalid_format("APR v2 header too short"));
        }

        if bytes.get(0..4) != Some(b"APR2".as_slice()) {
            return Err(CookbookError::invalid_format("Not an APR v2 file"));
        }

        let compression = match bytes[6] {
            0 => Compression::None,
            1 => Compression::Lz4,
            2 => Compression::Zstd,
            _ => return Err(CookbookError::invalid_format("Unknown compression type")),
        };

        let quantization = match bytes[7] {
            0 => Quantization::FP32,
            1 => Quantization::FP16,
            2 => Quantization::Int8,
            3 => Quantization::Int4,
            _ => return Err(CookbookError::invalid_format("Unknown quantization type")),
        };

        let tensor_count = u32::from_le_bytes([bytes[8], bytes[9], bytes[10], bytes[11]]);
        let has_signature = bytes[24] == 1;

        // Verify signature if present
        let signature_valid = if has_signature {
            Some(Self::verify_signature(bytes)?)
        } else {
            None
        };

        Ok(Self {
            bytes: bytes.to_vec(),
            compression,
            quantization,
            tensor_count,
            signature_valid,
        })
    }

    /// Verify Ed25519 signature.
    fn verify_signature(bytes: &[u8]) -> Result<bool> {
        use ed25519_dalek::{Signature, Verifier, VerifyingKey};

        if bytes.len() < 64 + 96 {
            return Err(CookbookError::invalid_format("Signature block missing"));
        }

        let sig_bytes: [u8; 64] = bytes[64..128]
            .try_into()
            .map_err(|_| CookbookError::invalid_format("Invalid signature"))?;
        let pub_bytes: [u8; 32] = bytes[128..160]
            .try_into()
            .map_err(|_| CookbookError::invalid_format("Invalid public key"))?;

        let signature = Signature::from_bytes(&sig_bytes);
        let public_key = VerifyingKey::from_bytes(&pub_bytes)
            .map_err(|_| CookbookError::invalid_format("Invalid public key"))?;

        // Get payload offset and compute hash
        let payload_offset =
            u32::from_le_bytes([bytes[16], bytes[17], bytes[18], bytes[19]]) as usize;
        let payload_size =
            u32::from_le_bytes([bytes[20], bytes[21], bytes[22], bytes[23]]) as usize;

        if bytes.len() < payload_offset + payload_size {
            return Err(CookbookError::invalid_format("Payload truncated"));
        }

        // Decompress to get original payload for hash
        let compressed_payload = &bytes[payload_offset..];
        let original_payload = match bytes[6] {
            1 => lz4_flex::decompress_size_prepended(compressed_payload)
                .map_err(|_| CookbookError::invalid_format("LZ4 decompression failed"))?,
            2 => zstd::decode_all(compressed_payload)
                .map_err(|_| CookbookError::invalid_format("ZSTD decompression failed"))?,
            _ => compressed_payload.to_vec(),
        };

        let mut hasher = blake3::Hasher::new();
        hasher.update(&original_payload);
        let hash = hasher.finalize();

        Ok(public_key.verify(hash.as_bytes(), &signature).is_ok())
    }

    /// Check if signature is valid.
    #[must_use]
    pub fn signature_valid(&self) -> Option<bool> {
        self.signature_valid
    }

    /// Get compression type.
    #[must_use]
    pub fn compression(&self) -> Compression {
        self.compression
    }

    /// Get quantization type.
    #[must_use]
    pub fn quantization(&self) -> Quantization {
        self.quantization
    }

    /// Get tensor count.
    #[must_use]
    pub fn tensor_count(&self) -> u32 {
        self.tensor_count
    }

    /// Decompress the payload.
    pub fn decompress(&self) -> Result<Vec<u8>> {
        let payload_offset = u32::from_le_bytes([
            self.bytes[16],
            self.bytes[17],
            self.bytes[18],
            self.bytes[19],
        ]) as usize;

        let compressed_payload = &self.bytes[payload_offset..];

        match self.compression {
            Compression::None => Ok(compressed_payload.to_vec()),
            Compression::Lz4 => {
                lz4_flex::decompress_size_prepended(compressed_payload).map_err(|e| {
                    CookbookError::invalid_format(format!("LZ4 decompression failed: {e}"))
                })
            }
            Compression::Zstd => zstd::decode_all(compressed_payload).map_err(|e| {
                CookbookError::invalid_format(format!("ZSTD decompression failed: {e}"))
            }),
        }
    }
}

#[cfg(test)]
#[allow(clippy::disallowed_methods)]
mod tests {
    use super::*;

    // ========================================================================
    // APR v2 Tests (EXTREME TDD - These must pass!)
    // ========================================================================

    #[test]
    fn test_v2_lz4_compression() {
        // F1: LZ4 decompression must be fast
        let payload = vec![42u8; 100_000]; // 100KB of data

        let bundle = ModelBundleV2::new()
            .with_name("test-lz4")
            .with_compression(Compression::Lz4)
            .add_tensor("weights", vec![100, 1000], payload.clone())
            .build();

        let loaded = BundledModelV2::from_bytes(&bundle).unwrap();
        assert_eq!(loaded.compression(), Compression::Lz4);

        let decompressed = loaded.decompress().unwrap();
        assert_eq!(decompressed, payload);
    }

    #[test]
    fn test_v2_zstd_compression() {
        let payload = vec![42u8; 100_000];

        let bundle = ModelBundleV2::new()
            .with_name("test-zstd")
            .with_compression(Compression::Zstd)
            .add_tensor("weights", vec![100, 1000], payload.clone())
            .build();

        let loaded = BundledModelV2::from_bytes(&bundle).unwrap();
        assert_eq!(loaded.compression(), Compression::Zstd);

        let decompressed = loaded.decompress().unwrap();
        assert_eq!(decompressed, payload);
    }

    #[test]
    fn test_v2_quantization_types() {
        for quant in [
            Quantization::FP32,
            Quantization::FP16,
            Quantization::Int8,
            Quantization::Int4,
        ] {
            let bundle = ModelBundleV2::new()
                .with_quantization(quant)
                .add_tensor("weights", vec![10, 10], vec![0u8; 100])
                .build();

            let loaded = BundledModelV2::from_bytes(&bundle).unwrap();
            assert_eq!(loaded.quantization(), quant);
        }
    }

    #[test]
    fn test_v2_ed25519_signature() {
        // Generate a test key pair
        let signing_key: [u8; 32] = [
            0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec,
            0x2c, 0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03,
            0x1c, 0xae, 0x7f, 0x60,
        ];

        let payload = vec![1u8, 2, 3, 4, 5];
        let bundle = ModelBundleV2::new()
            .with_name("signed-model")
            .add_tensor("data", vec![5], payload)
            .sign(&signing_key)
            .build();

        let loaded = BundledModelV2::from_bytes(&bundle).unwrap();
        assert_eq!(loaded.signature_valid(), Some(true));
    }

    #[test]
    fn test_v2_invalid_signature_rejected() {
        let signing_key: [u8; 32] = [1u8; 32];

        let bundle = ModelBundleV2::new()
            .add_tensor("data", vec![5], vec![1, 2, 3, 4, 5])
            .sign(&signing_key)
            .build();

        // Tamper with the payload
        let mut tampered = bundle.clone();
        if let Some(last) = tampered.last_mut() {
            *last ^= 0xFF;
        }

        let loaded = BundledModelV2::from_bytes(&tampered).unwrap();
        assert_eq!(loaded.signature_valid(), Some(false));
    }

    #[test]
    fn test_v2_tensor_index() {
        let bundle = ModelBundleV2::new()
            .add_tensor("layer1.weight", vec![768, 768], vec![0u8; 768 * 768 * 4])
            .add_tensor("layer1.bias", vec![768], vec![0u8; 768 * 4])
            .add_tensor("layer2.weight", vec![768, 768], vec![0u8; 768 * 768 * 4])
            .build();

        let loaded = BundledModelV2::from_bytes(&bundle).unwrap();
        assert_eq!(loaded.tensor_count(), 3);
    }

    #[test]
    fn test_v2_header_format() {
        let bundle = ModelBundleV2::new()
            .with_compression(Compression::Lz4)
            .with_quantization(Quantization::Int8)
            .build();

        // Check magic bytes
        assert_eq!(&bundle[0..4], b"APR2");
        // Check version
        assert_eq!(bundle[4], 2);
        assert_eq!(bundle[5], 0);
        // Check compression
        assert_eq!(bundle[6], 1); // LZ4
                                  // Check quantization
        assert_eq!(bundle[7], 2); // Int8
    }

    #[test]
    fn test_v2_rejects_v1_format() {
        let v1_bundle = ModelBundle::new().with_payload(vec![1, 2, 3]).build();

        let result = BundledModelV2::from_bytes(&v1_bundle);
        assert!(result.is_err());
    }

    #[test]
    fn test_v2_compression_ratio() {
        // Highly compressible data
        let payload = vec![0u8; 1_000_000]; // 1MB of zeros

        let uncompressed = ModelBundleV2::new()
            .with_compression(Compression::None)
            .add_tensor("data", vec![1_000_000], payload.clone())
            .build();

        let compressed = ModelBundleV2::new()
            .with_compression(Compression::Lz4)
            .add_tensor("data", vec![1_000_000], payload)
            .build();

        // LZ4 should achieve significant compression on zeros
        assert!(compressed.len() < uncompressed.len() / 10);
    }

    // ========================================================================
    // APR v1 Tests (Keep existing tests)
    // ========================================================================

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
