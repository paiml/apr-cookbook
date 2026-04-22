//! APR v2 bundle format types.
//!
//! New in v2:
//! - Explicit compression type (LZ4/ZSTD)
//! - Quantization support (Int4/Int8/FP16)
//! - Ed25519 signature verification
//! - Binary tensor index for O(1) lookup

use crate::error::{CookbookError, Result};

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
#[path = "v2_tests.rs"]
mod tests;
