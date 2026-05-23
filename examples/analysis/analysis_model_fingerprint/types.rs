//! Support module for the sibling `main.rs` recipe.
//!
//! Contract: contracts/recipe-iiur-v1.yaml (inherited from main.rs — Invariant B)
#![allow(
    dead_code,
    unused_imports,
    unreachable_pub,
    clippy::trivially_copy_pass_by_ref,
    clippy::wildcard_imports
)]
#[allow(unused_imports)]
use apr_cookbook::prelude::*;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use proptest::prelude::*;
use rand::rngs::StdRng;
use rand::Rng;
use rand::SeedableRng;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Fingerprint of a single tensor within a model.
#[derive(Debug, Clone, Serialize)]
pub struct TensorFingerprint {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: String,
    pub size_bytes: usize,
    pub blake3_hash: String,
}

/// Fingerprint of an entire model file.
#[derive(Debug, Clone, Serialize)]
pub struct ModelFingerprint {
    pub model_hash: String,
    pub tensor_fingerprints: Vec<TensorFingerprint>,
    pub metadata_hash: String,
    pub total_bytes: usize,
    pub n_tensors: usize,
    pub signature: Option<String>,
    pub signer_public_key: Option<String>,
}

// ---------------------------------------------------------------------------
// Hex encoding / decoding
// ---------------------------------------------------------------------------

pub fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

pub fn hex_decode(s: &str) -> Result<Vec<u8>> {
    (0..s.len())
        .step_by(2)
        .map(|i| {
            u8::from_str_radix(&s[i..i + 2], 16)
                .map_err(|e| CookbookError::invalid_format(e.to_string()))
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Deterministic key generation
// ---------------------------------------------------------------------------

pub fn deterministic_signing_key(seed: u64) -> SigningKey {
    let mut key_bytes = [0u8; 32];
    for (i, byte) in key_bytes.iter_mut().enumerate() {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        let mut hasher = DefaultHasher::new();
        (seed, i).hash(&mut hasher);
        *byte = (hasher.finish() & 0xFF) as u8;
    }
    SigningKey::from_bytes(&key_bytes)
}

// ---------------------------------------------------------------------------
// Model creation
// ---------------------------------------------------------------------------

// Create a synthetic .apr-like byte payload with header + tensor chunks.
//
/// Layout: [magic(4)] [version(4)] [n_tensors(4)] [metadata(variable)] [tensor_data...]
pub fn create_test_model_bytes(rng: &mut StdRng, n_tensors: usize, tensor_size: usize) -> Vec<u8> {
    let mut data = Vec::new();

    // Magic bytes
    data.extend_from_slice(b"APR2");
    // Version
    data.extend_from_slice(&2u32.to_le_bytes());
    // Number of tensors
    data.extend_from_slice(&(n_tensors as u32).to_le_bytes());
    // Metadata placeholder (32 bytes of descriptive content)
    let metadata = b"model=fingerprint_test;dtype=f32";
    data.extend_from_slice(metadata);
    // Pad metadata to 32 bytes
    let pad_len = 32_usize.saturating_sub(metadata.len());
    data.extend(std::iter::repeat(0u8).take(pad_len));

    // Tensor data chunks
    for _ in 0..n_tensors {
        for _ in 0..tensor_size {
            let val: f32 = rng.gen_range(-1.0..1.0);
            data.extend_from_slice(&val.to_le_bytes());
        }
    }

    data
}

// ---------------------------------------------------------------------------
// Fingerprinting
// ---------------------------------------------------------------------------

// Hash the entire model and each tensor individually using blake3.
//
/// `tensor_offsets` contains tuples of (name, byte_offset, byte_length, shape).
pub fn fingerprint_model(
    data: &[u8],
    tensor_offsets: &[(String, usize, usize, Vec<usize>)],
) -> Result<ModelFingerprint> {
    if data.is_empty() {
        return Err(CookbookError::invalid_format("empty model data"));
    }

    // Whole-model hash
    let model_hash = blake3::hash(data).to_hex().to_string();

    // Metadata hash (bytes 12..44 in our synthetic format)
    let meta_start = 12_usize.min(data.len());
    let meta_end = 44_usize.min(data.len());
    let metadata_hash = blake3::hash(&data[meta_start..meta_end])
        .to_hex()
        .to_string();

    // Per-tensor fingerprints
    let mut tensor_fingerprints = Vec::with_capacity(tensor_offsets.len());
    for (name, offset, length, shape) in tensor_offsets {
        let end = (*offset + *length).min(data.len());
        if *offset >= data.len() {
            return Err(CookbookError::invalid_format(format!(
                "tensor '{}' offset {} exceeds data length {}",
                name,
                offset,
                data.len()
            )));
        }
        let tensor_bytes = &data[*offset..end];
        let blake3_hash = blake3::hash(tensor_bytes).to_hex().to_string();

        tensor_fingerprints.push(TensorFingerprint {
            name: name.clone(),
            shape: shape.clone(),
            dtype: "f32".to_string(),
            size_bytes: end - *offset,
            blake3_hash,
        });
    }

    Ok(ModelFingerprint {
        model_hash,
        tensor_fingerprints,
        metadata_hash,
        total_bytes: data.len(),
        n_tensors: tensor_offsets.len(),
        signature: None,
        signer_public_key: None,
    })
}

// ---------------------------------------------------------------------------
// Signing & verification
// ---------------------------------------------------------------------------

/// Sign the model hash with ed25519, returning (signature_hex, pubkey_hex).
pub fn sign_fingerprint(
    fingerprint: &ModelFingerprint,
    key: &SigningKey,
) -> Result<(String, String)> {
    let message = fingerprint.model_hash.as_bytes();
    let signature = key.sign(message);
    let pubkey = key.verifying_key();

    Ok((
        hex_encode(&signature.to_bytes()),
        hex_encode(pubkey.as_bytes()),
    ))
}

/// Verify an ed25519 signature against a model hash.
pub fn verify_signature(model_hash: &str, signature_hex: &str, pubkey_hex: &str) -> Result<bool> {
    let sig_bytes = hex_decode(signature_hex)?;
    let pub_bytes = hex_decode(pubkey_hex)?;

    if sig_bytes.len() != 64 {
        return Err(CookbookError::invalid_format("signature must be 64 bytes"));
    }
    if pub_bytes.len() != 32 {
        return Err(CookbookError::invalid_format("public key must be 32 bytes"));
    }

    let mut sig_arr = [0u8; 64];
    sig_arr.copy_from_slice(&sig_bytes);
    let signature = Signature::from_bytes(&sig_arr);

    let mut pub_arr = [0u8; 32];
    pub_arr.copy_from_slice(&pub_bytes);
    let verifying_key = VerifyingKey::from_bytes(&pub_arr)
        .map_err(|e| CookbookError::invalid_format(e.to_string()))?;

    Ok(verifying_key
        .verify(model_hash.as_bytes(), &signature)
        .is_ok())
}

// ---------------------------------------------------------------------------
// Tamper detection
// ---------------------------------------------------------------------------

/// Compare two fingerprints and report which tensors changed.
pub fn detect_tampering(original: &ModelFingerprint, current: &ModelFingerprint) -> Vec<String> {
    let mut changes = Vec::new();

    if original.model_hash != current.model_hash {
        changes.push("model-level hash mismatch".to_string());
    }

    if original.metadata_hash != current.metadata_hash {
        changes.push("metadata hash mismatch".to_string());
    }

    if original.n_tensors != current.n_tensors {
        changes.push(format!(
            "tensor count changed: {} -> {}",
            original.n_tensors, current.n_tensors
        ));
    }

    let pairs = original
        .tensor_fingerprints
        .iter()
        .zip(current.tensor_fingerprints.iter());

    for (orig, curr) in pairs {
        if orig.blake3_hash != curr.blake3_hash {
            changes.push(format!(
                "tensor '{}' modified (hash: {}... -> {}...)",
                orig.name,
                &orig.blake3_hash[..16],
                &curr.blake3_hash[..16],
            ));
        }
    }

    changes
}

// ---------------------------------------------------------------------------
// Tensor offset helper
// ---------------------------------------------------------------------------

/// Build tensor offset descriptors for our synthetic model layout.
pub fn build_tensor_offsets(
    n_tensors: usize,
    tensor_size: usize,
    header_size: usize,
) -> Vec<(String, usize, usize, Vec<usize>)> {
    let bytes_per_tensor = tensor_size * 4; // f32 = 4 bytes
    (0..n_tensors)
        .map(|i| {
            let name = format!("layer_{i}.weight");
            let offset = header_size + i * bytes_per_tensor;
            let shape = vec![tensor_size];
            (name, offset, bytes_per_tensor, shape)
        })
        .collect()
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------
