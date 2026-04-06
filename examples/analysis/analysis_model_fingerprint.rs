//! # Recipe: Model Fingerprinting & Tamper Detection
//!
//! **Category**: Analysis
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: blake3, ed25519-dalek
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Memory usage stable
//! 6. [x] WASM compatible (N/A)
//! 7. [x] Clippy clean
//! 8. [x] Rustfmt standard
//! 9. [x] No `unwrap()` in logic
//! 10. [x] Proptests pass (100+ cases)
//!
//! ## Learning Objective
//! Demonstrate content-addressable model fingerprinting with blake3 hashing
//! and ed25519 digital signatures for cryptographic model provenance. Shows
//! how to detect per-tensor tampering in ML model files.
//!
//! ## Toyota Way: 自働化 (Jidoka) - Built-in Quality
//! Cryptographic verification ensures model integrity automatically.
//! Any modification to model weights is detected at the tensor level,
//! preventing silent corruption from reaching production.
//!
//! ## Run Command
//! ```bash
//! cargo run --example analysis_model_fingerprint
//! ```
//!
//! ## References
//! - Paleyes, A. et al. (2022). *Challenges in Deploying Machine Learning*. ACM Computing Surveys. DOI: 10.1145/3533378

use apr_cookbook::prelude::*;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand::rngs::StdRng;
use rand::Rng;
use serde::Serialize;

// ---------------------------------------------------------------------------
// Domain types
// ---------------------------------------------------------------------------

/// Fingerprint of a single tensor within a model.
#[derive(Debug, Clone, Serialize)]
struct TensorFingerprint {
    name: String,
    shape: Vec<usize>,
    dtype: String,
    size_bytes: usize,
    blake3_hash: String,
}

/// Fingerprint of an entire model file.
#[derive(Debug, Clone, Serialize)]
struct ModelFingerprint {
    model_hash: String,
    tensor_fingerprints: Vec<TensorFingerprint>,
    metadata_hash: String,
    total_bytes: usize,
    n_tensors: usize,
    signature: Option<String>,
    signer_public_key: Option<String>,
}

// ---------------------------------------------------------------------------
// Hex encoding / decoding
// ---------------------------------------------------------------------------

fn hex_encode(bytes: &[u8]) -> String {
    use std::fmt::Write;
    bytes
        .iter()
        .fold(String::with_capacity(bytes.len() * 2), |mut s, b| {
            let _ = write!(s, "{b:02x}");
            s
        })
}

fn hex_decode(s: &str) -> Result<Vec<u8>> {
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

fn deterministic_signing_key(seed: u64) -> SigningKey {
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

/// Create a synthetic .apr-like byte payload with header + tensor chunks.
///
/// Layout: [magic(4)] [version(4)] [n_tensors(4)] [metadata(variable)] [tensor_data...]
fn create_test_model_bytes(rng: &mut StdRng, n_tensors: usize, tensor_size: usize) -> Vec<u8> {
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

/// Hash the entire model and each tensor individually using blake3.
///
/// `tensor_offsets` contains tuples of (name, byte_offset, byte_length, shape).
fn fingerprint_model(
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
fn sign_fingerprint(fingerprint: &ModelFingerprint, key: &SigningKey) -> Result<(String, String)> {
    let message = fingerprint.model_hash.as_bytes();
    let signature = key.sign(message);
    let pubkey = key.verifying_key();

    Ok((
        hex_encode(&signature.to_bytes()),
        hex_encode(pubkey.as_bytes()),
    ))
}

/// Verify an ed25519 signature against a model hash.
fn verify_signature(model_hash: &str, signature_hex: &str, pubkey_hex: &str) -> Result<bool> {
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
fn detect_tampering(original: &ModelFingerprint, current: &ModelFingerprint) -> Vec<String> {
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
fn build_tensor_offsets(
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

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("analysis_model_fingerprint")?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Demonstrating cryptographic model fingerprinting and tamper detection");
    println!();

    // ── Section 1: Model Creation ────────────────────────────────────────
    println!("1. Creating synthetic model with 10 tensors...");
    let n_tensors = 10;
    let tensor_size = 64;
    let header_size = 44; // 4 + 4 + 4 + 32
    let model_data = create_test_model_bytes(ctx.rng(), n_tensors, tensor_size);
    let tensor_offsets = build_tensor_offsets(n_tensors, tensor_size, header_size);

    println!("   Model size: {} bytes", model_data.len());
    println!("   Tensors: {}", n_tensors);
    println!("   Tensor size: {} floats each", tensor_size);
    println!();

    // ── Section 2: Content-Addressable Fingerprinting ────────────────────
    println!("2. Computing content-addressable fingerprint...");
    let fingerprint = fingerprint_model(&model_data, &tensor_offsets)?;

    println!("   Model hash: {}...", &fingerprint.model_hash[..32]);
    println!("   Metadata hash: {}...", &fingerprint.metadata_hash[..32]);
    println!("   Per-tensor hashes:");
    for tf in fingerprint.tensor_fingerprints.iter().take(3) {
        println!(
            "     {}: {}... ({} bytes)",
            tf.name,
            &tf.blake3_hash[..16],
            tf.size_bytes
        );
    }
    if fingerprint.n_tensors > 3 {
        println!("     ... ({} more tensors)", fingerprint.n_tensors - 3);
    }
    println!();

    // ── Section 3: Ed25519 Signing ───────────────────────────────────────
    println!("3. Signing model fingerprint with Ed25519...");
    let signing_key = deterministic_signing_key(42);
    let (signature_hex, pubkey_hex) = sign_fingerprint(&fingerprint, &signing_key)?;

    println!("   Signature: {}...", &signature_hex[..32]);
    println!("   Public key: {}...", &pubkey_hex[..32]);
    println!();

    // ── Section 4: Signature Verification ────────────────────────────────
    println!("4. Verifying signature...");
    let is_valid = verify_signature(&fingerprint.model_hash, &signature_hex, &pubkey_hex)?;
    println!(
        "   Signature valid: {}",
        if is_valid { "YES" } else { "NO" }
    );
    println!();

    // ── Section 5: Tamper Detection Demo ─────────────────────────────────
    println!("5. Tamper detection demo...");
    println!("   Modifying tensor 'layer_3.weight'...");
    let mut tampered_data = model_data.clone();
    // Corrupt tensor 3 (offset = header + 3 * tensor_bytes)
    let corrupt_offset = header_size + 3 * tensor_size * 4;
    if corrupt_offset + 4 <= tampered_data.len() {
        tampered_data[corrupt_offset..corrupt_offset + 4].copy_from_slice(&999.0_f32.to_le_bytes());
    }

    let tampered_fingerprint = fingerprint_model(&tampered_data, &tensor_offsets)?;
    let changes = detect_tampering(&fingerprint, &tampered_fingerprint);

    println!("   Changes detected: {}", changes.len());
    for change in &changes {
        println!("     - {}", change);
    }
    println!();

    // ── Section 6: Save & Record Metrics ─────────────────────────────────
    println!("6. Saving fingerprint and recording metrics...");

    let mut signed_fingerprint = fingerprint.clone();
    signed_fingerprint.signature = Some(signature_hex.clone());
    signed_fingerprint.signer_public_key = Some(pubkey_hex.clone());

    let json = serde_json::to_string_pretty(&signed_fingerprint)
        .map_err(|e| CookbookError::Serialization(e.to_string()))?;
    let output_path = ctx.path("model_fingerprint.json");
    std::fs::write(&output_path, &json)?;
    println!("   Saved to: {:?}", output_path);

    ctx.record_metric("total_bytes", model_data.len() as i64);
    ctx.record_metric("n_tensors", n_tensors as i64);
    ctx.record_metric("signature_valid", i64::from(is_valid));
    ctx.record_metric("tamper_changes_detected", changes.len() as i64);
    ctx.record_string_metric("model_hash", &fingerprint.model_hash[..32]);
    ctx.record_string_metric("public_key", &pubkey_hex[..32]);

    println!();
    ctx.report()?;

    Ok(())
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn test_rng() -> StdRng {
        StdRng::seed_from_u64(12345)
    }

    fn make_test_model() -> (Vec<u8>, Vec<(String, usize, usize, Vec<usize>)>) {
        let mut rng = test_rng();
        let n = 4;
        let size = 16;
        let header = 44;
        let data = create_test_model_bytes(&mut rng, n, size);
        let offsets = build_tensor_offsets(n, size, header);
        (data, offsets)
    }

    #[test]
    fn test_hex_encode_roundtrip() {
        let original = vec![0x00, 0x01, 0xAB, 0xFF];
        let encoded = hex_encode(&original);
        let decoded = hex_decode(&encoded).unwrap();
        assert_eq!(original, decoded);
    }

    #[test]
    fn test_hex_decode_invalid() {
        let result = hex_decode("zz");
        assert!(result.is_err());
    }

    #[test]
    fn test_create_test_model_bytes_structure() {
        let mut rng = test_rng();
        let data = create_test_model_bytes(&mut rng, 3, 8);
        // Magic
        assert_eq!(&data[0..4], b"APR2");
        // Version
        assert_eq!(u32::from_le_bytes([data[4], data[5], data[6], data[7]]), 2);
        // N tensors
        assert_eq!(
            u32::from_le_bytes([data[8], data[9], data[10], data[11]]),
            3
        );
        // Total size: 44 header + 3 * 8 * 4 bytes = 44 + 96 = 140
        assert_eq!(data.len(), 44 + 3 * 8 * 4);
    }

    #[test]
    fn test_fingerprint_model_hashes_all_tensors() {
        let (data, offsets) = make_test_model();
        let fp = fingerprint_model(&data, &offsets).unwrap();
        assert_eq!(fp.n_tensors, 4);
        assert_eq!(fp.tensor_fingerprints.len(), 4);
        assert_eq!(fp.total_bytes, data.len());
    }

    #[test]
    fn test_fingerprint_deterministic() {
        let (data, offsets) = make_test_model();
        let fp1 = fingerprint_model(&data, &offsets).unwrap();
        let fp2 = fingerprint_model(&data, &offsets).unwrap();
        assert_eq!(fp1.model_hash, fp2.model_hash);
        assert_eq!(fp1.metadata_hash, fp2.metadata_hash);
        for (a, b) in fp1
            .tensor_fingerprints
            .iter()
            .zip(fp2.tensor_fingerprints.iter())
        {
            assert_eq!(a.blake3_hash, b.blake3_hash);
        }
    }

    #[test]
    fn test_fingerprint_empty_data_error() {
        let result = fingerprint_model(&[], &[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_sign_and_verify() {
        let (data, offsets) = make_test_model();
        let fp = fingerprint_model(&data, &offsets).unwrap();
        let key = deterministic_signing_key(99);
        let (sig, pubkey) = sign_fingerprint(&fp, &key).unwrap();
        let valid = verify_signature(&fp.model_hash, &sig, &pubkey).unwrap();
        assert!(valid);
    }

    #[test]
    fn test_verify_wrong_key_fails() {
        let (data, offsets) = make_test_model();
        let fp = fingerprint_model(&data, &offsets).unwrap();
        let key1 = deterministic_signing_key(1);
        let key2 = deterministic_signing_key(2);
        let (sig, _) = sign_fingerprint(&fp, &key1).unwrap();
        let other_pubkey = hex_encode(key2.verifying_key().as_bytes());
        let valid = verify_signature(&fp.model_hash, &sig, &other_pubkey).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_verify_tampered_hash_fails() {
        let (data, offsets) = make_test_model();
        let fp = fingerprint_model(&data, &offsets).unwrap();
        let key = deterministic_signing_key(42);
        let (sig, pubkey) = sign_fingerprint(&fp, &key).unwrap();
        let valid = verify_signature("tampered_hash_value", &sig, &pubkey).unwrap();
        assert!(!valid);
    }

    #[test]
    fn test_detect_tampering_no_changes() {
        let (data, offsets) = make_test_model();
        let fp = fingerprint_model(&data, &offsets).unwrap();
        let changes = detect_tampering(&fp, &fp);
        assert!(changes.is_empty());
    }

    #[test]
    fn test_detect_tampering_modified_tensor() {
        let (data, offsets) = make_test_model();
        let fp_original = fingerprint_model(&data, &offsets).unwrap();

        let mut tampered = data.clone();
        // Modify tensor 1 (offset = 44 + 1 * 16 * 4 = 108)
        let offset = 44 + 16 * 4;
        tampered[offset] = tampered[offset].wrapping_add(1);

        let fp_tampered = fingerprint_model(&tampered, &offsets).unwrap();
        let changes = detect_tampering(&fp_original, &fp_tampered);

        assert!(!changes.is_empty());
        let has_model_change = changes.iter().any(|c| c.contains("model-level"));
        let has_tensor_change = changes.iter().any(|c| c.contains("layer_1.weight"));
        assert!(has_model_change);
        assert!(has_tensor_change);
    }

    #[test]
    fn test_detect_tampering_tensor_count_change() {
        let fp1 = ModelFingerprint {
            model_hash: "aaa".to_string(),
            tensor_fingerprints: vec![],
            metadata_hash: "bbb".to_string(),
            total_bytes: 100,
            n_tensors: 3,
            signature: None,
            signer_public_key: None,
        };
        let fp2 = ModelFingerprint {
            model_hash: "aaa".to_string(),
            tensor_fingerprints: vec![],
            metadata_hash: "bbb".to_string(),
            total_bytes: 100,
            n_tensors: 5,
            signature: None,
            signer_public_key: None,
        };
        let changes = detect_tampering(&fp1, &fp2);
        assert!(changes.iter().any(|c| c.contains("tensor count changed")));
    }

    #[test]
    fn test_signature_hex_lengths() {
        let key = deterministic_signing_key(7);
        let (data, offsets) = make_test_model();
        let fp = fingerprint_model(&data, &offsets).unwrap();
        let (sig, pubkey) = sign_fingerprint(&fp, &key).unwrap();
        // ed25519 signature = 64 bytes = 128 hex chars
        assert_eq!(sig.len(), 128);
        // ed25519 public key = 32 bytes = 64 hex chars
        assert_eq!(pubkey.len(), 64);
    }

    #[test]
    fn test_verify_invalid_signature_length() {
        let result = verify_signature("hash", "aabb", "cc");
        assert!(result.is_err());
    }

    #[test]
    fn test_fingerprint_serialization() {
        let (data, offsets) = make_test_model();
        let fp = fingerprint_model(&data, &offsets).unwrap();
        let json = serde_json::to_string_pretty(&fp);
        assert!(json.is_ok());
        let s = json.unwrap();
        assert!(s.contains("model_hash"));
        assert!(s.contains("tensor_fingerprints"));
        assert!(s.contains("blake3_hash"));
    }

    #[test]
    fn test_tensor_offset_out_of_bounds() {
        let data = vec![0u8; 50];
        let offsets = vec![("bad_tensor".to_string(), 9999, 100, vec![25])];
        let result = fingerprint_model(&data, &offsets);
        assert!(result.is_err());
    }
}

// ===========================================================================
// Proptests
// ===========================================================================

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_blake3_deterministic(size in 1usize..1024) {
            let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
            let hash1 = blake3::hash(&data).to_hex().to_string();
            let hash2 = blake3::hash(&data).to_hex().to_string();
            prop_assert_eq!(hash1, hash2);
        }

        #[test]
        fn prop_different_data_different_hash(seed1 in 0u64..10000, seed2 in 10000u64..20000) {
            let mut rng1 = StdRng::seed_from_u64(seed1);
            let mut rng2 = StdRng::seed_from_u64(seed2);
            let data1: Vec<u8> = (0..64).map(|_| rng1.gen()).collect();
            let data2: Vec<u8> = (0..64).map(|_| rng2.gen()).collect();
            let hash1 = blake3::hash(&data1).to_hex().to_string();
            let hash2 = blake3::hash(&data2).to_hex().to_string();
            prop_assert_ne!(hash1, hash2);
        }

        #[test]
        fn prop_signature_verification_roundtrip(seed in 1u64..10000) {
            let key = deterministic_signing_key(seed);
            let message = format!("model_hash_{}", seed);
            let signature = key.sign(message.as_bytes());
            let sig_hex = hex_encode(&signature.to_bytes());
            let pub_hex = hex_encode(key.verifying_key().as_bytes());
            let valid = verify_signature(&message, &sig_hex, &pub_hex).unwrap();
            prop_assert!(valid);
        }

        #[test]
        fn prop_hex_roundtrip(size in 1usize..256) {
            let data: Vec<u8> = (0..size).map(|i| (i & 0xFF) as u8).collect();
            let encoded = hex_encode(&data);
            let decoded = hex_decode(&encoded).unwrap();
            prop_assert_eq!(data, decoded);
        }

        #[test]
        fn prop_fingerprint_changes_on_mutation(byte_idx in 44usize..300) {
            let mut rng = StdRng::seed_from_u64(7777);
            let n_tensors = 4;
            let tensor_size = 64;
            let header = 44;
            let data = create_test_model_bytes(&mut rng, n_tensors, tensor_size);
            let offsets = build_tensor_offsets(n_tensors, tensor_size, header);

            let idx = byte_idx % data.len();
            let fp_original = fingerprint_model(&data, &offsets).unwrap();

            let mut modified = data.clone();
            modified[idx] = modified[idx].wrapping_add(1);

            let fp_modified = fingerprint_model(&modified, &offsets).unwrap();
            prop_assert_ne!(fp_original.model_hash, fp_modified.model_hash);
        }
    }
}
