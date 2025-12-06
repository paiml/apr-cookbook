//! # Recipe: Bundle Ed25519 Signed Model
//!
//! **Category**: Binary Bundling
//! **Isolation Level**: Full
//! **Idempotency**: Guaranteed
//! **Dependencies**: None (default features)
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
//! Bundle Ed25519 signed model with integrity verification.
//!
//! ## Run Command
//! ```bash
//! cargo run --example bundle_apr_signed
//! ```

use apr_cookbook::prelude::*;
use rand::Rng;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("bundle_apr_signed")?;

    // Generate model payload
    let n_params = 4096;
    let payload = generate_model_payload(hash_name_to_seed("signed_model"), n_params);

    // Create mock signature (in production, use actual Ed25519)
    let (public_key, signature) = create_mock_signature(ctx.rng(), &payload);

    ctx.record_metric("payload_size", payload.len() as i64);
    ctx.record_metric("signature_size", signature.len() as i64);
    ctx.record_metric("public_key_size", public_key.len() as i64);

    // Append signature and public key to payload
    let mut full_payload = payload.clone();
    full_payload.extend_from_slice(&signature);
    full_payload.extend_from_slice(&public_key);

    let signed_bundle = ModelBundle::new()
        .with_name("signed-model")
        .with_payload(full_payload)
        .with_compression(false);
    // Set signed flag manually
    let mut bytes = signed_bundle.build();
    bytes[6] |= 0x04; // Set signed flag

    // Verify signature
    let verification_result = verify_mock_signature(&payload, &signature, &public_key);
    ctx.record_string_metric(
        "verification_result",
        if verification_result {
            "VALID"
        } else {
            "INVALID"
        },
    );

    // Save signed model
    let apr_path = ctx.path("signed_model.apr");
    std::fs::write(&apr_path, &bytes)?;

    // Load and verify
    let loaded = BundledModel::from_bytes(&bytes)?;

    println!("=== Recipe: {} ===", ctx.name());
    println!("Signed Model Bundle:");
    println!("  Payload size: {} bytes", payload.len());
    println!("  Signature size: {} bytes (Ed25519)", signature.len());
    println!("  Public key size: {} bytes", public_key.len());
    println!("  Total bundle size: {} bytes", bytes.len());
    println!();
    println!(
        "Verification: {}",
        if verification_result {
            "VALID"
        } else {
            "INVALID"
        }
    );
    println!("Is signed flag: {}", loaded.is_signed());
    println!();
    println!("Saved to: {:?}", apr_path);

    Ok(())
}

/// Create a mock Ed25519 signature (for demonstration)
/// In production, use `ed25519-dalek` or similar
fn create_mock_signature(rng: &mut impl Rng, data: &[u8]) -> (Vec<u8>, Vec<u8>) {
    // Mock public key (32 bytes)
    let public_key: Vec<u8> = (0..32).map(|_| rng.gen()).collect();

    // Mock signature (64 bytes) - in reality, this would be computed from private key
    let mut signature = Vec::with_capacity(64);

    // Create deterministic "signature" based on data hash
    let data_hash = simple_hash(data);
    for i in 0..64 {
        signature.push((data_hash.wrapping_add(i as u64) & 0xFF) as u8);
    }

    (public_key, signature)
}

/// Verify a mock signature
fn verify_mock_signature(data: &[u8], signature: &[u8], _public_key: &[u8]) -> bool {
    if signature.len() != 64 {
        return false;
    }

    // Recreate expected signature
    let data_hash = simple_hash(data);
    for (i, &sig_byte) in signature.iter().enumerate().take(64) {
        let expected = (data_hash.wrapping_add(i as u64) & 0xFF) as u8;
        if sig_byte != expected {
            return false;
        }
    }

    true
}

/// Simple hash function for demonstration
fn simple_hash(data: &[u8]) -> u64 {
    let mut hash = 0u64;
    for (i, &byte) in data.iter().enumerate() {
        hash = hash.wrapping_add(u64::from(byte).wrapping_mul((i as u64).wrapping_add(1)));
        hash = hash.rotate_left(7);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_signature_creation() {
        let mut ctx = RecipeContext::new("test_sig_create").unwrap();
        let payload = vec![1u8, 2, 3, 4, 5];
        let (public_key, signature) = create_mock_signature(ctx.rng(), &payload);

        assert_eq!(public_key.len(), 32);
        assert_eq!(signature.len(), 64);
    }

    #[test]
    fn test_signature_verification() {
        let mut ctx = RecipeContext::new("test_sig_verify").unwrap();
        let payload = vec![1u8, 2, 3, 4, 5];
        let (public_key, signature) = create_mock_signature(ctx.rng(), &payload);

        assert!(verify_mock_signature(&payload, &signature, &public_key));
    }

    #[test]
    fn test_signature_tampering_detection() {
        let mut ctx = RecipeContext::new("test_tamper").unwrap();
        let payload = vec![1u8, 2, 3, 4, 5];
        let (public_key, signature) = create_mock_signature(ctx.rng(), &payload);

        // Tamper with payload
        let tampered_payload = vec![1u8, 2, 3, 4, 6]; // Changed last byte
        assert!(!verify_mock_signature(
            &tampered_payload,
            &signature,
            &public_key
        ));
    }

    #[test]
    fn test_signed_flag() {
        let mut bundle_bytes = ModelBundle::new().with_payload(vec![1, 2, 3]).build();

        // Initially not signed
        let model = BundledModel::from_bytes(&bundle_bytes).unwrap();
        assert!(!model.is_signed());

        // Set signed flag
        bundle_bytes[6] |= 0x04;
        let model = BundledModel::from_bytes(&bundle_bytes).unwrap();
        assert!(model.is_signed());
    }

    #[test]
    fn test_deterministic_signature() {
        let payload = vec![1u8, 2, 3, 4, 5];

        let (_, sig1) = create_mock_signature(&mut rand::rngs::StdRng::seed_from_u64(42), &payload);
        let (_, sig2) = create_mock_signature(&mut rand::rngs::StdRng::seed_from_u64(42), &payload);

        // Signatures from same seed should match
        // Note: public key is random, but signature is deterministic on data
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_hash_deterministic() {
        let data = vec![1u8, 2, 3, 4, 5];
        let hash1 = simple_hash(&data);
        let hash2 = simple_hash(&data);
        assert_eq!(hash1, hash2);
    }
}

#[cfg(test)]
mod proptests {
    use super::*;
    use proptest::prelude::*;
    use rand::SeedableRng;

    proptest! {
        #![proptest_config(ProptestConfig::with_cases(100))]

        #[test]
        fn prop_valid_signature_verifies(data in proptest::collection::vec(any::<u8>(), 1..100)) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let (public_key, signature) = create_mock_signature(&mut rng, &data);
            prop_assert!(verify_mock_signature(&data, &signature, &public_key));
        }

        #[test]
        fn prop_signature_sizes(data in proptest::collection::vec(any::<u8>(), 1..100)) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let (public_key, signature) = create_mock_signature(&mut rng, &data);
            prop_assert_eq!(public_key.len(), 32);
            prop_assert_eq!(signature.len(), 64);
        }

        #[test]
        fn prop_tampered_fails(
            data in proptest::collection::vec(any::<u8>(), 2..100),
            tamper_idx in 0usize..100
        ) {
            let mut rng = rand::rngs::StdRng::seed_from_u64(42);
            let (public_key, signature) = create_mock_signature(&mut rng, &data);

            let mut tampered = data.clone();
            let idx = tamper_idx % tampered.len();
            tampered[idx] = tampered[idx].wrapping_add(1);

            prop_assert!(!verify_mock_signature(&tampered, &signature, &public_key));
        }
    }
}
