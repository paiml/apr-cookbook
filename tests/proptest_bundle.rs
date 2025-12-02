//! Property-based tests for the bundle module.
//!
//! These tests verify invariants that should hold for any valid input.

use apr_cookbook::bundle::{BundledModel, ModelBundle};
use proptest::prelude::*;

/// Strategy for generating valid model dimensions
fn model_dimensions() -> impl Strategy<Value = (usize, usize)> {
    (1..100usize, 1..100usize)
}

/// Strategy for generating valid model names
fn model_name() -> impl Strategy<Value = String> {
    "[a-zA-Z][a-zA-Z0-9_-]{0,63}".prop_map(|s| s.clone())
}

proptest! {
    /// Property: Building a model bundle and getting bytes should produce valid APR data
    #[test]
    fn model_bundle_produces_valid_apr(
        name in model_name(),
        (input_dim, output_dim) in model_dimensions(),
    ) {
        let size = input_dim * output_dim;
        let weights: Vec<f32> = (0..size).map(|i| i as f32 * 0.01).collect();
        let data: Vec<u8> = weights.iter().flat_map(|f| f.to_le_bytes()).collect();

        let bundle = ModelBundle::new()
            .with_name(&name)
            .with_payload(data);

        let bytes = bundle.build();

        // Should start with APR magic
        prop_assert_eq!(&bytes[0..4], b"APRN");
        // Should have valid header length (at least magic + version + flags + payload size)
        prop_assert!(bytes.len() >= 16);
    }

    /// Property: Bundled model size matches the bundle bytes
    #[test]
    fn bundled_model_size_matches_bytes(data_len in 0..10000usize) {
        let data = vec![0u8; data_len];
        let bundle = ModelBundle::new().with_payload(data);

        let bytes = bundle.build();
        let loaded = BundledModel::from_bytes(&bytes).unwrap();

        // Size should match the full bundle size
        prop_assert_eq!(loaded.size(), bytes.len());
    }

    /// Property: Compression flag roundtrip
    #[test]
    fn compression_flag_roundtrip(compressed in any::<bool>()) {
        let data = vec![0u8; 100];
        let bundle = ModelBundle::new()
            .with_payload(data)
            .with_compression(compressed);

        let bytes = bundle.build();
        let loaded = BundledModel::from_bytes(&bytes).unwrap();

        prop_assert_eq!(loaded.is_compressed(), compressed);
    }

    /// Property: Version is always (1, 0) for new bundles
    #[test]
    fn version_is_always_1_0(data_len in 1..1000usize) {
        let data = vec![0u8; data_len];
        let bundle = ModelBundle::new().with_payload(data);

        let bytes = bundle.build();
        let loaded = BundledModel::from_bytes(&bytes).unwrap();

        prop_assert_eq!(loaded.version(), (1, 0));
    }

    /// Property: as_bytes returns the full bundle
    #[test]
    fn as_bytes_returns_bundle(data in prop::collection::vec(any::<u8>(), 0..1000)) {
        let bundle = ModelBundle::new().with_payload(data);

        let bytes = bundle.build();
        let loaded = BundledModel::from_bytes(&bytes).unwrap();

        // as_bytes returns the full bundle including header
        prop_assert_eq!(loaded.as_bytes(), &bytes[..]);
    }
}

#[cfg(test)]
mod edge_cases {
    use super::*;

    #[test]
    fn empty_payload_is_valid() {
        let bundle = ModelBundle::new().with_payload(vec![]);
        let bytes = bundle.build();
        let loaded = BundledModel::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.size(), bytes.len());
    }

    #[test]
    fn maximum_reasonable_payload() {
        // Test with 1MB payload
        let data = vec![0xABu8; 1024 * 1024];
        let bundle = ModelBundle::new().with_payload(data);
        let bytes = bundle.build();
        let loaded = BundledModel::from_bytes(&bytes).unwrap();
        assert_eq!(loaded.size(), bytes.len());
    }

    #[test]
    fn compression_flag_true() {
        let data = vec![0u8; 100];
        let bundle = ModelBundle::new()
            .with_payload(data)
            .with_compression(true);

        let bytes = bundle.build();
        let loaded = BundledModel::from_bytes(&bytes).unwrap();

        assert!(loaded.is_compressed());
    }

    #[test]
    fn compression_flag_false() {
        let data = vec![0u8; 100];
        let bundle = ModelBundle::new()
            .with_payload(data)
            .with_compression(false);

        let bytes = bundle.build();
        let loaded = BundledModel::from_bytes(&bytes).unwrap();

        assert!(!loaded.is_compressed());
    }
}
