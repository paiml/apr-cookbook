//! APR v2 tests (EXTREME TDD — these must pass!).

#![allow(clippy::disallowed_methods)]

use super::*;
use crate::bundle::v1::ModelBundle;

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
        0x9d, 0x61, 0xb1, 0x9d, 0xef, 0xfd, 0x5a, 0x60, 0xba, 0x84, 0x4a, 0xf4, 0x92, 0xec, 0x2c,
        0xc4, 0x44, 0x49, 0xc5, 0x69, 0x7b, 0x32, 0x69, 0x19, 0x70, 0x3b, 0xac, 0x03, 0x1c, 0xae,
        0x7f, 0x60,
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
