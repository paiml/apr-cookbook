# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2024-12-02

### Added
- Core library with bundle and convert modules
- `BundledModel` for loading APR models from bytes
- `ModelBundle` builder for creating APR bundles
- `AprConverter` for format conversion (SafeTensors, GGUF)
- Integration with `aprender` format module
- Property-based testing suite with proptest
- Examples:
  - `bundle_static_model` - Static model embedding
  - `bundle_quantized_model` - Quantized model bundling
  - `bundle_encrypted_model` - AES-256-GCM encryption (requires `encryption` feature)
  - `convert_safetensors_to_apr` - SafeTensors conversion
  - `convert_apr_to_gguf` - GGUF export
  - `convert_gguf_to_apr` - GGUF import
  - `simd_matrix_operations` - SIMD acceleration demo
  - `apr_info` - CLI model inspector
  - `apr_bench` - CLI benchmark tool

### Security
- AES-256-GCM authenticated encryption support
- Argon2id key derivation for password-based encryption

[Unreleased]: https://github.com/paiml/apr-cookbook/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paiml/apr-cookbook/releases/tag/v0.1.0
