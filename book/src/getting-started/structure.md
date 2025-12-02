# Project Structure

## Library Organization

```
apr-cookbook/
├── src/
│   ├── lib.rs                 # Public API exports
│   ├── bundle.rs              # Model bundling (ModelBundle, BundledModel)
│   ├── convert.rs             # Format conversion (AprConverter)
│   ├── aprender_integration.rs # aprender format integration
│   └── error.rs               # Error types
├── examples/
│   ├── bundling/              # Bundling recipes
│   │   ├── bundle_static_model.rs
│   │   ├── bundle_quantized_model.rs
│   │   └── bundle_encrypted_model.rs
│   ├── conversion/            # Format conversion
│   │   ├── convert_safetensors_to_apr.rs
│   │   ├── convert_apr_to_gguf.rs
│   │   └── convert_gguf_to_apr.rs
│   ├── acceleration/          # Performance
│   │   └── simd_matrix_operations.rs
│   └── cli/                   # Command-line tools
│       ├── apr_info.rs
│       └── apr_bench.rs
└── tests/
    ├── proptest_bundle.rs     # Property tests for bundling
    ├── proptest_convert.rs    # Property tests for conversion
    └── proptest_aprender.rs   # Property tests for integration
```

## Module Overview

### `bundle` - Model Bundling

Core types for creating and loading APR bundles:

- `ModelBundle` - Builder for creating APR files
- `BundledModel` - Zero-copy model loader

### `convert` - Format Conversion

Convert between formats:

- `AprConverter` - Multi-format converter
- `TensorData` - Tensor representation
- `ConversionFormat` - Supported formats (APR, SafeTensors, GGUF)

### `aprender_integration` - Format Integration

Direct integration with aprender's format module:

- `save_model()` / `load_model()` - File-based I/O
- `AprModelInfo` - Model metadata inspection

### `error` - Error Handling

Comprehensive error types:

- `CookbookError` - Main error enum
- `Result<T>` - Convenience type alias
