# APR Cookbook

Idiomatic Rust examples for the APR ML format, following Toyota Way principles.

[![Crates.io](https://img.shields.io/crates/v/apr-cookbook.svg)](https://crates.io/crates/apr-cookbook)
[![Documentation](https://docs.rs/apr-cookbook/badge.svg)](https://docs.rs/apr-cookbook)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Features

- **Zero-copy model loading** with `include_bytes!()` pattern
- **Format conversion** between SafeTensors, GGUF, and APR
- **Encryption support** with AES-256-GCM and Argon2id
- **SIMD acceleration** via trueno integration
- **WASM-ready** architecture

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
apr-cookbook = "0.1"
```

For encryption support:

```toml
[dependencies]
apr-cookbook = { version = "0.1", features = ["encryption"] }
```

## Quick Start

### Bundle a Model

```rust
use apr_cookbook::bundle::ModelBundle;

// Create a model bundle
let bundle = ModelBundle::new()
    .with_name("my-model")
    .with_payload(model_bytes)
    .with_compression(true)
    .build();

// Embed at compile time
const MODEL: &[u8] = include_bytes!("model.apr");
```

### Load a Model

```rust
use apr_cookbook::bundle::BundledModel;

let model = BundledModel::from_bytes(MODEL)?;
println!("Model: {} ({} bytes)", model.name(), model.size());
```

### Convert Formats

```rust
use apr_cookbook::convert::{AprConverter, TensorData, DataType};

let mut converter = AprConverter::new();
converter.add_tensor(TensorData {
    name: "weights".to_string(),
    shape: vec![768, 768],
    dtype: DataType::F32,
    data: weights_bytes,
});

let apr_bytes = converter.to_apr()?;
```

## Examples

Run the examples:

```bash
# Model bundling
cargo run --example bundle_static_model
cargo run --example bundle_quantized_model

# Format conversion
cargo run --example convert_safetensors_to_apr
cargo run --example convert_gguf_to_apr

# CLI tools
cargo run --example apr_info -- --demo
cargo run --example apr_bench -- --demo

# Encryption (requires feature)
cargo run --example bundle_encrypted_model --features encryption
```

## Development

```bash
# Run tests
make test-fast

# Run linter
make lint

# Generate coverage report
make coverage

# Full validation
make validate
```

## Architecture

```
apr-cookbook/
├── src/
│   ├── lib.rs              # Public API
│   ├── bundle.rs           # Model bundling
│   ├── convert.rs          # Format conversion
│   ├── aprender_integration.rs  # aprender format integration
│   └── error.rs            # Error types
├── examples/
│   ├── bundling/           # Bundling examples
│   ├── conversion/         # Conversion examples
│   ├── acceleration/       # SIMD examples
│   └── cli/                # CLI tools
└── tests/
    └── proptest_*.rs       # Property-based tests
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions welcome! Please read [CONTRIBUTING.md](CONTRIBUTING.md) first.
