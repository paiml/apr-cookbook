# Export to GGUF

Export APR models to GGUF format for use with llama.cpp ecosystem.

## Recipe

```rust
use apr_cookbook::convert::{AprConverter, ConversionFormat};

fn export_to_gguf(apr_bytes: &[u8]) -> apr_cookbook::Result<Vec<u8>> {
    assert!(AprConverter::is_conversion_supported(
        ConversionFormat::Apr,
        ConversionFormat::Gguf
    ));

    // Load APR model
    let model = apr_cookbook::bundle::BundledModel::from_bytes(apr_bytes)?;

    // Export to GGUF format
    // Implementation depends on model architecture
    todo!("GGUF export implementation")
}
```

## Run the Example

```bash
cargo run --example convert_apr_to_gguf
```

## Use Cases

- Run APR-trained models in Ollama
- Use with llama.cpp for CPU inference
- Share models with GGUF ecosystem
