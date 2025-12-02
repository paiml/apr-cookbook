# Convert from SafeTensors

Import Hugging Face SafeTensors models into APR format.

## Recipe

```rust
use apr_cookbook::convert::{AprConverter, TensorData, DataType, ConversionMetadata};

fn convert_safetensors(path: &str) -> apr_cookbook::Result<Vec<u8>> {
    // In production, use the safetensors crate to read the file
    // Here we demonstrate the conversion pattern

    let mut converter = AprConverter::new();

    converter.set_metadata(ConversionMetadata {
        name: Some("bert-base".to_string()),
        architecture: Some("transformer".to_string()),
        source_format: Some(apr_cookbook::convert::ConversionFormat::SafeTensors),
        ..Default::default()
    });

    // Add each tensor from SafeTensors
    converter.add_tensor(TensorData {
        name: "embeddings.word_embeddings.weight".to_string(),
        shape: vec![30522, 768],
        dtype: DataType::F32,
        data: vec![0u8; 30522 * 768 * 4], // Placeholder
    });

    converter.to_apr()
}
```

## Run the Example

```bash
cargo run --example convert_safetensors_to_apr
```

## Output

```
=== APR Cookbook: SafeTensors → APR Conversion ===

Conversion supported: true

Simulating SafeTensors model...
  Tensors: 5
  Total parameters: 110M

Converting to APR...
  ✓ Token embeddings: [30522, 768]
  ✓ Position embeddings: [512, 768]
  ✓ Attention weights: [768, 768]
  ✓ FFN weights: [768, 3072]
  ✓ Output weights: [768, 30522]

Conversion complete:
  APR size: 440MB
  Tensors: 5
  Parameters: 110M

[SUCCESS] SafeTensors → APR conversion complete!
```

## Why Convert?

| Feature | SafeTensors | APR |
|---------|-------------|-----|
| Zero-copy | ✅ | ✅ |
| Encryption | ❌ | ✅ |
| Compression | ❌ | ✅ |
| include_bytes! | ✅ | ✅ |
| Rust-native | ❌ | ✅ |
