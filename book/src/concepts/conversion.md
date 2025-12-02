# Format Conversion

Convert models between APR, SafeTensors, and GGUF formats.

## Supported Conversions

| From | To | Supported |
|------|-----|-----------|
| SafeTensors | APR | ✅ |
| GGUF | APR | ✅ |
| APR | GGUF | ✅ |
| APR | SafeTensors | ✅ |

## Using AprConverter

```rust
use apr_cookbook::convert::{AprConverter, TensorData, DataType, ConversionMetadata};

// Create converter
let mut converter = AprConverter::new();

// Set metadata
converter.set_metadata(ConversionMetadata {
    name: Some("my-model".to_string()),
    architecture: Some("transformer".to_string()),
    ..Default::default()
});

// Add tensors
converter.add_tensor(TensorData {
    name: "embed.weight".to_string(),
    shape: vec![32000, 4096],
    dtype: DataType::F16,
    data: embedding_bytes,
});

// Generate APR
let apr_bytes = converter.to_apr()?;
```

## Data Types

| Type | Size | Use Case |
|------|------|----------|
| `F32` | 4 bytes | Full precision |
| `F16` | 2 bytes | Half precision |
| `BF16` | 2 bytes | Brain float |
| `Q8_0` | 1 byte | 8-bit quantized |
| `Q4_0` | 0.5 byte | 4-bit quantized |

## Checking Support

```rust
use apr_cookbook::convert::{AprConverter, ConversionFormat};

let supported = AprConverter::is_conversion_supported(
    ConversionFormat::Gguf,
    ConversionFormat::Apr
);
assert!(supported);
```

## Format Detection

```rust
use apr_cookbook::convert::ConversionFormat;

let format = ConversionFormat::from_extension("safetensors");
assert_eq!(format, Some(ConversionFormat::SafeTensors));

let format = ConversionFormat::from_path("model.gguf");
assert_eq!(format, Some(ConversionFormat::Gguf));
```
