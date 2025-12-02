# Convert from GGUF

Import llama.cpp GGUF models into APR format.

## Why Import GGUF?

GGUF is the de-facto standard for quantized LLMs:
- Thousands of models on Hugging Face
- Ollama model library
- TheBloke quantizations

Converting to APR enables:
- Pure Rust inference (no C++ deps)
- WASM deployment
- Native trueno SIMD acceleration

## Recipe

```rust
use apr_cookbook::convert::{AprConverter, TensorData, DataType, ConversionMetadata, ConversionFormat};

fn main() -> apr_cookbook::Result<()> {
    // Verify conversion is supported
    assert!(AprConverter::is_conversion_supported(
        ConversionFormat::Gguf,
        ConversionFormat::Apr
    ));

    let mut converter = AprConverter::new();

    converter.set_metadata(ConversionMetadata {
        name: Some("llama-7b-q4".to_string()),
        architecture: Some("llama".to_string()),
        source_format: Some(ConversionFormat::Gguf),
        ..Default::default()
    });

    // Add tensors from GGUF
    // In production, parse GGUF header and extract tensors
    converter.add_tensor(TensorData {
        name: "token_embd.weight".to_string(),
        shape: vec![32000, 4096],
        dtype: DataType::Q8_0,
        data: vec![0u8; 32000 * 4096],
    });

    let apr_bytes = converter.to_apr()?;
    println!("Converted: {} bytes", apr_bytes.len());

    Ok(())
}
```

## Run the Example

```bash
cargo run --example convert_gguf_to_apr
```

## GGML Type Mapping

| GGML Type | APR Type |
|-----------|----------|
| F32 | F32 |
| F16 | F16 |
| Q4_0 | Q4_0 |
| Q4_1 | Q4_0 |
| Q8_0 | Q8_0 |
