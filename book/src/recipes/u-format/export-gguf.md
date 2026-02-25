# Export to GGUF

**CLI Equivalent:** `apr export --format gguf`

## What This Demonstrates

Converts an `.apr` model to the GGUF format used by llama.cpp and related inference engines, mapping APR tensor layouts and quantization schemes to their GGUF equivalents.

## Run

```bash
cargo run --example format_export_gguf
```

## Key APIs

- `AprModel::load(path)` — Load a native `.apr` model from disk
- `GgufExporter::new(&model)` — Create an exporter targeting GGUF
- `.with_quantization(GgufQuantType::Q4_0)` — Apply GGUF-native quantization during export
- `.export(output_path)` — Write the `.gguf` file with architecture metadata and tensors

## Source

[`examples/format/format_export_gguf.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_export_gguf.rs)
