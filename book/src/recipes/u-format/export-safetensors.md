# Export to SafeTensors

**CLI Equivalent:** `apr export --format safetensors`

## What This Demonstrates

Serializes an `.apr` model into the SafeTensors format, preserving tensor names and dtypes for interoperability with the HuggingFace ecosystem and Python inference frameworks.

## Run

```bash
cargo run --example format_export_safetensors
```

## Key APIs

- `AprModel::load(path)` — Load a native `.apr` model from disk
- `SafeTensorsExporter::new(&model)` — Create an exporter targeting SafeTensors
- `.export(output_path)` — Write the `.safetensors` file with all tensors and metadata
- `.with_metadata(map)` — Attach additional key-value metadata to the output header

## Source

[`examples/format/format_export_safetensors.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_export_safetensors.rs)
