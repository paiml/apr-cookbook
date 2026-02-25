# Batch Multi-Format Export

**CLI Equivalent:** `apr export --batch`

## What This Demonstrates

Exports a single `.apr` model to multiple target formats (SafeTensors, GGUF, ONNX) in one pass, reading the source tensors once and writing all outputs in parallel. This is significantly faster than running separate export commands.

## Run

```bash
cargo run --example format_batch_export
```

## Key APIs

- `BatchExporter::new(model_path)` — Create a batch exporter from a source `.apr` model
- `.add_target(format, output_path)` — Register a target format and output location
- `.with_shared_quantization(quant)` — Apply the same quantization to all targets
- `.export_all()` — Execute all exports in parallel, returning a summary of each result

## Source

[`examples/format/format_batch_export.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_batch_export.rs)
