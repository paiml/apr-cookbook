# Rosetta Cross-Format Conversion

**CLI Equivalent:** `apr rosetta convert`

## What This Demonstrates

Uses the Rosetta engine to perform a single-step conversion between any two supported formats (APR, SafeTensors, GGUF, ONNX). Rosetta handles tensor name remapping, dtype coercion, and metadata translation automatically.

## Run

```bash
cargo run --example format_rosetta_convert
```

## Key APIs

- `Rosetta::convert(input, output, config)` — One-shot conversion between formats
- `RosettaConfig::new(source_fmt, target_fmt)` — Specify source and target format pair
- `.with_tensor_map(map)` — Override default tensor name remapping rules
- `FormatDetector::detect(path)` — Auto-detect format from file magic bytes

## Source

[`examples/format/format_rosetta_convert.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_rosetta_convert.rs)
