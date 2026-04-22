# Model Metadata Inspection

**CLI Equivalent:** `apr inspect model.apr [--verbose] [--json]`

## What This Demonstrates

Inspects an APR model file to extract metadata, architecture details, tensor listing, size breakdown by category, and compression statistics. Essential for understanding model structure before inference or conversion.

## Run

```bash
cargo run --example analysis_inspect
```

## Key APIs

- `ModelBundleV2::new().with_name().add_tensor().build()` -- create a multi-tensor APR v2 bundle
- `inspect_apr(&bytes)` -- parse magic bytes, metadata, tensor directory from raw APR binary
- `size_breakdown(&tensors)` -- categorize tensors into embedding, attention, feed-forward, normalization
- `detect_compression(&bytes)` -- detect LZ4/Zstd compression from magic bytes in payload

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_inspect/main.rs}}
```

## Source

[`examples/analysis/analysis_inspect/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_inspect/main.rs)
