# Format-Aware Binary Forensics

**CLI Equivalent:** `apr hex model.apr`

## What This Demonstrates

Hex dump with APR format annotations, parsing magic bytes, version, metadata offsets, and tensor data regions. Produces a classic hex dump view with ASCII representation alongside annotated region labels and a format structure map showing the APR v2 binary layout.

## Run

```bash
cargo run --example analysis_hex
```

## Key APIs

- `annotated_hex_dump(&data, max_bytes)` -- produce `Vec<HexAnnotation>` with labeled format regions
- `parse_format_structure(&data)` -- extract `FormatStructure { magic, version, metadata_offset, tensor_data_offset }`
- `hex_dump_view(&data, max_bytes)` -- classic hex dump with offset, hex, and ASCII columns
- `bytes_to_hex(&data)` -- convert byte slice to space-separated hex string
- `read_u32_le(&data, offset)` -- read little-endian u32 from byte slice

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_hex.rs}}
```

## Source

[`examples/analysis/analysis_hex.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_hex.rs)
