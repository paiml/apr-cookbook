# Low-Level Debug

Parses raw APR model bytes to extract header fields: magic bytes, version, flags (compressed, signed, encrypted), dtype, and tensor count. Detects format from magic bytes and produces an annotated hex dump.

## CLI Equivalent
```bash
apr debug model.apr
```

## Key Concepts
- Binary header parsing with explicit error handling
- Flag bitmask extraction (compressed, signed, encrypted)
- Format detection from magic bytes (APR2, GGUF, SafeTensors)

## Run
```bash
cargo run --example analysis_debug
```

## Source
[`examples/analysis/analysis_debug.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_debug.rs)
