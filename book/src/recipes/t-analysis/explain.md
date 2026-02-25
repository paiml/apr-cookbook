# Error Code Explanation

**CLI Equivalent:** `apr explain E001`

## What This Demonstrates

Provides detailed explanations, causes, and solutions for APR error codes, similar to `rustc --explain`. Implements an error catalog with structured documentation (E001 through E006) covering invalid magic bytes, version mismatch, tensor corruption, size mismatch, unsupported quantization, and decompression failure. Supports case-insensitive lookup and related-error navigation.

## Run

```bash
cargo run --example analysis_explain
```

## Key APIs

- `explain_error(code)` -- look up an error code in the catalog, returns `Option<ErrorCode>`
- `all_error_codes()` -- list all defined error codes
- `format_explanation(&error)` -- render error with title, description, causes, solutions, related codes
- `ErrorCode { code, title, description, causes, solutions, related }` -- structured error documentation

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_explain.rs}}
```

## Source

[`examples/analysis/analysis_explain.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_explain.rs)
