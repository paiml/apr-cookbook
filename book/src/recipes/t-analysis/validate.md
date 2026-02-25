# 100-Point Integrity Validation

**CLI Equivalent:** `apr validate model.apr`

## What This Demonstrates

Performs a comprehensive 100-point model validation and integrity check. Each check (magic bytes, minimum size, version, metadata, tensor payload, NaN/Inf detection, compression, alignment, checksum) contributes to a scored pass/fail/warn result for deployment readiness.

## Run

```bash
cargo run --example analysis_validate
```

## Key APIs

- `validate_model(&bytes)` -- run all 10 validation checks, return scored `ValidationResult`
- `ValidationResult::score()` -- compute 0-100 score (pass=100, warn=50, fail=0 per check)
- `check_no_nan(&bytes, &mut result)` -- scan tensor payload for IEEE 754 NaN values
- `check_checksum(&bytes, &mut result)` -- FNV-1a checksum of entire file

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_validate.rs}}
```

## Source

[`examples/analysis/analysis_validate.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_validate.rs)
