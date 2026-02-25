# Canary Regression Testing

**CLI Equivalent:** `apr canary create model.apr` / `apr canary check model.apr`

## What This Demonstrates

Embeds deterministic test vectors (canaries) in a model and verifies outputs match expected values, detecting model drift and weight corruption. Supports tolerance-based drift detection, JSON serialization of canary test vectors, and roundtrip verification.

## Run

```bash
cargo run --example analysis_canary
```

## Key APIs

- `create_canaries(&model_weights, n, tolerance)` -- generate n deterministic canary test vectors from weights
- `check_canaries(&model_weights, &canaries)` -- verify all canaries pass within tolerance
- `compute_probe(&input, &weights)` -- dot-product probe function for expected output computation
- `canaries_to_json(&canaries)` / `canaries_from_json(&json)` -- JSON serialization roundtrip

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_canary.rs}}
```

## Source

[`examples/analysis/analysis_canary.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_canary.rs)
