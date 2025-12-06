# Incremental Training

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Add new training data to an existing model without full retraining.

## Run Command

```bash
cargo run --example continuous_train_incremental
```

## Code

```rust,ignore
{{#include ../../../../examples/training/continuous_train_incremental.rs}}
```
