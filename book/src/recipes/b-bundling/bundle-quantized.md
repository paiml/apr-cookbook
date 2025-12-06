# Bundle Quantized Model

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Reduce model size by quantizing weights before bundling.

## Run Command

```bash
cargo run --example bundle_quantized_model
```

## Code

```rust,ignore
{{#include ../../../../examples/bundling/bundle_quantized_model.rs}}
```

## Size Comparison

| Precision | Size | Accuracy Impact |
|-----------|------|-----------------|
| FP32 | 100% | Baseline |
| FP16 | 50% | Negligible |
| INT8 | 25% | <1% loss |
| Q4 | 12.5% | 1-2% loss |
