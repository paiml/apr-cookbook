# Q4 Quantization

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Apply 4-bit quantization for maximum size reduction.

## Run Command

```bash
cargo run --example bundle_apr_quantized_q4
```

## Code

```rust,ignore
{{#include ../../../../examples/bundling/bundle_apr_quantized_q4.rs}}
```

## Q4 Format

- 4 bits per weight value
- Block-wise scaling factors
- 8x size reduction from FP32
