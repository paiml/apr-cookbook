# Fake Quantization-Aware Training

**CLI Equivalent:** `apr quantize --method qat --bits 4 --fake model.apr`

## What This Demonstrates

Fake quantization-aware training (QAT) that simulates quantization error during training so the model learns to be robust to it. Inserts fake-quantize/dequantize operations in the forward pass while keeping full-precision weights for gradient computation.

## Run

```bash
cargo run --example quantize_fake_qat
```

## Key APIs

- `FakeQuantize::new(bits, per_channel)` -- create fake quantization module
- `.forward(tensor)` -- quantize then immediately dequantize (simulates error)
- `.observer()` -- track min/max ranges for calibration
- `convert_fake_to_real(model)` -- replace fake-quant ops with actual Int4 quantization

## Source

[`examples/optimize/quantize_fake_qat.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/quantize_fake_qat.rs)
