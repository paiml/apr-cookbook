# 4-bit Quantization

**CLI Equivalent:** `apr quantize --bits 4 model.apr -o model_q4.apr`

## What This Demonstrates

Int4 weight quantization that reduces model size by 4x (FP32 to Int4) with minimal accuracy loss. Quantizes weight tensors to 4-bit integers with per-group scaling factors.

## Run

```bash
cargo run --example quantize_4bit
```

## Key APIs

- `Quantization::Int4` -- select 4-bit quantization mode
- `quantize_tensor(tensor, Quantization::Int4)` -- quantize a single tensor
- `ModelBundleV2::new().with_quantization(Quantization::Int4)` -- build quantized .apr bundle
- `dequantize_tensor(qtensor)` -- reconstruct approximate FP32 values

## Source

[`examples/optimize/quantize_4bit.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/quantize_4bit.rs)
