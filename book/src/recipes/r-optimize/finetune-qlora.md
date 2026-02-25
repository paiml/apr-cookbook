# QLoRA Fine-Tuning

**CLI Equivalent:** `apr finetune --method qlora --rank 8 --alpha 16 model.apr`

## What This Demonstrates

Quantized LoRA (QLoRA) fine-tuning that keeps base model weights in 4-bit precision while training full-precision LoRA adapters. Enables fine-tuning of large models on limited VRAM.

## Run

```bash
cargo run --example finetune_qlora
```

## Key APIs

- `LoRAConfig::new(rank, alpha).target_qv_projections()` -- configure adapter dimensions
- `Quantization::Int4` -- quantize base weights to 4-bit
- `LoRALayer::new(quantized_base, d_out, d_in, rank, alpha)` -- attach adapters to quantized base
- `.trainable_params()` -- only adapter weights are trainable
- `MemoryPlanner::estimate_vram(config)` -- predict peak VRAM usage

## Source

[`examples/optimize/finetune_qlora.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/finetune_qlora.rs)
