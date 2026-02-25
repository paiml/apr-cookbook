# Full Optimization Pipeline

**CLI Equivalent:** `apr finetune ... && apr prune ... && apr distill ... && apr merge ... && apr quantize ...`

## What This Demonstrates

Composes the entire optimization pipeline in a single example: LoRA fine-tuning, magnitude pruning, KL distillation, SLERP merge, and 4-bit quantization applied sequentially to produce a smaller, faster model.

## Run

```bash
cargo run --example optimize_full_pipeline
```

## Key APIs

- `LoRALayer::new(base, d_out, d_in, rank, alpha)` -- fine-tune with low-rank adapters
- `prune_magnitude(tensor, sparsity)` -- remove small weights
- `DistillationLoss::new(temp, alpha).forward(...)` -- transfer knowledge from teacher
- `slerp_merge(&m1, &m2, &SlerpConfig::new(t))` -- interpolate two models
- `Quantization::Int4` -- quantize to 4-bit integers

## Source

[`examples/optimize/full_pipeline.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/full_pipeline.rs)
