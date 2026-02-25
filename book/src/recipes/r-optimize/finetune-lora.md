# LoRA Fine-Tuning

**CLI Equivalent:** `apr finetune --method lora --rank 8 --alpha 16 model.apr`

## What This Demonstrates

Low-Rank Adaptation (LoRA) fine-tuning that freezes the base model and trains small rank-decomposed adapter matrices. Dramatically reduces trainable parameter count while preserving model quality.

## Run

```bash
cargo run --example finetune_lora
```

## Key APIs

- `LoRAConfig::new(rank, alpha).target_qv_projections()` -- configure LoRA rank and target layers
- `LoRALayer::new(base_tensor, d_out, d_in, rank, alpha)` -- create trainable adapter
- `.trainable_params()` -- get only the adapter parameters for optimization
- `AdamW::default_params(lr)` -- optimizer for adapter training
- `.merge()` -- fold adapter weights into base model

## Source

[`examples/optimize/finetune_lora.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/finetune_lora.rs)
