# Plan VRAM

**CLI Equivalent:** `apr finetune --plan --rank 8 --method lora model.apr`

## What This Demonstrates

VRAM estimation and memory planning for fine-tuning jobs before committing GPU resources. Calculates peak memory for base weights, adapters, optimizer states, activations, and gradients.

## Run

```bash
cargo run --example finetune_plan_vram
```

## Key APIs

- `MemoryPlanner::new(model_config)` -- create planner for a given model size
- `.estimate_vram(lora_config)` -- calculate peak VRAM in bytes
- `.plan(lora_config)` -- detailed breakdown: base, adapter, optimizer, activations
- `entrenar_lora::plan(model_path, rank, method)` -- one-shot planning from CLI

## Source

[`examples/optimize/finetune_plan_vram/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/finetune_plan_vram/main.rs)
