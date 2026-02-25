# Merge Adapter

**CLI Equivalent:** `apr finetune --merge-adapter model.apr adapter.apr`

## What This Demonstrates

Merging and unmerging LoRA adapters with base model weights. Merge folds adapter matrices into the base for zero-overhead inference; unmerge restores the original base for further fine-tuning or adapter swapping.

## Run

```bash
cargo run --example finetune_merge_adapter
```

## Key APIs

- `LoRALayer::new(base, d_out, d_in, rank, alpha)` -- create adapter layer
- `.merge()` -- fold adapter into base weights (W' = W + BA)
- `.unmerge()` -- restore original base weights
- `MergeEngine::merge_and_save(base, adapter, path)` -- merge and serialize to .apr

## Source

[`examples/optimize/finetune_merge_adapter.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/finetune_merge_adapter.rs)
