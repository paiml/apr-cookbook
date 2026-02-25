# Average Merge

**CLI Equivalent:** `apr merge --method average model1.apr model2.apr model3.apr -o merged.apr`

## What This Demonstrates

Uniform average merging of multiple model weight tensors. The simplest merge strategy -- takes the element-wise mean across all models. Works well when models are fine-tuned from the same base.

## Run

```bash
cargo run --example merge_average
```

## Key APIs

- `average_merge(&[models])` -- element-wise mean of all model weights
- `ModelBundleV2::new().add_tensor(name, shape, merged_bytes)` -- save merged model

## Source

[`examples/optimize/merge_average.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/merge_average.rs)
