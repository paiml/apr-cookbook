# Weighted Merge

**CLI Equivalent:** `apr merge --method weighted --weights 0.6,0.3,0.1 model1.apr model2.apr model3.apr -o merged.apr`

## What This Demonstrates

Weighted average merging where each model contributes proportionally to its assigned weight. Allows emphasizing higher-quality or task-specific models in the final blend.

## Run

```bash
cargo run --example merge_weighted
```

## Key APIs

- `weighted_merge(&[models], &[weights])` -- weighted element-wise average
- `normalize_weights(&weights)` -- ensure weights sum to 1.0
- `ModelBundleV2::new().add_tensor(name, shape, merged_bytes)` -- save merged model

## Source

[`examples/optimize/merge_weighted.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/merge_weighted.rs)
