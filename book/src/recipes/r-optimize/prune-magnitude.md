# Magnitude Pruning

**CLI Equivalent:** `apr prune --method magnitude --sparsity 0.5 model.apr`

## What This Demonstrates

Unstructured magnitude pruning that zeros out weights below a threshold. The simplest pruning method -- removes weights with the smallest absolute values to achieve target sparsity.

## Run

```bash
cargo run --example prune_magnitude
```

## Key APIs

- `prune_magnitude(tensor, sparsity)` -- zero out smallest weights to reach target sparsity
- `sparsity_ratio(tensor)` -- measure fraction of zero weights
- `ModelBundleV2::new().with_quantization(Quantization::FP32)` -- save pruned model

## Source

[`examples/optimize/prune_magnitude.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/prune_magnitude.rs)
