# Wanda Pruning

**CLI Equivalent:** `apr prune --method wanda --sparsity 0.5 --calibration data.jsonl model.apr`

## What This Demonstrates

Wanda (Weights and Activations) pruning that uses calibration data to determine weight importance. Multiplies weight magnitude by input activation norm to prune weights that contribute least to outputs.

## Run

```bash
cargo run --example prune_wanda
```

## Key APIs

- `prune_wanda(tensor, activations, sparsity)` -- prune using weight * activation importance
- `collect_activations(model, calibration_data)` -- run calibration pass to gather activation norms
- `sparsity_ratio(tensor)` -- verify achieved sparsity

## Source

[`examples/optimize/prune_wanda.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/prune_wanda.rs)
