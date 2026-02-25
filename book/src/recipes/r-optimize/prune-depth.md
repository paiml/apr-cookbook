# Depth Pruning

**CLI Equivalent:** `apr prune --method depth --layers-to-remove 4,5,6 model.apr`

## What This Demonstrates

Depth pruning (Minitron-style) that removes entire transformer layers. Reduces model depth for faster inference while preserving the most important layers based on importance scoring.

## Run

```bash
cargo run --example prune_depth
```

## Key APIs

- `prune_depth(model, layers_to_remove)` -- remove specified layers
- `layer_importance(model)` -- rank layers by angular distance or gradient magnitude
- `reindex_layers(model)` -- renumber remaining layers after removal

## Source

[`examples/optimize/prune_depth.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/prune_depth.rs)
