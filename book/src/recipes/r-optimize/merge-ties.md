# TIES Merge

**CLI Equivalent:** `apr merge --method ties --density 0.5 --base base.apr model1.apr model2.apr -o merged.apr`

## What This Demonstrates

TIES (Trim, Elect Sign, and Merge) that resolves sign conflicts between task vectors before merging. Uses a density parameter to retain only the top-k most significant parameter changes relative to the base model.

## Run

```bash
cargo run --example merge_ties
```

## Key APIs

- `ties_merge(&models, &base, &TiesConfig::new(density))` -- TIES merge with conflict resolution
- `TiesConfig::new(density)` -- fraction of parameters to retain (0.0-1.0)

## Source

[`examples/optimize/merge_ties.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/merge_ties.rs)
