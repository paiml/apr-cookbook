# DARE Merge

**CLI Equivalent:** `apr merge --method dare --drop-prob 0.9 --base base.apr model1.apr model2.apr -o merged.apr`

## What This Demonstrates

DARE (Drop And REscale) merge that randomly drops a fraction of delta parameters and rescales the survivors. Reduces interference between task vectors by sparsifying the parameter deltas before merging.

## Run

```bash
cargo run --example merge_dare
```

## Key APIs

- `dare_merge(&models, &base, &DareConfig::new(drop_prob))` -- DARE merge with random dropout
- `DareConfig::new(drop_prob)` -- probability of dropping each delta parameter (0.0-1.0)

## Source

[`examples/optimize/merge_dare.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/merge_dare.rs)
