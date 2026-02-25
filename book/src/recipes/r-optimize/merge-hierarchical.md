# Hierarchical Merge

**CLI Equivalent:** `apr merge --method hierarchical --config merge_tree.toml -o merged.apr`

## What This Demonstrates

Multi-model hierarchical merge that applies different merge strategies at each level of a merge tree. For example, SLERP-merge domain-specific pairs first, then TIES-merge the results into a final generalist model.

## Run

```bash
cargo run --example merge_hierarchical
```

## Key APIs

- `MergeTree::new()` -- define a hierarchical merge plan
- `.add_level(strategy, &[model_pairs])` -- add a merge stage with its strategy
- `.execute()` -- run the full merge tree bottom-up
- `slerp_merge(...)`, `ties_merge(...)` -- composable strategies at each level

## Source

[`examples/optimize/merge_hierarchical.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/merge_hierarchical.rs)
