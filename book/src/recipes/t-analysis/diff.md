# Model Weight Diff

**CLI Equivalent:** `apr diff model_a.apr model_b.apr --weights --values`

## What This Demonstrates

Compares two APR models structurally and numerically. Reports tensor-level weight differences including L2 distance, max absolute diff, mean absolute diff, and cosine similarity. Essential for tracking fine-tuning impact, merge quality, and quantization drift.

## Run

```bash
cargo run --example analysis_diff
```

## Key APIs

- `diff_weights(&weights_a, &weights_b)` -- produce structural changes and per-tensor weight diffs
- `cosine_similarity(&a, &b)` -- compute cosine similarity between two float slices
- `l2_distance(&a, &b)` -- compute Euclidean distance between weight vectors
- `ChangeKind::{Added, Removed, ShapeChanged, Unchanged}` -- structural change classification

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_diff.rs}}
```

## Source

[`examples/analysis/analysis_diff.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_diff.rs)
