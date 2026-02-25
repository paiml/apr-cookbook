# Architecture Visualization

**CLI Equivalent:** `apr tree model.apr`

## What This Demonstrates

Renders model tensor hierarchy as an ASCII tree with box-drawing characters and parameter counts. Groups flat tensor names (e.g., `layers.0.attn.q_proj.weight`) into a hierarchical tree by splitting on `.` separators, with parameter count aggregation at each level.

## Run

```bash
cargo run --example analysis_tree
```

## Key APIs

- `build_tree(&tensors)` -- construct hierarchical `TreeNode` from flat `(name, shape)` pairs
- `render(&root)` -- render tree as ASCII string with box-drawing characters
- `TreeNode::total_params()` -- recursive parameter count aggregation
- `format_params(n)` -- human-readable parameter count (e.g., `1.5M`, `2.3B`)
- `format_shape(&shape)` -- dimension string with multiplication sign (e.g., `768x768`)

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_tree.rs}}
```

## Source

[`examples/analysis/analysis_tree.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_tree.rs)
