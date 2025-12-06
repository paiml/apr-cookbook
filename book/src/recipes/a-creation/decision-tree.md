# Decision Tree Model

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Build a decision tree classifier stored in APR format.

## Run Command

```bash
cargo run --example create_apr_decision_tree
```

## Code

```rust,ignore
{{#include ../../../../examples/creation/create_apr_decision_tree.rs}}
```

## Key Concepts

1. **Node Structure**: Each node contains split feature, threshold, and child indices
2. **Leaf Nodes**: Store class predictions
3. **Serialization**: Tree structure encoded as flat arrays for efficient storage
