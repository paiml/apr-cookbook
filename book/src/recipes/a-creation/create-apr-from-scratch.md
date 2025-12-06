# Create APR from Scratch

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Build a minimal APR model programmatically without external frameworks.

## Run Command

```bash
cargo run --example create_apr_from_scratch
```

## Code

```rust,ignore
{{#include ../../../../examples/creation/create_apr_from_scratch.rs}}
```

## Key Concepts

1. **Model Structure**: APR models consist of named tensors with typed data
2. **Deterministic Seeds**: Use `hash_name_to_seed()` for reproducible random initialization
3. **Zero-Copy Serialization**: APR format supports memory-mapped loading

## Output

```
=== Recipe: create_apr_from_scratch ===
Model created with 2 layers
Total parameters: 1,024
File size: 4,112 bytes
```
