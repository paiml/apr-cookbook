# Bundle Static Model

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Embed an APR model directly into your Rust binary using `include_bytes!()`.

## Run Command

```bash
cargo run --example bundle_static_model
```

## Code

```rust,ignore
{{#include ../../../../examples/bundling/bundle_static_model.rs}}
```

## Key Concepts

1. **Compile-Time Embedding**: Model bytes become part of the binary
2. **Zero Runtime I/O**: No file system access needed at runtime
3. **Single Binary**: Complete application with model in one file
