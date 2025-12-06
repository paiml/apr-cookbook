# APR to GGUF

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Export APR models to GGUF format for llama.cpp.

## Run Command

```bash
cargo run --example convert_apr_to_gguf
```

## Code

```rust,ignore
{{#include ../../../../examples/conversion/convert_apr_to_gguf.rs}}
```
