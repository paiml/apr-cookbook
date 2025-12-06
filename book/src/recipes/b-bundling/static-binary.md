# Static Binary Embedding

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Create fully static binaries with embedded models.

## Run Command

```bash
cargo run --example bundle_apr_static_binary
```

## Code

```rust,ignore
{{#include ../../../../examples/bundling/bundle_apr_static_binary.rs}}
```

## Deployment Benefits

- No runtime dependencies
- Works on minimal container images (scratch, distroless)
- Predictable behavior across environments
