# Signed Models

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Cryptographically sign models for integrity verification.

## Run Command

```bash
cargo run --example bundle_apr_signed
```

## Code

```rust,ignore
{{#include ../../../../examples/bundling/bundle_apr_signed.rs}}
```

## Verification Flow

1. Generate keypair
2. Sign model hash
3. Bundle signature with model
4. Verify before loading
