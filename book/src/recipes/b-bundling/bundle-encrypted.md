# Bundle Encrypted Model

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Protect model weights with encryption before bundling.

## Run Command

```bash
cargo run --example bundle_encrypted_model --features encryption
```

## Code

```rust,ignore
{{#include ../../../../examples/bundling/bundle_encrypted_model.rs}}
```

## Security Considerations

1. **Key Management**: Store decryption keys securely
2. **Runtime Decryption**: Models decrypted in memory only
3. **Obfuscation**: Additional protection against reverse engineering
