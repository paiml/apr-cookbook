# Encrypt a Model

Protect proprietary models with AES-256-GCM encryption.

## Security Features

- **AES-256-GCM** - Authenticated encryption (AEAD)
- **Argon2id** - Memory-hard key derivation (resists GPU attacks)
- **Random nonce** - Unique per encryption (prevents IV reuse)

## Recipe

```rust
use aprender::format::{
    save_encrypted, load_encrypted, load_from_bytes_encrypted,
    ModelType, SaveOptions
};

// Save encrypted
save_encrypted(
    &model,
    ModelType::Custom,
    "model.apr.enc",
    SaveOptions::default().with_name("proprietary-model"),
    "secure_password_123",
)?;

// Load encrypted
let model: MyModel = load_encrypted(
    "model.apr.enc",
    ModelType::Custom,
    "secure_password_123"
)?;

// Load from embedded bytes
const ENCRYPTED: &[u8] = include_bytes!("model.apr.enc");
let model: MyModel = load_from_bytes_encrypted(
    ENCRYPTED,
    ModelType::Custom,
    password
)?;
```

## Run the Example

```bash
cargo run --example bundle_encrypted_model --features encryption
```

## Output

```
=== APR Cookbook: Encrypted Model Bundling ===

Created sentiment classifier:
  Vocabulary size: 1000
  Embedding dimension: 64

Saving encrypted model...
File sizes:
  Unencrypted: 256128 bytes
  Encrypted:   256176 bytes (+48 bytes overhead)

Loading encrypted model with correct password...
  ✓ Model loaded successfully
  ✓ Decryption verified

Testing wrong password...
  ✓ Correctly rejected wrong password

[SUCCESS] Encrypted model demonstration complete!
```

## Best Practices

1. **Use strong passwords** - 16+ characters, mixed case/numbers/symbols
2. **Rotate keys** - Re-encrypt periodically
3. **Secure key storage** - Use environment variables or secret managers
4. **Never hardcode passwords** - Pass at runtime

## Threat Model

| Threat | Protection |
|--------|------------|
| Disk theft | Encrypted at rest |
| Memory dump | Decrypted only when needed |
| Brute force | Argon2id slows attacks |
| Tampering | GCM authentication tag |
