# apr-decrypt

Decrypt model weights encrypted with `apr encrypt`. Uses BLAKE3-derived keystream with MAC verification. Demonstrates the full encrypt/decrypt roundtrip and wrong-password rejection.

```bash
cargo run --example cli_apr_decrypt -- --demo
```

**CLI equivalent**: `apr decrypt model.apr.enc -p my-secret -o model.apr`
