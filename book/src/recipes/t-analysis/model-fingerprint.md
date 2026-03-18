# Model Fingerprint

Content-addressable model hashing with blake3 and ed25519 digital signatures. Computes per-tensor and whole-model fingerprints, signs with Ed25519, verifies signatures, and detects single-tensor tampering.

**Device**: ![cpu](https://img.shields.io/badge/-cpu-lightgrey)

```bash
cargo run --example analysis_model_fingerprint
```

**Key concepts**: blake3 hashing, ed25519 signing/verification, per-tensor checksums, tamper detection, provenance chain.
