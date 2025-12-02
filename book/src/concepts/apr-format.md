# The APR Format

APR (Aprender Portable Runtime) is a binary format optimized for ML model deployment.

## Design Goals

1. **Zero-copy loading** - No parsing, direct memory access
2. **Compile-time embedding** - Works with `include_bytes!()`
3. **Cross-platform** - Native, WASM, embedded
4. **Security** - Optional encryption and signing

## File Structure

```
┌────────────────────────────────────────┐
│  Magic (4 bytes): "APRN"               │
├────────────────────────────────────────┤
│  Version (2 bytes): major.minor       │
├────────────────────────────────────────┤
│  Flags (2 bytes): compression, etc.   │
├────────────────────────────────────────┤
│  Header length (4 bytes)              │
├────────────────────────────────────────┤
│  Payload length (8 bytes)             │
├────────────────────────────────────────┤
│  Metadata (variable)                  │
│  - Name (null-terminated string)      │
│  - Description (optional)             │
│  - Custom fields                      │
├────────────────────────────────────────┤
│  Payload (variable)                   │
│  - Tensor data                        │
│  - Model weights                      │
│  - Optionally compressed (zstd)       │
└────────────────────────────────────────┘
```

## Flags

| Bit | Name | Description |
|-----|------|-------------|
| 0 | Compressed | Payload is zstd compressed |
| 1 | Encrypted | Payload is AES-256-GCM encrypted |
| 2 | Signed | Ed25519 signature present |
| 3-15 | Reserved | Future use |

## Version History

| Version | Features |
|---------|----------|
| 1.0 | Initial release, basic bundling |
| 1.1 | Compression support (zstd) |
| 1.2 | Encryption (AES-256-GCM) |

## Comparison with Other Formats

| Feature | APR | SafeTensors | GGUF | ONNX |
|---------|-----|-------------|------|------|
| Zero-copy | ✅ | ✅ | ❌ | ❌ |
| Rust-native | ✅ | ❌ | ❌ | ❌ |
| WASM support | ✅ | ✅ | ❌ | ❌ |
| Encryption | ✅ | ❌ | ❌ | ❌ |
| Quantization | ✅ | ❌ | ✅ | ✅ |
