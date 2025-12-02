# Introduction

**APR Cookbook** provides idiomatic Rust patterns for deploying machine learning models using the APR format. Built on Toyota Way principles, it emphasizes zero-defect quality and production readiness.

## What is APR?

APR (Aprender Portable Runtime) is a native Rust ML model format designed for:

- **Zero-copy loading** - Models load directly from memory without parsing overhead
- **Compile-time embedding** - Use `include_bytes!()` to bundle models in your binary
- **WASM compatibility** - Deploy the same model to browser and server
- **Security** - Optional AES-256-GCM encryption with Argon2id key derivation

## Why APR Cookbook?

| Challenge | Solution |
|-----------|----------|
| Large model files | Quantization (Q4, Q8) reduces size 4-8x |
| Slow cold starts | Zero-copy loading, no deserialization |
| Model theft | AES-256-GCM encryption at rest |
| Format lock-in | Convert from/to SafeTensors, GGUF |
| Platform limits | WASM-ready, no native dependencies |

## The Sovereign Stack

APR Cookbook integrates with the Sovereign AI Stack:

```
┌─────────────────────────────────────────┐
│           Your Application              │
├─────────────────────────────────────────┤
│  apr-cookbook  │  Recipes & patterns    │
├────────────────┼────────────────────────┤
│    aprender    │  ML algorithms         │
├────────────────┼────────────────────────┤
│     trueno     │  SIMD compute          │
├────────────────┼────────────────────────┤
│    entrenar    │  Training & optim      │
└─────────────────────────────────────────┘
```

## Quick Example

```rust
use apr_cookbook::bundle::{BundledModel, ModelBundle};

// Embed model at compile time
const MODEL: &[u8] = include_bytes!("model.apr");

fn main() -> apr_cookbook::Result<()> {
    // Zero-copy load
    let model = BundledModel::from_bytes(MODEL)?;

    println!("Loaded: {} ({} bytes)", model.name(), model.size());
    Ok(())
}
```

## Toyota Way Principles

This cookbook follows Toyota Way quality principles:

1. **Jidoka** - Build quality in, don't inspect it in
2. **Genchi Genbutsu** - Go see for yourself
3. **Kaizen** - Continuous improvement
4. **Muda elimination** - Remove waste (unnecessary copies, allocations)

Every recipe includes tests, benchmarks, and quality metrics.

## Next Steps

- [Installation](./getting-started/installation.md) - Add apr-cookbook to your project
- [Quick Start](./getting-started/quick-start.md) - Bundle your first model
- [Recipes](./recipes/bundle-static.md) - Production-ready patterns
