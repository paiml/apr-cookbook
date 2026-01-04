<div align="center">

# APR Cookbook

> Production recipes for ML model deployment — bundling, conversion, and acceleration in pure Rust

[![CI](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml)
[![Falsification](https://github.com/paiml/apr-cookbook/actions/workflows/falsification.yml/badge.svg)](https://github.com/paiml/apr-cookbook/actions/workflows/falsification.yml)
[![Book](https://github.com/paiml/apr-cookbook/actions/workflows/book.yml/badge.svg)](https://github.com/paiml/apr-cookbook/actions/workflows/book.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Coverage](https://img.shields.io/badge/coverage-95%25-brightgreen.svg)](https://github.com/paiml/apr-cookbook)
[![Book](https://img.shields.io/badge/book-online-blue)](https://paiml.github.io/apr-cookbook/)

</div>

![](.github/apr-cookbook-hero.svg)

[![APR Format Vision](https://img.youtube.com/vi/MoQ-kiOm57Q/maxresdefault.jpg)](https://www.youtube.com/live/MoQ-kiOm57Q?si=KGhHRY42YDV-_iRB)
> **Watch:** Vision for the `.apr` ML model format

## Table of Contents

- [Overview](#overview)
- [APR v2 Format](#apr-v2-format)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Recipe Categories](#recipe-categories)
- [Falsification Testing](#falsification-testing)
- [Examples](#examples)
- [Usage](#usage)
- [Contributing](#contributing)
- [Documentation](#documentation)
- [Development](#development)
- [License](#license)

## Overview

APR Cookbook provides **60+ production-ready recipes** across 15 categories for deploying ML models using the APR v2 format. Each recipe is isolated, idempotent, and verified with Popperian falsification tests following Toyota Way quality principles.

### Key Capabilities

- **APR v2 Format**: LZ4/ZSTD compression (3-13 GB/s), Int4/Int8 quantization, Ed25519 signatures
- **Zero-Copy Loading**: Embed models with `include_bytes!()` for <1ms startup
- **Format Conversion**: SafeTensors, GGUF, ONNX to/from APR v2
- **Speech Recognition**: whisper.apr integration for pure Rust ASR
- **GPU Acceleration**: FlashAttention, fused kernels via realizar
- **Distributed Computing**: Multi-node inference with repartir work-stealing
- **Training**: Autograd with entrenar (tape-based autodiff, SGD/Adam)

### Sovereign AI Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                   APR Cookbook v2.0                             │
├─────────────────────────────────────────────────────────────────┤
│  aprender 0.21 (APR v2, LZ4/ZSTD, Int4/Int8 quantization)      │
│  trueno 0.11 (SIMD/GPU, AVX-512/NEON, LZ4 tensors)             │
│  realizar 0.4 (FlashAttention, Q4K/Q5K/Q6K kernels)            │
│  whisper-apr 0.1 (WASM-first ASR, streaming)                   │
│  repartir 1.1 (distributed compute, work-stealing)             │
│  entrenar 0.3 (autograd, LoRA/QLoRA, model merge)              │
└─────────────────────────────────────────────────────────────────┘
```

### Live Demos

See APR models in action with GPU/SIMD-accelerated WebAssembly:

| Demo | Description | Link |
|------|-------------|------|
| 📈 **Monte Carlo S&P 500** | GPU/SIMD-accelerated portfolio simulation with 100K+ paths/sec. Real-time risk metrics (VaR, CVaR, Sharpe) using `.apr` model for historical data. | [Launch →](https://interactive.paiml.com/monte-carlo-sp500/) |
| 🐚 **Shell ML Autocomplete** | N-gram Markov Model shell autocomplete. Statistical language model bundled as `.apr` for instant browser loading. | [Launch →](https://interactive.paiml.com/shell-ml/) |
| 🎮 **Pong** | Classic Pong game powered by [jugar](https://github.com/paiml/jugar) game engine with `.apr` model integration for WASM deployment. | [Launch →](https://interactive.paiml.com/pong/index.html) |

## APR v2 Format

The APR v2 format introduces significant improvements over v1:

| Feature | APR v1 | APR v2 |
|---------|--------|--------|
| Tensor Index | JSON | Binary (O(1) lookup) |
| Compression | None/Gzip | LZ4/ZSTD (3-13 GB/s) |
| Zero-Copy Loading | Partial | Full (mmap) |
| Quantization | Int8 | Int4/Int8/FP16 |
| Streaming | No | Yes |
| Signature | Optional | Ed25519 default |

### Compression Throughput

| Algorithm | Decompression | Use Case |
|-----------|---------------|----------|
| LZ4 | 3+ GB/s | General purpose, fast loading |
| ZSTD | 1-2 GB/s | Better ratio, storage optimization |

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
apr-cookbook = "0.1"
```

With optional features:

```toml
[dependencies]
apr-cookbook = { version = "0.1", features = ["encryption", "full"] }
```

## Quick Start

```rust
use apr_cookbook::prelude::*;

// Bundle a model at compile time with APR v2 compression
const MODEL: &[u8] = include_bytes!("model.apr");

fn main() -> Result<()> {
    // Zero-copy load (<1ms for 100MB models)
    let model = BundledModel::from_bytes(MODEL)?;
    println!("Loaded: {} ({} bytes)", model.name(), model.size());

    // Run inference
    let output = model.infer(&input)?;
    Ok(())
}
```

## Recipe Categories

```
┌─────────────────────────────────────────────────────────────────┐
│                    APR Cookbook (60+ Recipes)                   │
├─────────────────────────────────────────────────────────────────┤
│  A: Creation (5)      │  B: Bundling (7)    │  C: Training (5)  │
│  D: Conversion (5)    │  E: Registry (4)    │  F: API (4)       │
│  G: Serverless (4)    │  H: WASM (5)        │  I: GPU (5)       │
│  J: SIMD (5)          │  K: Distillation (4)│  L: CLI (4)       │
│  M: Monitoring (2)    │  N: Speech (2)      │  O: Distributed(1)│
├─────────────────────────────────────────────────────────────────┤
│  Falsification Tests: F1-F7 (Popperian methodology)            │
│  Test Coverage: 95%+ │ Quality Gate: Zero violations           │
└─────────────────────────────────────────────────────────────────┘
```

### Category Overview

| Category | Recipes | Description |
|----------|---------|-------------|
| **A: Model Creation** | 5 | Build models from scratch: regression, trees, clustering, n-grams |
| **B: Binary Bundling** | 7 | Embed models: static, quantized, encrypted, signed, Lambda |
| **C: Continuous Training** | 5 | Incremental, online, federated, curriculum, **entrenar autograd** |
| **D: Format Conversion** | 5 | SafeTensors, GGUF, ONNX, Phi model conversion |
| **E: Model Registry** | 4 | Register, lineage, comparison, rollback |
| **F: API Integration** | 4 | Inference, streaming, batch, health checks |
| **G: Serverless** | 4 | Lambda, cold start, edge functions, containers |
| **H: WASM/Browser** | 5 | Browser inference, workers, progressive loading, WebGPU |
| **I: GPU Acceleration** | 5 | **FlashAttention**, CUDA, tensor cores, multi-GPU, memory |
| **J: SIMD Acceleration** | 5 | **trueno ops**, matrix ops, vectorized, quantized SIMD |
| **K: Model Distillation** | 4 | Knowledge transfer, layer matching, pruning-aware |
| **L: CLI Tools** | 4 | apr-info, apr-bench, apr-convert, apr-serve |
| **M: Monitoring** | 2 | Inference explainability, hash chain audit |
| **N: Speech Recognition** | 2 | **whisper.apr** transcription, streaming ASR |
| **O: Distributed** | 1 | **repartir** multi-node inference |

## Falsification Testing

Following Karl Popper's criterion of demarcation, every performance claim is testable and refutable:

| Claim | Metric | Threshold | Test |
|-------|--------|-----------|------|
| **F1** | LZ4 decompression | ≥3 GB/s | `cargo test --test falsification f1_` |
| **F2** | Zero-copy loading | <1ms p95 | `cargo test --test falsification f2_` |
| **F3** | Int4 quantization | <2% accuracy loss | `cargo test --test falsification f3_` |
| **F4** | AES-256-GCM | <5ms for 100MB | `cargo test --test falsification f4_` |
| **F5** | Speech WER | <10% | `cargo test --test falsification f5_` |
| **F6** | FlashAttention | ≥2x speedup | `cargo test --test falsification f6_` |
| **F7** | AVX-512 matmul | ≥80 GFLOPS | `cargo test --test falsification f7_` |

Run the complete falsification suite:

```bash
cargo test --test falsification --release -- --nocapture
```

## Examples

```bash
# Category A: Model Creation
cargo run --example create_apr_from_scratch
cargo run --example create_apr_linear_regression

# Category B: Binary Bundling
cargo run --example bundle_static_model
cargo run --example bundle_quantized_model
cargo run --example bundle_encrypted_model --features encryption

# Category C: Training (NEW)
cargo run --example entrenar_autograd_training

# Category D: Format Conversion
cargo run --example convert_safetensors_to_apr
cargo run --example convert_gguf_to_apr

# Category I: GPU Acceleration (NEW)
cargo run --example flash_attention_inference

# Category J: SIMD (NEW)
cargo run --example trueno_simd_ops

# Category N: Speech Recognition (NEW)
cargo run --example whisper_transcribe
cargo run --example whisper_streaming

# Category O: Distributed (NEW)
cargo run --example distributed_inference

# Category L: CLI Tools
cargo run --example cli_apr_info -- --demo
cargo run --example cli_apr_bench -- --demo
```

## Usage

### Basic Model Loading

```rust
use apr_cookbook::prelude::*;

fn main() -> Result<()> {
    // Load model from file
    let model = BundledModel::from_file("model.apr")?;

    // Or embed at compile time with APR v2 compression
    const EMBEDDED: &[u8] = include_bytes!("model.apr");
    let model = BundledModel::from_bytes(EMBEDDED)?;

    Ok(())
}
```

### APR v2 with Compression

```rust
use apr_cookbook::prelude::*;

fn main() -> Result<()> {
    // Create APR v2 bundle with LZ4 compression
    let bundle = ModelBundleV2::new()
        .with_name("my-model")
        .with_compression(Compression::Lz4)
        .with_quantization(Quantization::Int8)
        .with_payload(weights)
        .build();

    // Save to file
    std::fs::write("model.apr", bundle)?;

    Ok(())
}
```

### Training with entrenar

```rust
use entrenar::{Tensor, optim::SGD, Optimizer};

fn main() -> Result<()> {
    // Create tensors with gradients
    let weights = Tensor::from_vec(vec![0.0; 768], true);

    // SGD optimizer with momentum
    let mut optimizer = SGD::new(0.01, 0.9);

    // Training loop
    optimizer.zero_grad(&mut [weights.clone()]);
    // ... forward pass, compute loss, backward ...
    optimizer.step(&mut [weights]);

    Ok(())
}
```

## Contributing

Contributions are welcome. Please follow these guidelines:

1. **Fork and branch**: Create a feature branch from `main`
2. **Follow the recipe pattern**: Each example must include the QA checklist header
3. **Add falsification tests**: New claims must be testable and refutable
4. **Add tests**: Include unit tests and property-based tests (proptest)
5. **Run quality checks**: Ensure `cargo clippy` and `cargo fmt` pass
6. **Submit PR**: Open a pull request with a clear description

### Code Standards

- All examples must pass the 10-point QA checklist
- **Minimum 95% test coverage** for new code
- No `unwrap()` in production code paths
- Property-based tests for core functionality
- Falsifiable claims with F-codes

### Commit Messages

Use conventional commit format:
```
feat: Add new recipe for X
fix: Resolve issue with Y
docs: Update README with Z
test: Improve coverage for W
```

## Documentation

- **[The APR Cookbook](https://paiml.github.io/apr-cookbook/)** — Online book with all 60+ recipes
- **[API Documentation](https://docs.rs/apr-cookbook)** — Rust API reference
- **[Sovereign AI Stack](https://paiml.github.io/sovereign-ai-stack-book/)** — Complete stack tutorial
- **[Specification](docs/specifications/cookbook-spec.md)** — APR v2 format specification

## Design Principles

This cookbook applies Toyota Production System principles to ML deployment:

| Principle | Application |
|-----------|-------------|
| **Jidoka** | Built-in quality via Rust type system and Popperian falsification |
| **Muda** | Zero-dependency binaries eliminate deployment waste |
| **Heijunka** | Consistent recipe structure across all categories |
| **Poka-Yoke** | Compile-time model embedding prevents runtime failures |
| **Kaizen** | Continuous improvement via 95%+ test coverage and falsification |

## Quality Standards

Every recipe is verified against the QA checklist:

- `cargo run` succeeds (Exit Code 0)
- `cargo test` passes with property-based tests
- Falsification tests pass (F1-F7)
- Deterministic output (reproducible)
- No temp files leaked
- Memory usage stable
- Clippy clean (`-D warnings`)
- 95%+ test coverage

## Development

```bash
# Run tests
cargo test --lib

# Run falsification tests (release mode for accurate metrics)
cargo test --test falsification --release -- --nocapture

# Run linter
cargo clippy --all-targets -- -D warnings

# Generate coverage report (target: 95%)
cargo llvm-cov --lib

# Build documentation book
cd book && mdbook build

# Full validation
cargo fmt --check && cargo clippy -- -D warnings && cargo test
```

## Feature Flags

| Feature | Description |
|---------|-------------|
| `default` | Core bundling and conversion |
| `encryption` | AES-256-GCM model encryption |
| `full` | All features enabled |

## Architecture

```
apr-cookbook/
├── src/
│   ├── lib.rs                    # Public API and prelude
│   ├── bundle.rs                 # Model bundling (APR v2)
│   ├── convert.rs                # Format conversion
│   ├── recipe.rs                 # Recipe infrastructure
│   ├── aprender_integration.rs   # APR format integration
│   └── error.rs                  # Error types
├── examples/
│   ├── creation/                 # Category A: 5 recipes
│   ├── bundling/                 # Category B: 7 recipes
│   ├── training/                 # Category C: 5 recipes (incl. entrenar)
│   ├── conversion/               # Category D: 5 recipes
│   ├── registry/                 # Category E: 4 recipes
│   ├── api/                      # Category F: 4 recipes
│   ├── serverless/               # Category G: 4 recipes
│   ├── wasm/                     # Category H: 5 recipes
│   ├── gpu/                      # Category I: 5 recipes (incl. FlashAttention)
│   ├── simd/                     # Category J: 5 recipes (incl. trueno)
│   ├── distillation/             # Category K: 4 recipes
│   ├── cli/                      # Category L: 4 recipes
│   ├── monitoring/               # Category M: 2 recipes
│   ├── speech/                   # Category N: 2 recipes (whisper.apr)
│   └── distributed/              # Category O: 1 recipe (repartir)
├── tests/
│   └── falsification.rs          # Popperian falsification tests (F1-F7)
├── book/                         # mdbook documentation
└── docs/
    └── specifications/           # APR v2 specification
```

## License

MIT License — see [LICENSE](LICENSE) for details.

## Links

- [GitHub Repository](https://github.com/paiml/apr-cookbook)
- [Documentation Book](https://paiml.github.io/apr-cookbook/)
- [crates.io](https://crates.io/crates/apr-cookbook)
- [Sovereign AI Stack](https://github.com/paiml/sovereign-ai-stack-book)

---

**APR Cookbook** — Production recipes for ML model deployment with Popperian falsification.
