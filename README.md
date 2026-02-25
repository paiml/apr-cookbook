<div align="center">

# APR Cookbook

**Production recipes for ML model deployment in pure Rust**

[![CI](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Rust](https://img.shields.io/badge/rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![Book](https://img.shields.io/badge/book-online-blue)](https://paiml.github.io/apr-cookbook/)

</div>

![](.github/apr-cookbook-hero.svg)

169 executable examples across 24 categories covering model bundling, format conversion, training, optimization, inference, and deployment with the APR v2 format.

## Installation

```bash
# Clone the repository
git clone https://github.com/paiml/apr-cookbook.git
cd apr-cookbook

# Build all examples
cargo build --examples

# Run tests
cargo test --all-features
```

Requires Rust 1.75+ (`rustup update stable`).

## Usage

```bash
# Run any example by name
cargo run --example create_apr_from_scratch
cargo run --example bundle_static_model
cargo run --example optimize_full_pipeline

# Run example tests
cargo test --example finetune_lora

# List all examples
cargo run --example 2>&1 | head -50
```

## Stack

| Crate | Version | Role |
|-------|---------|------|
| [aprender](https://crates.io/crates/aprender) | 0.25 | APR v2 format, LZ4/ZSTD compression, Int4/Int8 quantization |
| [trueno](https://crates.io/crates/trueno) | 0.14 | SIMD/GPU tensor operations, AVX-512/NEON |
| [entrenar](https://crates.io/crates/entrenar) | 0.5 | Autograd, LoRA/QLoRA, model merge, distillation |

## Examples

| Category | Count | Highlights |
|----------|-------|------------|
| [Creation](examples/creation/) | 6 | Linear regression, decision trees, clustering, neural networks |
| [Bundling](examples/bundling/) | 7 | Static embedding, quantized, encrypted, signed, Lambda |
| [Training](examples/training/) | 17 | Autograd, LoRA, QLoRA, mixed-precision, few-shot, gradient accumulation |
| [Conversion](examples/conversion/) | 5 | SafeTensors, GGUF, ONNX, Phi |
| [Registry](examples/registry/) | 5 | Versioning, lineage, comparison, rollback |
| [API](examples/api/) | 5 | REST inference, streaming, batch, auth middleware |
| [Serverless](examples/serverless/) | 5 | Lambda, cold start, edge, containers, warmup |
| [WASM](examples/wasm/) | 6 | Browser inference, Web Workers, WebGPU, progressive loading |
| [GPU](examples/gpu/) | 6 | FlashAttention, CUDA, tensor cores, multi-GPU, memory pool |
| [SIMD](examples/simd/) | 5 | trueno ops, matrix operations, vectorized inference |
| [Distillation](examples/distillation/) | 5 | Knowledge transfer, pruning, quantization-aware |
| [CLI](examples/cli/) | 7 | apr-info, apr-bench, apr-convert, apr-serve, apr-diff |
| [Monitoring](examples/monitoring/) | 5 | Explainability, audit trail, cost tracking, drift detection |
| [Speech](examples/speech/) | 2 | whisper.apr transcription, streaming ASR |
| [Distributed](examples/distributed/) | 2 | Multi-node inference, model sharding |
| [Inference](examples/inference/) | 11 | Speculative decode, KV-cache, streaming, ensemble, tool use |
| [Serving](examples/serve/) | 5 | HTTP server, A/B testing, canary deploy, rate limiting |
| [Optimize](examples/optimize/) | 22 | Full pipeline, LoRA, QLoRA, pruning, distillation, merge, quantize |
| [Chat](examples/chat/) | 5 | ChatML, LLaMA 2, Mistral, multi-format, injection defense |
| [Analysis](examples/analysis/) | 11 | Inspect, validate, diff, bench, profile, QA gates, oracle |
| [Format](examples/format/) | 10 | Import, export, Rosetta convert/chain/verify, batch export |
| [Advanced](examples/advanced/) | 16 | RAG pipeline, style transfer, voice recognition, and more |

Run any example:

```bash
cargo run --example <name>
```

## APR v2 Format

| Feature | Spec |
|---------|------|
| Compression | LZ4 (3+ GB/s) / ZSTD |
| Quantization | FP32, FP16, Int8, Int4 |
| Signatures | Ed25519 |
| Tensor Index | Binary, O(1) lookup |
| Zero-Copy | Full mmap support |

```rust
use apr_cookbook::prelude::*;

let bundle = ModelBundleV2::new()
    .with_name("my-model")
    .with_compression(Compression::Lz4)
    .with_quantization(Quantization::Int8)
    .add_tensor("weights", vec![768, 768], weight_bytes)
    .build();

assert_eq!(&bundle[0..4], b"APR2");
```

## Development

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-features
```

## Live Demos

| Demo | Link |
|------|------|
| Monte Carlo S&P 500 | [Launch](https://interactive.paiml.com/monte-carlo-sp500/) |
| Shell ML Autocomplete | [Launch](https://interactive.paiml.com/shell-ml/) |

## Documentation

- [The APR Cookbook](https://paiml.github.io/apr-cookbook/) — Online book
- [API Reference](https://docs.rs/apr-cookbook) — Rust docs
- [APR v2 Spec](docs/specifications/cookbook-spec.md) — Format specification

## Contributing

1. Fork the repository
2. Create your example following the template in `docs/specifications/apr-cli-demos.md`
3. Ensure all quality gates pass:
   ```bash
   cargo fmt --check
   cargo clippy --all-targets -- -D warnings
   cargo test --all-features
   ```
4. Submit a pull request

Every example must include a doc header with QA checklist, use `RecipeContext` for isolation, and contain 8-15 unit tests.

## License

MIT
