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

APR Cookbook provides **121 production-ready recipes** across 20 categories for deploying ML models using the APR v2 format. Each recipe is isolated, idempotent, and verified with Popperian falsification tests following Toyota Way quality principles.

### Key Capabilities

- **APR v2 Format**: LZ4/ZSTD compression (3-13 GB/s), Int4/Int8 quantization, Ed25519 signatures
- **Zero-Copy Loading**: Embed models with `include_bytes!()` for <1ms startup
- **Format Conversion**: SafeTensors, GGUF, ONNX to/from APR v2
- **Speech Recognition**: whisper.apr integration for pure Rust ASR
- **GPU Acceleration**: FlashAttention, fused kernels via realizar
- **Distributed Computing**: Multi-node inference with repartir work-stealing
- **Training**: Autograd, LoRA/QLoRA, distillation, model merge with entrenar
- **Inference Patterns**: Speculative decoding, KV-cache, streaming, batching
- **Model Serving**: HTTP REST API with metrics, batching, health checks

### Sovereign AI Stack

```
┌─────────────────────────────────────────────────────────────────┐
│                   APR Cookbook v2.0                             │
├─────────────────────────────────────────────────────────────────┤
│  aprender 0.25 (APR v2, LZ4/ZSTD, Int4/Int8 quantization)      │
│  trueno 0.14 (SIMD/GPU, AVX-512/NEON, LZ4 tensors)             │
│  realizar 0.4 (FlashAttention, Q4K/Q5K/Q6K kernels)            │
│  whisper-apr 0.1 (WASM-first ASR, streaming)                   │
│  repartir 1.1 (distributed compute, work-stealing)             │
│  entrenar 0.5 (autograd, LoRA/QLoRA, model merge)              │
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
│                    APR Cookbook (121 Recipes)                    │
├─────────────────────────────────────────────────────────────────┤
│  A: Creation (6)      │  B: Bundling (7)    │  C: Training (17) │
│  D: Conversion (5)    │  E: Registry (5)    │  F: API (5)       │
│  G: Serverless (5)    │  H: WASM (6)        │  I: GPU (6)       │
│  J: SIMD (5)          │  K: Distillation (5)│  L: CLI (7)       │
│  M: Monitoring (5)    │  N: Speech (2)      │  O: Distributed(2)│
│  P: Inference (11)    │  Q: Serving (5)     │  Advanced (16)    │
├─────────────────────────────────────────────────────────────────┤
│  Falsification Tests: F1-F7 (Popperian methodology)            │
│  Test Coverage: 95%+ │ Quality Gate: Zero violations           │
└─────────────────────────────────────────────────────────────────┘
```

### Category Overview

| Category | Recipes | Description |
|----------|---------|-------------|
| **A: Model Creation** | 6 | Build models from scratch: regression, trees, clustering, n-grams, neural networks |
| **B: Binary Bundling** | 7 | Embed models: static, quantized, encrypted, signed, Lambda |
| **C: Training** | 17 | Autograd, LoRA, QLoRA, distillation, model merge, eval, sweep, checkpoint, mixed-precision, few-shot, gradient accumulation, LR schedules |
| **D: Format Conversion** | 5 | SafeTensors, GGUF, ONNX, Phi model conversion |
| **E: Model Registry** | 5 | Register, lineage, comparison, rollback, versioning |
| **F: API Integration** | 5 | Inference, streaming, batch, health checks, auth middleware |
| **G: Serverless** | 5 | Lambda, cold start, edge functions, containers, model warmup |
| **H: WASM/Browser** | 6 | Browser inference, workers, progressive loading, WebGPU, model loader |
| **I: GPU Acceleration** | 6 | **FlashAttention**, CUDA, tensor cores, multi-GPU, memory management, memory pool |
| **J: SIMD Acceleration** | 5 | **trueno ops**, matrix ops, vectorized, quantized SIMD |
| **K: Model Distillation** | 5 | Knowledge transfer, layer matching, pruning-aware, quantization-aware, structured pruning |
| **L: CLI Tools** | 7 | apr-info, apr-bench, apr-convert, apr-serve, apr-diff |
| **M: Monitoring** | 5 | Inference explainability, hash chain audit, cost tracking, latency histograms, drift detection |
| **N: Speech Recognition** | 2 | **whisper.apr** transcription, streaming ASR |
| **O: Distributed** | 2 | **repartir** multi-node inference, model sharding |
| **P: Inference Patterns** | 11 | Speculative decode, KV-cache, streaming, batching, pipeline, quantized, tool use, ensemble, dynamic SLA |
| **Q: Model Serving** | 5 | HTTP REST API, A/B testing, canary deploy, rate limiting, model selection router |

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
cargo run --example create_apr_decision_tree
cargo run --example create_apr_kmeans_clustering
cargo run --example create_apr_ngram_language_model
cargo run --example create_apr_neural_network

# Category B: Binary Bundling
cargo run --example bundle_static_model
cargo run --example bundle_quantized_model
cargo run --example bundle_encrypted_model --features encryption
cargo run --example bundle_apr_static_binary
cargo run --example bundle_apr_quantized_q4
cargo run --example bundle_apr_signed
cargo run --example bundle_apr_lambda_package

# Category C: Training
cargo run --example entrenar_autograd_training
cargo run --example entrenar_lora_finetune
cargo run --example entrenar_qlora_finetune
cargo run --example entrenar_distillation
cargo run --example entrenar_model_merge
cargo run --example entrenar_eval_metrics
cargo run --example hyperparameter_sweep
cargo run --example checkpoint_resume
cargo run --example mixed_precision_training
cargo run --example few_shot_finetune
cargo run --example gradient_accumulation
cargo run --example learning_rate_schedule
cargo run --example data_preprocessing

# Category D: Format Conversion
cargo run --example convert_safetensors_to_apr
cargo run --example convert_gguf_to_apr
cargo run --example convert_apr_to_gguf
cargo run --example convert_phi_to_apr
cargo run --example convert_onnx_to_apr

# Category E: Model Registry
cargo run --example registry_register_apr
cargo run --example registry_model_lineage
cargo run --example registry_model_comparison
cargo run --example registry_model_rollback
cargo run --example registry_model_versioning

# Category F: API Integration
cargo run --example api_call_model_inference
cargo run --example api_streaming_inference
cargo run --example api_batch_inference
cargo run --example api_model_health_check
cargo run --example api_auth_middleware

# Category G: Serverless
cargo run --example serverless_lambda_inference
cargo run --example serverless_cold_start_optimization
cargo run --example serverless_edge_function
cargo run --example serverless_container_image
cargo run --example serverless_model_warmup

# Category H: WASM/Browser
cargo run --example wasm_browser_inference
cargo run --example wasm_web_worker
cargo run --example wasm_progressive_loading
cargo run --example wasm_webgpu_acceleration
cargo run --example wasm_streaming_compilation
cargo run --example wasm_model_loader

# Category I: GPU Acceleration
cargo run --example flash_attention_inference
cargo run --example gpu_cuda_inference
cargo run --example gpu_tensor_core_optimization
cargo run --example gpu_multi_gpu_inference
cargo run --example gpu_memory_management
cargo run --example gpu_memory_pool

# Category J: SIMD Acceleration
cargo run --example trueno_simd_ops
cargo run --example simd_matrix_operations
cargo run --example simd_vectorized_inference
cargo run --example simd_quantized_operations
cargo run --example simd_auto_vectorization

# Category K: Model Distillation
cargo run --example distill_knowledge_transfer
cargo run --example distill_layer_matching
cargo run --example distill_pruning_aware
cargo run --example distill_quantization_aware
cargo run --example distill_structured_pruning

# Category L: CLI Tools
cargo run --example cli_apr_info -- --demo
cargo run --example cli_apr_bench -- --demo
cargo run --example cli_apr_convert
cargo run --example cli_apr_serve
cargo run --example cli_apr_diff

# Category M: Monitoring
cargo run --example inference_explainability
cargo run --example hash_chain_audit
cargo run --example inference_cost_tracking
cargo run --example latency_histogram
cargo run --example model_drift_detection

# Category N: Speech Recognition
cargo run --example whisper_transcribe
cargo run --example whisper_streaming

# Category O: Distributed
cargo run --example distributed_inference
cargo run --example distributed_model_sharding

# Category P: Inference Patterns
cargo run --example simple_inference
cargo run --example speculative_decode
cargo run --example chat_kv_cache
cargo run --example chat_multiturn
cargo run --example chat_tool_use
cargo run --example streaming_token_generator
cargo run --example adaptive_batch_inference
cargo run --example dynamic_batch_with_sla
cargo run --example ensemble_inference
cargo run --example model_pipeline
cargo run --example quantized_inference_comparison

# Category Q: Model Serving
cargo run --example http_model_server
cargo run --example model_ab_testing
cargo run --example model_canary_deploy
cargo run --example model_rate_limiter
cargo run --example model_selection_router
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

- **[The APR Cookbook](https://paiml.github.io/apr-cookbook/)** — Online book with all 121 recipes
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
│   ├── explainable.rs            # Inference explainability wrappers
│   └── error.rs                  # Error types
├── examples/
│   ├── creation/                 # Category A: 6 recipes
│   ├── bundling/                 # Category B: 7 recipes
│   ├── training/                 # Category C: 17 recipes (entrenar, LoRA, QLoRA, mixed-precision)
│   ├── conversion/               # Category D: 5 recipes
│   ├── registry/                 # Category E: 5 recipes
│   ├── api/                      # Category F: 5 recipes
│   ├── serverless/               # Category G: 5 recipes
│   ├── wasm/                     # Category H: 6 recipes
│   ├── gpu/                      # Category I: 6 recipes (incl. FlashAttention)
│   ├── simd/                     # Category J: 5 recipes (incl. trueno)
│   ├── distillation/             # Category K: 5 recipes
│   ├── cli/                      # Category L: 7 recipes
│   ├── monitoring/               # Category M: 5 recipes
│   ├── speech/                   # Category N: 2 recipes (whisper.apr)
│   ├── distributed/              # Category O: 2 recipes (repartir, sharding)
│   ├── inference/                # Category P: 11 recipes (speculative, KV-cache, streaming, ensemble)
│   └── serve/                    # Category Q: 5 recipes (HTTP, A/B, canary, rate limit, router)
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
