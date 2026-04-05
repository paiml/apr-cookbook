# APR Cookbook Specification

**Version**: 3.0.0
**Status**: ACTIVE
**MSRV**: 1.75
**Date**: 2026-03-17
**Repository**: [github.com/paiml/apr-cookbook](https://github.com/paiml/apr-cookbook)

---

## Executive Summary

The APR Cookbook is the technical manual for production ML deployment using the `.apr` format. It provides idiomatic Rust examples that demonstrate model bundling, format conversion, browser/edge deployment (WASM), SIMD/GPU acceleration, and CLI tooling — all as single-binary, zero-dependency artifacts.

**APR v2 Format** introduces binary tensor indices, LZ4/ZSTD compression, zero-copy loading, and Int4/Int8 quantization — achieving 3-10x size reduction with minimal accuracy loss.

**Design Principles**:
- **IIUR**: Every recipe is **Isolated**, **Idempotent**, **Useful**, and **Reproducible**
- **Toyota Way**: Muda (waste elimination), Jidoka (built-in quality), Genchi Genbutsu (edge deployment), Poka-Yoke (compile-time safety), Kaizen (measurable improvement)
- **Popperian Falsification**: Every performance claim is specific, testable, and refutable

**Target Audience**: Rust developers and ML Engineers deploying models without the Python/CUDA ecosystem.

---

## Technology Stack

```
APR Cookbook v3.0
├── Examples Layer (this repo)
│   ├── 52 IIUR recipes (Categories A-L)
│   ├── 48 CLI demo recipes (optimize, chat, analysis, format)
│   └── CLI tools (apr-info, apr-bench, apr-convert, apr-compile, apr-serve)
├── Framework Layer
│   ├── aprender 0.25  — ML algorithms, .apr format, quantization
│   ├── trueno 0.14    — SIMD/GPU tensor operations
│   ├── entrenar 0.5   — Training, monitoring, autograd
│   ├── realizar 0.4   — GPU kernels, FlashAttention
│   ├── repartir 1.1   — Distributed compute, work-stealing
│   └── whisper-apr 0.1 — WASM-first ASR
├── Compression Layer
│   ├── trueno-zram    — SIMD LZ4/ZSTD (3-13 GB/s)
│   └── trueno-ublk    — GPU block device (10-50 GB/s)
└── Runtime Layer
    ├── Native: x86_64 (AVX2/AVX-512), aarch64 (NEON)
    ├── WASM: wasm32-unknown-unknown
    └── GPU: wgpu (Vulkan/Metal/DX12/WebGPU)
```

---

## Table of Contents

Each component specification is in `components/` and is self-contained (max 500 lines).

### Core Architecture

| # | Component | Description |
|---|-----------|-------------|
| 1 | [Architecture](components/architecture.md) | APR v2 format spec, deployment targets, philosophy |
| 2 | [IIUR Principles](components/principles.md) | Isolation, idempotency, usefulness, reproducibility; recipe structure |

### Recipe Specifications

| # | Component | Description |
|---|-----------|-------------|
| 3 | [Recipe Catalog](components/recipe-catalog.md) | 52 IIUR recipes across 12 categories (A-L) |
| 4 | [CLI Demos](components/cli-demos.md) | 48 CLI-mirroring recipes: optimize, chat, analysis, format |

### Quality & Process

| # | Component | Description |
|---|-----------|-------------|
| 5 | [Quality Gates](components/quality-gates.md) | PMAT, falsification, provable contracts, CLI QA process |
| 6 | [Implementation Guidelines](components/implementation.md) | Toyota Way compliance, code style, error handling |

### Reference

| # | Component | Description |
|---|-----------|-------------|
| 7 | [References](components/references.md) | Peer-reviewed citations informing the specification |
| 8 | [Appendices](components/appendices.md) | Dependency matrix, feature flags, checklists, status |

---

## APR v2 Format Summary

| Feature | APR v1 | APR v2 |
|---------|--------|--------|
| Tensor Index | JSON | Binary (O(1) lookup) |
| Compression | None/Gzip | LZ4/ZSTD (3-13 GB/s) |
| Zero-Copy Loading | Partial | Full (mmap) |
| Quantization | Int8 | Int4/Int8/FP16 |
| Streaming | No | Yes |
| Signature | Optional | Ed25519 default |

## Deployment Targets

| Target | Binary Size | Acceleration | Cold Start |
|--------|-------------|--------------|------------|
| `x86_64-unknown-linux-gnu` | ~5MB | AVX-512 | <10ms |
| `aarch64-unknown-linux-gnu` | ~4MB | NEON | <10ms |
| `aarch64-apple-darwin` | ~4MB | NEON | <10ms |
| `wasm32-unknown-unknown` | ~500KB | SIMD128/WebGPU | <50ms |

---

## Recipe Overview

### IIUR Recipes (52 total)

| Category | Count | Scope |
|----------|-------|-------|
| A: Model Creation | 5 | Build models from scratch |
| B: Binary Bundling | 5 | Static embedding, quantized, encrypted, signed, Lambda |
| C: Continuous Training | 4 | Incremental, online, federated, curriculum |
| D: Format Conversion | 5 | Phi, SafeTensors, GGUF, ONNX |
| E: Model Registry | 4 | Pacha: register, lineage, comparison, rollback |
| F: API Integration | 4 | REST inference, streaming, batch, health |
| G: Serverless | 4 | Lambda inference, batch, edge, container |
| H: WASM & Browser | 5 | Browser inference, interactive, dashboard, autocomplete, web worker |
| I: GPU Acceleration | 4 | Matrix ops, model inference, batch, WebGPU fallback |
| J: SIMD Acceleration | 4 | Vector ops, matmul, convolution, softmax |
| K: Distillation | 4 | HF distill, knowledge transfer, pruning, QAT |
| L: CLI Tools | 4 | apr-info, apr-bench, apr-convert, apr-validate |

### CLI Demo Recipes (48 total)

| Category | Count | Scope |
|----------|-------|-------|
| optimize/ | 22 | Finetune, prune, distill, merge, quantize |
| chat/ | 5 | ChatML, LLaMA 2, Mistral, multi-format, injection defense |
| analysis/ | 11 | Inspect, validate, diff, bench, profile, QA, oracle, canary, tree, hex, explain |
| format/ | 10 | Import HF, export SafeTensors/GGUF, rosetta, convert, publish, pull, batch |

---

## Build Commands

```bash
# Build all examples
cargo build --examples

# Run a specific example
cargo run --example bundle_static_model
cargo run --example apr_info -- model.apr

# Run tests
cargo test --all-features

# Quality gates
pmat analyze defects --path .
pmat analyze tdg --path .
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
```

---

## Cargo.toml Configuration

```toml
[package]
name = "apr-cookbook"
version = "3.0.0"
edition = "2021"
rust-version = "1.75"
license = "MIT"
description = "APR Cookbook - Production ML deployment with IIUR recipes"

[dependencies]
aprender = { version = "0.25", features = ["format-compression", "format-signing"] }
trueno = "0.14"
entrenar = { version = "0.5", optional = true }
realizar = { version = "0.4", optional = true }
whisper-apr = { version = "0.1", optional = true }
repartir = { version = "1.1", optional = true, features = ["cpu"] }
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "2"
anyhow = "1"

[target.'cfg(target_arch = "wasm32")'.dependencies]
wasm-bindgen = "0.2"
console_error_panic_hook = "0.1"
web-sys = { version = "0.3", features = ["Performance", "Window", "console"] }

[dev-dependencies]
criterion = { version = "0.5", features = ["html_reports"] }
proptest = "1"
tempfile = "3"

[features]
default = []
encryption = ["aprender/format-encryption"]
gpu = ["realizar", "trueno/gpu"]
speech = ["whisper-apr"]
distributed = ["repartir"]
full = ["encryption", "gpu", "speech", "distributed"]
```

---

## Falsifiable Claims Summary

| Code | Claim | Threshold | Refutation | Contract |
|------|-------|-----------|------------|----------|
| F1 | LZ4 decompression throughput | >= 3 GB/s (AVX2) | < 2.5 GB/s | `lz4-decompression-v1` |
| F2 | Zero-copy mmap latency (<=100MB) | < 1ms | p95 > 2ms | `mmap-inference-v1` |
| F3 | Int4 quantization accuracy loss | < 2% | > 2.5% | `int4-quantization-v1` |
| F4 | AES-256-GCM decrypt latency (100MB) | < 5ms | p95 > 10ms | `aes256-gcm-decrypt-v1` |
| F5 | whisper.apr WER (LibriSpeech) | < 10% | > 12% | `whisper-wer-v1` |
| F6 | FlashAttention speedup (seq>=1024) | >= 2x | < 1.5x | `flash-attention-v1` |
| F7 | AVX-512 matmul GFLOPS (1024x1024) | >= 80 | < 60 | `avx512-matmul-v1` |

All claims are backed by provable-contracts YAML in `contracts/` with formal equations, proof obligations, and falsification tests. See [Quality Gates](components/quality-gates.md) for the full contract schema.

## CLI QA

The installed `apr` binary is tested via `/qa` (Claude Code skill in `.claude/skills/qa/`). This exercises all 40+ subcommands against a real model, detects bugs (panics, exit code lies, hangs, wrong data, missing fallbacks), and files GitHub issues automatically. See [Quality Gates](components/quality-gates.md) for the defect taxonomy and test matrix.

---

*Specification Version: 3.1.0 — Added provable contracts (F1–F7) and CLI QA process*
