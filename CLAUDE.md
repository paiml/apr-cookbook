# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APR Cookbook is a collection of idiomatic Rust examples demonstrating the `.apr` ML model format. Examples are executable via `cargo run --example <name>` and showcase model bundling, format conversion, browser deployment (WASM), and SIMD/GPU acceleration.

## Build Commands

```bash
# Build all examples
cargo build --examples

# Run a specific example
cargo run --example bundle_static_model
cargo run --example apr_info -- model.apr

# Run tests
cargo test --all-features

# Single test
cargo test test_name
```

## Quality Gates (PMAT)

This project uses `paiml-mcp-agent-toolkit` for quality enforcement. All code must pass:

```bash
# Pre-commit (required)
pmat analyze defects --path .
pmat analyze tdg --path .
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --all-features

# Pre-release
pmat rust-project-score --full --verbose
cargo llvm-cov --min-coverage 95
```

Minimum grade: **A**. Coverage target: **95%**.

## Code Search Policy

**NEVER use grep/glob for code search. ALWAYS prefer `pmat query`.**

`pmat query` returns quality-annotated, semantically ranked results with TDG grades, complexity, fault patterns, and call graphs.

| Task | Command |
|------|---------|
| Find functions by intent | `pmat query "error handling" --limit 10` |
| Find high-quality examples | `pmat query "serialize" --min-grade A` |
| Find with fault patterns | `pmat query "unwrap" --faults --exclude-tests` |
| Include source code | `pmat query "tokenize" --include-source` |
| Regex search | `pmat query --regex "fn\s+handle_\w+" --limit 10` |
| Literal string search | `pmat query --literal "unwrap()" --limit 10` |

When grep IS acceptable: Searching non-code files (TOML, YAML, Markdown).

## Architecture

```
Examples Layer (this repo)
    ↓
Framework Layer (dependencies)
├── aprender 0.25: ML algorithms, .apr format, quantization
├── trueno 0.14: SIMD/GPU tensor operations
└── entrenar 0.5: Training, monitoring, autograd
```

### Example Categories

| Category | Path | Purpose |
|----------|------|---------|
| creation | `examples/creation/` | Build models from scratch |
| bundling | `examples/bundling/` | Static model embedding via `include_bytes!()` |
| training | `examples/training/` | Incremental, online, federated, autograd |
| conversion | `examples/conversion/` | SafeTensors ↔ .apr ↔ GGUF ↔ ONNX |
| registry | `examples/registry/` | Model registry and lineage |
| api | `examples/api/` | Inference API patterns |
| serverless | `examples/serverless/` | Lambda, edge, container deployment |
| wasm | `examples/wasm/` | Browser inference, WebGPU |
| gpu | `examples/gpu/` | FlashAttention, CUDA, multi-GPU |
| simd | `examples/simd/` | trueno SIMD ops, vectorized inference |
| distillation | `examples/distillation/` | Knowledge transfer, pruning |
| cli | `examples/cli/` | apr-info, apr-bench, apr-convert, apr-compile, apr-serve |
| monitoring | `examples/monitoring/` | Inference explainability, hash chain audit |
| speech | `examples/speech/` | whisper.apr transcription |
| distributed | `examples/distributed/` | repartir multi-node inference |
| advanced | `examples/advanced/` | End-to-end demo applications |

### Key Dependencies

- `aprender`: Core ML library with `.apr` format (features: `format-compression`)
- `entrenar`: Training and inference monitoring (autograd, LoRA, collectors)
- `trueno`: SIMD tensor backend (always required)
- `clap`: CLI argument parsing

### Feature Flags

- `encryption`: Enable AES-256-GCM model encryption
- `full`: All features

## Philosophy

This project follows **Toyota Way** principles:
- **Muda** (waste elimination): Zero-dependency binaries, no Python/CUDA
- **Jidoka** (built-in quality): Rust type system + PMAT enforcement
- **Genchi Genbutsu** (go and see): Edge/WASM deployment
