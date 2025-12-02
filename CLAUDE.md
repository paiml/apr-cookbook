# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

APR Cookbook is a collection of idiomatic Rust examples demonstrating the `.apr` ML model format. Examples are executable via `cargo run --example <name>` and showcase model bundling, format conversion, browser deployment (WASM), and SIMD/GPU acceleration.

## Build Commands

```bash
# Build all examples
cargo build --examples

# Build with browser features
cargo build --examples --features browser

# Build with GPU features
cargo build --examples --features gpu

# Build for WASM
cargo build --target wasm32-unknown-unknown --features browser

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

## Architecture

```
Examples Layer (this repo)
    ↓
Framework Layer (dependencies)
├── aprender: ML algorithms, .apr format, quantization
├── presentar: WASM UI, widgets, YAML config
└── trueno: SIMD/GPU tensor operations
```

### Example Categories

| Category | Path | Purpose |
|----------|------|---------|
| bundling | `examples/bundling/` | Static model embedding via `include_bytes!()` |
| conversion | `examples/conversion/` | SafeTensors ↔ .apr ↔ GGUF |
| browser | `examples/browser/` | WASM apps with presentar widgets |
| acceleration | `examples/acceleration/` | SIMD/GPU with automatic fallback |
| cli | `examples/cli/` | Production CLI tools |

### Key Dependencies

- `aprender`: Core ML library with `.apr` format (features: `format-compression`)
- `entrenar`: Training infrastructure (optional, feature: `training`)
- `trueno`: SIMD tensor backend (always required)
- `clap`: CLI argument parsing

### Feature Flags

- `training`: Enable entrenar for training examples
- `full`: All features

### Available Examples

```bash
cargo run --example bundle_static_model      # Static model embedding
cargo run --example bundle_quantized_model   # Quantized model loading
cargo run --example convert_safetensors_to_apr  # Format conversion
cargo run --example convert_apr_to_gguf      # GGUF export
cargo run --example simd_matrix_operations   # SIMD benchmark
cargo run --example apr_info -- --demo       # Model inspection CLI
cargo run --example apr_bench -- --demo      # Inference benchmark
```

## Philosophy

This project follows **Toyota Way** principles:
- **Muda** (waste elimination): Zero-dependency binaries, no Python/CUDA
- **Jidoka** (built-in quality): Rust type system + PMAT enforcement
- **Genchi Genbutsu** (go and see): Edge/WASM deployment
