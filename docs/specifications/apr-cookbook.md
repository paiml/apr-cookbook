# APR Cookbook Specification

**Version**: 4.0.0
**Status**: ACTIVE
**MSRV**: 1.75
**Date**: 2026-04-06
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
APR Cookbook v4.0
├── Examples Layer (this repo)
│   ├── 91 IIUR recipes (Categories A-L)
│   ├── 64 CLI demo recipes (optimize, chat, analysis, format)
│   ├── 64 other recipes (acceleration, advanced, inference, etc.)
│   └── 219 total examples across 24 categories
├── Framework Layer
│   ├── aprender 0.27  — ML algorithms, .apr format, quantization
│   ├── trueno 0.16    — SIMD/GPU tensor operations
│   ├── entrenar 0.7   — Training, monitoring, autograd
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
| 3 | [Recipe Catalog](components/recipe-catalog.md) | 91 IIUR recipes across 12 categories (A-L) |
| 4 | [CLI Demos](components/cli-demos.md) | 64 CLI-mirroring recipes: optimize, chat, analysis, format |

### Quality & Process

| # | Component | Description |
|---|-----------|-------------|
| 5 | [Quality Gates](components/quality-gates.md) | PMAT, falsification, provable contracts, five invariants |
| 5b | [CLI QA](components/cli-qa.md) | Fleet testing, defect taxonomy, 12 QA protocols |
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

## Parity & POC Ecosystem

The cookbook does not exist in isolation. These companion repos provide head-to-head benchmarks against competing runtimes and proof-of-concept deployments. Recipes should link to these repos when demonstrating a workflow that has a competing-solution equivalent.

### Parity Repos (APR vs. competitors)

| Repo | What it proves | Competing runtime | Link |
|------|---------------|-------------------|------|
| `qwen-coder-deploy` | 5-runtime inference benchmark (Ollama, llama.cpp, vLLM, realizr, realizr-wgpu) | All major | [paiml/qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy) |
| `candle-vs-apr` | Candle vs realizr GGUF inference — realizr 1.63x faster | HuggingFace Candle | [paiml/candle-vs-apr](https://github.com/paiml/candle-vs-apr) |
| `qwen-train-canary` | Training throughput comparison across 5 runtimes | Ollama, vLLM, etc. | [paiml/qwen-train-canary](https://github.com/paiml/qwen-train-canary) |
| `apr-leaderboard` | HuggingFace leaderboard proving single `apr` binary matches Python benchmarks (HumanEval, MBPP) | Python ecosystem | [paiml/apr-leaderboard](https://github.com/paiml/apr-leaderboard) |

### POC Repos (proof-of-concept deployments)

| Repo | What it demonstrates | Link |
|------|---------------------|------|
| `sovereign-ai-cookbook` | Full sovereign stack: 17 Rust components, 10 deployment stacks, forjar configs | [paiml/sovereign-ai-cookbook](https://github.com/paiml/sovereign-ai-cookbook) |
| `whisper.apr` | Production Whisper in pure Rust — WASM-first speech-to-text | [paiml/whisper.apr](https://github.com/paiml/whisper.apr) |
| `tiny-model-ground-truth` | Token-identical greedy outputs across all apr subcommands and formats | [paiml/tiny-model-ground-truth](https://github.com/paiml/tiny-model-ground-truth) |
| `apr-model-qa-playbook` | Structured QA playbook for model validation with apr | [paiml/apr-model-qa-playbook](https://github.com/paiml/apr-model-qa-playbook) |

### Competing Runtimes (reference implementations)

Recipes that benchmark or compare against these runtimes should link to the upstream repo and the corresponding parity repo above.

| Runtime | Language | Strength | Link |
|---------|----------|----------|------|
| llama.cpp | C++ | Gold-standard CPU inference, GGUF format origin | [ggerganov/llama.cpp](https://github.com/ggerganov/llama.cpp) |
| Ollama | Go | Easy local deployment UX | [ollama/ollama](https://github.com/ollama/ollama) |
| vLLM | Python/CUDA | High-concurrency GPU serving | [vllm-project/vllm](https://github.com/vllm-project/vllm) |
| Candle | Rust | HuggingFace minimalist Rust ML framework | [huggingface/candle](https://github.com/huggingface/candle) |
| llamafile | C++ | Single-file executable distribution | [Mozilla-Ocho/llamafile](https://github.com/Mozilla-Ocho/llamafile) |
| TGI | Rust/Python | HuggingFace inference server (maintenance mode → vLLM) | [huggingface/text-generation-inference](https://github.com/huggingface/text-generation-inference) |
| SafeTensors | Rust/Python | Secure tensor format (complementary, not competing) | [huggingface/safetensors](https://github.com/huggingface/safetensors) |

### Recipe Cross-Reference Convention

When a recipe demonstrates a workflow that has a parity benchmark, it should include a doc-comment link:

```rust
//! ## Parity
//! - Benchmark: [qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy) — APR vs 4 runtimes
//! - Competing: `ollama run qwen2.5-coder:1.5b` (see parity repo for tok/s comparison)
```

---

## Recipe Overview

### IIUR Recipes (91 total)

| Category | Count | Scope |
|----------|-------|-------|
| A: Model Creation | 7 | Build models from scratch |
| B: Binary Bundling | 7 | Static embedding, quantized, encrypted, signed, Lambda |
| C: Continuous Training | 16 | Incremental, online, federated, curriculum, autograd |
| D: Format Conversion | 5 | Phi, SafeTensors, GGUF, ONNX |
| E: Model Registry | 5 | Pacha: register, lineage, comparison, rollback |
| F: API Integration | 5 | REST inference, streaming, batch, health |
| G: Serverless | 5 | Lambda inference, batch, edge, container |
| H: WASM & Browser | 6 | Browser inference, interactive, dashboard, autocomplete, web worker |
| I: GPU Acceleration | 8 | Matrix ops, model inference, batch, WebGPU fallback |
| J: SIMD Acceleration | 6 | Vector ops, matmul, convolution, softmax |
| K: Distillation | 5 | HF distill, knowledge transfer, pruning, QAT |
| L: CLI Tools | 16 | apr-info, apr-bench, apr-convert, apr-validate, and more |

### CLI Demo Recipes (64 total)

| Category | Count | Scope |
|----------|-------|-------|
| optimize/ | 23 | Finetune, prune, distill, merge, quantize |
| chat/ | 5 | ChatML, LLaMA 2, Mistral, multi-format, injection defense |
| analysis/ | 25 | Inspect, validate, diff, bench, profile, QA, oracle, canary, tree, hex, explain |
| format/ | 11 | Import HF, export SafeTensors/GGUF, rosetta, convert, publish, pull, batch |

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
version = "4.0.0"
edition = "2021"
rust-version = "1.75"
license = "MIT"
description = "APR Cookbook - Production ML deployment with IIUR recipes"

[dependencies]
aprender = { version = "0.27", features = ["format-compression"] }
trueno = "0.16"
entrenar = { version = "0.7", optional = true }
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

All claims are backed by 11 provable-contracts YAML in `contracts/` (0 errors, mean score 0.54, 0 Lean proofs). All 219 recipe files reference ≥1 contract via `Contract:` header (Invariant B: 100%). See [Quality Gates](components/quality-gates.md) for the full contract inventory and gap analysis.

## Five Coverage Invariants

The spec defines five coverage invariants. Each has a formal definition, a `make` target for enforcement, and a measured baseline. Invariants marked **TARGET** are not yet fully satisfied — the baseline shows the current gap.

### Invariant A — CLI Recipe Parity (F-CLIPARITY-001) — ENFORCED

Every one of the **57 non-help apr-cli subcommands** has ≥1 cookbook recipe.

```
∀ s ∈ apr.subcommands \ {help}: ∃ r ∈ recipes: r.cli_equivalent = s
```

**Baseline (2026-04-06)**: 57/57 = 100%. **Gate**: `make cli-parity` (exits non-zero on regression).

### Invariant B — Recipe Contract Grade (F-CONTRACT-GRADE-001) — TARGET

Every recipe should reference a provable-contract (`../provable-contracts` YAML) that passes `pv lint` at grade **A**.

```
∀ r ∈ recipes:
  ∃ c ∈ contracts/: r.contract = c
  pv grade(c) = A
  pv lean-status(c) ≥ L2
```

Grade A requires: complete `metadata` (incl. academic references), ≥3 `proof_obligations`, matching `falsification_tests`, ≥1 `kani_harness`, and a passing `qa_gate`.

**Baseline (2026-04-06)**: 219/219 = **100%**. 11 contracts exist, mean `pv lint` score 0.54. **Gate**: `make contract-grade` — **ENFORCED**.

### Invariant C — Model Format Coverage (F-FORMAT-COV-001) — TARGET

Every recipe that operates on a model file should demonstrate all three canonical formats where applicable: **APR** (`.apr`), **GGUF** (`.gguf`), **SafeTensors** (`.safetensors`).

```
∀ r ∈ recipes where r.accepts_model_input:
  ∀ fmt ∈ {apr, gguf, safetensors} where r.supports(fmt):
    ∃ variant v ∈ r: v.format = fmt
```

"Where applicable" means the subcommand accepts that format. For example:
- `apr run model.apr`, `apr run model.gguf`, `apr run model.safetensors` → 3 variants required
- `apr encrypt` only supports `.apr` → 1 variant sufficient
- `apr import hf://…` outputs `.apr` only → 1 variant sufficient

**Baseline (2026-04-06)**: 219/219 = **100%**. **Gate**: `make format-coverage` — **ENFORCED**.

### Invariant D — arXiv Citation (F-ARXIV-001) — TARGET

Every recipe should include ≥1 arXiv or peer-reviewed citation in its doc comment header linking the technique to the literature.

```
∀ r ∈ recipes:
  |r.citations ∩ (arXiv ∪ peer_reviewed)| ≥ 1
```

Doc comment format:

```rust
//! ## References
//! - Hu et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models*. arXiv:2106.09685
```

**Baseline (2026-04-06)**: 219/219 = **100%**. **Gate**: `make citation-check` — **ENFORCED**.

### Invariant E — Docs Contract Coverage (F-DOCS-CONTRACT-001) — TARGET

Every documentation artifact in the repo — `README.md`, `CLAUDE.md`, mdbook chapters, spec components — should be bound to a provable-contract that validates factual accuracy and structural integrity.

```
∀ d ∈ {README.md, CLAUDE.md, book/src/**/*.md, docs/specifications/**/*.md}:
  ∃ c ∈ contracts/: d ∈ c.bindings
  pv lint(c) = PASS
  pmat validate-readme(d) ⊨ {unverified = 0, contradictions = 0}
```

**Baseline (2026-04-06)**: 264/267 = **98.9%**. `make docs-validate` covers `README.md`, `CLAUDE.md`, `docs/specifications/**/*.md`, and `book/src/**/*.md`. 3 excluded: `CHANGELOG.md`, `deep-context.md` (generated), `docs/specifications-advanced-demos.md` (orphan). **Gate**: `make docs-validate` — **ENFORCED**.

---

## CLI QA — Fleet + Contract Coverage + Pattern-Driven Protocols

The installed `apr` binary is tested via `/qa` (Claude Code skill in `.claude/skills/qa/`) across the full **hardware fleet** (intel, yoga, jetson, lambda-labs + local), and every subcommand is audited against the `provable-contracts` registry (YAML + Lean 4 proofs via `pv` CLI). This exercises all 57 subcommands, detects arch-divergence bugs, contract drift, coverage gaps, and 12+ systemic bug patterns, and files GitHub issues automatically.

**Pattern-Driven QA Protocols**: 12 protocol-level checks derived from 500 historical paiml/aprender issues (#24–#607). The top bug classes — GPU/CUDA (8.2%), NaN/Inf (7.0%), silently-ignored flags (4.6%), hardcoded values (3.0%), wrong output (2.8%), cross-subcommand divergence, and cache inconsistency — are each caught by a dedicated protocol (Silent-Flag, Exit-Code Contradiction, Flag-Echo, Cross-Subcommand Consistency, Cache Registry Integrity, GPU/CPU Parity, NaN/Inf Sentinel, Version Sanity, Phantom Subcommand, JSON Schema Stability, Default-Defamation, Hardware Cascade).

See [Quality Gates](components/quality-gates.md) for the target fleet, full defect taxonomy with historical frequencies, test matrix, protocol definitions, docs schema, CLI parity invariant, and formal coverage metrics.

---

*Specification Version: 4.1.0 — Five Coverage Invariants (A-E) + Parity/POC ecosystem cross-references*
