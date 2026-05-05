# APR Cookbook Specification

**Version**: 5.1.0
**Status**: ACTIVE
**MSRV**: 1.89
**Date**: 2026-05-05
**Repository**: [github.com/paiml/apr-cookbook](https://github.com/paiml/apr-cookbook)
**Sovereign Stack**: APR-MONO v0.31.2 ([github.com/paiml/aprender](https://github.com/paiml/aprender))

**v5.1.0 (2026-05-05)**: Add **SHIP-TWO-001 fine-tune-from-init workflow** cross-reference (new §"SHIP-TWO-001 Workflow" section below). The aprender §50.4 cascade landed `apr pretrain --init <PATH>.apr` end-to-end on 2026-05-05 (PRs #1471-#1494, post-INTEGRATION-COMPLETE per aprender SPEC-SHIP-TWO-001 §53). Companion recipes for fine-tuning from public Qwen2.5-Coder-0.5B-Instruct via `apr tokenize import-hf` + `apr tokenize encode-corpus` + `apr pretrain --init` are tracked here as the operator-facing cookbook surface for the SHIP-TWO-001 spec.

---

## Executive Summary

The APR Cookbook is the technical manual for production ML deployment using the `.apr` format. It provides idiomatic Rust examples that demonstrate model bundling, format conversion, browser/edge deployment (WASM), SIMD/GPU acceleration, and CLI tooling — all as single-binary artifacts with zero Python/CUDA runtime dependency (CPU path; GPU path requires CUDA libraries at inference time).

**APR v2 Format** introduces binary tensor indices, LZ4/ZSTD compression, zero-copy mmap loading, and Int4/Int8 quantization. Fair BF16 → Int4+LZ4 size comparison yields ~2–3× reduction (measured in [aprender/docs/benchmarks/FORMAT_PARITY_REPORT.md:17](https://github.com/paiml/aprender/blob/main/docs/benchmarks/FORMAT_PARITY_REPORT.md)). Larger ratios (e.g. 6.3× for F32 → Q4_K_M) confuse precision drop with format advantage.

**Design Principles**:
- **IIUR**: Every recipe is **Isolated**, **Idempotent**, **Useful**, and **Reproducible**
- **Toyota Way**: Muda (waste elimination), Jidoka (built-in quality), Genchi Genbutsu (edge deployment), Poka-Yoke (compile-time safety), Kaizen (measurable improvement)
- **Popperian Falsification**: Every published performance claim is specific, testable, and refutable — and either (a) exercised in `tests/falsification.rs` with a committed refutation threshold, or (b) cited from a reproducible external harness with a source path and line number. No proxies, no simulated fixtures, no aspirational thresholds.

**Target Audience**: Rust developers and ML Engineers deploying models without the Python/CUDA ecosystem.

---

## Technology Stack

```
APR Cookbook v5.0 — APR-MONO Integration
├── Examples Layer (this repo)
│   ├── 91 IIUR recipes (Categories A-L)
│   ├── 64 CLI demo recipes (optimize, chat, analysis, format)
│   ├── 186 variant-depth recipes (PMAT-049/050/051 sprints — ≥3 per subcommand)
│   └── 341 total examples across 24 categories (2026-04-23)
├── Framework Layer — APR-MONO v0.31.2 crates from crates.io
│   │   (optional `[patch.crates-io]` override for local ../aprender co-dev)
│   ├── aprender-core 0.31.2         — package `aprender-core`, lib `aprender`
│   │                                  — ML algorithms, .apr format, quantization
│   │                                  — declared as `aprender = { package = "aprender-core" }`
│   ├── aprender-compute 0.31.2      — package `aprender-compute`, lib `trueno`
│   │                                  — SIMD/GPU tensor operations (was standalone `trueno`)
│   ├── aprender-train 0.31.2        — package `aprender-train`, lib `entrenar`
│   │                                  — Training, autograd, LoRA/QLoRA, merge, distill
│   │                                  — (was standalone `entrenar`)
│   ├── aprender-contracts 0.31.2    — package `aprender-contracts`, lib `provable_contracts`
│   │                                  — dev-dep: in-process YAML contract validation
│   │                                  — (was standalone `provable-contracts`)
│   └── ndarray 0.16                 — Tensor gradients (unchanged third-party)
├── Compression (third-party)
│   ├── lz4_flex 0.11  — LZ4 compression
│   └── zstd 0.13      — ZSTD compression
└── Runtime Layer
    ├── Native: x86_64 (AVX2/AVX-512), aarch64 (NEON)
    ├── WASM: wasm32-unknown-unknown
    └── GPU: simulated in cookbook demos; real kernels live in sibling
             repos (candle-vs-apr, apr-leaderboard) and are cited via
             claims N1–N4, not re-run in this repo's test suite.
```

**Backward compatibility**: None. The sovereign-stack repos `trueno`, `entrenar`, `realizar`, `batuta`, and `provable-contracts` were consolidated into the `aprender` monorepo at v0.31.2. The cookbook depends on the monorepo directly via path deps; published crates.io aliases (`trueno`, `entrenar`) are deprecation shims and are NOT used. Lib names (`aprender`, `trueno`, `entrenar`, `provable_contracts`) are preserved inside the monorepo, so Rust source imports (`use entrenar::...`) do not change — only package names in `Cargo.toml` do.

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
| Compression | None/Gzip | LZ4/ZSTD (throughput unmeasured in this repo; see apr-leaderboard for hardware-specific numbers) |
| Zero-Copy Loading | Partial | Full (mmap, < 0.1 ms p95 release — F2) |
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

## SHIP-TWO-001 Workflow — Fine-tune from APR Init (added v5.1.0)

The aprender SPEC-SHIP-TWO-001 (`docs/specifications/aprender-train/ship-two-models-spec.md` in [paiml/aprender](https://github.com/paiml/aprender)) drives the production of two artifacts:

- **MODEL-1**: A 7B-class Qwen2.5-Coder distilled / quantized teacher (currently 91% ship-ready; 9% blocked on SHIP-007 GPU root cause).
- **MODEL-2**: A 0.5B-class fine-tune of public Qwen2.5-Coder-0.5B-Instruct on a Python+permissive code corpus (currently 57% ship-ready; gated on val_loss < 9.38 evidence from a LIVE 500-step run).

The §50.4 cascade (PRs #1471-#1506 in aprender, 2026-05-04 to 2026-05-05) landed the operator-facing CLI surface for MODEL-2's fine-tune workflow. The cookbook documents this workflow as a 4-step recipe operators can run end-to-end. Each step is contract-bound and falsifier-pinned per [`feedback_falsifier_first_cascade_pattern.md`](https://github.com/paiml/aprender/blob/main/docs/specifications/aprender-train/ship-two-models-spec.md).

### 4-Step Recipe

| Step | Command | What it does | Contract |
|---|---|---|---|
| 5g.0 | `apr tokenize import-hf --input <Qwen-tokenizer.json> --output <DIR>` | Extract HF BPE → aprender vocab.json + merges.txt + manifest.json | [`apr-cli-tokenize-import-hf-v1`](https://github.com/paiml/aprender/blob/main/contracts/apr-cli-tokenize-import-hf-v1.yaml) v1.1.0 PARTIAL_ALGORITHM_LEVEL |
| 5g.1 | `apr tokenize encode-corpus --corpus <jsonl> --tokenizer <DIR> --output <shards>` | Encode corpus to flat u32 .bin shards consumable by `ShardBatchIter` | [`pretokenize-bin-v1`](https://github.com/paiml/aprender/blob/main/contracts/pretokenize-bin-v1.yaml) |
| 5g.2 | `apr pretrain --init <Qwen.apr> --tokenizer <DIR> --dataset <shards> --mode finetune --num-steps 500` | LIVE 500-step fine-tune from the pretrained checkpoint | [`apr-pretrain-from-init-v1`](https://github.com/paiml/aprender/blob/main/contracts/apr-pretrain-from-init-v1.yaml) v1.2.0 + [`apr-pretrain-arch-polymorphic-v1`](https://github.com/paiml/aprender/blob/main/contracts/apr-pretrain-arch-polymorphic-v1.yaml) v1.5.0 FUNCTIONAL |
| 5g.3 | (verdict step — operator inspects checkpoint val_loss) | Flips MODEL-2 ship % from 57% → ≥58% when val_loss < 9.38 | SHIP-TWO-001 §50.4 step 5g.3 |

### Why these steps exist (§54-§57 spec history)

The SHIP-TWO-001 §50.4 cascade re-scoped multiple times as live source inspection surfaced hidden coupling:

- **§54**: 5g originally framed as "1 dispatch, 0 LOC" — re-scoped to 5g.0/5g.1/5g.2/5g.3 after live smoke surfaced that HF tokenizers don't have aprender-compatible vocab.json layouts.
- **§55**: Polymorphic preflight relaxed from `tokenizer_vocab == model_vocab` to `tokenizer_vocab ≤ model_vocab` (RELAXED bound) when `--init` is set, because HF-distributed checkpoints (Qwen2.5/Llama2/Mistral) standardly declare `vocab_size` larger than tokenizer.json materializes (reserved/special slots).
- **§56**: 5g.1 LIVE smoke validated the chain end-to-end on a 5000-doc slice (13 valid u32 shards produced; ~110 sec / M-token throughput).
- **§57**: Drift sweep cleaned PV-VER-001 across §50.4 cascade contracts (PRs #1502/#1504/#1505/#1506); zero dangling test references remain across all 870+ contracts.

The recipe above maps each step to its contract; operators can run `cargo run --release -p aprender-contracts-cli --bin pv -- validate <contract>` to verify falsifier bindings before dispatching the step.

### Cross-Reference: SHIP-TWO-001 in apr-leaderboard

When MODEL-2 ships (val_loss < 9.38 attained), the resulting checkpoint will appear in [paiml/apr-leaderboard](https://github.com/paiml/apr-leaderboard) alongside competing Python-stack models. The leaderboard quantifies the load-bearing claim "PAIML stack alone produces a competitive 0.5B-class code model without Python/CUDA toolchain dependencies" — the same claim §50.4 cascade ships at the runtime level.

### Recipe Cross-Reference Convention (for SHIP-TWO-001)

```rust
//! ## SHIP-TWO-001
//! - Spec: [aprender §50.4](https://github.com/paiml/aprender/blob/main/docs/specifications/aprender-train/ship-two-models-spec.md)
//! - Step: 5g.X (where X ∈ {0, 1, 2, 3})
//! - Contract: <name>-v<N>.yaml (see table above)
//! - Leaderboard: see [apr-leaderboard](https://github.com/paiml/apr-leaderboard) for ship-% verification
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
version = "0.1.0"
edition = "2021"
rust-version = "1.89"
license = "MIT"
description = "Idiomatic Rust examples for the APR ML format - Toyota Way principles"

[dependencies]
# APR-MONO v0.31.2 — canonical crate names from crates.io.
# Root `aprender` facade doesn't forward features, so depend on aprender-core
# (whose lib name is "aprender") with a package rename.
aprender = { version = "0.31.2", package = "aprender-core", features = ["format-compression"] }
aprender-compute = "0.31.2"   # package `aprender-compute`, lib name `trueno`
aprender-train = "0.31.2"     # package `aprender-train`, lib name `entrenar`
ndarray = "0.16"
clap = { version = "4", features = ["derive"] }
serde = { version = "1", features = ["derive"] }
serde_json = "1"
thiserror = "1"
rand = { version = "0.8", features = ["std_rng"] }
tempfile = "3"
lz4_flex = "0.11.3"
zstd = "0.13.3"
ed25519-dalek = "2.1.1"
blake3 = "1"
flate2 = "1"

[dev-dependencies]
proptest = "1"
criterion = { version = "0.5", features = ["html_reports"] }
memmap2 = "0.9"   # F2 zero-copy falsification
# In-process contract validation (replaces external `pv` binary dep)
aprender-contracts = "0.31.2"   # lib = `provable_contracts`

[features]
default = []
encryption = ["aprender/format-encryption"]
full = ["encryption"]

# Local-development override: point the four APR-MONO deps at a co-checked-out
# ../aprender monorepo instead of crates.io. Uncomment when iterating on both
# repos simultaneously.
# [patch.crates-io]
# aprender-core = { path = "../aprender/crates/aprender-core" }
# aprender-compute = { path = "../aprender/crates/aprender-compute" }
# aprender-train = { path = "../aprender/crates/aprender-train" }
# aprender-contracts = { path = "../aprender/crates/aprender-contracts" }
```

**Note**: GPU (aprender-gpu / aprender-cuda-edge), distributed (aprender-distribute), and speech (aprender-core/audio) capabilities are **simulated** in cookbook examples. The recipes demonstrate algorithms and patterns without requiring GPU drivers at compile time. Real-hardware numbers for those paths live in sibling repos (candle-vs-apr, apr-leaderboard) and are cited as claims N1–N4, not re-run here.

---

## What the demos **don't** cover

Everything builds and runs (330/341 under 10s, 11 compute-heavy benchmarks under 60s, 0 failures — 2026-04-23). But several categories are *deliberately* simulated or feature-gated and are worth calling out so a reader doesn't mistake a passing demo for a real-hardware measurement.

| Category | What's simulated | Why | How to get the real thing |
|---|---|---|---|
| **GPU** (14 demos) | `cuda`, `vulkan`, `multi_gpu`, `tensor_cores`, `flash_attention_inference` all run a CPU tiled/scalar proxy. | Cookbook compiles without CUDA/ROCm drivers. | Run the same kernel via `aprender-compute` (`trueno`) on a CUDA-enabled host; see `candle-vs-apr/performance.md` for measured numbers. |
| **Distributed** (5 demos) | `distributed_inference`, `sharding`, `ring_allreduce`, `pipeline_parallel`, `gossip` simulate with in-process workers. | Zero-dependency demo; no sockets. | Use `aprender-distribute`/`repartir` on a real multi-node setup. |
| **Speech** (5 demos) | `whisper_transcribe` returns a deterministic "Hello, world!" from any audio; real Whisper-small weights never load. | No model checkpoint bundled. | Load a real whisper.apr bundle (out of scope for zero-dep cookbook) or use the LibriSpeech harness in sibling `whisper-apr`. |
| **WASM** (6 demos) | Build as native binaries with WASM-feature flags; browser deployment is not exercised here. | Cookbook focuses on Rust-side; browser deploy needs `wasm-pack` + page scaffolding. | `cargo install wasm-pack && wasm-pack build --target web` — see `book/src/advanced/wasm/` for a page template. |
| **Serverless** (5 demos) | Build Lambda package layouts, measure cold-start, but never deploy. | Would require AWS credentials and network. | Ship the generated `bootstrap` with the template; see `book/src/advanced/serverless/`. |
| **Encryption** (1 demo) | `bundle_encrypted_model` works end-to-end but requires `--features encryption`. | Optional feature-gate. | `cargo run --example bundle_encrypted_model --features encryption`. |
| **4 B-grade contracts** | `aes256-gcm-decrypt-v1`, `avx512-matmul-v1`, `flash-attention-v1`, `lz4-decompression-v1` each carry 2–3 `pending` bindings on runtime measurements (latency, throughput, FP equivalence). | The kernel lives upstream (aprender-core/compute benches) and the cookbook does not re-measure. | Wire an in-cookbook benchmark harness for each, then flip the `binding.yaml` status to `implemented`. Not currently ticketed — see `contracts/binding.yaml`. |
| **16 Lean `sorry` theorems** | Runtime/hardware claims (latency, throughput, FP numerical equivalence, AES correctness, O(N) memory scaling) remain `:= by sorry`. | Not derivable from pure Lean semantics without Mathlib + a cost-model. | Status kept as `wip` — honest; do not flip to `proved` without a real proof body. |

Everything else — algorithm correctness, round-trip preservation, structural invariants — is either Lean-proved, Kani-verified, or both.

---

## Falsifiable Claims Summary

Two categories: **in-process** claims are exercised by `tests/falsification.rs` in this repo; **cited** claims reference external measurements with reproducible source paths. All published thresholds match the actual test assertion or the cited source file.

### In-process claims (exercised here)

| Code | Claim | Threshold | Refutation | Test | Contract |
|------|-------|-----------|------------|------|----------|
| F2 | Zero-copy mmap-backed load | p95 < 0.1 ms (release) | p95 > 0.2 ms (release) or > 10 ms (debug) | `f2_zero_copy_loading_latency` | `mmap-inference-v1` |

### Cited claims (measured elsewhere — verified source paths)

| Code | Claim | Threshold | Measured | Source |
|------|-------|-----------|----------|--------|
| N1 | Decode throughput, c=1, RTX 4090, GGUF Q4_K_M | ≥ 270 tok/s | 273.8 tok/s | [candle-vs-apr/performance.md:85](https://github.com/paiml/candle-vs-apr/blob/main/performance.md) |
| N2 | Batch scaling, c=1 → c=32, v5 scheduler | ≥ 10× | 13.4× | [candle-vs-apr/performance.md:150](https://github.com/paiml/candle-vs-apr/blob/main/performance.md) |
| N3 | Load-time parity across APR / GGUF / SafeTensors | within 1.5× | 0.028 / 0.024 / 0.029 ms | [aprender/docs/benchmarks/FORMAT_PARITY_REPORT.md:88-93](https://github.com/paiml/aprender/blob/main/docs/benchmarks/FORMAT_PARITY_REPORT.md) |
| N4 | Decode advantage vs Candle on identical hardware | ≥ 1.15× | 1.20× (273.8 / 227.4 tok/s) | [candle-vs-apr/performance.md:85](https://github.com/paiml/candle-vs-apr/blob/main/performance.md) |

### Deleted claims (previously in spec — no evidence exists)

The following claims were asserted in prior spec versions but are **removed** from v5.0 because no measurement exists in any sibling repo. They must not be re-introduced without a committed harness.

| Code | Original claim | Why removed |
|------|----------------|-------------|
| F1 | LZ4 decompression ≥ 3 GB/s (AVX2) | No LZ4 throughput benchmark in trueno / aprender-compute |
| F3 | Int4 quantization NMSE < 2% | aprender-quant benches measure throughput only, never accuracy delta |
| F4 | AES-256-GCM decrypt ≥ 100 MB/s | Prior test used BLAKE3 as proxy; no crypto bench in any repo |
| F5 | Whisper WER < 10% on LibriSpeech | Threshold defined in [whisper.apr/THRESHOLDS.md:74](https://github.com/paiml/whisper.apr) but no measured WER logged; prior test simulated WER on hand-written strings, not audio. Re-add when whisper.apr publishes measured results. |
| F6 | FlashAttention ≥ 2× speedup | Prior CPU-tiled proxy never hits 2×; GPU harness not hosted here |
| F7 | AVX-512 matmul ≥ 80 GFLOPS | Trueno SDE infrastructure exists but no published GFLOPS numbers |

### Contract backing

All surviving claims (F2 + N1–N4) are backed by provable-contracts YAML in `contracts/` (11 files total). Every YAML parses and validates in-process via `cargo test --test contracts`, which replaces the prior external `pv validate` dependency. 341/341 cargo `[[example]]` recipes reference ≥1 contract via `//! Contract:` header (Invariant B). Contract inventory: see [Quality Gates](components/quality-gates.md).

**Provability scoreboard (2026-04-23, post-PMAT-046/047/048/056):**

| Dimension | Score |
|---|---|
| Codebase simple mean | **0.92 (A)** |
| PVScore 10-dim | **94.8 (A)** |
| Per-contract mean | 0.89 (B) |
| Kani harnesses landed | 39/39 |
| Lean theorems proved | 23/39 (16 honest `sorry` for runtime/hardware claims) |
| Binding coverage | 76% |
| Grade C contracts | **0** |

**Demo-run baseline (2026-04-23):** 330/341 examples pass in <10s; 11 compute-heavy benchmarks (matmul, SIMD, cache-tiling, distributed sim) require a longer timeout and are tracked in `docs/specifications/components/quality-gates.md#demo-run-baseline`. Zero failures.

## Six Coverage Invariants

The spec defines six coverage invariants. Each has a formal definition, a `make` target for enforcement, and a measured baseline.

### Invariant A — CLI Recipe Parity (F-CLIPARITY-001) — ENFORCED

Every one of the **66 non-help apr-cli subcommands** (APR-MONO v0.31.2) has ≥1 cookbook recipe.

```
∀ s ∈ apr.subcommands \ {help}: ∃ r ∈ recipes: r.cli_equivalent = s
```

**Baseline (2026-04-22)**: 56/66 = 85% (10 subcommands added by APR-MONO v0.31.2 still uncovered — see PMAT-049). **Gate**: `make cli-parity` (exits non-zero on regression).

### Invariant B — Recipe Contract Grade (F-CONTRACT-GRADE-001) — ENFORCED

Every recipe should reference a provable-contract (`../provable-contracts` YAML) that passes `pv lint` at grade **A**.

```
∀ r ∈ recipes:
  ∃ c ∈ contracts/: r.contract = c
  pv grade(c) = A
  pv lean-status(c) ≥ L2
```

Grade A requires: complete `metadata` (incl. academic references), ≥3 `proof_obligations`, matching `falsification_tests`, ≥1 `kani_harness`, and a passing `qa_gate`.

**Baseline (2026-04-22)**: 341/341 = **100%**. 11 contracts exist, mean `pv lint` score 0.54. **Gate**: `make contract-grade` — **ENFORCED**.

### Invariant C — Model Format Coverage (F-FORMAT-COV-001) — ENFORCED

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

**Baseline (2026-04-22)**: 341/341 = **100%**. **Gate**: `make format-coverage` — **ENFORCED**.

### Invariant D — arXiv Citation (F-ARXIV-001) — ENFORCED

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

**Baseline (2026-04-22)**: 341/341 = **100%**. **Gate**: `make citation-check` — **ENFORCED**.

### Invariant E — Docs Contract Coverage (F-DOCS-CONTRACT-001) — ENFORCED

Every documentation artifact in the repo — `README.md`, `CLAUDE.md`, mdbook chapters, spec components — should be bound to a provable-contract that validates factual accuracy and structural integrity.

```
∀ d ∈ {README.md, CLAUDE.md, book/src/**/*.md, docs/specifications/**/*.md}:
  ∃ c ∈ contracts/: d ∈ c.bindings
  pv lint(c) = PASS
  pmat validate-readme(d) ⊨ {unverified = 0, contradictions = 0}
```

### Invariant F — Variant Depth (F-VARIANT-DEPTH-001) — ENFORCED

Every apr-cli subcommand must have **≥3 distinct cookbook recipes** demonstrating different variants (different flag values, input shapes, deployment targets, or workflows). Single-example coverage is necessary (Invariant A) but not sufficient — real users need multiple worked examples per subcommand to learn idiomatic usage.

```
∀ s ∈ apr.subcommands \ {help}:
  |{ r ∈ recipes : r.cli_equivalent = s }| ≥ 3
```

**Rationale**: A single recipe demonstrates that a subcommand exists; three demonstrate the *variance* (happy path, edge case, composition). This maps directly to Toyota *kata* — three repetitions make the pattern concrete.

**Baseline (2026-04-22)**: 66/66 = **100%**. Closed via PMAT-049/050/051 (→ 128 new recipes written and merged). **Gate**: `make variant-depth` — **ENFORCED** (exits non-zero on regression).

**Baseline (2026-04-22)**: 264/267 = **98.9%**. `make docs-validate` covers `README.md`, `CLAUDE.md`, `docs/specifications/**/*.md`, and `book/src/**/*.md`. 3 excluded: `CHANGELOG.md`, `deep-context.md` (generated), `docs/specifications-advanced-demos.md` (orphan). **Gate**: `make docs-validate` — **ENFORCED**.

---

## CLI QA — Fleet + Contract Coverage + Pattern-Driven Protocols

The installed `apr` binary is tested via `/qa` (Claude Code skill in `.claude/skills/qa/`) across the full **hardware fleet** (intel, yoga, jetson, lambda-labs + local), and every subcommand is audited against the `provable-contracts` registry (YAML + Lean 4 proofs via `pv` CLI). This exercises all 57 subcommands, detects arch-divergence bugs, contract drift, coverage gaps, and 12+ systemic bug patterns, and files GitHub issues automatically.

**Pattern-Driven QA Protocols**: 12 protocol-level checks derived from 500 historical paiml/aprender issues (#24–#607). The top bug classes — GPU/CUDA (8.2%), NaN/Inf (7.0%), silently-ignored flags (4.6%), hardcoded values (3.0%), wrong output (2.8%), cross-subcommand divergence, and cache inconsistency — are each caught by a dedicated protocol (Silent-Flag, Exit-Code Contradiction, Flag-Echo, Cross-Subcommand Consistency, Cache Registry Integrity, GPU/CPU Parity, NaN/Inf Sentinel, Version Sanity, Phantom Subcommand, JSON Schema Stability, Default-Defamation, Hardware Cascade).

See [Quality Gates](components/quality-gates.md) for the target fleet, full defect taxonomy with historical frequencies, test matrix, protocol definitions, docs schema, CLI parity invariant, and formal coverage metrics.

---

*Specification Version: 5.0.0 — APR-MONO v0.31.2 Integration; falsification set trimmed to evidence-backed claims (F2 in-process, N1–N4 cited); six unsupported claims (F1/F3/F4/F5/F6/F7) removed.*
