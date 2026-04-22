# Architecture

## APR v2 Format Specification

| Feature | APR v1 | APR v2 |
|---------|--------|--------|
| Tensor Index | JSON | Binary (O(1) lookup) |
| Compression | None/Gzip | LZ4/ZSTD (throughput unmeasured in this repo; see apr-leaderboard) |
| Zero-Copy Loading | Partial | Full (mmap, < 0.1 ms p95 release — F2) |
| Quantization | Int8 | Int4/Int8/FP16 |
| Streaming | No | Yes |
| Signature | Optional | Ed25519 default |

### Falsifiable Claims

Only claims with reproducible measurements are listed. See [Quality Gates](quality-gates.md#falsifiable-claims-registry--v50) for the full in-process (F2) + cited (N1–N4) registry and for the list of deleted claims (F1/F3/F4/F5/F6/F7) that lack evidence.

**F2** (in-process): Zero-copy mmap-backed load completes in p95 < 0.1 ms (release) for the cookbook's `BundledModel` path.
- **Test**: `cargo test --test falsification -- f2_zero_copy_loading_latency`
- **Refutation**: p95 > 0.2 ms (release) or p95 > 10 ms (debug).
- **Evidence basis**: `aprender/docs/benchmarks/FORMAT_PARITY_REPORT.md:88-93` measured 0.028 ms.

**N1–N4** (cited): Decode throughput, batch scaling, load-time parity, and Candle comparison — see [Quality Gates §Cited](quality-gates.md).

---

## Technology Stack

```
┌─────────────────────────────────────────────────────────────┐
│                   APR Cookbook v4.0                          │
├─────────────────────────────────────────────────────────────┤
│  Examples Layer (219 recipes across 24 categories)          │
│  ├── Model Bundling (include_bytes!, APR v2 compression)    │
│  ├── Format Conversion (SafeTensors -> APR v2 -> GGUF)     │
│  ├── Speech Recognition (simulated whisper pipelines)       │
│  ├── Browser Apps (WASM + WebGPU acceleration)              │
│  ├── CLI Tools (inference, conversion, benchmarking)        │
│  └── Optimization (finetune, prune, distill, merge, quant) │
├─────────────────────────────────────────────────────────────┤
│  Framework Layer — APR-MONO v0.31.2 (../aprender monorepo) │
│  ├── aprender-core  (pkg) / aprender (lib) — APR v2, LZ4, Int4│
│  ├── aprender-compute (pkg) / trueno (lib) — SIMD/GPU tensors │
│  ├── aprender-train  (pkg) / entrenar (lib) — Train, autograd │
│  ├── aprender-contracts (dev) — In-process YAML validation    │
│  └── ndarray 0.16 — Tensor gradients (third-party, unchanged) │
├─────────────────────────────────────────────────────────────┤
│  Compression Layer                                          │
│  ├── lz4_flex 0.11 (LZ4, throughput unmeasured in this repo)│
│  └── zstd 0.13 (ZSTD, throughput unmeasured in this repo)   │
├─────────────────────────────────────────────────────────────┤
│  Runtime Layer                                              │
│  ├── Native: x86_64 (AVX2/AVX-512), aarch64 (NEON)        │
│  ├── WASM: wasm32-unknown-unknown (browser, edge)          │
│  └── GPU: wgpu (Vulkan/Metal/DX12/WebGPU)                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Deployment Targets

| Target | Binary Size | Acceleration | Cold Start |
|--------|-------------|--------------|------------|
| `x86_64-unknown-linux-gnu` | ~5MB | AVX-512 | <10ms |
| `aarch64-unknown-linux-gnu` | ~4MB | NEON | <10ms |
| `aarch64-apple-darwin` | ~4MB | NEON | <10ms |
| `wasm32-unknown-unknown` | ~500KB | SIMD128/WebGPU | <50ms |

---

## Key Dependencies

All sovereign-stack deps are path deps to the APR-MONO monorepo at `../aprender` (v0.31.2). Package names (Cargo.toml keys) differ from lib names (Rust `use` paths) because the monorepo preserved historical lib names for source-compat.

| Cargo package | Rust lib ident | Version | Role | Features |
|---------------|----------------|---------|------|----------|
| aprender-core | `aprender` | 0.31.2 | Core ML library, .apr format | `format-compression`, `format-encryption` (via `encryption` feature) |
| aprender-compute | `trueno` | 0.31.2 | SIMD tensor backend (was standalone `trueno` crate) | — |
| aprender-train | `entrenar` | 0.31.2 | Training, autograd, LoRA, monitoring (was standalone `entrenar`) | autograd, LoRA, collectors |
| aprender-contracts (dev-dep) | `provable_contracts` | 0.31.2 | In-process YAML contract validation | — |
| ndarray | `ndarray` | 0.16 | Tensor gradients | — |
| clap | `clap` | 4 | CLI argument parsing | `derive` |
| serde | `serde` | 1 | Serialization | `derive` |
| ed25519-dalek | `ed25519_dalek` | 2.1 | Model signing | — |
| lz4_flex | `lz4_flex` | 0.11 | LZ4 compression | — |
| zstd | `zstd` | 0.13 | ZSTD compression | — |
| memmap2 (dev-dep) | `memmap2` | 0.9 | F2 zero-copy falsification | — |

GPU (aprender-gpu / aprender-cuda-edge), distributed (aprender-distribute), and speech (aprender-core/audio) are part of the broader APR-MONO workspace but are **simulated** in cookbook examples — not actual Cargo.toml dependencies of this repo. Real-hardware numbers live in sibling repos (candle-vs-apr, apr-leaderboard) and are cited as claims N1–N4 in the root spec, not re-run here.

---

## Philosophy: Lean AI & The Toyota Way

### Muda (Waste Elimination)

The Python interpreter, heavy containers, and gigabyte-sized runtime environments are *Muda*. APR v2 binaries are single-file with zero Python/CUDA runtime dependency (CPU path), eliminating "transport waste" of massive Docker images. GPU inference still requires CUDA libraries at runtime — that's a driver dependency, not a Python one.

### Jidoka (Built-in Quality)

Rust's type system and Popperian falsification tests stop defects automatically. A model that doesn't type-check is a defect stopped immediately, not debugged in production.

### Genchi Genbutsu (Go and See)

By deploying models to the Edge (WASM/Embedded), we process data where it originates rather than shipping it to a central cloud.

### Poka-Yoke (Error-Proofing)

Compile-time model embedding via `include_bytes!()` makes runtime file-not-found errors impossible. APR v2 checksums detect corruption before inference.

### Kaizen (Continuous Improvement)

Each release must demonstrate measurable improvement via falsifiable benchmarks. No "improvements" without evidence.

---

## Example Directory Layout

```
examples/
├── creation/       Build models from scratch
├── bundling/       Static model embedding via include_bytes!()
├── training/       Incremental, online, federated, autograd
├── conversion/     SafeTensors <-> .apr <-> GGUF <-> ONNX
├── registry/       Model registry and lineage
├── api/            Inference API patterns
├── serverless/     Lambda, edge, container deployment
├── wasm/           Browser inference, WebGPU
├── gpu/            FlashAttention, CUDA, multi-GPU
├── simd/           trueno SIMD ops, vectorized inference
├── distillation/   Knowledge transfer, pruning
├── cli/            apr-info, apr-bench, apr-convert, apr-compile, apr-serve
├── monitoring/     Inference explainability, hash chain audit
├── speech/         Speech recognition pipelines (simulated)
├── distributed/    Multi-node inference (simulated)
├── advanced/       End-to-end demo applications
├── optimize/       Finetune, prune, distill, merge, quantize (CLI demos)
├── chat/           Chat template formatting (CLI demos)
├── analysis/       Model analysis tools (CLI demos)
└── format/         Format import/export operations (CLI demos)
```
