# Architecture

## APR v2 Format Specification

| Feature | APR v1 | APR v2 |
|---------|--------|--------|
| Tensor Index | JSON | Binary (O(1) lookup) |
| Compression | None/Gzip | LZ4/ZSTD (3-13 GB/s) |
| Zero-Copy Loading | Partial | Full (mmap) |
| Quantization | Int8 | Int4/Int8/FP16 |
| Streaming | No | Yes |
| Signature | Optional | Ed25519 default |

### Falsifiable Claims

**F1**: APR v2 with LZ4 compression achieves >= 3 GB/s decompression on x86_64 with AVX2.
- **Test**: `cargo bench --bench compression -- --baseline`
- **Refutation**: If measured throughput < 2.5 GB/s on reference hardware (AMD EPYC 7763), claim is falsified.

**F2**: Zero-copy loading via mmap adds < 1ms latency for models <= 100MB.
- **Test**: `cargo bench --bench loading -- --size 100mb`
- **Refutation**: If p95 latency > 2ms, claim is falsified.

**F3**: Int4 quantization (Q4_K) achieves < 2% accuracy loss on standard benchmarks.
- **Test**: `cargo test --test quantization_accuracy`
- **Refutation**: If accuracy loss > 2.5% on GLUE benchmark subset, claim is falsified.

**F4**: AES-256-GCM decryption adds < 5ms latency for 100MB models.
- **Test**: `cargo bench --bench encryption -- --size 100mb`
- **Refutation**: If p95 latency > 10ms, claim is falsified.

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
│  Framework Layer (Sovereign AI Stack)                       │
│  ├── aprender 0.27 (APR v2 format, LZ4/ZSTD, Int4/Int8)   │
│  ├── trueno 0.16 (SIMD/GPU, LZ4 tensors)                   │
│  ├── entrenar 0.7 (Training, autograd, LoRA, monitoring)    │
│  └── ndarray 0.16 (Tensor gradients)                        │
├─────────────────────────────────────────────────────────────┤
│  Compression Layer                                          │
│  ├── trueno-zram (SIMD LZ4/ZSTD, 3-13 GB/s)               │
│  └── trueno-ublk (GPU block device, 10-50 GB/s)            │
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

| Crate | Version | Role | Features |
|-------|---------|------|----------|
| aprender | 0.27 | Core ML library, .apr format | `format-compression` |
| trueno | 0.16 | SIMD tensor backend | — |
| entrenar | 0.7 | Training and inference monitoring | autograd, LoRA, collectors |
| ndarray | 0.16 | Tensor gradients for entrenar | — |
| clap | 4 | CLI argument parsing | `derive` |
| serde | 1 | Serialization | `derive` |
| ed25519-dalek | 2.1 | Model signing | — |
| lz4_flex | 0.11 | LZ4 compression | — |
| zstd | 0.13 | ZSTD compression | — |

GPU (realizar), distributed (repartir), and speech (whisper-apr) are part of the broader sovereign stack but are **simulated** in cookbook examples — not actual Cargo.toml dependencies.

---

## Philosophy: Lean AI & The Toyota Way

### Muda (Waste Elimination)

The Python interpreter, heavy containers, and gigabyte-sized runtime environments are *Muda*. APR v2 binaries are single-file and zero-dependency, eliminating "transport waste" of massive Docker images.

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
