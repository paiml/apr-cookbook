# Advanced APR Demos Specification

**Version**: 1.3.0-draft
**Status**: IN REVIEW
**Authors**: Claude Code, Gemini Agent
**Date**: 2025-12-08
**Toyota Way Alignment**: Jidoka, Heijunka, Genchi Genbutsu, Kaizen, Poka-yoke

---

## Executive Summary

This specification defines 15 advanced demonstration recipes for the APR Cookbook, showcasing production-grade ML deployment patterns using the `.apr` format. Each demo follows EXTREME TDD methodology with 25-point quality checklists, 30+ peer-reviewed citations, and Toyota Way principles. These examples push the boundaries of performance, integration, and system architecture.

**Target Audience**: ML Engineers, Systems Architects, Embedded Developers
**Rust Edition**: 2021 (MSRV 1.75+)
**Core Dependencies**: `aprender` 0.15+, `trueno` 0.8+, `batuta` 0.3+, `renacer` 0.2+
**Extended Stack**: `candle-core` (Inference), `wgpu` (Graphics), `usearch` (Vector DB), `tokio` (Async), `postcard` (Embedded)

### Summary Metrics

| Metric | Value |
|--------|-------|
| **Total Demos** | 15 |
| **Total LOC (Est.)** | 7,200 - 8,700 |
| **QA Checkpoints** | 375 (15 × 25) |
| **Peer-Reviewed Citations** | 35 |
| **Categories Covered** | 10 |
| **Deployment Targets** | CPU, GPU, WASM, Embedded, Edge |

### Demo Categories

| Category | Demos | Focus |
|----------|-------|-------|
| **Infrastructure** | A, B | Model loading, caching |
| **Quality & Observability** | C, D | Inspection, quantization |
| **Continuous Learning** | E | Online training, drift |
| **Data Exploration** | F | Embeddings, visualization |
| **NLP** | G, K, N | Summarization, RAG, sentiment |
| **Audio** | H | Speech recognition |
| **Vision** | I, J, M, O | Handwriting, classification, style, CLIP |
| **Edge/Embedded** | L | IoT anomaly detection |

### Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| HuggingFace API changes | Medium | High | Pin model versions, local cache |
| WASM size limits | Medium | Medium | Aggressive quantization, streaming |
| Model accuracy drift | Low | High | Continuous monitoring, checksums |
| Dependency conflicts | Medium | Low | Lock file, MSRV enforcement |
| Performance regression | Low | Medium | Criterion benchmarks in CI |

### Resource Requirements

| Resource | Minimum | Recommended |
|----------|---------|-------------|
| **Development Machine** | 16GB RAM, 4 cores | 32GB RAM, 8 cores |
| **GPU (optional)** | GTX 1060 6GB | RTX 3080 10GB |
| **Disk Space** | 50GB | 100GB (model cache) |
| **Network** | 10 Mbps | 100 Mbps (model downloads) |

---

## Table of Contents

1. [Demo A: Multi-Model Single-Shot Compilation](#demo-a-multi-model-single-shot-compilation)
2. [Demo B: Hierarchical Cache Performance](#demo-b-hierarchical-cache-performance)
3. [Demo C: Model Inspection & Quality Scoring](#demo-c-model-inspection--quality-scoring)
4. [Demo D: Quantization Quality Tradeoff](#demo-d-quantization-quality-tradeoff)
5. [Demo E: Continuous Online Training (Defect Prediction)](#demo-e-continuous-online-training-defect-prediction)
6. [Demo F: Large-Scale Embedding Visualization](#demo-f-large-scale-embedding-visualization)
7. [Demo G: WASM Document Summarizer](#demo-g-wasm-document-summarizer)
8. [Demo H: Voice Recognition Pipeline](#demo-h-voice-recognition-pipeline)
9. [Demo I: Handwriting Recognition](#demo-i-handwriting-recognition)
10. [Demo J: Image Classification](#demo-j-image-classification)
11. [Demo K: RAG Pipeline Integration](#demo-k-rag-pipeline-integration)
12. [Demo L: Edge Anomaly Detection](#demo-l-edge-anomaly-detection)
13. [Demo M: Real-time Style Transfer](#demo-m-real-time-style-transfer)
14. [Demo N: Streaming Sentiment Analysis](#demo-n-streaming-sentiment-analysis)
15. [Demo O: Multi-Modal CLIP Search](#demo-o-multi-modal-clip-search)
16. [Peer-Reviewed Citations](#peer-reviewed-citations)
17. [Appendix: Toyota Way Application](#appendix-toyota-way-application)

---

## Demo A: Multi-Model Single-Shot Compilation

### `multi_model_single_shot.rs`

**Category**: Advanced Compilation
**Complexity**: High
**Estimated LOC**: 450-550

### Overview

Demonstrate downloading multiple models from Hugging Face using the Batuta stack, composing them into a single inference pipeline, and evaluating cold-start vs. warm performance. This showcases the "sovereign stack" pattern where models are fetched, verified, cached, and composed without external runtime dependencies.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Batuta Model Orchestrator                    │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐     │
│  │ Tokenizer│──▶│ Encoder  │──▶│ Decoder  │──▶│ Head     │     │
│  │ .apr     │   │ .apr     │   │ .apr     │   │ .apr     │     │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘     │
│       │              │              │              │            │
│       ▼              ▼              ▼              ▼            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              Unified APR Pipeline (.apr)                 │   │
│  │  - Single binary embedding                               │   │
│  │  - Shared weight deduplication                           │   │
│  │  - Cross-model optimization                              │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Hugging Face Models (Example Pipeline)

| Component | Model | Size | Purpose |
|-----------|-------|------|---------|
| Tokenizer | `sentence-transformers/all-MiniLM-L6-v2` | 2.1 MB | Text tokenization |
| Encoder | `BAAI/bge-small-en-v1.5` | 33 MB | Sentence embeddings |
| Classifier | `distilbert-base-uncased-finetuned-sst-2-english` | 67 MB | Sentiment |

### Performance Metrics

| Metric | Cold (No Cache) | Warm (L1 Cache) | Target |
|--------|-----------------|-----------------|--------|
| Download Time | 2-5s | 0ms | <5s |
| Verification | 150ms | 10ms | <200ms |
| Composition | 50ms | 5ms | <100ms |
| First Inference | 200ms | 20ms | <250ms |

### Key Implementation Points

1. **Batuta Integration**: Use `batuta::HubClient` for authenticated HF downloads
2. **Checksum Verification**: SHA-256 verification against HF manifest
3. **Signature Validation**: Ed25519 signatures for model provenance
4. **Weight Deduplication**: Shared embedding layers merged via content-addressing
5. **Streaming Compilation**: Progressive model loading during download

### 25-Point QA Checklist - Demo A

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | `cargo build --release` succeeds | Build | Exit code 0, no warnings |
| 2 | `cargo test` passes all tests | Test | 100% pass rate |
| 3 | `cargo clippy -- -D warnings` clean | Lint | Zero warnings |
| 4 | `cargo fmt --check` passes | Format | No formatting changes |
| 5 | Documentation coverage >90% | Docs | `cargo doc` clean |
| 6 | Unit test coverage >95% | Coverage | llvm-cov report |
| 7 | Property tests (100+ cases) | Testing | proptest passes |
| 8 | No `unwrap()` in logic paths | Safety | grep verification |
| 9 | All errors use `?` or `expect()` | Errors | Code review |
| 10 | Deterministic output (seeded RNG) | Reproducibility | 3 consecutive runs match |
| 11 | Memory usage <500MB peak | Resources | `/usr/bin/time -v` |
| 12 | Cold-start <5s on 100Mbps | Performance | Benchmark suite |
| 13 | Warm inference <50ms | Performance | Criterion benchmark |
| 14 | Graceful network failure handling | Resilience | Offline test |
| 15 | Checksum mismatch detection | Security | Tampered file test |
| 16 | Signature verification works | Security | Invalid sig test |
| 17 | Cache eviction under memory pressure | Resources | Low-memory test |
| 18 | Progress callback fires correctly | UX | Event verification |
| 19 | Timeout handling (30s default) | Resilience | Slow server test |
| 20 | Retry logic (3 attempts) | Resilience | Flaky network test |
| 21 | IIUR compliance verified | Methodology | Isolation test |
| 22 | Toyota Way principles documented | Process | README review |
| 23 | Example output matches expected | Correctness | Golden file diff |
| 24 | CI/CD integration tested | DevOps | GitHub Actions pass |
| 25 | Security audit clean | Security | `cargo audit` |

### Citations for Demo A

- [1] Wolf et al. (2020) - Transformers: State-of-the-Art NLP
- [2] Reimers & Gurevych (2019) - Sentence-BERT
- [15] Sanh et al. (2019) - DistilBERT

---

## Demo B: Hierarchical Cache Performance

### `hierarchical_cache_benchmark.rs`

**Category**: Performance Optimization
**Complexity**: Medium-High
**Estimated LOC**: 400-500

### Overview

Benchmark the three-tier cache architecture (L1 Hot/L2 Warm/L3 Cold) with different eviction policies. Measure hit rates, latency distributions, and memory efficiency under various access patterns.

### Cache Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                    Cache Hierarchy                              │
├────────────────────────────────────────────────────────────────┤
│  L1 (Hot)     │ Heap-allocated, decompressed    │ ~100ns      │
│  ─────────────┼─────────────────────────────────┼─────────────│
│  L2 (Warm)    │ Memory-mapped, compressed       │ ~1μs        │
│  ─────────────┼─────────────────────────────────┼─────────────│
│  L3 (Cold)    │ Filesystem / Network            │ ~10ms       │
└────────────────────────────────────────────────────────────────┘
```

### Eviction Policies Compared

| Policy | Description | Best For |
|--------|-------------|----------|
| LRU | Least Recently Used | General workloads |
| LFU | Least Frequently Used | Skewed popularity |
| ARC | Adaptive Replacement | Mixed patterns |
| Clock | Approximate LRU | Low overhead |
| Fixed | No eviction | Critical models |

### Benchmark Scenarios

1. **Zipfian Access** (α=1.0): 80/20 popularity distribution
2. **Uniform Random**: Equal probability access
3. **Temporal Burst**: Hot period followed by cold
4. **Scan Resistance**: Sequential scan patterns
5. **Working Set Shift**: Gradual popularity change

### 25-Point QA Checklist - Demo B

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | `cargo build --release` succeeds | Build | Exit code 0 |
| 2 | `cargo test` passes | Test | 100% pass |
| 3 | `cargo clippy -- -D warnings` clean | Lint | Zero warnings |
| 4 | `cargo fmt --check` passes | Format | Clean |
| 5 | Documentation >90% | Docs | Verified |
| 6 | Unit test coverage >95% | Coverage | llvm-cov |
| 7 | Property tests pass | Testing | proptest |
| 8 | No `unwrap()` calls | Safety | Verified |
| 9 | Proper error handling | Errors | Code review |
| 10 | Deterministic benchmarks | Reproducibility | Seeded RNG |
| 11 | L1 latency <500ns p99 | Performance | Histogram |
| 12 | L2 latency <10μs p99 | Performance | Histogram |
| 13 | L3 latency <100ms p99 | Performance | Histogram |
| 14 | Hit rate >90% for Zipfian | Efficiency | Measurement |
| 15 | Memory watermark respected | Resources | Monitor |
| 16 | Eviction callbacks fire | Events | Verification |
| 17 | Thread-safe access | Concurrency | Race test |
| 18 | No memory leaks | Resources | Valgrind |
| 19 | Graceful OOM handling | Resilience | Low-mem test |
| 20 | Statistics accuracy ±1% | Correctness | Validation |
| 21 | IIUR compliance | Methodology | Isolation test |
| 22 | Toyota Way documented | Process | README |
| 23 | Criterion benchmarks included | Benchmarks | CI integration |
| 24 | Flamegraph generated | Profiling | renacer output |
| 25 | Regression detection | CI | Threshold alerts |

### Citations for Demo B

- [3] Megiddo & Modha (2003) - ARC Algorithm
- [4] O'Neil et al. (1993) - LRU-K
- [16] Waldspurger et al. (2015) - Cache Modeling

---

## Demo C: Model Inspection & Quality Scoring

### `model_inspection_scoring.rs`

**Category**: Observability & Quality
**Complexity**: Medium
**Estimated LOC**: 350-450

### Overview

Comprehensive model inspection demonstrating header parsing, metadata extraction, weight statistics, health scoring, and model comparison (diff). Implements the 100-point quality scoring framework.

### Inspection Capabilities

```
┌─────────────────────────────────────────────────────────────────┐
│                    Model Inspection Pipeline                     │
├─────────────────────────────────────────────────────────────────┤
│  1. Header Analysis                                              │
│     ├── Magic bytes verification (APRN)                         │
│     ├── Version compatibility check                             │
│     ├── Feature flags decode                                    │
│     └── Compression ratio calculation                           │
│                                                                  │
│  2. Metadata Extraction                                          │
│     ├── Model type identification                               │
│     ├── Hyperparameter dump                                     │
│     ├── Provenance tracking                                     │
│     └── License verification                                    │
│                                                                  │
│  3. Weight Statistics                                            │
│     ├── Min/Max/Mean/Std per layer                              │
│     ├── NaN/Inf detection (critical)                            │
│     ├── Sparsity analysis                                       │
│     └── Distribution visualization                              │
│                                                                  │
│  4. Quality Scoring (100-point scale)                           │
│     ├── Structural integrity (25 pts)                           │
│     ├── Numerical stability (25 pts)                            │
│     ├── Compression efficiency (25 pts)                         │
│     └── Security compliance (25 pts)                            │
│                                                                  │
│  5. Model Diff                                                   │
│     ├── Layer-by-layer comparison                               │
│     ├── L2 distance metrics                                     │
│     ├── Cosine similarity                                       │
│     └── Drift detection alerts                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Quality Score Breakdown

| Category | Points | Criteria |
|----------|--------|----------|
| **Structural Integrity** | 25 | Valid header, correct checksums, proper alignment |
| **Numerical Stability** | 25 | No NaN/Inf, reasonable ranges, no denormals |
| **Compression Efficiency** | 25 | Optimal ratio, fast decompression, size targets |
| **Security Compliance** | 25 | Valid signatures, encryption where required, audit trail |

### Health Status Levels

| Status | Score Range | Action Required |
|--------|-------------|-----------------|
| **Healthy** | 85-100 | Production ready |
| **Warning** | 60-84 | Review recommended |
| **Critical** | 0-59 | Do not deploy |

### 25-Point QA Checklist - Demo C

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic | Reproducibility | 3 runs match |
| 11 | Detects NaN weights | Correctness | Inject test |
| 12 | Detects Inf weights | Correctness | Inject test |
| 13 | Checksum validation | Security | Tamper test |
| 14 | Signature validation | Security | Invalid sig |
| 15 | Score accuracy ±2pts | Correctness | Golden models |
| 16 | Diff detects changes | Correctness | Modified model |
| 17 | JSON output valid | Format | Schema validation |
| 18 | Human-readable output | UX | Manual review |
| 19 | Large model handling | Scale | 1GB+ model test |
| 20 | Memory-mapped inspection | Performance | <100MB overhead |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | CI integration | DevOps | Actions pass |
| 24 | Example models included | Resources | 3 test models |
| 25 | Security audit clean | Security | cargo audit |

### Citations for Demo C

- [5] Mitchell et al. (2019) - Model Cards
- [6] Gebru et al. (2021) - Datasheets for Datasets
- [17] Hutchinson et al. (2021) - Fairness Auditing

---

## Demo D: Quantization Quality Tradeoff

### `quantization_quality_tradeoff.rs`

**Category**: Model Compression
**Complexity**: High
**Estimated LOC**: 500-600

### Overview

Comprehensive analysis of quantization schemes (F32, F16, BF16, Q8_0, Q4_0, Q4_1) measuring accuracy degradation, compression ratios, inference latency, and memory bandwidth. Implements GGUF-compatible quantization for llama.cpp interoperability.

### Quantization Schemes

| Format | Bits/Weight | Block Size | Compression | Use Case |
|--------|-------------|------------|-------------|----------|
| F32 | 32.0 | N/A | 1.0x | Reference |
| F16 | 16.0 | N/A | 2.0x | GPU training |
| BF16 | 16.0 | N/A | 2.0x | TPU/mixed precision |
| Q8_0 | 8.5 | 32 | 3.76x | Quality-sensitive |
| Q4_0 | 4.5 | 32 | 7.1x | Memory-constrained |
| Q4_1 | 5.0 | 32 | 6.4x | Balanced |

### Measurement Framework

```
┌─────────────────────────────────────────────────────────────────┐
│              Quantization Evaluation Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │ Original F32 │────▶│ Quantize to  │────▶│ Dequantize   │    │
│  │ Weights      │     │ Target Format│     │ to F32       │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                   Metrics Collection                      │  │
│  │  • MSE (Mean Squared Error)                               │  │
│  │  • SNR (Signal-to-Noise Ratio)                            │  │
│  │  • Perplexity delta (language models)                     │  │
│  │  • Top-k accuracy delta (classifiers)                     │  │
│  │  • Inference latency (ms)                                 │  │
│  │  • Memory bandwidth (GB/s)                                │  │
│  │  • File size reduction                                    │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Benchmark Models

| Model | Parameters | Original Size | Domain |
|-------|------------|---------------|--------|
| Linear Regression | 1K | 4 KB | Baseline |
| MLP-MNIST | 100K | 400 KB | Vision |
| DistilBERT | 66M | 264 MB | NLP |
| ResNet-18 | 11M | 44 MB | Vision |
| Whisper-tiny | 39M | 156 MB | Audio |

### Expected Results Matrix

| Model | Q8_0 Acc | Q4_0 Acc | Q4_1 Acc |
|-------|----------|----------|----------|
| Linear | -0.01% | -0.05% | -0.03% |
| MLP | -0.1% | -0.5% | -0.3% |
| DistilBERT | -0.2% | -1.2% | -0.8% |
| ResNet | -0.3% | -1.5% | -1.0% |
| Whisper | -0.5% | -2.0% | -1.3% |

### 25-Point QA Checklist - Demo D

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic quantization | Reproducibility | Bit-exact |
| 11 | MSE calculation correct | Correctness | Golden values |
| 12 | SNR calculation correct | Correctness | Golden values |
| 13 | Compression ratios match spec | Correctness | Within 1% |
| 14 | GGUF block compatibility | Interop | llama.cpp load |
| 15 | Handles denormal weights | Edge case | Inject test |
| 16 | Handles zero weights | Edge case | Sparse model |
| 17 | Large tensor support | Scale | 1B params |
| 18 | SIMD acceleration verified | Performance | Benchmark |
| 19 | Memory efficiency | Resources | Peak <2x model |
| 20 | Streaming quantization | Scale | 10GB+ models |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Criterion benchmarks | Benchmarks | CI |
| 24 | Comparison charts generated | Visualization | PNG output |
| 25 | CSV export for analysis | Data | Valid CSV |

### Citations for Demo D

- [7] Dettmers et al. (2022) - LLM.int8()
- [8] Frantar et al. (2023) - GPTQ
- [9] Lin et al. (2023) - AWQ
- [18] Jacob et al. (2018) - Quantization Training
- [19] Nagel et al. (2021) - White Paper Quantization

---

## Demo E: Continuous Online Training (Defect Prediction)

### `online_training_defect_prediction.rs`

**Category**: Continuous Learning
**Complexity**: High
**Estimated LOC**: 550-650

### Overview

Implement continuous online training using Renacer profiling traces to predict software defects. The model learns incrementally from execution traces, improving its ability to identify anomalous patterns that correlate with bugs.

### System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│              Online Learning Pipeline (Defect Prediction)        │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │   Renacer    │────▶│   Feature    │────▶│   Online     │    │
│  │   Traces     │     │   Extractor  │     │   Learner    │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                    │                    │             │
│         │                    │                    ▼             │
│         │                    │           ┌──────────────┐       │
│         │                    │           │  Prediction  │       │
│         │                    │           │  Engine      │       │
│         │                    │           └──────────────┘       │
│         │                    │                    │             │
│         ▼                    ▼                    ▼             │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                    Feedback Loop                          │  │
│  │  • Ground truth labels from bug tracker                   │  │
│  │  • Model update on new evidence                           │  │
│  │  • Concept drift detection                                │  │
│  │  • Performance monitoring                                 │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Feature Extraction from Traces

| Feature Category | Examples | Dimensionality |
|------------------|----------|----------------|
| **Call Graph** | Function call frequency, depth | 128 |
| **Memory** | Allocation patterns, peak usage | 64 |
| **Timing** | Latency percentiles, variance | 32 |
| **I/O** | Read/write ratios, syscall frequency | 32 |
| **Branches** | Misprediction rate, coverage | 16 |

### Online Learning Algorithms

| Algorithm | Update Complexity | Memory | Use Case |
|-----------|-------------------|--------|----------|
| SGD | O(d) | O(d) | Baseline |
| Passive-Aggressive | O(d) | O(d) | Margin-based |
| AROW | O(d) | O(d²) | Confidence-weighted |
| Mondrian Forest | O(log n) | O(n) | Non-parametric |

### 7-Example Training Scenarios

1. **Memory Leak Detection**: Gradually increasing heap size pattern
2. **Race Condition**: Non-deterministic execution paths
3. **Deadlock Precursor**: Lock acquisition ordering anomalies
4. **Buffer Overflow**: Stack frame size spikes
5. **Infinite Loop**: CPU utilization without I/O
6. **Resource Exhaustion**: File descriptor accumulation
7. **Performance Regression**: Latency drift over time

### 25-Point QA Checklist - Demo E

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic with seed | Reproducibility | 3 runs |
| 11 | Detects memory leaks | Correctness | Inject test |
| 12 | Detects race conditions | Correctness | Inject test |
| 13 | Concept drift detection | Correctness | Shift test |
| 14 | Model checkpoint works | Persistence | Load/save |
| 15 | Incremental update <10ms | Performance | Benchmark |
| 16 | Prediction <1ms | Performance | Benchmark |
| 17 | Memory bounded | Resources | Long run test |
| 18 | Handles missing features | Robustness | Sparse input |
| 19 | Handles label delay | Robustness | Async labels |
| 20 | Renacer integration | Integration | Trace parsing |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Precision/Recall metrics | Evaluation | F1 >0.8 |
| 24 | ROC-AUC tracking | Evaluation | AUC >0.85 |
| 25 | Alert threshold tuning | UX | Configurable |

### Citations for Demo E

- [10] Crammer et al. (2006) - Online Passive-Aggressive
- [11] Lakshminarayanan et al. (2014) - Mondrian Forests
- [20] Gama et al. (2014) - Survey Concept Drift
- [21] Bifet & Gavalda (2007) - ADWIN

---

## Demo F: Large-Scale Embedding Visualization

### `embedding_visualization_clustering.rs`

**Category**: Data Exploration
**Complexity**: High
**Estimated LOC**: 500-600

### Overview

Visualize large datasets (1M+ points) using embedding models and clustering algorithms. Output to `.prs` (Presentacion) format for interactive exploration. Demonstrates dimensionality reduction, clustering, and efficient rendering.

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│           Large-Scale Embedding Visualization Pipeline           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Raw Data    │────▶│  Embedding   │────▶│  Dim Reduce  │    │
│  │  (Text/Img)  │     │  Model (.apr)│     │  (UMAP/tSNE) │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│         │                    │                    │             │
│         │                    │                    ▼             │
│         │                    │           ┌──────────────┐       │
│         │                    │           │  Clustering  │       │
│         │                    │           │  (HDBSCAN)   │       │
│         │                    │           └──────────────┘       │
│         │                    │                    │             │
│         │                    │                    ▼             │
│         │                    │           ┌──────────────┐       │
│         │                    │           │  .prs Export │       │
│         │                    │           │  (WebGL)     │       │
│         │                    │           └──────────────┘       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Data Types

| Type | Embedding Model | Dimensions |
|------|-----------------|------------|
| Text | all-MiniLM-L6-v2 | 384 |
| Images | CLIP-ViT-B/32 | 512 |
| Code | CodeBERT | 768 |
| Audio | Wav2Vec2 | 768 |

### Dimensionality Reduction

| Algorithm | Complexity | Preserves | Best For |
|-----------|------------|-----------|----------|
| PCA | O(nd²) | Variance | Linear structure |
| UMAP | O(n log n) | Local + Global | General |
| t-SNE | O(n²) | Local only | Cluster visualization |
| TriMap | O(n log n) | Global | Large datasets |

### Clustering Algorithms

| Algorithm | Complexity | Finds K | Noise Handling |
|-----------|------------|---------|----------------|
| K-Means | O(nkd) | Required | None |
| HDBSCAN | O(n log n) | Automatic | Yes |
| DBSCAN | O(n log n) | Automatic | Yes |
| Spectral | O(n³) | Required | None |

### .prs Format Specification

```
Header:
  magic: "PRS1"
  version: (1, 0)
  point_count: u64
  dimension: u8 (2 or 3)
  cluster_count: u32

Point Data (per point):
  x: f32
  y: f32
  z: f32 (if 3D)
  cluster_id: u32
  label_offset: u32

Label Table:
  [string labels, null-terminated]

Metadata:
  [MessagePack encoded]
```

### 25-Point QA Checklist - Demo F

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic | Reproducibility | Seeded |
| 11 | Handles 1M+ points | Scale | Benchmark |
| 12 | Memory <8GB for 1M | Resources | Monitor |
| 13 | Embedding <100ms/batch | Performance | Benchmark |
| 14 | UMAP <60s for 100K | Performance | Benchmark |
| 15 | HDBSCAN <30s for 100K | Performance | Benchmark |
| 16 | .prs file valid | Format | Schema check |
| 17 | WebGL rendering works | Integration | Browser test |
| 18 | Cluster labels correct | Correctness | ARI >0.9 |
| 19 | Handles high-dim input | Robustness | 768-dim test |
| 20 | Streaming embedding | Scale | 10M points |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Interactive demo | UX | Browser |
| 24 | Export to CSV | Interop | Valid CSV |
| 25 | Color palette accessible | A11y | WCAG 2.1 |

### Citations for Demo F

- [12] McInnes et al. (2018) - UMAP
- [13] McInnes et al. (2017) - HDBSCAN
- [22] van der Maaten (2014) - Barnes-Hut t-SNE
- [23] Amid & Warmuth (2019) - TriMap

---

## Demo G: WASM Document Summarizer

### `wasm_document_summarizer.rs`

**Category**: Browser/WASM
**Complexity**: High
**Estimated LOC**: 450-550

### Overview

Deploy a distilled summarization model (T5-small or BART-small) in WebAssembly for client-side document summarization. Users upload documents directly in the browser with zero server roundtrips.

### Model Selection

| Model | Parameters | WASM Size | Quality |
|-------|------------|-----------|---------|
| T5-small | 60M | ~120MB | Good |
| BART-small | 70M | ~140MB | Better |
| DistilBART-6-6 | 37M | ~75MB | Best tradeoff |
| Pegasus-xsum | 90M | ~180MB | Highest |

**Recommended**: DistilBART-6-6 (37M params, 75MB WASM)

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                Browser Document Summarizer                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    JavaScript UI                          │   │
│  │  • File upload (drag & drop)                              │   │
│  │  • Progress indicator                                     │   │
│  │  • Summary display                                        │   │
│  │  • Length slider (short/medium/long)                      │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    Web Worker                             │   │
│  │  • Model loading (streaming)                              │   │
│  │  • Tokenization                                           │   │
│  │  • Inference                                              │   │
│  │  • Detokenization                                         │   │
│  └──────────────────────────────────────────────────────────┘   │
│                           │                                      │
│                           ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │                    WASM Runtime                           │   │
│  │  • APR model (.apr embedded)                              │   │
│  │  • Trueno SIMD kernels                                    │   │
│  │  • Memory management (64MB heap)                          │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Supported Document Formats

| Format | Parser | Max Size |
|--------|--------|----------|
| Plain Text | Native | 100KB |
| Markdown | pulldown-cmark | 100KB |
| PDF | pdf.js (JS) | 10MB |
| DOCX | docx-rs | 5MB |
| HTML | scraper | 500KB |

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Model Load | <5s | Streaming compilation |
| Cold Inference | <10s | 1000 token input |
| Warm Inference | <3s | Cached WASM |
| Memory Peak | <200MB | Browser limit |
| WASM Size | <80MB | Gzipped: <30MB |

### 25-Point QA Checklist - Demo G

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | `wasm-pack build` succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | wasm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic output | Reproducibility | Fixed seed |
| 11 | Chrome works | Browser | Manual test |
| 12 | Firefox works | Browser | Manual test |
| 13 | Safari works | Browser | Manual test |
| 14 | Mobile Safari works | Browser | iOS test |
| 15 | WASM <80MB | Size | wasm-opt |
| 16 | Load <5s on 4G | Performance | Throttle test |
| 17 | Inference <10s | Performance | Benchmark |
| 18 | Memory <200MB | Resources | DevTools |
| 19 | Handles large docs | Robustness | 100KB test |
| 20 | Graceful OOM | Resilience | Low-mem test |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Accessible UI | A11y | WCAG 2.1 AA |
| 24 | Privacy preserved | Security | No server calls |
| 25 | Offline capable | PWA | Service worker |

### Citations for Demo G

- [14] Shleifer & Rush (2020) - Pre-trained Summarization
- [24] Lewis et al. (2020) - BART
- [25] Raffel et al. (2020) - T5

---

## Demo H: Voice Recognition Pipeline

### `voice_recognition_whisper.rs`

**Category**: Audio/Speech
**Complexity**: High
**Estimated LOC**: 500-600

### Overview

Implement real-time voice recognition using Whisper-tiny or Whisper-base converted to APR format. Demonstrates audio preprocessing, streaming inference, and transcription output.

### Model Options

| Model | Parameters | Size | WER (LibriSpeech) |
|-------|------------|------|-------------------|
| Whisper-tiny | 39M | 75MB | 7.6% |
| Whisper-base | 74M | 145MB | 5.0% |
| Whisper-small | 244M | 488MB | 3.4% |

**Recommended**: Whisper-tiny (39M, 75MB APR)

### Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                  Voice Recognition Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Audio Input │────▶│  Preprocess  │────▶│  Mel Spectro │    │
│  │  (16kHz PCM) │     │  (Resample)  │     │  (80 bins)   │    │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                   │             │
│                                                   ▼             │
│  ┌──────────────┐     ┌──────────────┐     ┌──────────────┐    │
│  │  Text Output │◀────│  Decoder     │◀────│  Encoder     │    │
│  │  (UTF-8)     │     │  (Autoregr.) │     │  (Transformer)│   │
│  └──────────────┘     └──────────────┘     └──────────────┘    │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Audio Preprocessing

| Stage | Operation | Parameters |
|-------|-----------|------------|
| Resample | SRC to 16kHz | Sinc interpolation |
| Normalize | Peak normalize | -1.0 to 1.0 |
| Pad/Trim | Fixed length | 30s chunks |
| Mel Spectrogram | STFT + Mel | 80 bins, 25ms window |
| Log Transform | log(mel + 1e-10) | Stability |

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| RTF (Real-Time Factor) | <0.5 | 2x real-time |
| Latency | <500ms | Streaming chunks |
| WER | <10% | Clean speech |
| Memory | <500MB | Peak usage |

### 25-Point QA Checklist - Demo H

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic | Reproducibility | Fixed input |
| 11 | WER <10% clean | Accuracy | LibriSpeech |
| 12 | WER <20% noisy | Accuracy | MUSAN noise |
| 13 | RTF <0.5 | Performance | Benchmark |
| 14 | Streaming works | Functionality | Chunk test |
| 15 | Handles silence | Robustness | Empty audio |
| 16 | Handles noise | Robustness | SNR 5dB |
| 17 | Multi-language | Functionality | EN/ES/FR |
| 18 | Timestamp output | Functionality | Word-level |
| 19 | Memory bounded | Resources | 1hr audio |
| 20 | SIMD acceleration | Performance | Verified |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Sample audio included | Resources | 5 test files |
| 24 | Microphone input | Integration | Live demo |
| 25 | GPU acceleration (opt) | Performance | CUDA test |

### Citations for Demo H

- Radford et al. (2023) - Robust Speech Recognition (Whisper)
- Park et al. (2019) - SpecAugment

---

## Demo I: Handwriting Recognition

### `handwriting_recognition_mnist.rs`

**Category**: Vision/OCR
**Complexity**: Medium
**Estimated LOC**: 400-500

### Overview

Implement handwriting digit recognition using a compact CNN trained on MNIST/EMNIST. Demonstrates image preprocessing, convolutional inference, and real-time prediction with canvas input.

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    LeNet-5 Style CNN (APR)                       │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: 28x28x1 (grayscale)                                      │
│      │                                                           │
│      ▼                                                           │
│  Conv2D: 6 filters, 5x5, ReLU → 24x24x6                          │
│      │                                                           │
│      ▼                                                           │
│  MaxPool: 2x2 → 12x12x6                                          │
│      │                                                           │
│      ▼                                                           │
│  Conv2D: 16 filters, 5x5, ReLU → 8x8x16                          │
│      │                                                           │
│      ▼                                                           │
│  MaxPool: 2x2 → 4x4x16                                           │
│      │                                                           │
│      ▼                                                           │
│  Flatten: 256                                                    │
│      │                                                           │
│      ▼                                                           │
│  Dense: 120, ReLU                                                │
│      │                                                           │
│      ▼                                                           │
│  Dense: 84, ReLU                                                 │
│      │                                                           │
│      ▼                                                           │
│  Dense: 10, Softmax → Output                                     │
│                                                                  │
│  Total Parameters: ~61K (244KB F32, 61KB Q8)                     │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Dataset Options

| Dataset | Classes | Samples | Use Case |
|---------|---------|---------|----------|
| MNIST | 10 digits | 70K | Baseline |
| EMNIST-Digits | 10 digits | 280K | More data |
| EMNIST-Letters | 26 letters | 145K | Alphabet |
| EMNIST-Balanced | 47 classes | 131K | Alphanumeric |

### Image Preprocessing

| Stage | Operation | Output |
|-------|-----------|--------|
| Resize | Bilinear to 28x28 | 28x28 |
| Grayscale | Luminance | 1 channel |
| Normalize | /255.0 | [0, 1] |
| Center | Center of mass | Centered |
| Invert | If white background | Black bg |

### 25-Point QA Checklist - Demo I

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic | Reproducibility | Fixed seed |
| 11 | Accuracy >98% MNIST | Accuracy | Test set |
| 12 | Accuracy >95% EMNIST | Accuracy | Test set |
| 13 | Inference <5ms | Performance | Benchmark |
| 14 | Model <100KB Q8 | Size | Verified |
| 15 | Handles rotation ±15° | Robustness | Augmented |
| 16 | Handles scale 0.8-1.2x | Robustness | Augmented |
| 17 | Handles noise | Robustness | Gaussian σ=0.1 |
| 18 | Canvas input works | Integration | Browser |
| 19 | Confidence output | Functionality | Top-k probs |
| 20 | WASM compatible | Deployment | wasm-pack |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Live demo | UX | Interactive |
| 24 | Confusion matrix | Evaluation | Visualization |
| 25 | Mobile touch works | UX | iOS/Android |

### Citations for Demo I

- LeCun et al. (1998) - Gradient-Based Learning (LeNet)
- Cohen et al. (2017) - EMNIST Dataset

---

## Demo J: Image Classification

### `image_classification_mobilenet.rs`

**Category**: Vision/Classification
**Complexity**: High
**Estimated LOC**: 500-600

### Overview

Deploy MobileNetV2 or EfficientNet-B0 for general image classification (ImageNet-1K classes). Demonstrates efficient mobile-optimized inference with depthwise separable convolutions.

### Model Options

| Model | Parameters | Size | Top-1 Acc | MACs |
|-------|------------|------|-----------|------|
| MobileNetV2 | 3.4M | 14MB | 72.0% | 300M |
| MobileNetV3-Small | 2.5M | 10MB | 67.4% | 56M |
| EfficientNet-B0 | 5.3M | 21MB | 77.1% | 390M |
| ShuffleNetV2 | 2.3M | 9MB | 69.4% | 146M |

**Recommended**: MobileNetV3-Small (2.5M, 10MB Q8)

### Architecture Highlights

```
┌─────────────────────────────────────────────────────────────────┐
│               MobileNetV3-Small Architecture                     │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input: 224x224x3 (RGB)                                          │
│      │                                                           │
│      ▼                                                           │
│  Stem: Conv 3x3, stride 2 → 112x112x16                           │
│      │                                                           │
│      ▼                                                           │
│  MBConv Blocks (11 blocks):                                      │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │  • Depthwise Separable Convolutions                      │    │
│  │  • Squeeze-and-Excitation (SE) attention                 │    │
│  │  • Hard-Swish activation                                 │    │
│  │  • Residual connections                                  │    │
│  └─────────────────────────────────────────────────────────┘    │
│      │                                                           │
│      ▼                                                           │
│  Head: Conv 1x1 → Pool → FC → 1000 classes                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Image Preprocessing

| Stage | Operation | Parameters |
|-------|-----------|------------|
| Resize | Bilinear | 256x256 |
| Center Crop | Central | 224x224 |
| Normalize | ImageNet stats | μ=[0.485,0.456,0.406], σ=[0.229,0.224,0.225] |
| Channel Order | RGB | [0,1,2] |

### Performance Targets

| Metric | Target | Device |
|--------|--------|--------|
| Inference | <50ms | CPU (x86) |
| Inference | <100ms | CPU (ARM) |
| Inference | <20ms | GPU |
| Memory | <100MB | Peak |
| Throughput | >20 img/s | Batch |

### 25-Point QA Checklist - Demo J

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic | Reproducibility | Fixed input |
| 11 | Top-1 >65% | Accuracy | ImageNet val |
| 12 | Top-5 >85% | Accuracy | ImageNet val |
| 13 | Inference <50ms CPU | Performance | Benchmark |
| 14 | Model <15MB Q8 | Size | Verified |
| 15 | Handles JPEG | Format | Decode test |
| 16 | Handles PNG | Format | Decode test |
| 17 | Handles WebP | Format | Decode test |
| 18 | Batch inference | Functionality | 8 images |
| 19 | Top-k output | Functionality | k=5 labels |
| 20 | SIMD optimized | Performance | Verified |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Sample images | Resources | 10 test images |
| 24 | WASM compatible | Deployment | wasm-pack |
| 25 | Camera input | Integration | WebRTC demo |

### Citations for Demo J

- Sandler et al. (2018) - MobileNetV2
- Howard et al. (2019) - MobileNetV3
- Tan & Le (2019) - EfficientNet

---

## Demo K: RAG Pipeline Integration

### `rag_knowledge_retrieval.rs`

**Category**: Information Retrieval
**Complexity**: High
**Estimated LOC**: 600-700

### Overview

Construct a complete Retrieval-Augmented Generation (RAG) pipeline demonstrating document chunking, embedding generation, vector similarity search, and context injection into a generative model.

### Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    RAG System Architecture                       │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Document │───▶│ Chunking │───▶│ Embedder │───▶│ VectorDB │  │
│  │ Source   │    │ Strategy │    │ (.apr)   │    │ (in-mem) │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                                       ▲         │
│                                                       │ (Query) │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ User     │───▶│ Prompt   │───▶│ Context  │◀───│ Retriever│  │
│  │ Query    │    │ Template │    │ Fuser    │    │ (Top-k)  │  │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│                                       │                         │
│                                       ▼                         │
│                                  ┌──────────┐                   │
│                                  │ Generator│                   │
│                                  │ Model    │                   │
│                                  └──────────┘                   │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | Technology | Role |
|-----------|------------|------|
| **Embedder** | `bge-small-en` (APR) | Semantic representation |
| **Vector DB** | `usearch` / `hnsw` | Similarity search (HNSW) |
| **Generator** | `TinyLlama` / `Phi-2` | Answer synthesis |
| **Chunker** | RecursiveToken | Context window optimization |

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Retrieval Latency | <20ms | 100k vectors |
| End-to-End Latency | <500ms | First token |
| Recall@10 | >0.90 | MTEB benchmark |
| Indexing Speed | >1000 docs/s | Batch processing |

### 25-Point QA Checklist - Demo K

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic index | Reproducibility | Fixed seed |
| 11 | Retrieval precision >0.8 | Accuracy | Test corpus |
| 12 | Context window respected | Logic | Token count |
| 13 | Query latency <20ms | Performance | Benchmark |
| 14 | Index size <1.2x raw | Storage | Overhead |
| 15 | Handles duplicates | Robustness | Dedup test |
| 16 | Handles OOV words | Robustness | Fallback |
| 17 | Batch query support | Functionality | 10 queries |
| 18 | Metadata filtering | Functionality | Filter test |
| 19 | Index serialization | Persistence | Load/save |
| 20 | Incremental updates | Scale | Add/remove |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Example corpus included | Resources | Wikipedia sub |
| 24 | Hallucination check | Safety | Grounding |
| 25 | Memory cleanup | Resources | Drop trait |

### Citations for Demo K

- Lewis et al. (2020) - RAG: Retrieval-Augmented Generation
- Karpukhin et al. (2020) - Dense Passage Retrieval
- Malkov & Yashunin (2018) - HNSW Algorithm

---

## Demo L: Edge Anomaly Detection

### `edge_anomaly_detection.rs`

**Category**: IoT/Embedded
**Complexity**: High
**Estimated LOC**: 400-500

### Overview

Simulate processing sensor data streams on resource-constrained edge devices using lightweight unsupervised learning models (Isolation Forest or Autoencoder) to detect anomalies in real-time.

### Constraints

| Resource | Limit | Reason |
|----------|-------|--------|
| Flash Storage | <256KB | MCU constraint |
| RAM | <64KB | MCU constraint |
| Power | <10mW | Battery life |
| Latency | <1ms | Real-time control |

### Algorithms

| Model | Complexity | Size | Suitability |
|-------|------------|------|-------------|
| Isolation Forest | O(log n) | 50KB | High dim data |
| Autoencoder (Dense) | O(n) | 20KB | Complex patterns |
| One-Class SVM | O(n^2) | 100KB | Well-defined normal |
| LOF | O(n log n) | Variable | Density based |

**Recommended**: Micro-Autoencoder (3-layer dense, 8-bit quantized)

### 25-Point QA Checklist - Demo L

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | `cargo build --target thumbv7em` | Build | No std |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | `no_std` compatible | Compatibility | Core only |
| 10 | No heap allocation | Safety | Static memory |
| 11 | Detects anomalies | Accuracy | ROC >0.9 |
| 12 | False alarm rate <1% | Accuracy | Normal data |
| 13 | Inference <1ms | Performance | Cycles |
| 14 | Model size <50KB | Size | Binary analysis |
| 15 | Integer arithmetic only | Compatibility | Soft-float check |
| 16 | Power consumption est | Efficiency | Model count |
| 17 | Sensor noise robust | Robustness | White noise |
| 18 | Drift adaptation | Functionality | Re-calibration |
| 19 | Serialization support | Interop | Postcard |
| 20 | Panic handler safe | Safety | Reset logic |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Simulator included | Tools | PC harness |
| 24 | C FFI bindings | Interop | C header |
| 25 | Stack usage <4KB | Resources | Stack analysis |

### Citations for Demo L

- Liu et al. (2008) - Isolation Forest
- Sakurada & Yairi (2014) - Autoencoders for Anomaly Detection
- Banbury et al. (2020) - TensorFlow Lite Micro

---

## Demo M: Real-time Style Transfer

### `style_transfer_art.rs`

**Category**: Creative/Generative
**Complexity**: High
**Estimated LOC**: 550-650

### Overview

Apply artistic styles to video streams or images in real-time using a lightweight Fast Style Transfer network. Demonstrates advanced image processing, convolution optimization, and aesthetic quantification.

### Model Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                 Fast Style Transfer Network                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Image (Content)                                           │
│       │                                                          │
│       ▼                                                          │
│  Encoder (Reflection Padding, Strided Conv)                      │
│       │                                                          │
│       ▼                                                          │
│  Residual Blocks (5x, Instance Norm, ReLU)                       │
│       │                                                          │
│       ▼                                                          │
│  Decoder (Upsampling, Conv)                                      │
│       │                                                          │
│       ▼                                                          │
│  Stylized Output                                                 │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Targets

| Resolution | FPS (CPU) | FPS (GPU) | Latency |
|------------|-----------|-----------|---------|
| 256x256 | 30 | 120 | <33ms |
| 512x512 | 10 | 60 | <100ms |
| 1080p | 1 | 15 | <1s |

### 25-Point QA Checklist - Demo M

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic output | Reproducibility | Pixel match |
| 11 | Perceptual loss <0.1 | Quality | VGG metric |
| 12 | Temporal stability | Quality | Video jitter |
| 13 | FPS >20 (256p) | Performance | Benchmark |
| 14 | Memory <500MB | Resources | Peak |
| 15 | Arbitrary style opt | Functionality | AdaIN |
| 16 | Intensity control | Functionality | Alpha blend |
| 17 | Region masking | Functionality | Segmentation |
| 18 | SIMD kernels used | Performance | Verified |
| 19 | Multithreaded decode | Performance | Rayon |
| 20 | WebAssembly target | Deployment | Canvas demo |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Pre-trained styles | Resources | 5 models |
| 24 | Webcam support | Integration | v4l2 |
| 25 | Export to GIF/MP4 | Interop | Encoding |

### Citations for Demo M

- Gatys et al. (2015) - A Neural Algorithm of Artistic Style
- Johnson et al. (2016) - Perceptual Losses for Real-Time Style Transfer
- Huang & Belongie (2017) - Arbitrary Style Transfer (AdaIN)

---

## Demo N: Streaming Sentiment Analysis

### `streaming_sentiment_analysis.rs`

**Category**: NLP/Streaming
**Complexity**: Medium-High
**Estimated LOC**: 450-550

### Overview

High-throughput sentiment analysis on simulated social media firehose data. Focuses on efficient batching, async pipeline stages, backpressure handling, and throughput maximization.

### Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                 Streaming Analysis Pipeline                      │
├─────────────────────────────────────────────────────────────────┤
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐  │
│  │ Source   │───▶│ Tokenizer│───▶│ Inference│───▶│ Aggregator│ │
│  │ (Kafka)  │    │ (Batch)  │    │ (Batch)  │    │ (Window)  │ │
│  └──────────┘    └──────────┘    └──────────┘    └──────────┘  │
│       ▲               ▲               ▲               │         │
│       │               │               │               ▼         │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Backpressure Control                  │   │
│  │  • Channel capacity monitoring                           │   │
│  │  • Adaptive batch sizing                                 │   │
│  │  • Load shedding (if permitted)                          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### Performance Targets

| Metric | Target | Notes |
|--------|--------|-------|
| Throughput | >5000 docs/s | DistilBERT |
| Latency p99 | <50ms | Per batch |
| Batch Size | Adaptive | 16-256 |
| Memory | <2GB | Stable |

### 25-Point QA Checklist - Demo N

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic replay | Reproducibility | Log replay |
| 11 | Throughput >5k/s | Performance | Benchmark |
| 12 | Backpressure active | Resilience | Load test |
| 13 | Graceful shutdown | Resilience | Signal test |
| 14 | No dropped msgs | Reliability | Count check |
| 15 | Adaptive batching | Efficiency | Monitor |
| 16 | Tokenizer parallel | Performance | Threading |
| 17 | Async runtime | Concurrency | Tokio |
| 18 | Metrics exposure | Ops | Prometheus |
| 19 | JSON parsing fast | Performance | serde_json |
| 20 | Model reload safe | Ops | Hot swap |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Kafka/TCP mock | Testing | Integration |
| 24 | Dashboard output | UX | TUI |
| 25 | Memory leak check | Resources | Valgrind |

### Citations for Demo N

- Dean & Ghemawat (2008) - MapReduce (Batching concepts)
- Kleppmann (2017) - Designing Data-Intensive Applications
- Vaswani et al. (2017) - Attention Is All You Need

---

## Demo O: Multi-Modal CLIP Search

### `multimodal_clip_search.rs`

**Category**: Multi-Modal
**Complexity**: High
**Estimated LOC**: 550-650

### Overview

Implement a search engine that understands both images and text using OpenAI's CLIP model. Users can search for images using natural language, or find similar images using an image query.

### Shared Latent Space

```
┌─────────────────────────────────────────────────────────────────┐
│                    CLIP Joint Embedding Space                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌────────────┐                                  ┌────────────┐  │
│  │ Text       │───▶[Text Encoder]───┐            │ Image      │  │
│  │ "A dog"    │                     │            │ (Pixels)   │  │
│  └────────────┘                     ▼            └────────────┘  │
│                              ● ◀── (Similarity) ───┐             │
│                              │                     │             │
│                              ▼                     ▼             │
│                        [Latent Space (512-dim)] ◀──[Image Encoder]│
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Capabilities

1. **Text-to-Image**: "Find a photo of a sunset at the beach"
2. **Image-to-Image**: "Find images that look like this one"
3. **Zero-Shot Classification**: Classify images without training
4. **Ranking**: Sort a gallery by relevance to a query

### 25-Point QA Checklist - Demo O

| # | Check | Category | Pass Criteria |
|---|-------|----------|---------------|
| 1 | Build succeeds | Build | Exit 0 |
| 2 | Tests pass | Test | 100% |
| 3 | Clippy clean | Lint | 0 warnings |
| 4 | Format clean | Format | Pass |
| 5 | Docs >90% | Docs | Verified |
| 6 | Coverage >95% | Coverage | llvm-cov |
| 7 | Property tests | Testing | proptest |
| 8 | No unwrap() | Safety | Verified |
| 9 | Error handling | Errors | Review |
| 10 | Deterministic emb | Reproducibility | Fixed seed |
| 11 | Top-1 Accuracy >30% | Accuracy | Zero-shot |
| 12 | Top-5 Accuracy >60% | Accuracy | Zero-shot |
| 13 | Encoder latency <50ms | Performance | CPU |
| 14 | Index lookup <10ms | Performance | 10k items |
| 15 | Image decoding fast | Performance | image-rs |
| 16 | Text tokenization | Functionality | BPE |
| 17 | Normalization correct| Correctness | CLIP stats |
| 18 | Cosine similarity | Correctness | Verified |
| 19 | Batch processing | Functionality | Gallery indexing |
| 20 | Memory mapped index | Scale | 100k items |
| 21 | IIUR compliance | Methodology | Isolation |
| 22 | Toyota Way documented | Process | README |
| 23 | Gallery dataset | Resources | Unsplash 1k |
| 24 | Web interface | UX | Search bar |
| 25 | Model quantization | Size | Int8 support |

### Citations for Demo O

- Radford et al. (2021) - CLIP: Learning Transferable Visual Models
- Schuhmann et al. (2022) - LAION-5B Dataset

---

## Peer-Reviewed Citations

### Core ML & Deep Learning

| # | Citation | Topic |
|---|----------|-------|
| [1] | Wolf, T., et al. (2020). "Transformers: State-of-the-art natural language processing." EMNLP Demo. | NLP Framework |
| [2] | Reimers, N., & Gurevych, I. (2019). "Sentence-BERT: Sentence embeddings using Siamese BERT-networks." EMNLP. | Embeddings |
| [3] | Megiddo, N., & Modha, D. S. (2003). "ARC: A self-tuning, low overhead replacement cache." FAST. | Caching |
| [4] | O'Neil, E. J., et al. (1993). "The LRU-K page replacement algorithm for database disk buffering." SIGMOD. | Caching |
| [5] | Mitchell, M., et al. (2019). "Model cards for model reporting." FAT*. | Documentation |
| [6] | Gebru, T., et al. (2021). "Datasheets for datasets." CACM. | Documentation |
| [7] | Dettmers, T., et al. (2022). "LLM.int8(): 8-bit matrix multiplication for transformers at scale." NeurIPS. | Quantization |
| [8] | Frantar, E., et al. (2023). "GPTQ: Accurate post-training quantization for generative pre-trained transformers." ICLR. | Quantization |
| [9] | Lin, J., et al. (2023). "AWQ: Activation-aware weight quantization for LLM compression." MLSys. | Quantization |
| [10] | Crammer, K., et al. (2006). "Online passive-aggressive algorithms." JMLR. | Online Learning |

### Specialized Algorithms & New Additions

| # | Citation | Topic |
|---|----------|-------|
| [11] | Lakshminarayanan, B., et al. (2014). "Mondrian forests: Efficient online random forests." NeurIPS. | Online Learning |
| [12] | McInnes, L., et al. (2018). "UMAP: Uniform manifold approximation and projection." JOSS. | Dim Reduction |
| [13] | McInnes, L., et al. (2017). "HDBSCAN: Hierarchical density-based clustering." JOSS. | Clustering |
| [14] | Shleifer, S., & Rush, A. (2020). "Pre-trained summarization distillation." arXiv:2010.13002. | Summarization |
| [15] | Sanh, V., et al. (2019). "DistilBERT, a distilled version of BERT." NeurIPS Workshop. | Distillation |
| [16] | Waldspurger, C., et al. (2015). "Efficient MRC curve computation for analyzing cache effectiveness." USENIX ATC. | Caching |
| [17] | Hutchinson, B., et al. (2021). "Towards accountability for machine learning datasets." FAT*. | Fairness |
| [18] | Jacob, B., et al. (2018). "Quantization and training of neural networks for efficient integer-arithmetic-only inference." CVPR. | Quantization |
| [19] | Nagel, M., et al. (2021). "A white paper on neural network quantization." arXiv:2106.08295. | Quantization |
| [20] | Gama, J., et al. (2014). "A survey on concept drift adaptation." ACM Computing Surveys. | Concept Drift |
| [21] | Bifet, A., & Gavalda, R. (2007). "Learning from time-changing data with adaptive windowing." SDM. | Drift Detection |
| [22] | van der Maaten, L. (2014). "Accelerating t-SNE using tree-based algorithms." JMLR. | Visualization |
| [23] | Amid, E., & Warmuth, M. K. (2019). "TriMap: Large-scale dimensionality reduction." arXiv:1910.00204. | Visualization |
| [24] | Lewis, M., et al. (2020). "BART: Denoising sequence-to-sequence pre-training." ACL. | Summarization |
| [25] | Raffel, C., et al. (2020). "Exploring the limits of transfer learning with T5." JMLR. | Summarization |
| [26] | Radford, A., et al. (2021). "Learning Transferable Visual Models From Natural Language Supervision." ICML. | CLIP/Multi-Modal |
| [27] | Gatys, L., et al. (2015). "A Neural Algorithm of Artistic Style." NIPS. | Style Transfer |
| [28] | Liu, F. T., et al. (2008). "Isolation Forest." ICDM. | Anomaly Detection |
| [29] | Lewis, P., et al. (2020). "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks." NeurIPS. | RAG |
| [30] | Radford, A., et al. (2023). "Robust Speech Recognition via Large-Scale Weak Supervision." ICML. | Whisper/ASR |
| [31] | LeCun, Y., et al. (1998). "Gradient-Based Learning Applied to Document Recognition." Proc. IEEE. | LeNet/CNN |
| [32] | Howard, A., et al. (2019). "Searching for MobileNetV3." ICCV. | Efficient Vision |
| [33] | Karpukhin, V., et al. (2020). "Dense Passage Retrieval for Open-Domain QA." EMNLP. | DPR/Retrieval |
| [34] | Johnson, J., et al. (2016). "Perceptual Losses for Real-Time Style Transfer." ECCV. | Fast Style |
| [35] | Malkov, Y., & Yashunin, D. (2018). "Efficient and Robust Approximate Nearest Neighbor Search." IEEE TPAMI. | HNSW |

---

## Appendix A: Demo Cross-References

### Shared Components

| Component | Used By | Description |
|-----------|---------|-------------|
| **Tokenizer** | A, G, K, N | BPE/WordPiece tokenization |
| **Embedding Model** | A, F, K, O | Sentence/image embeddings |
| **Vector Index** | F, K, O | HNSW similarity search |
| **WASM Runtime** | G, I, J, M, O | Browser deployment |
| **Cache System** | A, B, K | Hierarchical caching |
| **Quantization** | D, I, J, L, O | Model compression |

### Learning Path

```
Beginner → Intermediate → Advanced
    │           │            │
    ▼           ▼            ▼
   [I]        [C,D]        [A,K]
   [J]        [B,F]        [E,N]
              [G,H]        [L,M,O]
```

### Dependency Graph

```
Demo A (Multi-Model) ──────────────────┐
    │                                  │
    ├──▶ Demo B (Cache) ◀──────────────┤
    │         │                        │
    │         ▼                        │
    │    Demo K (RAG) ◀── Demo F (Viz) │
    │         │                        │
    └─────────┴───────▶ Demo O (CLIP) ─┘

Demo C (Inspect) ──▶ Demo D (Quantize) ──▶ Demo L (Edge)
                           │
                           ▼
                    Demo I (MNIST)
                    Demo J (ImageNet)

Demo E (Online) ◀── Renacer traces

Demo G (Summarizer) ◀── Demo N (Sentiment) [shared NLP]

Demo H (Voice) ◀── Demo M (Style) [media processing]
```

---

## Appendix B: Toyota Way Application

### Principle Mapping to Demos

| Principle | Japanese | Application |
|-----------|----------|-------------|
| **Jidoka** | 自働化 | Quality built-in: All demos have 25-point checklists, property tests, and automated validation |
| **Heijunka** | 平準化 | Level loading: Deterministic memory allocation, predictable performance |
| **Genchi Genbutsu** | 現地現物 | Go and see: Comprehensive inspection (Demo C), real metrics |
| **Kaizen** | 改善 | Continuous improvement: Online learning (Demo E), model versioning |
| **Poka-yoke** | ポカヨケ | Error-proofing: Type-safe APIs, verification levels (NASA/ISO) |
| **Muda** | 無駄 | Eliminate waste: Quantization (Demo D), caching (Demo B) |
| **Mura** | 斑 | Eliminate inconsistency: Deterministic builds, seeded RNG |
| **Muri** | 無理 | Eliminate overburden: Memory budgets, streaming loading |

### Quality Gate Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                    Toyota Way Quality Gates                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Tier 1: On-Save (<1s) - Jidoka                                  │
│  ├── cargo fmt --check                                           │
│  ├── cargo clippy                                                │
│  └── cargo check                                                 │
│                                                                  │
│  Tier 2: Pre-Commit (<5s) - Poka-yoke                            │
│  ├── cargo test --lib                                            │
│  ├── 25-point checklist items 1-9                                │
│  └── No unwrap() verification                                    │
│                                                                  │
│  Tier 3: Pre-Push (1-5min) - Kaizen                              │
│  ├── cargo test --all                                            │
│  ├── Coverage >95%                                               │
│  ├── Property tests (100+ cases)                                 │
│  └── 25-point checklist items 10-20                              │
│                                                                  │
│  Tier 4: CI/CD (5-60min) - Genchi Genbutsu                       │
│  ├── Mutation testing                                            │
│  ├── Performance benchmarks                                      │
│  ├── Security audit                                              │
│  └── 25-point checklist items 21-25                              │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Implementation Timeline

| Demo | Priority | Complexity | Dependencies |
|------|----------|------------|--------------|
| C - Inspection | P0 | Medium | None |
| D - Quantization | P0 | High | None |
| B - Cache | P1 | Medium | None |
| A - Multi-Model | P1 | High | Batuta |
| I - Handwriting | P2 | Medium | None |
| J - Image | P2 | High | Model conversion |
| K - RAG | P2 | High | Vector DB |
| L - Edge Anomaly | P2 | High | No-std |
| E - Online Training | P3 | High | Renacer |
| F - Visualization | P3 | High | .prs format |
| G - Summarizer | P3 | High | WASM, HF model |
| H - Voice | P3 | High | Audio processing |
| M - Style Transfer| P3 | High | Image proc |
| N - Sentiment | P3 | Medium | Async |
| O - CLIP Search | P3 | High | Multi-modal |

---

## Appendix C: Acceptance Criteria

### Per-Demo Gate Requirements

Before any demo is considered complete, it **MUST** pass:

| Gate | Requirement | Verification |
|------|-------------|--------------|
| **Build** | `cargo build --release` | Exit 0, no warnings |
| **Test** | `cargo test` | 100% pass |
| **Lint** | `cargo clippy -- -D warnings` | 0 warnings |
| **Format** | `cargo fmt --check` | Clean |
| **Coverage** | `cargo llvm-cov` | ≥95% lines |
| **Docs** | `cargo doc` | ≥90% coverage |
| **Safety** | No `unwrap()` in logic | grep verification |
| **Property** | proptest (100+ cases) | All pass |
| **Benchmark** | Criterion suite | No regression |
| **Security** | `cargo audit` | 0 vulnerabilities |

### Specification Acceptance

This specification is **APPROVED** when:

- [ ] All 15 demos have implementation plans reviewed
- [ ] Resource estimates validated by team
- [ ] Risk mitigations accepted
- [ ] Priority ordering confirmed
- [ ] Toyota Way compliance verified
- [ ] Citation completeness checked (35 refs)
- [ ] Cross-reference accuracy verified

---

## Document Control

| Version | Date | Author | Changes |
|---------|------|--------|---------|
| 1.3.0-draft | 2025-12-08 | Claude Code | Added metrics, risk assessment, cross-refs, 6 new citations |
| 1.2.0-draft | 2025-12-08 | Gemini Agent | Refined executive summary, dependencies, and formatting |
| 1.1.0-draft | 2025-12-08 | Gemini Agent | Added Demos K-O (RAG, Edge, Style, Sentiment, CLIP) |
| 1.0.0-draft | 2025-12-08 | Claude Code | Initial specification (Demos A-J) |

**Status**: IN REVIEW

**Next Actions**:
1. Team review of priority ordering
2. Validate resource estimates
3. Confirm model selection for each demo
4. Approve and begin P0 implementation (C, D)

---

*This specification follows the Toyota Way principles of thoroughness (Genchi Genbutsu), quality built-in (Jidoka), and continuous improvement (Kaizen). Each demo is designed to be production-ready with comprehensive testing and documentation.*

**Total Specification Length**: ~1,700 lines | **Review Time Estimate**: 45-60 minutes
