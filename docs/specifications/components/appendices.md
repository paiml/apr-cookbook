# Appendices

---

## A. Feature Flag Matrix

| Feature | Description | Recipes |
|---------|-------------|---------|
| `default` | Core functionality — all recipes except B.3 | A.*, B.1-B.2/B.4-B.7, C.*, D.*, E.*, F.*, G.*, H.*, I.*, J.*, K.*, L.* |
| `encryption` | AES-256-GCM model encryption | B.3 |
| `full` | All features (`encryption`) | All recipes |

**Note**: Only two feature flags exist in Cargo.toml. GPU, distributed, and speech patterns are **simulated** in cookbook examples — no feature flag needed.

---

## B. Device Tier Classification

Every recipe is classified by hardware requirements. The runtime fallback chain is automatic: GPU -> SIMD -> Scalar.

### Tier Definitions

| Tier | Label | Device | Description |
|------|-------|--------|-------------|
| T0 | `cpu` | Any CPU | Universal scalar baseline — runs everywhere |
| T1a | `x86_64` | Intel/AMD | SIMD acceleration: SSE4.2, AVX2, AVX-512 |
| T1b | `aarch64` | ARM | SIMD acceleration: NEON (always available on aarch64) |
| T2a | `cuda` | NVIDIA GPU | CUDA cores, PTX kernels, tensor cores |
| T2b | `wgpu` | WebGPU | Browser/native GPU compute via wgpu (Vulkan/Metal/DX12/WebGPU) |
| T3 | `wasm` | wasm32 | Browser/edge via WebAssembly + SIMD128 |
| T4 | `serverless` | Lambda/Edge | Constrained CPU, cold start budget, memory cap |
| T5 | `distributed` | Multi-node | Cluster compute (simulated in cookbook) |

### Detection Mechanisms

| Tier | Detection | Fallback |
|------|-----------|----------|
| T1a | `is_x86_feature_detected!("avx2")` / `"avx512f"` | T0 scalar |
| T1b | `cfg!(target_arch = "aarch64")` — NEON always present | T0 scalar |
| T2a | `APR_GPU_ENABLED` env var, CUDA device enumeration | T1 SIMD -> T0 |
| T2b | wgpu adapter detection | T1 SIMD -> T0 |
| T3 | `cfg(target_arch = "wasm32")` | N/A (different target) |
| T4 | Runtime environment (Lambda bootstrap) | N/A |
| T5 | Node topology config | N/A |

### Category-to-Device Mapping

| Category | T0 cpu | T1a x86_64 | T1b aarch64 | T2a cuda | T2b wgpu | T3 wasm | T4 serverless | T5 distributed |
|----------|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Creation | + | | | | | | | |
| Bundling | + | | | | | | | |
| Training | + | | | | | | | |
| Conversion | + | | | | | | | |
| Registry | + | | | | | | | |
| API | + | | | | | | | |
| Monitoring | + | | | | | | | |
| Speech | + | | | | | | | |
| Inference | + | | | | | | | |
| Serving | + | | | | | | | |
| Optimize | + | | | | | | | |
| Chat | + | | | | | | | |
| Analysis | + | | | | | | | |
| Format | + | | | | | | | |
| Distillation | + | | | | | | | |
| SIMD | + | + | + | | | | | |
| Acceleration | + | + | + | | | | | |
| GPU | + | + | + | + | + | | | |
| CLI | + | + | + | | | + | | |
| WASM | | | | | + | + | | |
| Advanced | + | | | | | + | | |
| Serverless | + | | | | | | + | |
| Distributed | + | | | | | | | + |

---

## C. Recipe Compliance Checklist

Before submitting a recipe, verify:

**Isolation**
- [ ] Uses `tempfile::tempdir()` for all file I/O
- [ ] No global/static mutable state

**Idempotency**
- [ ] Fixed RNG seed via `RecipeContext`
- [ ] Running twice produces identical output

**Usefulness**
- [ ] Addresses real production use case
- [ ] Copy-paste ready code

**Reproducibility**
- [ ] Works on Linux, macOS, WASM
- [ ] Pinned dependency versions

**Testing**
- [ ] 95%+ line coverage
- [ ] 3+ proptest properties
- [ ] Idempotency test present
- [ ] Isolation test present

**Documentation**
- [ ] Module doc with run command
- [ ] Learning objective stated

**Quality**
- [ ] No unnecessary abstraction (Muda)
- [ ] Error handling via types (Jidoka)
- [ ] 10-point QA checklist included and verified

---

## C. Implementation Status

### Recipe Implementation Matrix

| Category | Recipes | Status |
|----------|---------|--------|
| A: Model Creation | 7 | Implemented |
| B: Binary Bundling | 7 | Implemented |
| C: Continuous Training | 16 | Implemented |
| D: Format Conversion | 5 | Implemented |
| E: Model Registry | 5 | Implemented |
| F: API Integration | 5 | Implemented |
| G: Serverless Deployment | 5 | Implemented |
| H: WASM & Browser | 6 | Implemented |
| I: GPU Acceleration | 8 | Implemented |
| J: SIMD Acceleration | 6 | Implemented |
| K: Model Distillation | 5 | Implemented |
| L: CLI Tools | 16 | Implemented |
| **Total IIUR** | **91** | **100%** |

### CLI Demo Status

| Category | Recipes | Status |
|----------|---------|--------|
| optimize/ | 23 | Implemented |
| chat/ | 5 | Implemented |
| analysis/ | 25 | Implemented |
| format/ | 11 | Implemented |
| **Total CLI** | **64** | **100%** |

### Five Invariants Status

| Invariant | Gate | Baseline | Status |
|-----------|------|----------|--------|
| A: CLI Recipe Parity | `make cli-parity` | 66/66 (100%) | **ENFORCED** |
| B: Recipe Contract Grade A | `make contract-grade` | 341/341 (100%) | **ENFORCED** |
| C: Model Format Coverage | `make format-coverage` | 341/341 (100%) | **ENFORCED** |
| D: arXiv Citation | `make citation-check` | 341/341 (100%) | **ENFORCED** |
| E: Docs Contract Coverage | `make docs-validate` | 264/267 (98.9%) | **ENFORCED** |
| F: Variant Depth (≥3 recipes/sub) | `make variant-depth` | 66/66 (100%) | **ENFORCED** |

### Quality Gates Summary

```
Pre-commit:   O(1) checks, <30s    Passing
Pre-push:     Full test suite       Passing
CI:           Multi-platform        Passing
Coverage:     95%+ minimum          Verified
```

---

## D. Approval

**Status**: ACTIVE

| Role | Name | Date |
|------|------|------|
| Author | Sovereign AI Stack Team | 2026-03-17 |

### QA Review Checklist

- [x] All 341 recipes execute without error (330/341 pass <10s, 11 compute-heavy benchmarks need 60s — verified 2026-04-23)
- [ ] mdbook builds successfully
- [ ] CI pipeline passes on all platforms
- [ ] Git hooks enforce quality gates
- [ ] Documentation matches implementation
- [ ] IIUR principles verified per recipe
- [ ] Security review of encryption/signing recipes
- [ ] Performance benchmarks validated
