# Appendices

---

## A. Feature Flag Matrix

| Feature | Description | Recipes |
|---------|-------------|---------|
| `default` | Core functionality | A.*, B.1-B.2, C.*, D.*, J.*, L.1-L.2 |
| `encryption` | AES-256-GCM | B.3, E.* |
| `signing` | Ed25519 signatures | B.4, E.*, L.4 |
| `gpu` | GPU acceleration | I.* |
| `browser` | WASM target | H.* |
| `pacha` | Model registry | E.* |
| `realizar` | Model serving | F.*, G.* |
| `presentar` | UI widgets | H.2-H.4 |
| `hf-hub` | HuggingFace integration | D.1, K.* |
| `lambda` | AWS Lambda support | B.5, G.* |
| `full` | All features | All recipes |

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
| T5 | `distributed` | Multi-node | Cluster via repartir work-stealing scheduler |

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
| A: Model Creation | 5 | Implemented |
| B: Binary Bundling | 5 | Implemented |
| C: Continuous Training | 4 | Implemented |
| D: Format Conversion | 5 | Implemented |
| E: Model Registry | 4 | Implemented |
| F: API Integration | 4 | Implemented |
| G: Serverless Deployment | 4 | Implemented |
| H: WASM & Browser | 5 | Implemented |
| I: GPU Acceleration | 4 | Implemented |
| J: SIMD Acceleration | 4 | Implemented |
| K: Model Distillation | 4 | Implemented |
| L: CLI Tools | 4 | Implemented |
| **Total IIUR** | **52** | **100%** |

### CLI Demo Status

| Category | Recipes | Status |
|----------|---------|--------|
| optimize/ | 22 | In progress |
| chat/ | 5 | In progress |
| analysis/ | 11 | In progress |
| format/ | 10 | In progress |
| **Total CLI** | **48** | **Phased rollout** |

### Five Invariants Status

| Invariant | Gate | Status |
|-----------|------|--------|
| A: CLI Recipe Parity | `make cli-parity` | 57/57 = 100% |
| B: Recipe Contract Grade A | `make contract-grade` | 11 contracts (in progress) |
| C: Model Format Coverage | `make format-coverage` | TBD |
| D: arXiv Citation | `make citation-check` | TBD |
| E: Docs Contract Coverage | `make docs-validate` | partial |

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

- [ ] All 52 IIUR recipes execute without error
- [ ] mdbook builds successfully
- [ ] CI pipeline passes on all platforms
- [ ] Git hooks enforce quality gates
- [ ] Documentation matches implementation
- [ ] IIUR principles verified per recipe
- [ ] Security review of encryption/signing recipes
- [ ] Performance benchmarks validated
