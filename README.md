<p align="center">
  <img src=".github/apr-cookbook-hero.svg" width="800" alt="apr-cookbook">
</p>

<h1 align="center">apr-cookbook</h1>

<p align="center">
  <strong>The umbrella cookbook for the PAIML sovereign AI stack — model bundling, data loading, deployment-as-recipe, and visualization, all in pure Rust.</strong>
</p>

<p align="center">
  <em>v6.4.0 (2026-05-18) — <strong>canonical model-publish workflow now lives in aprender</strong>. <a href="https://github.com/paiml/aprender/blob/main/docs/specifications/aprender-train/model-hf-publish-pipeline-spec.md">SPEC-HF-PUBLISH-001</a> (in the aprender repo) codifies the 12-file minimum, YAML schema, NDJSON-commit rule, LFS-batch flow, 13-tier crates.io cascade, and three-path verification (Rust + HF Transformers + llama.cpp) for any model published to HuggingFace Hub via <code>apr publish</code>. First applied 2026-05-18 to ship <a href="https://huggingface.co/paiml/albor-370m-v1"><code>paiml/albor-370m-v1</code></a> end-to-end. The cookbook's <code>cli_publish_*</code> + <code>hub_publishing</code> recipes (recipes 388-392, 759) operate within this spec for the full workflow context.</em>
</p>

<p align="center">
  <em>v6.3.0 (2026-05-10) closes the <a href="docs/specifications/fine-tuning-cookbook.md">fine-tuning-cookbook</a> sprint: <strong>155 new fine-tuning recipes</strong> across a 4-tier curriculum (Tier 1 SFT/eval/tabular × 25, Tier 2 LoRA/QLoRA/PEFT/CP/merge × 45, Tier 3 calibration/multimodal/QAT/etc × 48, Tier 4 DPO/ORPO/KTO/GRPO/PPO/RLAIF × 37). Mirrors Ludwig + Unsloth + TRL + LLaMA-Factory + Axolotl. Each recipe ships 4 tests (recipe_runs / falsifier_holds / falsifier_breaks / deterministic). 100% pass rate when run as standalone binaries. See <a href="examples/finetune/README.md"><code>examples/finetune/README.md</code></a>.</em>
</p>

<p align="center">
  <em>v6.2.0 (2026-05-07) closes the <a href="docs/specifications/architecture-demos.md">architecture-demos</a> sprint: 18 new family-smoke recipes + provable-contracts covering the full HF Transformers architecture surface that <code>aprender::rosetta</code> ships descriptors for (Llama, Mistral, Qwen2/3/3.5, Phi, Gemma, GPT-2, GPT-NeoX, DeepSeek, Falcon-H1, RWKV-7, OpenELM, OPT, MAMBA, BERT, plus Whisper + Moonshine in <code>examples/speech/</code>). Manifest-driven CI gate (<code>make architecture-demos-coverage</code>) reconciles upstream loader support against on-disk recipes and contracts.</em>
</p>

<p align="center">
  <em>v6.1.0 (2026-05-05) closes the <a href="docs/specifications/expand-cookbooks.md">expand-cookbooks</a> sprint: 44 new recipes covering Claude Code parity (<code>apr code</code>), GPU/CPU oracle bisection, MCP M5 transports, Anthropic Messages API drop-in, end-to-end publish, and 6 sister crates (<code>aprender-{mcp,tsp,shell,monte-carlo,cgp,contracts-macros}</code>). 420 recipes across 34 categories.</em>
</p>

<p align="center">
  <em>v6.0.0 (2026-05-05) consolidated sovereign-ai-cookbook, alimentar examples, and presentar examples into this repository per <a href="docs/specifications/centralize-cookbooks.md">centralize-cookbooks</a>. Source repositories archived 2026-05-05 (PMAT-070): <s><a href="https://github.com/paiml/sovereign-ai-cookbook">sovereign-ai-cookbook</a></s>, <s><a href="https://github.com/paiml/alimentar">alimentar</a></s>, <s><a href="https://github.com/paiml/presentar">presentar</a></s>. The alimentar and presentar Rust crates remain published on crates.io as <code>aprender-data</code> and <code>presentar</code> respectively.</em>
</p>

<p align="center">
  <a href="https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml">
    <img src="https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml/badge.svg" alt="CI">
  </a>
  <a href="book/src/introduction.md">
    <img src="https://img.shields.io/badge/book-local-blue" alt="Book">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="MIT License">
  </a>
  <a href="https://www.rust-lang.org/">
    <img src="https://img.shields.io/badge/rust-1.89%2B-orange.svg" alt="Rust 1.89+">
  </a>
</p>

<p align="center">
  <a href="#what-is-apr-cookbook">What</a> |
  <a href="#installation">Install</a> |
  <a href="#quick-start">Quick Start</a> |
  <a href="#examples">Examples</a> |
  <a href="#apr-v2-format">APR v2</a> |
  <a href="#sovereign-ai-stack">Stack</a> |
  <a href="#documentation">Docs</a>
</p>

---

## What is apr-cookbook?

341 executable examples across 25 categories covering the full ML
model lifecycle in pure Rust: creation, bundling, format conversion,
training, optimization, inference, serving, monitoring, and deployment
-- all built on the APR v2 model format with LZ4/ZSTD compression,
Int4/Int8 quantization, and Ed25519 signatures.

Every example compiles, runs, and includes unit tests. Device fallback
is automatic: GPU -> SIMD -> Scalar.

## Table of Contents

- [Installation](#installation)
- [Quick Start](#quick-start)
- [Examples](#examples)
- [APR v2 Format](#apr-v2-format)
- [Quality](#quality)
- [Sovereign AI Stack](#sovereign-ai-stack)
- [Live Demos](#live-demos)
- [Documentation](#documentation)
- [License](#license)

## Installation

```bash
git clone https://github.com/paiml/apr-cookbook.git
cd apr-cookbook
cargo build --examples
cargo test --all-features
```

Requires Rust 1.89+ (`rustup update stable`).

## Quick Start

```bash
# Create a model from scratch
cargo run --example create_apr_from_scratch

# Bundle for deployment
cargo run --example bundle_static_model

# Run the full optimization pipeline
cargo run --example optimize_full_pipeline

# List all 341 examples
cargo run --example 2>&1 | head -50
```

## Examples

| Category | Count | Description |
|----------|------:|-------------|
| [Analysis](examples/analysis/) | 64 | Inspect, validate, diff, bench, profile, QA gates, oracle, trace, lint, fingerprint |
| [CLI](examples/cli/) | 36 | apr-info, bench, convert, compile, serve, diff, tui, decrypt, diagnose, ptx-map |
| [Optimize](examples/optimize/) | 27 | LoRA, QLoRA, pruning, distillation, merge (SLERP/TIES/DARE), quantize |
| [Advanced](examples/advanced/) | 23 | RAG, CI/CD pipeline, A/B experiment, compliance audit, edge anomaly, style transfer |
| [Training](examples/training/) | 23 | Autograd, custom ops, gradient clipping, mixed-precision, curriculum, federated |
| [Format](examples/format/) | 20 | Rosetta convert/chain/verify, batch export, HF import, migration pipeline |
| [Lint](examples/lint/) | 18 | AWQ, GBNF, naming rules, DRY sampling, OOM, tool-use, validate-manifest |
| [Inference](examples/inference/) | 16 | Speculative decode, KV-cache, streaming tokens, mmap lazy load, ensemble, tool use |
| [GPU](examples/gpu/) | 14 | FlashAttention, CUDA, Vulkan/Intel Arc, tensor cores, multi-GPU, PTX analysis |
| [Monitoring](examples/monitoring/) | 12 | Explainability, audit trail, cost tracking, drift detection, RAPL energy, memory profiler |
| [Serving](examples/serve/) | 12 | HTTP server, A/B testing, canary deploy, rate limiting, selection router |
| [Bundling](examples/bundling/) | 9 | Static binary, quantized, encrypted, signed, Lambda package |
| [Registry](examples/registry/) | 8 | Versioning, lineage, comparison, rollback |
| [Acceleration](examples/acceleration/) | 7 | Autotuner, kernel fusion, mmap, quantized matmul, LZ4/ZSTD bench, cache tiling |
| [Creation](examples/creation/) | 7 | Linear regression, decision trees, clustering, neural networks, n-gram LM |
| [SIMD](examples/simd/) | 6 | trueno ops, AVX-VNNI Int8, matrix operations, vectorized inference |
| [WASM](examples/wasm/) | 6 | Browser inference, Web Workers, WebGPU, progressive loading, streaming compilation |
| [API](examples/api/) | 5 | REST inference, streaming, batch, auth middleware, health check |
| [Chat](examples/chat/) | 5 | ChatML, LLaMA 2, Mistral, multi-format, injection defense |
| [Conversion](examples/conversion/) | 5 | SafeTensors, GGUF, ONNX, Phi |
| [Distillation](examples/distillation/) | 5 | Knowledge transfer, attention transfer, layer matching, self-distillation |
| [Distributed](examples/distributed/) | 5 | Multi-node inference, sharding, ring-allreduce, pipeline parallel, gossip |
| [Serverless](examples/serverless/) | 5 | Lambda, cold start optimization, edge functions, containers, warmup |
| [Speech](examples/speech/) | 5 | Whisper transcription, streaming ASR, VAD, diarization, multilingual |
| [MCP](examples/mcp/) | 3 | Stdio server, HTTP server, tool schema advertisement |
| [**Fine-Tuning**](examples/finetune/) | **155** | **4-tier curriculum: SFT → LoRA/QLoRA/PEFT/CP/merge → calibration/multimodal/QAT → DPO/ORPO/KTO/GRPO/PPO/RLAIF (Ludwig + Unsloth + TRL + LLaMA-Factory + Axolotl mirror)** |

**Total: 496 recipes.** Run any example: `cargo run --example <name>`. The 155
fine-tuning recipes ship a closed-form falsifier each (see [`examples/finetune/README.md`](examples/finetune/README.md)).

**Demo-run baseline (2026-04-23):** 330 / 341 pass under 10s; the remaining 11 are compute-heavy benchmarks (see `docs/specifications/components/quality-gates.md#demo-run-baseline`) that require a longer timeout. 0 failures.

### Full recipe table

<!-- RECIPE-TABLE-START -->
<!-- Auto-generated by scripts/generate-recipe-table.sh — do not edit manually -->
<!-- Re-generate: ./scripts/generate-recipe-table.sh --update -->
<!-- CI validates: recipe-table workflow ensures this table matches source -->

**1825 recipes** | Build: [![CI](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml)

<details>
<summary>Full recipe table (click to expand)</summary>

| # | Example | Category | Devices | Build |
|--:|---------|----------|---------|:-----:|
| 1 | `acceleration_arithmetic_intensity` | acceleration | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 2 | `acceleration_autotuner` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 3 | `acceleration_cache_tiling` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 4 | `acceleration_compression_benchmark` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 5 | `acceleration_kernel_dispatch_planner` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 6 | `acceleration_kernel_fusion` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 7 | `acceleration_mmap_inference` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 8 | `acceleration_mmap_per_tensor_diff_bench` | acceleration | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 9 | `acceleration_moe_rayon_dispatch_bench` | acceleration | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 10 | `acceleration_quantized_matmul` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 11 | `acceleration_thread_pool_sizer` | acceleration | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 12 | `simd_matrix_operations` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 13 | `ab_experiment` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 14 | `adv_admission_quota_per_tenant` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 15 | `adv_canary_promotion_gate` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 16 | `adv_capability_manifest_validator` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 17 | `adv_chain_of_thought` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 18 | `adv_chunked_prefill` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 19 | `adv_chunked_response_buffer` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 20 | `adv_circuit_breaker` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 21 | `adv_continuous_batching` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 22 | `adv_correlation_id_propagator` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 23 | `adv_idempotency_key` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 24 | `adv_iiur_compliance_scorer` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 25 | `adv_kv_cache_eviction` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 26 | `adv_kv_quantization` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 27 | `adv_long_context_retrieval_split` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 28 | `adv_multimodal_router` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 29 | `adv_payload_size_limit` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 30 | `adv_pipeline_dag` | advanced | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 31 | `adv_priority_queue_eviction` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 32 | `adv_provenance_chain` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 33 | `adv_quantize_pipeline` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 34 | `adv_quota_token_bucket` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 35 | `adv_recipe_dependency_dag` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 36 | `adv_replica_failover` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 37 | `adv_request_coalescing` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 38 | `adv_request_dedup_cache` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 39 | `adv_request_id_format` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 40 | `adv_request_priority_aging` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 41 | `adv_request_priority_router` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 42 | `adv_response_compression_picker` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 43 | `adv_response_diff_validator` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 44 | `adv_response_redactor` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 45 | `adv_response_schema_match` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 46 | `adv_retry_backoff` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 47 | `adv_safety_classifier_threshold` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 48 | `adv_session_affinity_router` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 49 | `adv_speculative_decode_window` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 50 | `adv_speculative_tree_attention` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 51 | `adv_token_budget` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 52 | `adv_tool_call_validator` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 53 | `adv_warmup_classifier` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 54 | `cicd_model_pipeline` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 55 | `clip_search` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 56 | `code_defect_oracle` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 57 | `compliance_audit` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 58 | `debug_fix_loop` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 59 | `edge_anomaly_detection` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 60 | `embedding_visualization` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 61 | `handwriting_recognition` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 62 | `hierarchical_cache_benchmark` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 63 | `image_classification` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 64 | `model_inspection_scoring` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 65 | `model_showcase` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 66 | `online_training_defect` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 67 | `quantization_quality_tradeoff` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 68 | `rag_pipeline` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 69 | `showcase_gallery` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 70 | `showcase_markdown` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 71 | `spanish_tutor` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 72 | `streaming_sentiment` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 73 | `style_transfer` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 74 | `voice_recognition` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 75 | `wasm_summarizer` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 76 | `analysis_bench` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 77 | `analysis_canary` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 78 | `analysis_check` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 79 | `analysis_compare_hf` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 80 | `analysis_compare_hf_threshold` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 81 | `analysis_contract_algorithm_binding_pattern` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 82 | `analysis_cpu_vs_gpu_parity_gate` | analysis | ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 83 | `analysis_debug` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 84 | `analysis_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 85 | `analysis_eval` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 86 | `analysis_explain` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 87 | `analysis_flow` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 88 | `analysis_hex` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 89 | `analysis_inspect` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 90 | `analysis_json_schema_draft7_meta_validation` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 91 | `analysis_latency_breakdown` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 92 | `analysis_lint` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 93 | `analysis_memory_leak_detector` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 94 | `analysis_model_fingerprint` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 95 | `analysis_oracle` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 96 | `analysis_p99_throughput_gate` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 97 | `analysis_parity` | analysis | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 98 | `analysis_probar` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 99 | `analysis_profile` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 100 | `analysis_pv_check_parity_authoring` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 101 | `analysis_qa_capability` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 102 | `analysis_qa_gates` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 103 | `analysis_qa_report` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 104 | `analysis_qualify` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 105 | `analysis_slice` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 106 | `analysis_tensors` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 107 | `analysis_tensors_stats` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 108 | `analysis_trace` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 109 | `analysis_tree` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 110 | `analysis_validate` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 111 | `bench_batch_sweep` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 112 | `bench_quantization_compare` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 113 | `canary_regression` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 114 | `canary_rolling_window` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 115 | `check_batch` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 116 | `check_json_report` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 117 | `debug_activation_dist` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 118 | `debug_nan_trace` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 119 | `eval_benchmark_suite` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 120 | `eval_pass_at_k` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 121 | `experiment_ab_test` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 122 | `experiment_multi_seed` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 123 | `explain_error_codes` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 124 | `explain_shape_mismatch` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 125 | `flow_arch_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 126 | `flow_depth_sweep` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 127 | `hex_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 128 | `hex_pattern_search` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 129 | `inspect_layer_params` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 130 | `inspect_quantization_stats` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 131 | `lint_naming_rules` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 132 | `lint_suppression` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 133 | `oracle_classifier` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 134 | `oracle_ensemble_vote` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 135 | `parity_format` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 136 | `parity_quantization` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 137 | `probar_regression_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 138 | `probar_suite_runner` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 139 | `profile_memory_layers` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 140 | `profile_roofline` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 141 | `qualify_remediation` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 142 | `qualify_scorecard` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 143 | `trace_per_op_latency` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 144 | `trace_run_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 145 | `tree_arch_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 146 | `tree_param_rollup` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 147 | `api_admission_control_queue` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 148 | `api_auth_middleware` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 149 | `api_batch_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 150 | `api_call_model_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 151 | `api_circuit_breaker_state` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 152 | `api_idempotency_key_dedup` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 153 | `api_model_health_check` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 154 | `api_oauth_token_refresh_window` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 155 | `api_request_id_correlation` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 156 | `api_request_timeout_classifier` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 157 | `api_response_cache_ttl` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 158 | `api_streaming_chunk_size` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 159 | `api_streaming_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 160 | `api_token_bucket_rate_limiter` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 161 | `bundle_apr_lambda_package` | bundling | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 162 | `bundle_apr_quantized_q4` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 163 | `bundle_apr_signed` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 164 | `bundle_apr_static_binary` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 165 | `bundle_cache_key_deriver` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 166 | `bundle_compression_picker` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 167 | `bundle_compression_ratio_calc` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 168 | `bundle_encrypted_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 169 | `bundle_integrity_checksum` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 170 | `bundle_manifest_header_validator` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 171 | `bundle_metadata_versioning` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 172 | `bundle_mmap_offset_calculator` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 173 | `bundle_partial_load` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 174 | `bundle_pre_load_warmup` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 175 | `bundle_quantized_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 176 | `bundle_signing_attestation` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 177 | `bundle_signing_chain_builder` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 178 | `bundle_static_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 179 | `bundle_streaming_q4k_large_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 180 | `bundle_streaming_unpack` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 181 | `bundle_table_of_contents` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 182 | `bundle_tensor_dedup` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 183 | `bundle_zero_copy_handoff` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 184 | `encrypt_kdf_sweep` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 185 | `encrypt_signed` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 186 | `cgp_baseline_diff_classifier` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 187 | `cgp_kernel_metric_aggregator` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 188 | `cgp_proof_status_dispatcher` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 189 | `cgp_regression_detector_baseline_vs_current` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 190 | `cgp_roofline_classify_kernel` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 191 | `cgp_roofline_ridge_point_per_precision` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 192 | `chat_chatml` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 193 | `chat_injection_defense` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 194 | `chat_llama2` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 195 | `chat_mistral` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 196 | `chat_multi_format` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 197 | `chat_role_state_machine` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 198 | `chat_template_renderer` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 199 | `chat_token_budget_truncation` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 200 | `apr_bench` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 201 | `apr_info` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 202 | `cli_apr_bench` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 203 | `cli_apr_compile` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 204 | `cli_apr_convert` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 205 | `cli_apr_decrypt` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 206 | `cli_apr_diagnose` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 207 | `cli_apr_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 208 | `cli_apr_info` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 209 | `cli_apr_list` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 210 | `cli_apr_ptx_map` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 211 | `cli_apr_rm` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 212 | `cli_apr_runs` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 213 | `cli_apr_serve` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 214 | `cli_apr_tokenize` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 215 | `cli_apr_tui` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 216 | `cli_bench_batch_sweep_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 217 | `cli_bench_cv_stability_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 218 | `cli_bench_h12_throughput_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 219 | `cli_bench_percentiles_csv_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 220 | `cli_bench_unit_normalizer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 221 | `cli_bench_warmup_iterations_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 222 | `cli_canary_check_verdict` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 223 | `cli_canary_create_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 224 | `cli_canary_directory_layout` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 225 | `cli_cbtop_ci_threshold_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 226 | `cli_cbtop_headless_json_schema` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 227 | `cli_cbtop_speculative_decoding_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 228 | `cli_check_json_output_schema` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 229 | `cli_check_pipeline_integrity_smoke` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 230 | `cli_check_skip_contract_diagnostic` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 231 | `cli_compare_hf_offline_safety` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 232 | `cli_compare_hf_tensor_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 233 | `cli_compare_hf_threshold_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 234 | `cli_compile_optimization_flags` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 235 | `cli_compile_output_path_resolver` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 236 | `cli_compile_target_triple_validator` | cli | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 237 | `cli_data_balance_strategy_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 238 | `cli_data_decontaminate_ngram_overlap` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 239 | `cli_data_split_stratified_ratios` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 240 | `cli_debug_breakpoint_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 241 | `cli_debug_drama_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 242 | `cli_debug_layer_glob_matcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 243 | `cli_debug_limit_truncator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 244 | `cli_debug_string_extractor` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 245 | `cli_debug_tensor_diff_tolerance` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 246 | `cli_decrypt_aead_tag_verification` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 247 | `cli_decrypt_invocation_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 248 | `cli_decrypt_key_format_validation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 249 | `cli_decrypt_key_rotation_grace` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 250 | `cli_decrypt_output_collision_detector` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 251 | `cli_decrypt_verify_ordering` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 252 | `cli_diagnose_five_whys_chain` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 253 | `cli_diagnose_grad_nan_scanner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 254 | `cli_diagnose_jsonl_corpus_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 255 | `cli_diagnose_model_size_inference` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 256 | `cli_diagnose_param_count_sanity` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 257 | `cli_diagnose_weight_histogram_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 258 | `cli_diff_magnitude_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 259 | `cli_diff_shape_compatibility` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 260 | `cli_diff_structural_renderer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 261 | `cli_diff_values_aprt_stage` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 262 | `cli_distill_ensemble_weighter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 263 | `cli_distill_layer_pairer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 264 | `cli_distill_loss_combiner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 265 | `cli_distill_stage_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 266 | `cli_distill_strategy_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 267 | `cli_distill_temperature_alpha_band` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 268 | `cli_encrypt_aad_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 269 | `cli_encrypt_force_overwrite_policy` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 270 | `cli_encrypt_kdf_iterations_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 271 | `cli_encrypt_keystream_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 272 | `cli_encrypt_nonce_uniqueness_checker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 273 | `cli_encrypt_passphrase_strength` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 274 | `cli_eval_bleu_score_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 275 | `cli_eval_dataset_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 276 | `cli_eval_metric_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 277 | `cli_eval_pass_at_k_temperature_pairing` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 278 | `cli_eval_perplexity_threshold_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 279 | `cli_eval_top_k_accuracy_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 280 | `cli_experiment_hypothesis_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 281 | `cli_experiment_metric_compare_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 282 | `cli_experiment_param_diff_renderer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 283 | `cli_experiment_run_id_collision` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 284 | `cli_experiment_view_loss_curve_render` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 285 | `cli_experiment_view_run_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 286 | `cli_explain_ablation_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 287 | `cli_explain_error_code_lookup` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 288 | `cli_explain_ig_steps_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 289 | `cli_explain_kernel_dispatch_pipeline` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 290 | `cli_explain_proof_status_aggregator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 291 | `cli_explain_saliency_rank_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 292 | `cli_export_batch_csv_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 293 | `cli_export_format_allowlist` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 294 | `cli_export_opset_compat_matrix` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 295 | `cli_export_output_naming_pattern` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 296 | `cli_export_plan_mode_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 297 | `cli_export_target_dtype_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 298 | `cli_finetune_checkpoint_format_csv` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 299 | `cli_finetune_grad_accum_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 300 | `cli_finetune_lora_rank_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 301 | `cli_finetune_lr_scheduler_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 302 | `cli_finetune_merge_mode_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 303 | `cli_finetune_method_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 304 | `cli_flow_component_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 305 | `cli_flow_dot_renderer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 306 | `cli_flow_layer_aggregation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 307 | `cli_gpu_device_capability_filter` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 308 | `cli_gpu_fp8_capability_checker` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 309 | `cli_gpu_nccl_strategy_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 310 | `cli_gpu_oom_recovery_advisor` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 311 | `cli_gpu_peer_access_topology` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 312 | `cli_gpu_vram_reservation_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 313 | `cli_hex_offset_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 314 | `cli_hex_slice_range_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 315 | `cli_hex_view_mode_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 316 | `cli_import_dtype_coercion_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 317 | `cli_import_format_auto_detector` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 318 | `cli_import_no_config_inference_risk` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 319 | `cli_import_provenance_chain_enforcement` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 320 | `cli_import_sharding_plan_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 321 | `cli_import_strict_mode_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 322 | `cli_inspect_view_mode_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 323 | `cli_inspect_vocab_token_query` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 324 | `cli_inspect_weight_stats_aggregator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 325 | `cli_mcp_batch_request_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 326 | `cli_mcp_error_response_codes` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 327 | `cli_mcp_jsonrpc_request_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 328 | `cli_mcp_resource_uri_scheme` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 329 | `cli_mcp_tool_manifest_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 330 | `cli_mcp_transport_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 331 | `cli_merge_dare_drop_rate_band` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 332 | `cli_merge_signed_conflict_resolver` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 333 | `cli_merge_slerp_t_clamp` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 334 | `cli_merge_strategy_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 335 | `cli_merge_ties_density_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 336 | `cli_merge_weights_csv_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 337 | `cli_monitor_drift_threshold_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 338 | `cli_monitor_format_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 339 | `cli_monitor_log_rotation_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 340 | `cli_monitor_metrics_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 341 | `cli_monitor_quantile_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 342 | `cli_monitor_refresh_throttle` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 343 | `cli_ollama_chat_lint_eval_count_consistency` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 344 | `cli_ollama_chat_lint_message_content_validation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 345 | `cli_ollama_chat_lint_role_state_machine` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 346 | `cli_ollama_embed_dim_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 347 | `cli_ollama_model_name_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 348 | `cli_ollama_token_rate_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 349 | `cli_oracle_compliance_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 350 | `cli_oracle_family_introspection` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 351 | `cli_oracle_size_constraint_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 352 | `cli_parity_assert_mode_exit_codes` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 353 | `cli_parity_default_prompt_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 354 | `cli_parity_token_divergence_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 355 | `cli_pipeline_concurrency_limiter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 356 | `cli_pipeline_dag_topological_sort` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 357 | `cli_pipeline_retry_policy_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 358 | `cli_pipeline_stage_skip_predicate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 359 | `cli_pipeline_status_state_machine` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 360 | `cli_pipeline_validate_manifest_schema` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 361 | `cli_pretrain_curriculum_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 362 | `cli_pretrain_divergence_guard` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 363 | `cli_pretrain_epoch_budget_calc` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 364 | `cli_pretrain_grad_clip_threshold` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 365 | `cli_pretrain_mode_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 366 | `cli_pretrain_run_dir_layout` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 367 | `cli_probar_export_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 368 | `cli_probar_golden_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 369 | `cli_probar_layer_pattern_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 370 | `cli_profile_flame_depth_limit` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 371 | `cli_profile_format_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 372 | `cli_profile_hot_function_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 373 | `cli_profile_naive_detection_threshold` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 374 | `cli_profile_perf_grade_thresholds` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 375 | `cli_profile_sampling_rate_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 376 | `cli_prune_lottery_ticket_warmup` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 377 | `cli_prune_method_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 378 | `cli_prune_remove_layers_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 379 | `cli_prune_sparsity_ramp_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 380 | `cli_prune_target_ratio_band` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 381 | `cli_prune_wanda_activation_scorer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 382 | `cli_ptx_kernel_name_parser` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 383 | `cli_ptx_map_kernel_filter` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 384 | `cli_ptx_map_prefill_vs_decode` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 385 | `cli_ptx_map_reverse_lookup` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 386 | `cli_ptx_register_pressure_threshold` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 387 | `cli_ptx_strict_mode_whitelist` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 388 | `cli_publish_dry_run_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 389 | `cli_publish_manifest_full` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 390 | `cli_publish_parent_chain_termination` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 391 | `cli_publish_pipeline_tag_allowlist` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 392 | `cli_publish_repo_id_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 393 | `cli_pull_alias_resolver` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 394 | `cli_pull_dataset_glob_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 395 | `cli_pull_revision_pin_resolver` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 396 | `cli_qa_assertion_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 397 | `cli_qa_parallel_test_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 398 | `cli_qa_regression_delta_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 399 | `cli_qa_safetensors_parity_required` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 400 | `cli_qa_tier_aggregator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 401 | `cli_qa_warmup_iteration_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 402 | `cli_qualify_skip_list_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 403 | `cli_qualify_tier_progression` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 404 | `cli_qualify_timeout_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 405 | `cli_quantize_batch_csv_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 406 | `cli_quantize_calibration_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 407 | `cli_quantize_format_compatibility` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 408 | `cli_quantize_mixed_precision_selector` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 409 | `cli_quantize_scale_zero_point_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 410 | `cli_quantize_scheme_size_predictor` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 411 | `cli_registry_aliases_collision_detection` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 412 | `cli_registry_aliases_json_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 413 | `cli_registry_aliases_yaml_loader` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 414 | `cli_registry_lineage_cycle_detector` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 415 | `cli_registry_semver_tag_matcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 416 | `cli_registry_uri_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 417 | `cli_rosetta_chain_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 418 | `cli_rosetta_compare_inference_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 419 | `cli_rosetta_compare_inference_logit_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 420 | `cli_rosetta_compare_inference_temperature_modes` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 421 | `cli_rosetta_convert_extension_inference` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 422 | `cli_rosetta_convert_external_tokenizer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 423 | `cli_rosetta_convert_quantize_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 424 | `cli_rosetta_diff_tensors_layout_check` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 425 | `cli_rosetta_diff_tensors_pad_token_signal` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 426 | `cli_rosetta_diff_tensors_value_sampler` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 427 | `cli_rosetta_fingerprint_diff_mode` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 428 | `cli_rosetta_fingerprint_filter_pattern` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 429 | `cli_rosetta_fingerprint_json_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 430 | `cli_rosetta_fingerprint_stats` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 431 | `cli_rosetta_inspect_format_detector` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 432 | `cli_rosetta_inspect_hexdump_window` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 433 | `cli_rosetta_inspect_tensor_table` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 434 | `cli_rosetta_round_trip_verify` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 435 | `cli_rosetta_validate_stats_per_tensor_report` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 436 | `cli_rosetta_validate_stats_reference_or_fingerprints` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 437 | `cli_rosetta_validate_stats_threshold_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 438 | `cli_runs_diff_two_runs` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 439 | `cli_runs_ls_sparkline_renderer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 440 | `cli_runs_show_metric_summary` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 441 | `cli_serve_kv_cache_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 442 | `cli_serve_max_tokens_cap` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 443 | `cli_serve_plan_capacity_estimator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 444 | `cli_serve_run_endpoint_router` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 445 | `cli_serve_run_port_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 446 | `cli_serve_streaming_chunk_size` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 447 | `cli_showcase_runs_floor_enforcement` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 448 | `cli_showcase_step_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 449 | `cli_showcase_tier_baseline_matrix` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 450 | `cli_stamp_preserves_tensor_bytes` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 451 | `cli_stamp_provenance_basic` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 452 | `cli_stamp_spdx_validation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 453 | `cli_tensors_filter_pattern` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 454 | `cli_tensors_limit_truncator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 455 | `cli_tensors_stats_aggregator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 456 | `cli_tokenize_corpus_shard_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 457 | `cli_tokenize_hf_import_validation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 458 | `cli_tokenize_plan_estimator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 459 | `cli_trace_diff_mode_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 460 | `cli_trace_save_tensor_layer0` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 461 | `cli_trace_save_tensor_layer_range` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 462 | `cli_trace_stage_csv_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 463 | `cli_train_checkpoint_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 464 | `cli_train_early_stop_patience` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 465 | `cli_train_halving_round_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 466 | `cli_train_lr_finder_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 467 | `cli_train_sweep_grid_generator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 468 | `cli_train_watch_restart_policy` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 469 | `cli_tui_color_theme_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 470 | `cli_tui_keybinding_matcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 471 | `cli_tui_pager_buffer_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 472 | `cli_tui_panel_layout_calculator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 473 | `cli_tui_resize_event_throttle` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 474 | `cli_tui_search_filter_predicate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 475 | `cli_tune_budget_compat_matrix` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 476 | `cli_tune_scheduler_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 477 | `cli_tune_strategy_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 478 | `cli_validate_manifest_falsify_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 479 | `cli_validate_manifest_offline_safety` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 480 | `cli_validate_manifest_safetensors_dtype` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 481 | `cli_validate_manifest_sha256_format` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 482 | `cli_validate_min_score_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 483 | `cli_validate_quality_score_aggregator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 484 | `cli_validate_strict_warning_promoter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 485 | `compile_ptx` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 486 | `compile_size_optimized` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 487 | `decrypt_batch` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 488 | `decrypt_key_rotation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 489 | `diagnose_hardware` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 490 | `diagnose_multi_model` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 491 | `diff_quantization` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 492 | `diff_topology` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 493 | `list_json_export` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 494 | `list_size_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 495 | `rm_dry_run` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 496 | `rm_retention_policy` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 497 | `runs_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 498 | `runs_filter_sort` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 499 | `tokenize_bpe_trace` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 500 | `tokenize_compare_vocabs` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 501 | `tui_health_dashboard` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 502 | `tui_log_tail` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 503 | `validate_batch` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 504 | `validate_fix_suggestions` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 505 | `code_custom_agent_definition` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 506 | `code_diff_hunk_parser` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 507 | `code_hook_session_start` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 508 | `code_indent_normalizer` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 509 | `code_lint_severity_aggregator` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 510 | `code_mcp_client_config` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 511 | `code_skill_discovery` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 512 | `code_slash_command_extension` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 513 | `code_subagent_spawn_payload` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 514 | `code_worktree_isolation_permission_mode` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 515 | `contracts_macros_alias_resolver` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 516 | `contracts_macros_alphabet_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 517 | `contracts_macros_arxiv_citation_lint` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 518 | `contracts_macros_attribute_basic` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 519 | `contracts_macros_attribute_round_trip` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 520 | `contracts_macros_constant_propagation` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 521 | `contracts_macros_coverage_band_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 522 | `contracts_macros_dependency_graph` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 523 | `contracts_macros_deprecation_window_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 524 | `contracts_macros_dispatch_fallback` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 525 | `contracts_macros_env_key_convention` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 526 | `contracts_macros_falsification_witness` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 527 | `contracts_macros_field_default_inference` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 528 | `contracts_macros_invariant_atomicity` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 529 | `contracts_macros_invariant_baseline_diff` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 530 | `contracts_macros_invariant_chain` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 531 | `contracts_macros_invariant_compose_chain` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 532 | `contracts_macros_invariant_drift_alert` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 533 | `contracts_macros_invariant_drift_window` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 534 | `contracts_macros_invariant_negated_classify` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 535 | `contracts_macros_invariant_priority_band` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 536 | `contracts_macros_invariant_proof_lang` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 537 | `contracts_macros_invariant_split_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 538 | `contracts_macros_invariant_witness_pair` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 539 | `contracts_macros_inverse_postcond` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 540 | `contracts_macros_kani_status` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 541 | `contracts_macros_lean_axiom_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 542 | `contracts_macros_lean_compile_status` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 543 | `contracts_macros_lean_filename` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 544 | `contracts_macros_lemma_reuse_count` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 545 | `contracts_macros_metric_threshold` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 546 | `contracts_macros_module_binding` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 547 | `contracts_macros_multi_equation_dispatch` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 548 | `contracts_macros_no_op_degradation` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 549 | `contracts_macros_obligation_age_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 550 | `contracts_macros_obligation_arity` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 551 | `contracts_macros_obligation_assignee_balance` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 552 | `contracts_macros_obligation_audit_log` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 553 | `contracts_macros_obligation_blame_chain` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 554 | `contracts_macros_obligation_breaking_change_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 555 | `contracts_macros_obligation_chain_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 556 | `contracts_macros_obligation_checksum` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 557 | `contracts_macros_obligation_compose` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 558 | `contracts_macros_obligation_count_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 559 | `contracts_macros_obligation_coverage_pct` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 560 | `contracts_macros_obligation_dedupe` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 561 | `contracts_macros_obligation_diff` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 562 | `contracts_macros_obligation_export_csv` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 563 | `contracts_macros_obligation_history_log` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 564 | `contracts_macros_obligation_join_status` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 565 | `contracts_macros_obligation_lifecycle_state` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 566 | `contracts_macros_obligation_lock_file` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 567 | `contracts_macros_obligation_namespace_collision` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 568 | `contracts_macros_obligation_owner_filter` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 569 | `contracts_macros_obligation_parametric` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 570 | `contracts_macros_obligation_priority_inheritance` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 571 | `contracts_macros_obligation_renaming` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 572 | `contracts_macros_obligation_review_age` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 573 | `contracts_macros_obligation_satisfied_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 574 | `contracts_macros_obligation_severity` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 575 | `contracts_macros_obligation_severity_escalate` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 576 | `contracts_macros_obligation_split_grouping` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 577 | `contracts_macros_obligation_subset_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 578 | `contracts_macros_obligation_tag_filter` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 579 | `contracts_macros_obligation_test_traceability` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 580 | `contracts_macros_obligation_type_consistency` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 581 | `contracts_macros_phase_chain` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 582 | `contracts_macros_phase_dag_order` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 583 | `contracts_macros_phase_dependency_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 584 | `contracts_macros_pre_post_envelope` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 585 | `contracts_macros_pre_post_partition` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 586 | `contracts_macros_pre_violation_classify` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 587 | `contracts_macros_priority_inversion` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 588 | `contracts_macros_priority_sort` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 589 | `contracts_macros_proof_age_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 590 | `contracts_macros_proof_dependency` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 591 | `contracts_macros_proof_module_index` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 592 | `contracts_macros_proof_obligation_score` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 593 | `contracts_macros_proof_status_transition` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 594 | `contracts_macros_pure_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 595 | `contracts_macros_recipe_archive_safe` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 596 | `contracts_macros_recipe_attestation_chain` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 597 | `contracts_macros_recipe_benchmark_envelope` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 598 | `contracts_macros_recipe_breaking_change_log` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 599 | `contracts_macros_recipe_categorize_by_keyword` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 600 | `contracts_macros_recipe_change_log` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 601 | `contracts_macros_recipe_changelog_entry` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 602 | `contracts_macros_recipe_changelog_versi` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 603 | `contracts_macros_recipe_compat_matrix` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 604 | `contracts_macros_recipe_complete_pct` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 605 | `contracts_macros_recipe_decomposition` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 606 | `contracts_macros_recipe_dependency_height` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 607 | `contracts_macros_recipe_dependent_count` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 608 | `contracts_macros_recipe_diff_minimum` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 609 | `contracts_macros_recipe_estimated_complexity` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 610 | `contracts_macros_recipe_freeze_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 611 | `contracts_macros_recipe_freshness_window` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 612 | `contracts_macros_recipe_generated_marker` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 613 | `contracts_macros_recipe_hash_consistency` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 614 | `contracts_macros_recipe_id_canon` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 615 | `contracts_macros_recipe_id_format` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 616 | `contracts_macros_recipe_id_uniqueness` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 617 | `contracts_macros_recipe_label_consistency` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 618 | `contracts_macros_recipe_lifecycle` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 619 | `contracts_macros_recipe_link_health` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 620 | `contracts_macros_recipe_locale_consistency` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 621 | `contracts_macros_recipe_meta_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 622 | `contracts_macros_recipe_metadata_min` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 623 | `contracts_macros_recipe_namespace_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 624 | `contracts_macros_recipe_orphan_detector` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 625 | `contracts_macros_recipe_outdated_dep` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 626 | `contracts_macros_recipe_owner_assignment` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 627 | `contracts_macros_recipe_owner_email_valid` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 628 | `contracts_macros_recipe_owner_handoff` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 629 | `contracts_macros_recipe_owner_routing` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 630 | `contracts_macros_recipe_pre_publish_lint` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 631 | `contracts_macros_recipe_priority_classifier` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 632 | `contracts_macros_recipe_priority_decay` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 633 | `contracts_macros_recipe_publish_gate` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 634 | `contracts_macros_recipe_quarantine_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 635 | `contracts_macros_recipe_release_blocker` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 636 | `contracts_macros_recipe_review_round_count` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 637 | `contracts_macros_recipe_reviewer_assign` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 638 | `contracts_macros_recipe_revision_bump` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 639 | `contracts_macros_recipe_risk_score` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 640 | `contracts_macros_recipe_runtime_budget` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 641 | `contracts_macros_recipe_severity_aggregate` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 642 | `contracts_macros_recipe_signature` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 643 | `contracts_macros_recipe_signature_diff` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 644 | `contracts_macros_recipe_signoff_required` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 645 | `contracts_macros_recipe_status_summary` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 646 | `contracts_macros_recipe_tag_consistency` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 647 | `contracts_macros_recipe_tag_taxonomy` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 648 | `contracts_macros_recipe_test_count_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 649 | `contracts_macros_recipe_test_naming` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 650 | `contracts_macros_recipe_test_runtime_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 651 | `contracts_macros_recipe_test_skip_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 652 | `contracts_macros_recipe_ttl_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 653 | `contracts_macros_runtime_validator_bridge` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 654 | `contracts_macros_severity_decay_curve` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 655 | `contracts_macros_severity_propagation` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 656 | `contracts_macros_severity_score_normalize` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 657 | `contracts_macros_signature_hash` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 658 | `contracts_macros_spec_drift_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 659 | `contracts_macros_spec_hash_pin` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 660 | `contracts_macros_spec_release_notes` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 661 | `contracts_macros_status_provenance` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 662 | `contracts_macros_tolerance_propagation` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 663 | `contracts_macros_violation_history` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 664 | `contracts_macros_witness_aging_decay` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 665 | `contracts_macros_witness_count_min` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 666 | `contracts_macros_witness_dag_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 667 | `contracts_macros_witness_origin_trace` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 668 | `contracts_macros_witness_replay_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 669 | `contracts_macros_yaml_alias_dup_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 670 | `contracts_macros_yaml_alias_resolution_depth` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 671 | `contracts_macros_yaml_alias_resolve_chain` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 672 | `contracts_macros_yaml_anchor_alphabetize` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 673 | `contracts_macros_yaml_anchor_chain_depth` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 674 | `contracts_macros_yaml_anchor_cycle` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 675 | `contracts_macros_yaml_anchor_namespace` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 676 | `contracts_macros_yaml_anchor_naming` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 677 | `contracts_macros_yaml_anchor_resolver` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 678 | `contracts_macros_yaml_anchor_unused` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 679 | `contracts_macros_yaml_block_indent_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 680 | `contracts_macros_yaml_block_scalar_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 681 | `contracts_macros_yaml_block_scalar_norm` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 682 | `contracts_macros_yaml_canonical_form` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 683 | `contracts_macros_yaml_collection_size_max` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 684 | `contracts_macros_yaml_comment_density` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 685 | `contracts_macros_yaml_default_field_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 686 | `contracts_macros_yaml_default_value_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 687 | `contracts_macros_yaml_doc_separator_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 688 | `contracts_macros_yaml_doc_seq_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 689 | `contracts_macros_yaml_dotted_path_lookup` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 690 | `contracts_macros_yaml_emit_format_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 691 | `contracts_macros_yaml_envelope_quote` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 692 | `contracts_macros_yaml_envelope_size_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 693 | `contracts_macros_yaml_explicit_tag_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 694 | `contracts_macros_yaml_field_alphabetize` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 695 | `contracts_macros_yaml_indent_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 696 | `contracts_macros_yaml_indent_normalize` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 697 | `contracts_macros_yaml_inline_array_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 698 | `contracts_macros_yaml_kebab_case_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 699 | `contracts_macros_yaml_lint` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 700 | `contracts_macros_yaml_macros_expansion_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 701 | `contracts_macros_yaml_max_nesting_depth` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 702 | `contracts_macros_yaml_path_normalize` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 703 | `contracts_macros_yaml_quote_style_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 704 | `contracts_macros_yaml_quoted_keys_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 705 | `contracts_macros_yaml_required_field_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 706 | `contracts_macros_yaml_required_top_level` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 707 | `contracts_macros_yaml_reserved_word_audit` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 708 | `contracts_macros_yaml_root_node_check` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 709 | `contracts_macros_yaml_schema_diff` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 710 | `contracts_macros_yaml_seq_dedupe` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 711 | `contracts_macros_yaml_seq_padding_norm` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 712 | `contracts_macros_yaml_string_truncate` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 713 | `contracts_macros_yaml_tab_indent_reject` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 714 | `contracts_macros_yaml_unicode_normalize` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 715 | `contracts_macros_yaml_url_validation` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 716 | `conversion_gguf_legacy_quant_fallback` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 717 | `convert_apr_to_gguf` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 718 | `convert_dtype_loss_estimator` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 719 | `convert_dtype_promote` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 720 | `convert_dtype_widener` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 721 | `convert_endianness_swapper` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 722 | `convert_format_version_matrix` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 723 | `convert_gguf_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 724 | `convert_layout_transposer` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 725 | `convert_lossy_check` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 726 | `convert_metadata_passthrough` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 727 | `convert_onnx_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 728 | `convert_phi_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 729 | `convert_quant_calibration` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 730 | `convert_quantization_rescaler` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 731 | `convert_safetensors_header` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 732 | `convert_safetensors_header_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 733 | `convert_safetensors_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 734 | `convert_sparse_csr_to_dense` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 735 | `convert_tensor_name_remapper` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 736 | `convert_tensor_view_strider` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 737 | `create_apr_decision_tree` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 738 | `create_apr_from_scratch` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 739 | `create_apr_kmeans_clustering` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 740 | `create_apr_linear_regression` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 741 | `create_apr_neural_network` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 742 | `create_apr_ngram_language_model` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 743 | `create_demo_model` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 744 | `create_embedding_tying_envelope` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 745 | `create_init_scheme_dispatcher` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 746 | `create_vocab_size_validator` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 747 | `basic_loading` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 748 | `cli_batch_commands` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 749 | `data_compression_codec` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 750 | `data_row_dedup_strategy` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 751 | `data_sample_quota_balancer` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 752 | `data_shuffle_seed_validator` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 753 | `data_streaming_buffer_size` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 754 | `data_train_val_test_split` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 755 | `dataloader_batching` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 756 | `doctest_extraction` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 757 | `drift_detection` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 758 | `federated_split` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 759 | `hub_publishing` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 760 | `prose_detection` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 761 | `quality_check` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 762 | `registry_publish` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 763 | `repl_commands` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 764 | `repl_completer` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 765 | `repl_display_config` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 766 | `repl_health_status` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 767 | `repl_session` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 768 | `streaming_large` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 769 | `transforms_pipeline` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 770 | `tui_viewer` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 771 | `alimentar_ingest` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 772 | `apr_inference_server` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 773 | `batuta_agent` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 774 | `deploy_blue_green_cutover` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 775 | `deploy_canary_traffic_ramp` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 776 | `deploy_pod_replica_scheduler` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 777 | `entrenar_train` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 778 | `jetson_edge_base` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 779 | `pacha_registry` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 780 | `pepita_sandbox` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 781 | `realizar_serve` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 782 | `renacer_observability` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 783 | `repartir_worker` | deployment-stacks | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 784 | `sovereign_ai_stack` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 785 | `trueno_db_analytics` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 786 | `trueno_rag_pipeline` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 787 | `whisper_apr_asr` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 788 | `distill_against_contract_v1` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 789 | `distill_attention_head_align` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 790 | `distill_attention_transfer` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 791 | `distill_attention_transfer_loss` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 792 | `distill_block_skip` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 793 | `distill_capacity_match` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 794 | `distill_continual_replay` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 795 | `distill_cross_layer_match` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 796 | `distill_curriculum_difficulty` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 797 | `distill_data_augmentation` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 798 | `distill_dataset_filter` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 799 | `distill_dataset_synth_ratio` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 800 | `distill_dropout_match` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 801 | `distill_grad_accum_picker` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 802 | `distill_grad_clip_picker` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 803 | `distill_intermediate_feature_match` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 804 | `distill_kl_floor` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 805 | `distill_knowledge_transfer` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 806 | `distill_layer_alignment_planner` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 807 | `distill_layer_matching` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 808 | `distill_layer_skip_connection` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 809 | `distill_logit_smoothing` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 810 | `distill_loss_mask_padding` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 811 | `distill_loss_weight_schedule` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 812 | `distill_lr_schedule_match` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 813 | `distill_per_class_temperature` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 814 | `distill_progressive_freeze` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 815 | `distill_quantile_calibration` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 816 | `distill_quantization_aware` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 817 | `distill_response_pruning` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 818 | `distill_self_consistency_filter` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 819 | `distill_self_distill_bootstrap` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 820 | `distill_self_distillation` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 821 | `distill_skill_cluster_router` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 822 | `distill_softlabel_topk` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 823 | `distill_teacher_ensemble` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 824 | `distill_temperature_anneal` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 825 | `distill_temperature_picker` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 826 | `distill_temperature_search_envelope` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 827 | `distill_token_level_kd` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 828 | `distill_token_skipping` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 829 | `distill_warmup_kd_only` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 830 | `distill_white_box_logit_match` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 831 | `distributed_allreduce_strategy` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 832 | `distributed_byzantine_quorum` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 833 | `distributed_collective_overlap` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 834 | `distributed_consistent_hash` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 835 | `distributed_eventual_consistency_window` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 836 | `distributed_failure_recovery` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 837 | `distributed_gossip_protocol` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 838 | `distributed_grad_compression` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 839 | `distributed_inference` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 840 | `distributed_lamport_clock` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 841 | `distributed_model_sharding` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 842 | `distributed_phi_failure_detector` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 843 | `distributed_pipeline_microbatch` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 844 | `distributed_pipeline_parallel` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 845 | `distributed_priority_queue` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 846 | `distributed_ring_allreduce` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 847 | `distributed_ring_chunks` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 848 | `distributed_shard_partition_planner` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 849 | `distributed_split_brain_detector` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 850 | `distributed_zero_copy_rdma` | distributed | ![cuda](https://img.shields.io/badge/-cuda-76b900) ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 851 | `t1_eval_accuracy` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 852 | `t1_eval_bleu` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 853 | `t1_eval_f1` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 854 | `t1_eval_perplexity` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 855 | `t1_eval_rouge_l` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 856 | `t1_sft_minimal_gemma` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 857 | `t1_sft_minimal_llama` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 858 | `t1_sft_minimal_mistral` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 859 | `t1_sft_minimal_phi` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 860 | `t1_sft_minimal_qwen` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 861 | `t1_smoke_bench` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 862 | `t1_smoke_dry_run` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 863 | `t1_smoke_early_stop` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 864 | `t1_smoke_plan` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 865 | `t1_smoke_resume` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 866 | `t1_tabular_100class` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 867 | `t1_tabular_3class` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 868 | `t1_tabular_7class` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 869 | `t1_tabular_binary` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 870 | `t1_tabular_imbalanced` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 871 | `t1_tabular_regression_energy` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 872 | `t1_tabular_regression_housing` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 873 | `t1_tabular_regression_missing` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 874 | `t1_tabular_regression_multitarget` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 875 | `t1_tabular_regression_timeseries` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 876 | `t2_adapter_merge_average` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 877 | `t2_adapter_merge_dare` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 878 | `t2_adapter_merge_multilora` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 879 | `t2_adapter_merge_slerp` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 880 | `t2_adapter_merge_ties` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 881 | `t2_apollo` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 882 | `t2_badam` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 883 | `t2_continued_pretrain_code` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 884 | `t2_continued_pretrain_codeswitch` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 885 | `t2_continued_pretrain_legal` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 886 | `t2_continued_pretrain_medical` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 887 | `t2_continued_pretrain_scientific` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 888 | `t2_dora` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 889 | `t2_freeze_tuning` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 890 | `t2_galore` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 891 | `t2_lisa` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 892 | `t2_ln_tuning` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 893 | `t2_lora_aqlm` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 894 | `t2_lora_awq` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 895 | `t2_lora_gptq` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 896 | `t2_lora_rank32_gemma` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 897 | `t2_lora_rank32_llama` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 898 | `t2_lora_rank32_mistral` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 899 | `t2_lora_rank32_phi` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 900 | `t2_lora_rank32_qwen` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 901 | `t2_lora_rank8_gemma` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 902 | `t2_lora_rank8_llama` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 903 | `t2_lora_rank8_mistral` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 904 | `t2_lora_rank8_phi` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 905 | `t2_lora_rank8_qwen` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 906 | `t2_neftune` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 907 | `t2_oft` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 908 | `t2_peft_corda_init` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 909 | `t2_peft_eva_init` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 910 | `t2_peft_loftq_init` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 911 | `t2_peft_pissa_init` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 912 | `t2_qlora_4bit_rank16_mistral` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 913 | `t2_qlora_4bit_rank32_phi` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 914 | `t2_qlora_4bit_rank8_llama` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 915 | `t2_qlora_double_quant_off_gemma` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 916 | `t2_qlora_double_quant_qwen` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 917 | `t2_regex_freeze` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 918 | `t2_relora` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 919 | `t2_tinylora` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 920 | `t2_vblora` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 921 | `t3_anomaly_deep_sad` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 922 | `t3_anomaly_deep_svdd` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 923 | `t3_anomaly_drocc` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 924 | `t3_calibration_conformal` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 925 | `t3_calibration_ensemble` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 926 | `t3_calibration_isotonic` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 927 | `t3_calibration_platt` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 928 | `t3_calibration_temperature` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 929 | `t3_fsdp_lora` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 930 | `t3_hypernetwork` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 931 | `t3_hyperopt_asha` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 932 | `t3_hyperopt_grid` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 933 | `t3_hyperopt_hyperband` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 934 | `t3_hyperopt_random` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 935 | `t3_hyperopt_tpe` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 936 | `t3_image_encoder_clip` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 937 | `t3_image_encoder_dinov2_lp` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 938 | `t3_image_encoder_siglip` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 939 | `t3_imbalance_costsensitive` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 940 | `t3_imbalance_focal` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 941 | `t3_imbalance_smote` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 942 | `t3_imbalance_threshold` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 943 | `t3_imbalance_weighted` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 944 | `t3_instruction_alpaca` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 945 | `t3_instruction_chat_template` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 946 | `t3_instruction_openassistant` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 947 | `t3_instruction_sharegpt` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 948 | `t3_instruction_system_prompt` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 949 | `t3_kfold_cv` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 950 | `t3_lbfgs` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 951 | `t3_mamba_encoder_text` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 952 | `t3_multimodal_multitask` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 953 | `t3_multimodal_text_image` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 954 | `t3_multimodal_text_tabular` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 955 | `t3_multimodal_zero_shot` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 956 | `t3_multitask_famo` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 957 | `t3_open_set_baseline` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 958 | `t3_open_set_entropic` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 959 | `t3_open_set_objectosphere` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 960 | `t3_optimizer_muon` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 961 | `t3_optimizer_schedule_free` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 962 | `t3_qat_fp8` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 963 | `t3_qat_mxfp4` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 964 | `t3_sample_packing` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 965 | `t3_semantic_segmentation_segformer` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 966 | `t3_structured_output_json` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 967 | `t3_uncertainty_calibrated` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 968 | `t3_uncertainty_mc_dropout` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 969 | `t4_async_grpo` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 970 | `t4_bco` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 971 | `t4_cpo` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 972 | `t4_dpo_gemma` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 973 | `t4_dpo_llama` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 974 | `t4_dpo_mistral` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 975 | `t4_dpo_phi` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 976 | `t4_dpo_qwen` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 977 | `t4_gkd` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 978 | `t4_grpo_classification` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 979 | `t4_grpo_code_exec` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 980 | `t4_grpo_format_match` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 981 | `t4_grpo_length_budget` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 982 | `t4_grpo_math` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 983 | `t4_gspo` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 984 | `t4_kto_gemma` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 985 | `t4_kto_llama` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 986 | `t4_kto_phi` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 987 | `t4_mpo` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 988 | `t4_nash_md` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 989 | `t4_online_dpo` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 990 | `t4_orpo_llama` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 991 | `t4_orpo_mistral` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 992 | `t4_orpo_qwen` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 993 | `t4_prm` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 994 | `t4_reward_ensemble` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 995 | `t4_reward_pairwise` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 996 | `t4_reward_scalar` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 997 | `t4_rlaif_constitutional` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 998 | `t4_rlaif_judge` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 999 | `t4_rlaif_self_critique` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1000 | `t4_rlhf_ppo_llama` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1001 | `t4_rlhf_ppo_mistral` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1002 | `t4_rlhf_ppo_qwen` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1003 | `t4_rloo` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1004 | `t4_simpo` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1005 | `t4_xpo` | finetune | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1006 | `format_alignment_padding` | format | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 1007 | `format_arrow_ipc` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1008 | `format_avro_schema_resolver` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1009 | `format_batch_export` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1010 | `format_bf16_vs_fp16_mantissa` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1011 | `format_convert_quantize` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1012 | `format_endianness_detector` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1013 | `format_export_gguf` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1014 | `format_export_safetensors` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1015 | `format_gguf_metadata_key` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1016 | `format_import_hf` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1017 | `format_magic_bytes_validator` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1018 | `format_manifest_tensor_count` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1019 | `format_migration_pipeline` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1020 | `format_npy_header` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1021 | `format_pickle_safety` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1022 | `format_protobuf_varint` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1023 | `format_publish` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1024 | `format_pull_cache` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1025 | `format_rosetta_chain` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1026 | `format_rosetta_convert` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1027 | `format_rosetta_verify` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1028 | `format_stride_encoder` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1029 | `format_tensor_name_canonicalizer` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1030 | `format_version_compat_matrix` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1031 | `format_zip_eocd` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1032 | `import_hf_cache` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1033 | `import_multi_format` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1034 | `publish_dry_run` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1035 | `publish_multi_registry` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1036 | `pull_resume` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1037 | `pull_verify_decompress` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1038 | `validate_manifest_happy` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1039 | `validate_manifest_live_check` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1040 | `validate_manifest_sha_mismatch` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1041 | `flash_attention_inference` | gpu | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 1042 | `gpu_async_memcpy_overlap` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1043 | `gpu_capability_detect` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1044 | `gpu_collective_op_picker` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1045 | `gpu_cuda_graph_capture` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1046 | `gpu_cuda_inference` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1047 | `gpu_dynamic_shape_handler` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1048 | `gpu_gqa_group_picker` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1049 | `gpu_kernel_autotune` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1050 | `gpu_kernel_fusion` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1051 | `gpu_kv_cache_paging` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1052 | `gpu_l1_shared_mem_partition` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1053 | `gpu_memory_bandwidth_roofline` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1054 | `gpu_memory_management` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1055 | `gpu_memory_planner` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1056 | `gpu_memory_pool` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1057 | `gpu_multi_gpu_inference` | gpu | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 1058 | `gpu_occupancy_calculator` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1059 | `gpu_pcie_bandwidth_tier` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1060 | `gpu_pcie_p2p_topology` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1061 | `gpu_persistent_kernel` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1062 | `gpu_pinned_memory_pipeline` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1063 | `gpu_ptx_analysis` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1064 | `gpu_register_spill` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1065 | `gpu_shared_mem_bank` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1066 | `gpu_tensor_core_alignment` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1067 | `gpu_tensor_core_optimization` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1068 | `gpu_tensor_parallel_split` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1069 | `gpu_vulkan_inference` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 1070 | `gpu_warp_divergence_classifier` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1071 | `gpu_warp_scheduler_dispatch` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1072 | `gpu_warp_size_dispatch` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1073 | `gpu_warp_specialization` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1074 | `ptx_disassembly` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1075 | `ptx_map_hot_regions` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1076 | `ptx_map_sass_to_ptx` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1077 | `ptx_register_usage` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1078 | `adaptive_batch_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1079 | `chat_kv_cache` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1080 | `chat_multiturn` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1081 | `chat_tool_use` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1082 | `dynamic_batch_with_sla` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1083 | `ensemble_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1084 | `inference_apr_run` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1085 | `inference_arch_alias_resolver` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1086 | `inference_arch_compare` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1087 | `inference_arch_detector` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1088 | `inference_arch_quirk_audit` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1089 | `inference_arch_resolution_pipeline` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1090 | `inference_arch_summary` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1091 | `inference_beam_search_width` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1092 | `inference_bert_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1093 | `inference_deepseek_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1094 | `inference_falcon_h1_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1095 | `inference_gemma_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1096 | `inference_gpt2_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1097 | `inference_gptneox_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1098 | `inference_kv_cache_lru` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1099 | `inference_llama_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1100 | `inference_mamba_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1101 | `inference_mistral_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1102 | `inference_mmap_lazy_load` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1103 | `inference_openelm_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1104 | `inference_opt_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1105 | `inference_phi_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1106 | `inference_qwen2_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1107 | `inference_qwen3_5_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1108 | `inference_qwen3_moe_numerical_parity_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1109 | `inference_qwen3_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1110 | `inference_run_temperature_sweep` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1111 | `inference_rwkv7_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1112 | `inference_top_p_nucleus` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1113 | `model_pipeline` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1114 | `pipeline_3stage` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1115 | `pipeline_resilient` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1116 | `quantized_inference_comparison` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1117 | `simple_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1118 | `speculative_decode` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1119 | `streaming_token_generator` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1120 | `awq_lint_batch` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1121 | `awq_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1122 | `awq_lint_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1123 | `dry_sampling_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1124 | `dry_sampling_lint_pipeline` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1125 | `dry_sampling_lint_repetition` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1126 | `embeddings_lint_dim_consistency` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1127 | `embeddings_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1128 | `embeddings_lint_l2_norm_check` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1129 | `fp8_lint_capability_gate` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1130 | `fp8_lint_happy` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1131 | `fp8_lint_saturation_violation` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1132 | `gbnf_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1133 | `gbnf_lint_malformed` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1134 | `gbnf_lint_pipeline` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1135 | `gptq_lint_cosine_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1136 | `gptq_lint_flag_combinations` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1137 | `gptq_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1138 | `grad_norm_divergence_run` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1139 | `grad_norm_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1140 | `grad_norm_spike_detection` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1141 | `imatrix_lint_corpus_entropy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1142 | `imatrix_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1143 | `imatrix_lint_nan_detection` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1144 | `lint_cors_origin_allowlist` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1145 | `lint_prompt_pii_redactor` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1146 | `lint_schema_drift_detector` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1147 | `nf4_lint_codebook_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1148 | `nf4_lint_double_quant_parity` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1149 | `nf4_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1150 | `ollama_chat_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1151 | `ollama_chat_lint_schema_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1152 | `ollama_chat_lint_stream` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1153 | `ollama_tools_lint_allowlist_gate` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1154 | `ollama_tools_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1155 | `ollama_tools_lint_streaming_ndjson` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1156 | `oom_lint_allocation_trace` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1157 | `oom_lint_happy` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1158 | `oom_lint_missing_breadcrumb` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1159 | `registry_quota_lint_atomic_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1160 | `registry_quota_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1161 | `registry_quota_lint_tenant_overage` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1162 | `rm_gc_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1163 | `rm_gc_lint_orphan_detection` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1164 | `rm_gc_lint_refcount_conservation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1165 | `shared_cache_lint_dedup_audit` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1166 | `shared_cache_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1167 | `shared_cache_lint_permission_matrix` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1168 | `tool_use_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1169 | `tool_use_lint_invalid_args` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1170 | `tool_use_lint_streaming` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1171 | `typical_p_lint_entropy_truncation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1172 | `typical_p_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1173 | `typical_p_lint_min_keep_floor` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1174 | `unified_search_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1175 | `unified_search_lint_offline_consistency` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1176 | `unified_search_lint_rrf_recompute` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1177 | `mcp_byte_parity_dispatcher_swap` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1178 | `mcp_capability_handshake_diff` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1179 | `mcp_client_simulation` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1180 | `mcp_embedded_initialize_handshake` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1181 | `mcp_embedded_protocol_invariants` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1182 | `mcp_embedded_tools_list_discovery` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1183 | `mcp_error_code_classifier` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1184 | `mcp_notification_progress_token` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1185 | `mcp_request_id_correlator` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1186 | `mcp_resource_uri_resolver` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1187 | `mcp_session_lifecycle` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1188 | `mcp_sse_event_envelope` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1189 | `mcp_stdio_server` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1190 | `mcp_tool_discovery` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1191 | `mcp_tool_signature_validator` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1192 | `mcp_websocket_frame_envelope` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1193 | `cbtop_headless` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1194 | `cbtop_histogram` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1195 | `cbtop_streaming` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1196 | `hash_chain_audit` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1197 | `inference_cost_tracking` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1198 | `inference_explainability` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1199 | `latency_histogram` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1200 | `model_drift_detection` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1201 | `monitor_aggregation_window` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1202 | `monitor_aggregation_window_dispatcher` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1203 | `monitor_alert_dedup_window` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1204 | `monitor_alerting` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1205 | `monitor_anomaly_z_score_classifier` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1206 | `monitor_cache_hit_rate` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1207 | `monitor_canary_metric` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1208 | `monitor_capacity_planner` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1209 | `monitor_circuit_log_emit` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1210 | `monitor_concurrent_inflight` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1211 | `monitor_correlated_failures` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1212 | `monitor_db_pool_exhaustion` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1213 | `monitor_disk_io_pressure` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1214 | `monitor_drift_psi` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1215 | `monitor_dropped_request_classifier` | monitoring | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1216 | `monitor_gpu_thermal_throttle` | monitoring | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1217 | `monitor_gradient_norm_alert` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1218 | `monitor_health_check_endpoint` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1219 | `monitor_inference_cost` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1220 | `monitor_ingest_backpressure` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1221 | `monitor_log_correlation` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1222 | `monitor_log_pii_redact` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1223 | `monitor_log_sampling` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1224 | `monitor_log_volume_anomaly` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1225 | `monitor_metric_cardinality` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1226 | `monitor_oom_predict` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1227 | `monitor_p50_drift` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1228 | `monitor_p99_outlier` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1229 | `monitor_packet_loss_rate` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1230 | `monitor_percentile_summary` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1231 | `monitor_query_pattern` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1232 | `monitor_realtime` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1233 | `monitor_request_size_histogram` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1234 | `monitor_seasonal_decompose` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1235 | `monitor_session_drop` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1236 | `monitor_sla_uptime_calc` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1237 | `monitor_slo_burn_rate` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1238 | `monitor_synthetic_probe` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1239 | `monitor_thread_pool_saturation` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1240 | `monitor_token_throughput` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1241 | `monitor_trace_sampling` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1242 | `monitoring_energy_estimation` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1243 | `monitoring_memory_profiler` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1244 | `mc_a_star_path_cost` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1245 | `mc_antithetic_variance_reduce` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1246 | `mc_battery_discharge_curve` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1247 | `mc_bayesian_ab_winrate` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1248 | `mc_birthday_collision` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1249 | `mc_blackjack_house_edge` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1250 | `mc_blockchain_fork_resolve` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1251 | `mc_bloom_filter_false_positive` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1252 | `mc_boltzmann_distribution` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1253 | `mc_bond_percolation_threshold` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1254 | `mc_bootstrap_resample_mean` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1255 | `mc_brownian_bridge_path` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1256 | `mc_brownian_motion_path` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1257 | `mc_buffon_needle_pi` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1258 | `mc_burst_buffer_overflow` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1259 | `mc_business_revenue_forecast` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1260 | `mc_cache_warmup_curve` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1261 | `mc_caching_eviction_oldest` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1262 | `mc_caching_layer_warmup` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1263 | `mc_calibration_drift` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1264 | `mc_capacity_planner` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1265 | `mc_card_shuffle_riffle` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1266 | `mc_chaos_monkey_failures` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1267 | `mc_chi_squared_uniformity` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1268 | `mc_chinese_restaurant_process` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1269 | `mc_circuit_breaker_recovery` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1270 | `mc_circuit_open_rate` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1271 | `mc_circuit_recovery_time` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1272 | `mc_clock_drift_compensation` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1273 | `mc_coalescent_tree_sample` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1274 | `mc_coin_flip_max_streak` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1275 | `mc_concurrency_collapse` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1276 | `mc_consensus_quorum_failure` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1277 | `mc_consensus_round_latency` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1278 | `mc_control_variate_reduction` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1279 | `mc_correlated_burst` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1280 | `mc_correlated_portfolio_var` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1281 | `mc_count_min_sketch_estimate` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1282 | `mc_coupon_collector` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1283 | `mc_data_freshness_lag` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1284 | `mc_dataset_split_uniformity` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1285 | `mc_db_lock_wait_time` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1286 | `mc_decode_branching_path` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1287 | `mc_deterministic_replay_diverge` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1288 | `mc_dice_throw_distribution` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1289 | `mc_disk_failure_raid` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1290 | `mc_disk_io_queue_depth` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1291 | `mc_disk_seek_pattern` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1292 | `mc_distill_label_corruption` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1293 | `mc_distributed_lock_contention` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1294 | `mc_dns_resolver_cache` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1295 | `mc_drop_in_replacement_test` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1296 | `mc_dropout_dependency` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1297 | `mc_dropout_resilience` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1298 | `mc_elementary_ca_rule30` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1299 | `mc_elephant_walk_memory` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1300 | `mc_epidemic_sir_model` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1301 | `mc_eval_split_significance` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1302 | `mc_event_correlation_window` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1303 | `mc_eviction_thrashing` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1304 | `mc_eviction_under_pressure` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1305 | `mc_exponential_backoff_jitter` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1306 | `mc_failure_chain_propagation` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1307 | `mc_firefly_optimization` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1308 | `mc_forest_fire_spread` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1309 | `mc_full_shuffle_quality` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1310 | `mc_galton_board_distribution` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1311 | `mc_gamblers_ruin` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1312 | `mc_garbage_collection_pause` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1313 | `mc_gaussian_kde_estimate` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1314 | `mc_genetic_algorithm_convergence` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1315 | `mc_genetic_drift_population` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1316 | `mc_geo_distance_routing` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1317 | `mc_geomean_estimate` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1318 | `mc_german_tank_problem` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1319 | `mc_gibbs_sampler_bivariate` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1320 | `mc_grid_walk_2d_diffusion` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1321 | `mc_hash_collision_birthday` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1322 | `mc_hawkes_self_exciting_process` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1323 | `mc_hit_or_miss_integration` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1324 | `mc_idempotency_collision` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1325 | `mc_inference_burst_buffer` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1326 | `mc_inference_jitter` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1327 | `mc_inference_p99_estimator` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1328 | `mc_inhomog_poisson_thinning` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1329 | `mc_inventory_replenishment` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1330 | `mc_inverse_cdf_sample` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1331 | `mc_ising_model_2d` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1332 | `mc_island_model_migration` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1333 | `mc_jidoka_guard_failure_rate` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1334 | `mc_jit_amortization` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1335 | `mc_jitter_buffer_underrun` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1336 | `mc_jobqueue_completion_time` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1337 | `mc_kademlia_routing_lookup` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1338 | `mc_kafka_partition_skew` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1339 | `mc_kalman_filter_smoothing` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1340 | `mc_kelly_criterion_bet_size` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1341 | `mc_knapsack_random` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1342 | `mc_kv_cache_hit_under_eviction` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1343 | `mc_kv_eviction_compare` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1344 | `mc_kv_zipf_hit_rate` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1345 | `mc_lazy_replication_lag` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1346 | `mc_levy_flight_step` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1347 | `mc_load_balancer_least_conn` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1348 | `mc_load_shed_threshold` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1349 | `mc_load_test_ramp_up` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1350 | `mc_loadbalancer_health_check` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1351 | `mc_log_aggregation_window` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1352 | `mc_lru_admission_filter` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1353 | `mc_lru_vs_fifo_eviction` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1354 | `mc_m_m_1_queue_wait` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1355 | `mc_markov_text_generator` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1356 | `mc_metropolis_hastings_target` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1357 | `mc_mm1_queue_little_law` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1358 | `mc_monty_hall_problem` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1359 | `mc_negative_binomial_overdisp` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1360 | `mc_neural_dropout_inference` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1361 | `mc_p_value_under_null` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1362 | `mc_packet_loss_burst_pattern` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1363 | `mc_pareto_principle_80_20` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1364 | `mc_password_brute_force_eta` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1365 | `mc_pi_estimation_buffon` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1366 | `mc_pipeline_stage_bottleneck` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1367 | `mc_polya_ballot_problem` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1368 | `mc_polya_urn_color_balance` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1369 | `mc_preferential_attachment_growth` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1370 | `mc_priority_inversion_detect` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1371 | `mc_priority_queue_aging` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1372 | `mc_priority_queue_wait` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1373 | `mc_priority_starvation` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1374 | `mc_prisoner_box_strategy` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1375 | `mc_quantization_loss_estimator` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1376 | `mc_quantized_round_trip` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1377 | `mc_quorum_read_consistency` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1378 | `mc_random_binary_tree` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1379 | `mc_random_forest_voting` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1380 | `mc_random_graph_connectedness` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1381 | `mc_random_partition_set` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1382 | `mc_random_polygon_area` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1383 | `mc_random_walk_2d_diffusion` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1384 | `mc_random_walk_drunkard` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1385 | `mc_rejection_sampling_normal` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1386 | `mc_replica_failover_time` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1387 | `mc_replica_lag_distribution` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1388 | `mc_request_arrival_poisson` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1389 | `mc_request_coalescing` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1390 | `mc_request_collision_dedup` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1391 | `mc_request_lifetime` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1392 | `mc_request_routing_a_b` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1393 | `mc_request_signature_verification_throughput` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1394 | `mc_request_size_log_normal` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1395 | `mc_request_size_pareto` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1396 | `mc_reservoir_sample` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1397 | `mc_resource_contention` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1398 | `mc_retry_chain_success` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1399 | `mc_revenue_max_pricing` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1400 | `mc_rng_period_check` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1401 | `mc_rumor_spread_dk_model` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1402 | `mc_safety_stock_planner` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1403 | `mc_secretary_problem` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1404 | `mc_secretary_two_choice` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1405 | `mc_self_avoiding_walk` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1406 | `mc_session_affinity_routing` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1407 | `mc_sim_clock_drift_detect` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1408 | `mc_simulated_annealing_optimum` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1409 | `mc_simulated_pi_dart` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1410 | `mc_skewed_load_distribution` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1411 | `mc_speculative_decode_acceptance` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1412 | `mc_sphere_volume_estimate` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1413 | `mc_spinning_wheel_payout` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1414 | `mc_st_petersburg_paradox` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1415 | `mc_stock_price_simulation_gbm` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1416 | `mc_stratified_sampling` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1417 | `mc_streaming_underflow` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1418 | `mc_supply_chain_lead_time_jitter` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1419 | `mc_texas_holdem_pocket_pair` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1420 | `mc_throttle_token_bucket_fairness` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1421 | `mc_throughput_concurrency_curve` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1422 | `mc_token_billing_estimator` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1423 | `mc_token_compress_ratio` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1424 | `mc_token_dropout_resilience` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1425 | `mc_token_pricing_arbitrage` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1426 | `mc_token_refill_rate_match` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1427 | `mc_token_streaming_jitter` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1428 | `mc_token_throughput_sim` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1429 | `mc_traffic_jam_density` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1430 | `mc_traffic_light_intersection` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1431 | `mc_traveling_salesman_random` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1432 | `mc_tsp_random_search` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1433 | `mc_two_armed_bandit_thompson` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1434 | `mc_value_at_risk_historical_vs_parametric` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1435 | `mc_voronoi_cell_area` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1436 | `mc_voting_majority_consensus` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1437 | `mc_walking_meander` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1438 | `mc_warehouse_pick_path_length` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1439 | `mc_warm_vs_cold_cache` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1440 | `mc_warmup_sample_efficiency` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1441 | `mc_warmup_to_target` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1442 | `mc_work_steal_scheduler_balance` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1443 | `mc_zipf_law_word_freq` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1444 | `mc_zipf_request_distribution` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1445 | `distill_checkpoint` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1446 | `distill_ensemble` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1447 | `distill_progressive` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1448 | `distill_standard_kl` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1449 | `finetune_lora` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1450 | `finetune_merge_adapter` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1451 | `finetune_plan_vram` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1452 | `finetune_qlora` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1453 | `merge_average` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1454 | `merge_dare` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1455 | `merge_hierarchical` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1456 | `merge_slerp` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1457 | `merge_ties` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1458 | `merge_weighted` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1459 | `optimize_adamw_vs_lion_picker` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1460 | `optimize_full_pipeline` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1461 | `optimize_micro_batch_grad_accum` | optimize | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1462 | `optimize_tune` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1463 | `optimize_warmup_cosine_lr` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1464 | `prune_depth` | optimize | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1465 | `prune_gradual_schedule` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1466 | `prune_magnitude` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1467 | `prune_structured` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1468 | `prune_wanda` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1469 | `quantize_4bit` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1470 | `quantize_fake_qat` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1471 | `quantize_gptq` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1472 | `quantize_mixed_precision` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1473 | `tune_bayesian` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1474 | `tune_grid_early_stop` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1475 | `registry_alias_resolver_chain` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1476 | `registry_aliases_diff` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1477 | `registry_aliases_list` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1478 | `registry_aliases_resolve` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1479 | `registry_artifact_size_quota` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1480 | `registry_dependency_resolver` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1481 | `registry_gc_orphans` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1482 | `registry_gc_policy` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1483 | `registry_hash_pin_enforcer` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1484 | `registry_immutable_tag_policy` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1485 | `registry_manifest_schema` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1486 | `registry_metadata_search` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1487 | `registry_model_comparison` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1488 | `registry_model_lineage` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1489 | `registry_model_rollback` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1490 | `registry_model_versioning` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1491 | `registry_oci_index` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1492 | `registry_provenance_attestation` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1493 | `registry_pull_secret_validator` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1494 | `registry_pull_throttle` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1495 | `registry_quota_per_user` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1496 | `registry_register_apr` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1497 | `registry_signature_chain` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1498 | `http_model_server` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1499 | `model_ab_testing` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1500 | `model_canary_deploy` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1501 | `model_rate_limiter` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1502 | `model_selection_router` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1503 | `serve_anthropic_messages_api_drop_in` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1504 | `serve_grpc_stream` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1505 | `serve_plan_hf_dryrun_no_weights` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1506 | `serve_rate_limited` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1507 | `serverless_cold_start_classifier` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1508 | `serverless_cold_start_optimization` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1509 | `serverless_concurrency_limit` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1510 | `serverless_concurrency_validator` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1511 | `serverless_container_image` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1512 | `serverless_edge_function` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1513 | `serverless_image_layer_caching` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1514 | `serverless_lambda_inference` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1515 | `serverless_lambda_pricing_matrix` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1516 | `serverless_model_warmup` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1517 | `serverless_provisioned_concurrency` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1518 | `serverless_step_function_router` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1519 | `serverless_timeout_budget_picker` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1520 | `serverless_vpc_cold_start` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 1521 | `shell_brace_expander` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1522 | `shell_corpus_from_string` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1523 | `shell_history_parse_zsh` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1524 | `shell_pipe_redirection_parser` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1525 | `shell_quote_state_classifier` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1526 | `shell_trie_prefix_completion` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1527 | `simd_alignment_strategy` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 1528 | `simd_alignment_validator` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 1529 | `simd_aos_to_soa` | simd | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1530 | `simd_auto_vectorization` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 1531 | `simd_avx_vnni_int8_inference` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 1532 | `simd_count_trailing_zeros_lane` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 1533 | `simd_dot_product_lane` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 1534 | `simd_fma_fusion_gate` | simd | ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 1535 | `simd_horizontal_max` | simd | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1536 | `simd_horizontal_min_lanes` | simd | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1537 | `simd_horizontal_reduce_dispatcher` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 1538 | `simd_lane_swizzle_validate` | simd | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1539 | `simd_loop_carried_dep_detector` | simd | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1540 | `simd_mask_lane_predicate` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 1541 | `simd_matrix_ops` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 1542 | `simd_pop_count_lane` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 1543 | `simd_prefetch_distance_picker` | simd | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1544 | `simd_quantized_operations` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 1545 | `simd_saturating_arithmetic` | simd | ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 1546 | `simd_strided_load` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 1547 | `simd_sum_lanes_reduce` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 1548 | `simd_unroll_factor_picker` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 1549 | `simd_vectorized_inference` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 1550 | `simd_zigzag_packer` | simd | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1551 | `trueno_simd_ops` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 1552 | `speech_audio_format_validator` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1553 | `speech_audio_resampler` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1554 | `speech_chunk_overlap_planner` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1555 | `speech_diarization` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1556 | `speech_diarization_count` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1557 | `speech_language_id_confidence` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1558 | `speech_multilingual` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1559 | `speech_punctuation_restorer` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1560 | `speech_vad` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1561 | `speech_vad_threshold` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1562 | `speech_vad_threshold_classifier` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1563 | `speech_word_timestamp_align` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1564 | `whisper_streaming` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1565 | `whisper_transcribe` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1566 | `autograd_backprop_viz` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1567 | `autograd_custom_ops` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1568 | `autograd_gradient_clipping` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1569 | `checkpoint_resume` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1570 | `continuous_train_curriculum` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1571 | `continuous_train_federated_simulation` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1572 | `continuous_train_incremental` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1573 | `continuous_train_online_learning` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1574 | `data_preprocessing` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1575 | `data_sharded_shuffle` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1576 | `data_streaming_tokens` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1577 | `entrenar_autograd_training` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1578 | `entrenar_eval_metrics` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1579 | `few_shot_finetune` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1580 | `gradient_accumulation` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1581 | `hyperparameter_sweep` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1582 | `learning_rate_schedule` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1583 | `mixed_precision_training` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1584 | `pretrain_checkpoint_resume` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1585 | `pretrain_nan_guard` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1586 | `pretrain_synthetic_decreasing` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1587 | `train_distributed_sim` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1588 | `train_grad_accum` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1589 | `training_attention_dropout` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1590 | `training_curriculum_filter` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1591 | `training_eval_cadence` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1592 | `training_grad_accum_steps` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1593 | `training_grad_clip_norm` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1594 | `training_label_smoothing` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1595 | `training_long_context_extension` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1596 | `training_loss_scaler` | training | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 1597 | `training_loss_spike_detector` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1598 | `training_lr_combo_scheduler` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1599 | `training_lr_warmup_decay` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1600 | `training_optimizer_state_memory` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1601 | `training_pretrain_data_mix` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1602 | `training_warmup_steps` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1603 | `training_zero3_partitioning` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1604 | `tsp_christofides_ratio` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1605 | `tsp_compare_tabu_vs_genetic` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1606 | `tsp_distance_matrix_explicit` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1607 | `tsp_nearest_neighbor` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1608 | `tsp_solve_with_tabu` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1609 | `tsp_two_opt_swap_improver` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1610 | `tui_action_undo_stack` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1611 | `tui_active_indicator_dot` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1612 | `tui_alert_dismiss_state` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1613 | `tui_alphabetical_index_jump` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1614 | `tui_animation_easing_curve` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1615 | `tui_arrow_key_navigation` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1616 | `tui_ascii_histogram_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1617 | `tui_aspect_ratio_grid` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1618 | `tui_banner_alert_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1619 | `tui_braces_match_highlight` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1620 | `tui_breadcrumb_click_target` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1621 | `tui_breadcrumb_collapse` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1622 | `tui_breadcrumb_path` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1623 | `tui_breadcrumb_search_filter` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1624 | `tui_breadcrumb_separator_style` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1625 | `tui_button_kbd_mnemonic` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1626 | `tui_calendar_grid` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1627 | `tui_carousel_advance_step` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1628 | `tui_chart_axis_ticks` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1629 | `tui_chart_bar_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1630 | `tui_chart_legend_swatch` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1631 | `tui_chart_threshold_overlay` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1632 | `tui_clip_to_visible` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1633 | `tui_clipboard_buffer` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1634 | `tui_clipboard_history_circular` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1635 | `tui_clipboard_paste_filter` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1636 | `tui_clipboard_paste_pending` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1637 | `tui_color_contrast_pass` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1638 | `tui_color_palette_quantize` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1639 | `tui_color_scheme_validate` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1640 | `tui_color_swatch_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1641 | `tui_column_header_sort_cycle` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1642 | `tui_command_history` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1643 | `tui_command_history_search_pattern` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1644 | `tui_command_line_history_dedupe` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1645 | `tui_command_palette` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1646 | `tui_command_palette_score` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1647 | `tui_context_menu_show` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1648 | `tui_cursor_blink_phase` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1649 | `tui_cursor_shape_phase` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1650 | `tui_date_picker_validate` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1651 | `tui_dialog_button_focus` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1652 | `tui_diff_renderer` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1653 | `tui_diff_three_way_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1654 | `tui_diff_word_level` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1655 | `tui_drag_drop_constrain` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1656 | `tui_drawer_slide_compute` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1657 | `tui_dropdown_filter` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1658 | `tui_duration_humanize` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1659 | `tui_emoji_picker_filter` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1660 | `tui_empty_state_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1661 | `tui_find_replace_count` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1662 | `tui_focus_ring_traversal` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1663 | `tui_fold_block_collapse` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1664 | `tui_form_field_tab_order` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1665 | `tui_form_layout_grid` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1666 | `tui_form_validation` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1667 | `tui_fps_meter_smooth` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1668 | `tui_gauge_meter_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1669 | `tui_gradient_fill_compute` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1670 | `tui_grid_focus_navigation` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1671 | `tui_help_overlay_dismiss` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1672 | `tui_help_panel_state` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1673 | `tui_history_navigation` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1674 | `tui_horizontal_scroll` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1675 | `tui_indent_guides_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1676 | `tui_inline_code_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1677 | `tui_inline_diff_view` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1678 | `tui_inline_emoji_replace` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1679 | `tui_input_autocomplete` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1680 | `tui_input_buffer` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1681 | `tui_input_history_navigate` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1682 | `tui_input_max_length_indicator` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1683 | `tui_input_validation_state` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1684 | `tui_kbd_chord_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1685 | `tui_kbd_shortcuts_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1686 | `tui_keybinding_dispatch` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1687 | `tui_keyboard_repeat_rate` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1688 | `tui_keymap_dispatch` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1689 | `tui_keymap_register_chord` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1690 | `tui_layout_constraint_solver` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1691 | `tui_layout_split` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1692 | `tui_line_number_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1693 | `tui_live_preview_throttle` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1694 | `tui_loading_dots_animation` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1695 | `tui_loading_skeleton` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1696 | `tui_log_level_filter` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1697 | `tui_log_tail_buffer` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1698 | `tui_marquee_scroll` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1699 | `tui_marquee_speed_control` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1700 | `tui_menu_render_aligned` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1701 | `tui_message_box_align` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1702 | `tui_minilist_filter_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1703 | `tui_minimap_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1704 | `tui_modal_focus_stack` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1705 | `tui_modal_keymap` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1706 | `tui_modal_slide_transition` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1707 | `tui_notification_toast_queue` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1708 | `tui_overflow_marker_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1709 | `tui_overlay_z_order` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1710 | `tui_pane_split_compute` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1711 | `tui_paragraph_wrap` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1712 | `tui_password_input_mask` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1713 | `tui_password_mask` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1714 | `tui_password_strength_meter` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1715 | `tui_pill_badge_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1716 | `tui_pixel_art_palette_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1717 | `tui_popup_dialog_modal` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1718 | `tui_progress_band_color` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1719 | `tui_progress_estimate_remaining` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1720 | `tui_progress_eta_format` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1721 | `tui_progress_indeterminate` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1722 | `tui_progress_multi_bar` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1723 | `tui_progress_smooth_eta` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1724 | `tui_progress_state` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1725 | `tui_progress_throughput_avg` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1726 | `tui_quote_block_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1727 | `tui_radio_group_select` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1728 | `tui_range_slider_validate` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1729 | `tui_resize_constraints` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1730 | `tui_resize_grid_layout` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1731 | `tui_resize_overflow_compact` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1732 | `tui_resize_safe_truncate` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1733 | `tui_resize_split_constraint` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1734 | `tui_scroll_state` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1735 | `tui_search_index_navigate` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1736 | `tui_search_input_box` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1737 | `tui_search_jump_list` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1738 | `tui_search_replace_preview` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1739 | `tui_segmented_control_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1740 | `tui_select_box_arrow_nav` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1741 | `tui_select_popup` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1742 | `tui_severity_color` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1743 | `tui_sidebar_toggle_state` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1744 | `tui_smart_indent_continue` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1745 | `tui_smooth_scroll_velocity` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1746 | `tui_sparkline_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1747 | `tui_spell_check_underline` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1748 | `tui_spinner_frame` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1749 | `tui_split_button_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1750 | `tui_split_pane_resize` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1751 | `tui_status_bar_compose` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1752 | `tui_status_dot_indicator` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1753 | `tui_status_history_compact` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1754 | `tui_status_pulse_dot` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1755 | `tui_sticky_header_pin` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1756 | `tui_syntax_token_classify` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1757 | `tui_tabbed_view_switch` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1758 | `tui_table_cell_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1759 | `tui_table_column_resize` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1760 | `tui_table_pager` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1761 | `tui_table_pagination_indicator` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1762 | `tui_table_sort_indicator` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1763 | `tui_table_zebra_stripe` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1764 | `tui_tag_chip_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1765 | `tui_tag_cloud_size_compute` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1766 | `tui_task_status_check` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1767 | `tui_terminal_bell_throttle` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1768 | `tui_text_alignment` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1769 | `tui_text_search_highlight` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1770 | `tui_text_search_match_count` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1771 | `tui_theme_switch_apply` | tui | ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 1772 | `tui_toast_notification_render` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1773 | `tui_toolbar_btn_overflow` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1774 | `tui_tooltip_show_delay` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1775 | `tui_trailing_ws_highlight` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1776 | `tui_tree_view_collapse` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1777 | `tui_truncate_ellipsis` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1778 | `tui_truncate_path_with_dots` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1779 | `tui_typeahead_completion` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1780 | `tui_undo_redo_stack` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1781 | `tui_unicode_emoji_width` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1782 | `tui_widget_focus_chain` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1783 | `tui_window_layout_anchor` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1784 | `tui_word_count_status` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1785 | `tui_word_wrap` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1786 | `tui_wrap_strategy_compute` | tui | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1787 | `load_visualization` | visualization | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1788 | `viz_axis_scale_classifier` | visualization | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1789 | `viz_color_palette_picker` | visualization | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1790 | `viz_legend_placement_optimizer` | visualization | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 1791 | `wasm_atomic_wait_timeout` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1792 | `wasm_browser_compat_matrix` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1793 | `wasm_browser_inference` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1794 | `wasm_capability_check` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1795 | `wasm_component_export` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1796 | `wasm_custom_section_emit` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1797 | `wasm_data_segment_overlap` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1798 | `wasm_export_section_size` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1799 | `wasm_export_table_validator` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1800 | `wasm_features_compat_check` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1801 | `wasm_function_table_growth` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1802 | `wasm_globals_section` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1803 | `wasm_import_count_budget` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1804 | `wasm_imported_function_count` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1805 | `wasm_magic_bytes_check` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1806 | `wasm_memory_grow_step` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1807 | `wasm_memory_growth_policy` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1808 | `wasm_memory_growth_strategy` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1809 | `wasm_model_loader` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1810 | `wasm_module_cache_strategy` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1811 | `wasm_module_size_budget` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1812 | `wasm_name_section_validate` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1813 | `wasm_progressive_loading` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1814 | `wasm_relaxed_simd` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1815 | `wasm_simd128_dispatch` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1816 | `wasm_simd_lane_dispatcher` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1817 | `wasm_start_section_validate` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1818 | `wasm_streaming_compilation` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1819 | `wasm_table_index_resolver` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1820 | `wasm_table_max_size_check` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1821 | `wasm_tail_call_dispatch` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1822 | `wasm_threads_atomics_gate` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1823 | `wasm_wasi_capability_grant` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1824 | `wasm_web_worker` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 1825 | `wasm_webgpu_acceleration` | wasm | ![wgpu](https://img.shields.io/badge/-wgpu-green) ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |

</details>
<!-- RECIPE-TABLE-END -->

## APR v2 Format

| Feature | Spec |
|---------|------|
| Compression | LZ4 (3+ GB/s) / ZSTD |
| Quantization | FP32, FP16, Int8, Int4 |
| Signatures | Ed25519 |
| Tensor Index | Binary, O(1) lookup |
| Zero-Copy | Full mmap support |

```rust
use apr_cookbook::prelude::*;

let bundle = ModelBundleV2::new()
    .with_name("my-model")
    .with_compression(Compression::Lz4)
    .with_quantization(Quantization::Int8)
    .add_tensor("weights", vec![768, 768], weight_bytes)
    .build();

assert_eq!(&bundle[0..4], b"APR2");
```

## Quality

```bash
cargo fmt --check
cargo clippy --all-targets -- -D warnings
cargo test --all-features
```

Every example includes a doc header with QA checklist, uses
`RecipeContext` for isolation, and contains 8-15 unit tests.

### Provable-contract scoreboard (`pv score --binding contracts/binding.yaml`)

| Metric | Value | Grade |
|---|---|---|
| Codebase (simple mean) | **0.92** | **A** |
| PVScore (10-dim composite) | **94.8** | **A** |
| Mean per-contract | 0.89 | B |
| Binding coverage | 76% | — |
| Proof depth (Kani + Lean) | 0.92 | — |
| Drift | 1.00 | — |

**Per-contract breakdown (11 contracts, 2026-04-23):**
recipe-iiur 0.96 (A), whisper-wer 0.94 (A), docs-schema 0.93 (A), cli-parity 0.90 (A),
int4-quantization 0.90 (A), apr-format-roundtrip 0.90 (A), mmap-inference 0.84 (B),
lz4-decompression 0.83 (B), flash-attention 0.81 (B), aes256-gcm-decrypt 0.79 (B),
avx512-matmul 0.79 (B). **No Grade C contracts remain.**

### Formal verification

- **23/39 Lean 4 theorems proved** — structural invariants, WER non-negativity, tensor round-trip preservation. Runtime/hardware claims (latency, throughput, crypto correctness) remain `:= by sorry` honestly. See `lean/ProvableContracts/` + `make lean-build`.
- **39/39 Kani harnesses** — `#[kani::proof]` for every contract obligation, strategies: `exhaustive | bounded_int | compositional | stub_float`. See `kani/src/lib.rs`.
- **11/11 contract YAMLs validate** — `cargo test --test contracts` enforces parse + schema in-process via `provable_contracts::schema`.

## Sovereign AI Stack

All stack crates are consolidated in the APR-MONO workspace at v0.31.2. Cargo **package** names differ from Rust **lib** names — the monorepo preserved historical lib idents for source compatibility.

| Cargo package | Rust lib | Version | Role |
|---|---|---|---|
| [aprender-core](https://crates.io/crates/aprender-core) | `aprender` | 0.31.2 | APR v2 format, LZ4/ZSTD compression, Int4/Int8 quantization |
| [aprender-compute](https://crates.io/crates/aprender-compute) | `trueno` | 0.31.2 | SIMD/GPU tensor operations (AVX-512/NEON/wgpu) |
| [aprender-train](https://crates.io/crates/aprender-train) | `entrenar` | 0.31.2 | Autograd, LoRA/QLoRA, model merge, distillation |
| [aprender-contracts](https://crates.io/crates/aprender-contracts) | `provable_contracts` | 0.31.2 | YAML contract validation (dev-dep, in-process) |

## Live Demos

| Demo | Link |
|------|------|
| Monte Carlo S&P 500 | [Launch](https://interactive.paiml.com/monte-carlo-sp500/) |
| Shell ML Autocomplete | [Launch](https://interactive.paiml.com/shell-ml/) |

## Documentation

- [The APR Cookbook](book/src/introduction.md) -- mdBook source (build with `mdbook serve book/`)
- [Specification](docs/specifications/apr-cookbook.md) -- Unified spec with component docs
- [API Reference](book/src/reference/api.md) -- module-level API overview

## Usage

```bash
# Run any example by name
cargo run --example create_apr_from_scratch
cargo run --example optimize_full_pipeline
cargo run --example analysis_inspect

# Build all examples
cargo build --examples

# Run tests
cargo test --all-features

# With encryption feature
cargo run --example bundle_encrypted_model --features encryption
```

## Contributing

1. Fork and clone the repository
2. Create examples following IIUR principles (see `docs/specifications/components/principles.md`)
3. Ensure `cargo clippy --all-targets -- -D warnings` passes
4. Ensure `cargo test --all-features` passes
5. Every recipe file must be under 500 lines; split into `main.rs` + `types.rs` + `helpers.rs` if needed

## License

MIT
