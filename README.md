<p align="center">
  <img src=".github/apr-cookbook-hero.svg" width="800" alt="apr-cookbook">
</p>

<h1 align="center">apr-cookbook</h1>

<p align="center">
  <strong>The umbrella cookbook for the PAIML sovereign AI stack — model bundling, data loading, deployment-as-recipe, and visualization, all in pure Rust.</strong>
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

**Total: 341 recipes.** Run any example: `cargo run --example <name>`

**Demo-run baseline (2026-04-23):** 330 / 341 pass under 10s; the remaining 11 are compute-heavy benchmarks (see `docs/specifications/components/quality-gates.md#demo-run-baseline`) that require a longer timeout. 0 failures.

### Full recipe table

<!-- RECIPE-TABLE-START -->
<!-- Auto-generated by scripts/generate-recipe-table.sh — do not edit manually -->
<!-- Re-generate: ./scripts/generate-recipe-table.sh --update -->
<!-- CI validates: recipe-table workflow ensures this table matches source -->

**915 recipes** | Build: [![CI](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml)

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
| 14 | `adv_canary_promotion_gate` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 15 | `adv_capability_manifest_validator` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 16 | `adv_chunked_prefill` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 17 | `adv_continuous_batching` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 18 | `adv_iiur_compliance_scorer` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 19 | `adv_kv_cache_eviction` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 20 | `adv_pipeline_dag` | advanced | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 21 | `adv_quantize_pipeline` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 22 | `adv_recipe_dependency_dag` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 23 | `adv_replica_failover` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 24 | `adv_request_priority_router` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 25 | `adv_speculative_decode_window` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 26 | `cicd_model_pipeline` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 27 | `clip_search` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 28 | `code_defect_oracle` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 29 | `compliance_audit` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 30 | `debug_fix_loop` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 31 | `edge_anomaly_detection` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 32 | `embedding_visualization` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 33 | `handwriting_recognition` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 34 | `hierarchical_cache_benchmark` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 35 | `image_classification` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 36 | `model_inspection_scoring` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 37 | `model_showcase` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 38 | `online_training_defect` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 39 | `quantization_quality_tradeoff` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 40 | `rag_pipeline` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 41 | `showcase_gallery` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 42 | `showcase_markdown` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 43 | `spanish_tutor` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 44 | `streaming_sentiment` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 45 | `style_transfer` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 46 | `voice_recognition` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 47 | `wasm_summarizer` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 48 | `analysis_bench` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 49 | `analysis_canary` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 50 | `analysis_check` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 51 | `analysis_compare_hf` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 52 | `analysis_compare_hf_threshold` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 53 | `analysis_contract_algorithm_binding_pattern` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 54 | `analysis_cpu_vs_gpu_parity_gate` | analysis | ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 55 | `analysis_debug` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 56 | `analysis_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 57 | `analysis_eval` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 58 | `analysis_explain` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 59 | `analysis_flow` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 60 | `analysis_hex` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 61 | `analysis_inspect` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 62 | `analysis_json_schema_draft7_meta_validation` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 63 | `analysis_latency_breakdown` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 64 | `analysis_lint` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 65 | `analysis_memory_leak_detector` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 66 | `analysis_model_fingerprint` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 67 | `analysis_oracle` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 68 | `analysis_p99_throughput_gate` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 69 | `analysis_parity` | analysis | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 70 | `analysis_probar` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 71 | `analysis_profile` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 72 | `analysis_pv_check_parity_authoring` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 73 | `analysis_qa_capability` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 74 | `analysis_qa_gates` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 75 | `analysis_qa_report` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 76 | `analysis_qualify` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 77 | `analysis_slice` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 78 | `analysis_tensors` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 79 | `analysis_tensors_stats` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 80 | `analysis_trace` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 81 | `analysis_tree` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 82 | `analysis_validate` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 83 | `bench_batch_sweep` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 84 | `bench_quantization_compare` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 85 | `canary_regression` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 86 | `canary_rolling_window` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 87 | `check_batch` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 88 | `check_json_report` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 89 | `debug_activation_dist` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 90 | `debug_nan_trace` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 91 | `eval_benchmark_suite` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 92 | `eval_pass_at_k` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 93 | `experiment_ab_test` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 94 | `experiment_multi_seed` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 95 | `explain_error_codes` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 96 | `explain_shape_mismatch` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 97 | `flow_arch_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 98 | `flow_depth_sweep` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 99 | `hex_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 100 | `hex_pattern_search` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 101 | `inspect_layer_params` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 102 | `inspect_quantization_stats` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 103 | `lint_naming_rules` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 104 | `lint_suppression` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 105 | `oracle_classifier` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 106 | `oracle_ensemble_vote` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 107 | `parity_format` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 108 | `parity_quantization` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 109 | `probar_regression_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 110 | `probar_suite_runner` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 111 | `profile_memory_layers` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 112 | `profile_roofline` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 113 | `qualify_remediation` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 114 | `qualify_scorecard` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 115 | `trace_per_op_latency` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 116 | `trace_run_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 117 | `tree_arch_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 118 | `tree_param_rollup` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 119 | `api_admission_control_queue` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 120 | `api_auth_middleware` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 121 | `api_batch_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 122 | `api_call_model_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 123 | `api_circuit_breaker_state` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 124 | `api_idempotency_key_dedup` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 125 | `api_model_health_check` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 126 | `api_oauth_token_refresh_window` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 127 | `api_request_id_correlation` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 128 | `api_request_timeout_classifier` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 129 | `api_response_cache_ttl` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 130 | `api_streaming_chunk_size` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 131 | `api_streaming_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 132 | `api_token_bucket_rate_limiter` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 133 | `bundle_apr_lambda_package` | bundling | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 134 | `bundle_apr_quantized_q4` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 135 | `bundle_apr_signed` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 136 | `bundle_apr_static_binary` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 137 | `bundle_cache_key_deriver` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 138 | `bundle_compression_ratio_calc` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 139 | `bundle_encrypted_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 140 | `bundle_integrity_checksum` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 141 | `bundle_manifest_header_validator` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 142 | `bundle_metadata_versioning` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 143 | `bundle_mmap_offset_calculator` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 144 | `bundle_quantized_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 145 | `bundle_signing_chain_builder` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 146 | `bundle_static_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 147 | `bundle_streaming_q4k_large_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 148 | `bundle_table_of_contents` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 149 | `bundle_tensor_dedup` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 150 | `encrypt_kdf_sweep` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 151 | `encrypt_signed` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 152 | `cgp_baseline_diff_classifier` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 153 | `cgp_kernel_metric_aggregator` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 154 | `cgp_proof_status_dispatcher` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 155 | `cgp_regression_detector_baseline_vs_current` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 156 | `cgp_roofline_classify_kernel` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 157 | `cgp_roofline_ridge_point_per_precision` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 158 | `chat_chatml` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 159 | `chat_injection_defense` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 160 | `chat_llama2` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 161 | `chat_mistral` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 162 | `chat_multi_format` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 163 | `chat_role_state_machine` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 164 | `chat_template_renderer` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 165 | `chat_token_budget_truncation` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 166 | `apr_bench` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 167 | `apr_info` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 168 | `cli_apr_bench` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 169 | `cli_apr_compile` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 170 | `cli_apr_convert` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 171 | `cli_apr_decrypt` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 172 | `cli_apr_diagnose` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 173 | `cli_apr_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 174 | `cli_apr_info` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 175 | `cli_apr_list` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 176 | `cli_apr_ptx_map` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 177 | `cli_apr_rm` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 178 | `cli_apr_runs` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 179 | `cli_apr_serve` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 180 | `cli_apr_tokenize` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 181 | `cli_apr_tui` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 182 | `cli_bench_batch_sweep_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 183 | `cli_bench_cv_stability_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 184 | `cli_bench_h12_throughput_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 185 | `cli_bench_percentiles_csv_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 186 | `cli_bench_unit_normalizer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 187 | `cli_bench_warmup_iterations_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 188 | `cli_canary_check_verdict` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 189 | `cli_canary_create_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 190 | `cli_canary_directory_layout` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 191 | `cli_cbtop_ci_threshold_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 192 | `cli_cbtop_headless_json_schema` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 193 | `cli_cbtop_speculative_decoding_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 194 | `cli_check_json_output_schema` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 195 | `cli_check_pipeline_integrity_smoke` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 196 | `cli_check_skip_contract_diagnostic` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 197 | `cli_compare_hf_offline_safety` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 198 | `cli_compare_hf_tensor_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 199 | `cli_compare_hf_threshold_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 200 | `cli_compile_optimization_flags` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 201 | `cli_compile_output_path_resolver` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 202 | `cli_compile_target_triple_validator` | cli | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 203 | `cli_data_balance_strategy_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 204 | `cli_data_decontaminate_ngram_overlap` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 205 | `cli_data_split_stratified_ratios` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 206 | `cli_debug_breakpoint_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 207 | `cli_debug_drama_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 208 | `cli_debug_layer_glob_matcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 209 | `cli_debug_limit_truncator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 210 | `cli_debug_string_extractor` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 211 | `cli_debug_tensor_diff_tolerance` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 212 | `cli_decrypt_aead_tag_verification` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 213 | `cli_decrypt_invocation_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 214 | `cli_decrypt_key_format_validation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 215 | `cli_decrypt_key_rotation_grace` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 216 | `cli_decrypt_output_collision_detector` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 217 | `cli_decrypt_verify_ordering` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 218 | `cli_diagnose_five_whys_chain` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 219 | `cli_diagnose_grad_nan_scanner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 220 | `cli_diagnose_jsonl_corpus_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 221 | `cli_diagnose_model_size_inference` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 222 | `cli_diagnose_param_count_sanity` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 223 | `cli_diagnose_weight_histogram_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 224 | `cli_diff_magnitude_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 225 | `cli_diff_shape_compatibility` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 226 | `cli_diff_structural_renderer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 227 | `cli_diff_values_aprt_stage` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 228 | `cli_distill_ensemble_weighter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 229 | `cli_distill_layer_pairer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 230 | `cli_distill_loss_combiner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 231 | `cli_distill_stage_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 232 | `cli_distill_strategy_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 233 | `cli_distill_temperature_alpha_band` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 234 | `cli_encrypt_aad_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 235 | `cli_encrypt_force_overwrite_policy` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 236 | `cli_encrypt_kdf_iterations_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 237 | `cli_encrypt_keystream_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 238 | `cli_encrypt_nonce_uniqueness_checker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 239 | `cli_encrypt_passphrase_strength` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 240 | `cli_eval_bleu_score_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 241 | `cli_eval_dataset_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 242 | `cli_eval_metric_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 243 | `cli_eval_pass_at_k_temperature_pairing` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 244 | `cli_eval_perplexity_threshold_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 245 | `cli_eval_top_k_accuracy_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 246 | `cli_experiment_hypothesis_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 247 | `cli_experiment_metric_compare_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 248 | `cli_experiment_param_diff_renderer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 249 | `cli_experiment_run_id_collision` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 250 | `cli_experiment_view_loss_curve_render` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 251 | `cli_experiment_view_run_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 252 | `cli_explain_ablation_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 253 | `cli_explain_error_code_lookup` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 254 | `cli_explain_ig_steps_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 255 | `cli_explain_kernel_dispatch_pipeline` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 256 | `cli_explain_proof_status_aggregator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 257 | `cli_explain_saliency_rank_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 258 | `cli_export_batch_csv_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 259 | `cli_export_format_allowlist` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 260 | `cli_export_opset_compat_matrix` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 261 | `cli_export_output_naming_pattern` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 262 | `cli_export_plan_mode_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 263 | `cli_export_target_dtype_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 264 | `cli_finetune_checkpoint_format_csv` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 265 | `cli_finetune_grad_accum_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 266 | `cli_finetune_lora_rank_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 267 | `cli_finetune_lr_scheduler_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 268 | `cli_finetune_merge_mode_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 269 | `cli_finetune_method_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 270 | `cli_flow_component_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 271 | `cli_flow_dot_renderer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 272 | `cli_flow_layer_aggregation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 273 | `cli_gpu_device_capability_filter` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 274 | `cli_gpu_fp8_capability_checker` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 275 | `cli_gpu_nccl_strategy_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 276 | `cli_gpu_oom_recovery_advisor` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 277 | `cli_gpu_peer_access_topology` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 278 | `cli_gpu_vram_reservation_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 279 | `cli_hex_offset_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 280 | `cli_hex_slice_range_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 281 | `cli_hex_view_mode_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 282 | `cli_import_dtype_coercion_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 283 | `cli_import_format_auto_detector` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 284 | `cli_import_no_config_inference_risk` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 285 | `cli_import_provenance_chain_enforcement` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 286 | `cli_import_sharding_plan_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 287 | `cli_import_strict_mode_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 288 | `cli_inspect_view_mode_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 289 | `cli_inspect_vocab_token_query` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 290 | `cli_inspect_weight_stats_aggregator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 291 | `cli_mcp_batch_request_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 292 | `cli_mcp_error_response_codes` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 293 | `cli_mcp_jsonrpc_request_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 294 | `cli_mcp_resource_uri_scheme` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 295 | `cli_mcp_tool_manifest_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 296 | `cli_mcp_transport_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 297 | `cli_merge_dare_drop_rate_band` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 298 | `cli_merge_signed_conflict_resolver` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 299 | `cli_merge_slerp_t_clamp` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 300 | `cli_merge_strategy_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 301 | `cli_merge_ties_density_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 302 | `cli_merge_weights_csv_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 303 | `cli_monitor_drift_threshold_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 304 | `cli_monitor_format_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 305 | `cli_monitor_log_rotation_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 306 | `cli_monitor_metrics_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 307 | `cli_monitor_quantile_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 308 | `cli_monitor_refresh_throttle` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 309 | `cli_ollama_chat_lint_eval_count_consistency` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 310 | `cli_ollama_chat_lint_message_content_validation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 311 | `cli_ollama_chat_lint_role_state_machine` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 312 | `cli_ollama_embed_dim_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 313 | `cli_ollama_model_name_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 314 | `cli_ollama_token_rate_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 315 | `cli_oracle_compliance_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 316 | `cli_oracle_family_introspection` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 317 | `cli_oracle_size_constraint_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 318 | `cli_parity_assert_mode_exit_codes` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 319 | `cli_parity_default_prompt_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 320 | `cli_parity_token_divergence_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 321 | `cli_pipeline_concurrency_limiter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 322 | `cli_pipeline_dag_topological_sort` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 323 | `cli_pipeline_retry_policy_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 324 | `cli_pipeline_stage_skip_predicate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 325 | `cli_pipeline_status_state_machine` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 326 | `cli_pipeline_validate_manifest_schema` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 327 | `cli_pretrain_curriculum_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 328 | `cli_pretrain_divergence_guard` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 329 | `cli_pretrain_epoch_budget_calc` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 330 | `cli_pretrain_grad_clip_threshold` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 331 | `cli_pretrain_mode_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 332 | `cli_pretrain_run_dir_layout` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 333 | `cli_probar_export_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 334 | `cli_probar_golden_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 335 | `cli_probar_layer_pattern_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 336 | `cli_profile_flame_depth_limit` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 337 | `cli_profile_format_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 338 | `cli_profile_hot_function_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 339 | `cli_profile_naive_detection_threshold` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 340 | `cli_profile_perf_grade_thresholds` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 341 | `cli_profile_sampling_rate_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 342 | `cli_prune_lottery_ticket_warmup` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 343 | `cli_prune_method_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 344 | `cli_prune_remove_layers_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 345 | `cli_prune_sparsity_ramp_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 346 | `cli_prune_target_ratio_band` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 347 | `cli_prune_wanda_activation_scorer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 348 | `cli_ptx_kernel_name_parser` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 349 | `cli_ptx_map_kernel_filter` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 350 | `cli_ptx_map_prefill_vs_decode` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 351 | `cli_ptx_map_reverse_lookup` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 352 | `cli_ptx_register_pressure_threshold` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 353 | `cli_ptx_strict_mode_whitelist` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 354 | `cli_publish_dry_run_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 355 | `cli_publish_manifest_full` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 356 | `cli_publish_parent_chain_termination` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 357 | `cli_publish_pipeline_tag_allowlist` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 358 | `cli_publish_repo_id_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 359 | `cli_pull_alias_resolver` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 360 | `cli_pull_dataset_glob_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 361 | `cli_pull_revision_pin_resolver` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 362 | `cli_qa_assertion_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 363 | `cli_qa_parallel_test_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 364 | `cli_qa_regression_delta_classifier` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 365 | `cli_qa_safetensors_parity_required` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 366 | `cli_qa_tier_aggregator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 367 | `cli_qa_warmup_iteration_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 368 | `cli_qualify_skip_list_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 369 | `cli_qualify_tier_progression` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 370 | `cli_qualify_timeout_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 371 | `cli_quantize_batch_csv_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 372 | `cli_quantize_calibration_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 373 | `cli_quantize_format_compatibility` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 374 | `cli_quantize_mixed_precision_selector` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 375 | `cli_quantize_scale_zero_point_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 376 | `cli_quantize_scheme_size_predictor` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 377 | `cli_registry_aliases_collision_detection` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 378 | `cli_registry_aliases_json_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 379 | `cli_registry_aliases_yaml_loader` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 380 | `cli_registry_lineage_cycle_detector` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 381 | `cli_registry_semver_tag_matcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 382 | `cli_registry_uri_parser` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 383 | `cli_rosetta_chain_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 384 | `cli_rosetta_compare_inference_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 385 | `cli_rosetta_compare_inference_logit_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 386 | `cli_rosetta_compare_inference_temperature_modes` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 387 | `cli_rosetta_convert_extension_inference` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 388 | `cli_rosetta_convert_external_tokenizer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 389 | `cli_rosetta_convert_quantize_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 390 | `cli_rosetta_diff_tensors_layout_check` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 391 | `cli_rosetta_diff_tensors_pad_token_signal` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 392 | `cli_rosetta_diff_tensors_value_sampler` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 393 | `cli_rosetta_fingerprint_diff_mode` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 394 | `cli_rosetta_fingerprint_filter_pattern` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 395 | `cli_rosetta_fingerprint_json_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 396 | `cli_rosetta_fingerprint_stats` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 397 | `cli_rosetta_inspect_format_detector` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 398 | `cli_rosetta_inspect_hexdump_window` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 399 | `cli_rosetta_inspect_tensor_table` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 400 | `cli_rosetta_round_trip_verify` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 401 | `cli_rosetta_validate_stats_per_tensor_report` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 402 | `cli_rosetta_validate_stats_reference_or_fingerprints` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 403 | `cli_rosetta_validate_stats_threshold_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 404 | `cli_runs_diff_two_runs` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 405 | `cli_runs_ls_sparkline_renderer` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 406 | `cli_runs_show_metric_summary` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 407 | `cli_serve_kv_cache_budget` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 408 | `cli_serve_max_tokens_cap` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 409 | `cli_serve_plan_capacity_estimator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 410 | `cli_serve_run_endpoint_router` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 411 | `cli_serve_run_port_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 412 | `cli_serve_streaming_chunk_size` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 413 | `cli_showcase_runs_floor_enforcement` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 414 | `cli_showcase_step_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 415 | `cli_showcase_tier_baseline_matrix` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 416 | `cli_stamp_preserves_tensor_bytes` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 417 | `cli_stamp_provenance_basic` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 418 | `cli_stamp_spdx_validation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 419 | `cli_tensors_filter_pattern` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 420 | `cli_tensors_limit_truncator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 421 | `cli_tensors_stats_aggregator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 422 | `cli_tokenize_corpus_shard_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 423 | `cli_tokenize_hf_import_validation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 424 | `cli_tokenize_plan_estimator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 425 | `cli_trace_diff_mode_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 426 | `cli_trace_save_tensor_layer0` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 427 | `cli_trace_save_tensor_layer_range` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 428 | `cli_trace_stage_csv_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 429 | `cli_train_checkpoint_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 430 | `cli_train_early_stop_patience` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 431 | `cli_train_halving_round_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 432 | `cli_train_lr_finder_validator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 433 | `cli_train_sweep_grid_generator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 434 | `cli_train_watch_restart_policy` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 435 | `cli_tui_color_theme_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 436 | `cli_tui_keybinding_matcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 437 | `cli_tui_pager_buffer_planner` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 438 | `cli_tui_panel_layout_calculator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 439 | `cli_tui_resize_event_throttle` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 440 | `cli_tui_search_filter_predicate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 441 | `cli_tune_budget_compat_matrix` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 442 | `cli_tune_scheduler_dispatcher` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 443 | `cli_tune_strategy_picker` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 444 | `cli_validate_manifest_falsify_envelope` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 445 | `cli_validate_manifest_offline_safety` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 446 | `cli_validate_manifest_safetensors_dtype` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 447 | `cli_validate_manifest_sha256_format` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 448 | `cli_validate_min_score_gate` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 449 | `cli_validate_quality_score_aggregator` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 450 | `cli_validate_strict_warning_promoter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 451 | `compile_ptx` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 452 | `compile_size_optimized` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 453 | `decrypt_batch` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 454 | `decrypt_key_rotation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 455 | `diagnose_hardware` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 456 | `diagnose_multi_model` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 457 | `diff_quantization` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 458 | `diff_topology` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 459 | `list_json_export` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 460 | `list_size_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 461 | `rm_dry_run` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 462 | `rm_retention_policy` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 463 | `runs_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 464 | `runs_filter_sort` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 465 | `tokenize_bpe_trace` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 466 | `tokenize_compare_vocabs` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 467 | `tui_health_dashboard` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 468 | `tui_log_tail` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 469 | `validate_batch` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 470 | `validate_fix_suggestions` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 471 | `code_custom_agent_definition` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 472 | `code_diff_hunk_parser` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 473 | `code_hook_session_start` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 474 | `code_indent_normalizer` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 475 | `code_lint_severity_aggregator` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 476 | `code_mcp_client_config` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 477 | `code_skill_discovery` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 478 | `code_slash_command_extension` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 479 | `code_subagent_spawn_payload` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 480 | `code_worktree_isolation_permission_mode` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 481 | `contracts_macros_attribute_basic` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 482 | `contracts_macros_env_key_convention` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 483 | `contracts_macros_multi_equation_dispatch` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 484 | `contracts_macros_no_op_degradation` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 485 | `contracts_macros_pre_post_envelope` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 486 | `contracts_macros_runtime_validator_bridge` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 487 | `conversion_gguf_legacy_quant_fallback` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 488 | `convert_apr_to_gguf` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 489 | `convert_dtype_loss_estimator` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 490 | `convert_dtype_widener` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 491 | `convert_endianness_swapper` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 492 | `convert_format_version_matrix` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 493 | `convert_gguf_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 494 | `convert_layout_transposer` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 495 | `convert_onnx_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 496 | `convert_phi_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 497 | `convert_quantization_rescaler` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 498 | `convert_safetensors_header` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 499 | `convert_safetensors_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 500 | `convert_sparse_csr_to_dense` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 501 | `convert_tensor_name_remapper` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 502 | `create_apr_decision_tree` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 503 | `create_apr_from_scratch` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 504 | `create_apr_kmeans_clustering` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 505 | `create_apr_linear_regression` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 506 | `create_apr_neural_network` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 507 | `create_apr_ngram_language_model` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 508 | `create_demo_model` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 509 | `create_embedding_tying_envelope` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 510 | `create_init_scheme_dispatcher` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 511 | `create_vocab_size_validator` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 512 | `basic_loading` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 513 | `cli_batch_commands` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 514 | `data_compression_codec` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 515 | `data_row_dedup_strategy` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 516 | `data_sample_quota_balancer` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 517 | `data_shuffle_seed_validator` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 518 | `data_streaming_buffer_size` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 519 | `data_train_val_test_split` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 520 | `dataloader_batching` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 521 | `doctest_extraction` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 522 | `drift_detection` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 523 | `federated_split` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 524 | `hub_publishing` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 525 | `prose_detection` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 526 | `quality_check` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 527 | `registry_publish` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 528 | `repl_commands` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 529 | `repl_completer` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 530 | `repl_display_config` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 531 | `repl_health_status` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 532 | `repl_session` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 533 | `streaming_large` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 534 | `transforms_pipeline` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 535 | `tui_viewer` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 536 | `alimentar_ingest` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 537 | `apr_inference_server` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 538 | `batuta_agent` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 539 | `deploy_blue_green_cutover` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 540 | `deploy_canary_traffic_ramp` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 541 | `deploy_pod_replica_scheduler` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 542 | `entrenar_train` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 543 | `jetson_edge_base` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 544 | `pacha_registry` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 545 | `pepita_sandbox` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 546 | `realizar_serve` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 547 | `renacer_observability` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 548 | `repartir_worker` | deployment-stacks | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 549 | `sovereign_ai_stack` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 550 | `trueno_db_analytics` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 551 | `trueno_rag_pipeline` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 552 | `whisper_apr_asr` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 553 | `distill_against_contract_v1` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 554 | `distill_attention_transfer` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 555 | `distill_attention_transfer_loss` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 556 | `distill_dataset_filter` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 557 | `distill_dataset_synth_ratio` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 558 | `distill_intermediate_feature_match` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 559 | `distill_knowledge_transfer` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 560 | `distill_layer_alignment_planner` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 561 | `distill_layer_matching` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 562 | `distill_quantile_calibration` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 563 | `distill_quantization_aware` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 564 | `distill_self_distill_bootstrap` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 565 | `distill_self_distillation` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 566 | `distill_temperature_picker` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 567 | `distill_temperature_search_envelope` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 568 | `distributed_allreduce_strategy` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 569 | `distributed_byzantine_quorum` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 570 | `distributed_eventual_consistency_window` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 571 | `distributed_failure_recovery` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 572 | `distributed_gossip_protocol` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 573 | `distributed_inference` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 574 | `distributed_lamport_clock` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 575 | `distributed_model_sharding` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 576 | `distributed_phi_failure_detector` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 577 | `distributed_pipeline_microbatch` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 578 | `distributed_pipeline_parallel` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 579 | `distributed_ring_allreduce` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 580 | `distributed_shard_partition_planner` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 581 | `distributed_zero_copy_rdma` | distributed | ![cuda](https://img.shields.io/badge/-cuda-76b900) ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 582 | `format_alignment_padding` | format | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 583 | `format_batch_export` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 584 | `format_bf16_vs_fp16_mantissa` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 585 | `format_convert_quantize` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 586 | `format_endianness_detector` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 587 | `format_export_gguf` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 588 | `format_export_safetensors` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 589 | `format_gguf_metadata_key` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 590 | `format_import_hf` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 591 | `format_magic_bytes_validator` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 592 | `format_manifest_tensor_count` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 593 | `format_migration_pipeline` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 594 | `format_publish` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 595 | `format_pull_cache` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 596 | `format_rosetta_chain` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 597 | `format_rosetta_convert` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 598 | `format_rosetta_verify` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 599 | `format_stride_encoder` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 600 | `format_tensor_name_canonicalizer` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 601 | `format_version_compat_matrix` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 602 | `import_hf_cache` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 603 | `import_multi_format` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 604 | `publish_dry_run` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 605 | `publish_multi_registry` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 606 | `pull_resume` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 607 | `pull_verify_decompress` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 608 | `validate_manifest_happy` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 609 | `validate_manifest_live_check` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 610 | `validate_manifest_sha_mismatch` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 611 | `flash_attention_inference` | gpu | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 612 | `gpu_async_memcpy_overlap` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 613 | `gpu_capability_detect` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 614 | `gpu_collective_op_picker` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 615 | `gpu_cuda_inference` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 616 | `gpu_kv_cache_paging` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 617 | `gpu_l1_shared_mem_partition` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 618 | `gpu_memory_bandwidth_roofline` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 619 | `gpu_memory_management` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 620 | `gpu_memory_planner` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 621 | `gpu_memory_pool` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 622 | `gpu_multi_gpu_inference` | gpu | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 623 | `gpu_occupancy_calculator` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 624 | `gpu_pcie_bandwidth_tier` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 625 | `gpu_ptx_analysis` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 626 | `gpu_register_spill` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 627 | `gpu_tensor_core_optimization` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 628 | `gpu_tensor_parallel_split` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 629 | `gpu_vulkan_inference` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 630 | `gpu_warp_divergence_classifier` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 631 | `gpu_warp_scheduler_dispatch` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 632 | `gpu_warp_specialization` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 633 | `ptx_disassembly` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 634 | `ptx_map_hot_regions` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 635 | `ptx_map_sass_to_ptx` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 636 | `ptx_register_usage` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 637 | `adaptive_batch_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 638 | `chat_kv_cache` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 639 | `chat_multiturn` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 640 | `chat_tool_use` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 641 | `dynamic_batch_with_sla` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 642 | `ensemble_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 643 | `inference_apr_run` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 644 | `inference_beam_search_width` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 645 | `inference_kv_cache_lru` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 646 | `inference_mmap_lazy_load` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 647 | `inference_qwen3_moe_numerical_parity_smoke` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 648 | `inference_run_temperature_sweep` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 649 | `inference_top_p_nucleus` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 650 | `model_pipeline` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 651 | `pipeline_3stage` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 652 | `pipeline_resilient` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 653 | `quantized_inference_comparison` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 654 | `simple_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 655 | `speculative_decode` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 656 | `streaming_token_generator` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 657 | `awq_lint_batch` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 658 | `awq_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 659 | `awq_lint_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 660 | `dry_sampling_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 661 | `dry_sampling_lint_pipeline` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 662 | `dry_sampling_lint_repetition` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 663 | `embeddings_lint_dim_consistency` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 664 | `embeddings_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 665 | `embeddings_lint_l2_norm_check` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 666 | `fp8_lint_capability_gate` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 667 | `fp8_lint_happy` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 668 | `fp8_lint_saturation_violation` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 669 | `gbnf_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 670 | `gbnf_lint_malformed` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 671 | `gbnf_lint_pipeline` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 672 | `gptq_lint_cosine_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 673 | `gptq_lint_flag_combinations` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 674 | `gptq_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 675 | `grad_norm_divergence_run` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 676 | `grad_norm_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 677 | `grad_norm_spike_detection` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 678 | `imatrix_lint_corpus_entropy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 679 | `imatrix_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 680 | `imatrix_lint_nan_detection` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 681 | `lint_cors_origin_allowlist` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 682 | `lint_prompt_pii_redactor` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 683 | `lint_schema_drift_detector` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 684 | `nf4_lint_codebook_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 685 | `nf4_lint_double_quant_parity` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 686 | `nf4_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 687 | `ollama_chat_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 688 | `ollama_chat_lint_schema_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 689 | `ollama_chat_lint_stream` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 690 | `ollama_tools_lint_allowlist_gate` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 691 | `ollama_tools_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 692 | `ollama_tools_lint_streaming_ndjson` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 693 | `oom_lint_allocation_trace` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 694 | `oom_lint_happy` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 695 | `oom_lint_missing_breadcrumb` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 696 | `registry_quota_lint_atomic_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 697 | `registry_quota_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 698 | `registry_quota_lint_tenant_overage` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 699 | `rm_gc_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 700 | `rm_gc_lint_orphan_detection` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 701 | `rm_gc_lint_refcount_conservation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 702 | `shared_cache_lint_dedup_audit` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 703 | `shared_cache_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 704 | `shared_cache_lint_permission_matrix` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 705 | `tool_use_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 706 | `tool_use_lint_invalid_args` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 707 | `tool_use_lint_streaming` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 708 | `typical_p_lint_entropy_truncation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 709 | `typical_p_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 710 | `typical_p_lint_min_keep_floor` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 711 | `unified_search_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 712 | `unified_search_lint_offline_consistency` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 713 | `unified_search_lint_rrf_recompute` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 714 | `mcp_byte_parity_dispatcher_swap` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 715 | `mcp_capability_handshake_diff` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 716 | `mcp_client_simulation` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 717 | `mcp_embedded_initialize_handshake` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 718 | `mcp_embedded_protocol_invariants` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 719 | `mcp_embedded_tools_list_discovery` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 720 | `mcp_error_code_classifier` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 721 | `mcp_notification_progress_token` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 722 | `mcp_request_id_correlator` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 723 | `mcp_resource_uri_resolver` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 724 | `mcp_session_lifecycle` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 725 | `mcp_sse_event_envelope` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 726 | `mcp_stdio_server` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 727 | `mcp_tool_discovery` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 728 | `mcp_tool_signature_validator` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 729 | `mcp_websocket_frame_envelope` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 730 | `cbtop_headless` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 731 | `cbtop_histogram` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 732 | `cbtop_streaming` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 733 | `hash_chain_audit` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 734 | `inference_cost_tracking` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 735 | `inference_explainability` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 736 | `latency_histogram` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 737 | `model_drift_detection` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 738 | `monitor_aggregation_window_dispatcher` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 739 | `monitor_alert_dedup_window` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 740 | `monitor_alerting` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 741 | `monitor_anomaly_z_score_classifier` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 742 | `monitor_capacity_planner` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 743 | `monitor_circuit_log_emit` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 744 | `monitor_drift_psi` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 745 | `monitor_dropped_request_classifier` | monitoring | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 746 | `monitor_inference_cost` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 747 | `monitor_log_sampling` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 748 | `monitor_percentile_summary` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 749 | `monitor_realtime` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 750 | `monitor_sla_uptime_calc` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 751 | `monitor_slo_burn_rate` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 752 | `monitoring_energy_estimation` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 753 | `monitoring_memory_profiler` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 754 | `mc_business_revenue_forecast` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 755 | `mc_correlated_portfolio_var` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 756 | `mc_mm1_queue_little_law` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 757 | `mc_safety_stock_planner` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 758 | `mc_stock_price_simulation_gbm` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 759 | `mc_value_at_risk_historical_vs_parametric` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 760 | `distill_checkpoint` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 761 | `distill_ensemble` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 762 | `distill_progressive` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 763 | `distill_standard_kl` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 764 | `finetune_lora` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 765 | `finetune_merge_adapter` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 766 | `finetune_plan_vram` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 767 | `finetune_qlora` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 768 | `merge_average` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 769 | `merge_dare` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 770 | `merge_hierarchical` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 771 | `merge_slerp` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 772 | `merge_ties` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 773 | `merge_weighted` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 774 | `optimize_adamw_vs_lion_picker` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 775 | `optimize_full_pipeline` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 776 | `optimize_micro_batch_grad_accum` | optimize | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 777 | `optimize_tune` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 778 | `optimize_warmup_cosine_lr` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 779 | `prune_depth` | optimize | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 780 | `prune_gradual_schedule` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 781 | `prune_magnitude` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 782 | `prune_structured` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 783 | `prune_wanda` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 784 | `quantize_4bit` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 785 | `quantize_fake_qat` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 786 | `quantize_gptq` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 787 | `quantize_mixed_precision` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 788 | `tune_bayesian` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 789 | `tune_grid_early_stop` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 790 | `registry_alias_resolver_chain` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 791 | `registry_aliases_diff` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 792 | `registry_aliases_list` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 793 | `registry_aliases_resolve` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 794 | `registry_gc_orphans` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 795 | `registry_gc_policy` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 796 | `registry_hash_pin_enforcer` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 797 | `registry_immutable_tag_policy` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 798 | `registry_manifest_schema` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 799 | `registry_model_comparison` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 800 | `registry_model_lineage` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 801 | `registry_model_rollback` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 802 | `registry_model_versioning` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 803 | `registry_pull_throttle` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 804 | `registry_quota_per_user` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 805 | `registry_register_apr` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 806 | `registry_signature_chain` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 807 | `http_model_server` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 808 | `model_ab_testing` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 809 | `model_canary_deploy` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 810 | `model_rate_limiter` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 811 | `model_selection_router` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 812 | `serve_anthropic_messages_api_drop_in` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 813 | `serve_grpc_stream` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 814 | `serve_plan_hf_dryrun_no_weights` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 815 | `serve_rate_limited` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 816 | `serverless_cold_start_classifier` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 817 | `serverless_cold_start_optimization` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 818 | `serverless_concurrency_validator` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 819 | `serverless_container_image` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 820 | `serverless_edge_function` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 821 | `serverless_lambda_inference` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 822 | `serverless_lambda_pricing_matrix` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 823 | `serverless_model_warmup` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 824 | `serverless_provisioned_concurrency` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 825 | `serverless_timeout_budget_picker` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 826 | `serverless_vpc_cold_start` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 827 | `shell_brace_expander` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 828 | `shell_corpus_from_string` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 829 | `shell_history_parse_zsh` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 830 | `shell_pipe_redirection_parser` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 831 | `shell_quote_state_classifier` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 832 | `shell_trie_prefix_completion` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 833 | `simd_alignment_strategy` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 834 | `simd_alignment_validator` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 835 | `simd_aos_to_soa` | simd | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 836 | `simd_auto_vectorization` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 837 | `simd_avx_vnni_int8_inference` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 838 | `simd_fma_fusion_gate` | simd | ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 839 | `simd_horizontal_reduce_dispatcher` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 840 | `simd_loop_carried_dep_detector` | simd | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 841 | `simd_mask_lane_predicate` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 842 | `simd_matrix_ops` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 843 | `simd_prefetch_distance_picker` | simd | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 844 | `simd_quantized_operations` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 845 | `simd_unroll_factor_picker` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 846 | `simd_vectorized_inference` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 847 | `trueno_simd_ops` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 848 | `speech_audio_format_validator` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 849 | `speech_audio_resampler` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 850 | `speech_chunk_overlap_planner` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 851 | `speech_diarization` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 852 | `speech_language_id_confidence` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 853 | `speech_multilingual` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 854 | `speech_vad` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 855 | `speech_vad_threshold_classifier` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 856 | `speech_word_timestamp_align` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 857 | `whisper_streaming` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 858 | `whisper_transcribe` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 859 | `autograd_backprop_viz` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 860 | `autograd_custom_ops` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 861 | `autograd_gradient_clipping` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 862 | `checkpoint_resume` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 863 | `continuous_train_curriculum` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 864 | `continuous_train_federated_simulation` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 865 | `continuous_train_incremental` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 866 | `continuous_train_online_learning` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 867 | `data_preprocessing` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 868 | `data_sharded_shuffle` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 869 | `data_streaming_tokens` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 870 | `entrenar_autograd_training` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 871 | `entrenar_eval_metrics` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 872 | `few_shot_finetune` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 873 | `gradient_accumulation` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 874 | `hyperparameter_sweep` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 875 | `learning_rate_schedule` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 876 | `mixed_precision_training` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 877 | `pretrain_checkpoint_resume` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 878 | `pretrain_nan_guard` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 879 | `pretrain_synthetic_decreasing` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 880 | `train_distributed_sim` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 881 | `train_grad_accum` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 882 | `training_curriculum_filter` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 883 | `training_eval_cadence` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 884 | `training_grad_clip_norm` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 885 | `training_loss_spike_detector` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 886 | `training_lr_combo_scheduler` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 887 | `training_lr_warmup_decay` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 888 | `tsp_christofides_ratio` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 889 | `tsp_compare_tabu_vs_genetic` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 890 | `tsp_distance_matrix_explicit` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 891 | `tsp_nearest_neighbor` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 892 | `tsp_solve_with_tabu` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 893 | `tsp_two_opt_swap_improver` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 894 | `load_visualization` | visualization | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 895 | `viz_axis_scale_classifier` | visualization | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 896 | `viz_color_palette_picker` | visualization | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 897 | `viz_legend_placement_optimizer` | visualization | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 898 | `wasm_browser_compat_matrix` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 899 | `wasm_browser_inference` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 900 | `wasm_capability_check` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 901 | `wasm_export_section_size` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 902 | `wasm_export_table_validator` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 903 | `wasm_imported_function_count` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 904 | `wasm_memory_growth_policy` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 905 | `wasm_memory_growth_strategy` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 906 | `wasm_model_loader` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 907 | `wasm_module_cache_strategy` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 908 | `wasm_module_size_budget` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 909 | `wasm_progressive_loading` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 910 | `wasm_simd128_dispatch` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 911 | `wasm_streaming_compilation` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 912 | `wasm_threads_atomics_gate` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 913 | `wasm_wasi_capability_grant` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 914 | `wasm_web_worker` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 915 | `wasm_webgpu_acceleration` | wasm | ![wgpu](https://img.shields.io/badge/-wgpu-green) ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |

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
