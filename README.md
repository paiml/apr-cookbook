<p align="center">
  <img src=".github/apr-cookbook-hero.svg" width="800" alt="apr-cookbook">
</p>

<h1 align="center">apr-cookbook</h1>

<p align="center">
  <strong>Production Recipes for ML Model Deployment in Pure Rust</strong>
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

**341 recipes** | Build: [![CI](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml)

<details>
<summary>Full recipe table (click to expand)</summary>

| # | Example | Category | Devices | Build |
|--:|---------|----------|---------|:-----:|
| 1 | `acceleration_autotuner` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 2 | `acceleration_cache_tiling` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 3 | `acceleration_compression_benchmark` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 4 | `acceleration_kernel_fusion` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 5 | `acceleration_mmap_inference` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 6 | `acceleration_quantized_matmul` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 7 | `simd_matrix_operations` | acceleration | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 8 | `ab_experiment` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 9 | `cicd_model_pipeline` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 10 | `clip_search` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 11 | `code_defect_oracle` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 12 | `compliance_audit` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 13 | `debug_fix_loop` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 14 | `edge_anomaly_detection` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 15 | `embedding_visualization` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 16 | `handwriting_recognition` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 17 | `hierarchical_cache_benchmark` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 18 | `image_classification` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 19 | `model_inspection_scoring` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 20 | `model_showcase` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 21 | `online_training_defect` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 22 | `quantization_quality_tradeoff` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 23 | `rag_pipeline` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 24 | `showcase_gallery` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 25 | `showcase_markdown` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 26 | `spanish_tutor` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 27 | `streaming_sentiment` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 28 | `style_transfer` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 29 | `voice_recognition` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 30 | `wasm_summarizer` | advanced | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 31 | `analysis_bench` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 32 | `analysis_canary` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 33 | `analysis_check` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 34 | `analysis_compare_hf` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 35 | `analysis_compare_hf_threshold` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 36 | `analysis_debug` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 37 | `analysis_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 38 | `analysis_eval` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 39 | `analysis_explain` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 40 | `analysis_flow` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 41 | `analysis_hex` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 42 | `analysis_inspect` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 43 | `analysis_lint` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 44 | `analysis_model_fingerprint` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 45 | `analysis_oracle` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 46 | `analysis_parity` | analysis | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 47 | `analysis_probar` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 48 | `analysis_profile` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 49 | `analysis_qa_capability` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 50 | `analysis_qa_gates` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 51 | `analysis_qa_report` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 52 | `analysis_qualify` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 53 | `analysis_slice` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 54 | `analysis_tensors` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 55 | `analysis_tensors_stats` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 56 | `analysis_trace` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 57 | `analysis_tree` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 58 | `analysis_validate` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 59 | `bench_batch_sweep` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 60 | `bench_quantization_compare` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 61 | `canary_regression` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 62 | `canary_rolling_window` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 63 | `check_batch` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 64 | `check_json_report` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 65 | `debug_activation_dist` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 66 | `debug_nan_trace` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 67 | `eval_benchmark_suite` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 68 | `eval_pass_at_k` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 69 | `experiment_ab_test` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 70 | `experiment_multi_seed` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 71 | `explain_error_codes` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 72 | `explain_shape_mismatch` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 73 | `flow_arch_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 74 | `flow_depth_sweep` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 75 | `hex_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 76 | `hex_pattern_search` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 77 | `inspect_layer_params` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 78 | `inspect_quantization_stats` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 79 | `lint_naming_rules` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 80 | `lint_suppression` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 81 | `oracle_classifier` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 82 | `oracle_ensemble_vote` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 83 | `parity_format` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 84 | `parity_quantization` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 85 | `probar_regression_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 86 | `probar_suite_runner` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 87 | `profile_memory_layers` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 88 | `profile_roofline` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 89 | `qualify_remediation` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 90 | `qualify_scorecard` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 91 | `trace_per_op_latency` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 92 | `trace_run_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 93 | `tree_arch_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 94 | `tree_param_rollup` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 95 | `api_auth_middleware` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 96 | `api_batch_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 97 | `api_call_model_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 98 | `api_model_health_check` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 99 | `api_streaming_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 100 | `bundle_apr_lambda_package` | bundling | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 101 | `bundle_apr_quantized_q4` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 102 | `bundle_apr_signed` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 103 | `bundle_apr_static_binary` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 104 | `bundle_encrypted_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 105 | `bundle_quantized_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 106 | `bundle_static_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 107 | `encrypt_kdf_sweep` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 108 | `encrypt_signed` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 109 | `chat_chatml` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 110 | `chat_injection_defense` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 111 | `chat_llama2` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 112 | `chat_mistral` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 113 | `chat_multi_format` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 114 | `apr_bench` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 115 | `apr_info` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 116 | `cli_apr_bench` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 117 | `cli_apr_compile` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 118 | `cli_apr_convert` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 119 | `cli_apr_decrypt` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 120 | `cli_apr_diagnose` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 121 | `cli_apr_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 122 | `cli_apr_info` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 123 | `cli_apr_list` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 124 | `cli_apr_ptx_map` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 125 | `cli_apr_rm` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 126 | `cli_apr_runs` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 127 | `cli_apr_serve` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 128 | `cli_apr_tokenize` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 129 | `cli_apr_tui` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 130 | `compile_ptx` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 131 | `compile_size_optimized` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 132 | `decrypt_batch` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 133 | `decrypt_key_rotation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 134 | `diagnose_hardware` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 135 | `diagnose_multi_model` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 136 | `diff_quantization` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 137 | `diff_topology` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 138 | `list_json_export` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 139 | `list_size_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 140 | `rm_dry_run` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 141 | `rm_retention_policy` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 142 | `runs_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 143 | `runs_filter_sort` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 144 | `tokenize_bpe_trace` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 145 | `tokenize_compare_vocabs` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 146 | `tui_health_dashboard` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 147 | `tui_log_tail` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 148 | `validate_batch` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 149 | `validate_fix_suggestions` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 150 | `convert_apr_to_gguf` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 151 | `convert_gguf_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 152 | `convert_onnx_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 153 | `convert_phi_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 154 | `convert_safetensors_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 155 | `create_apr_decision_tree` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 156 | `create_apr_from_scratch` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 157 | `create_apr_kmeans_clustering` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 158 | `create_apr_linear_regression` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 159 | `create_apr_neural_network` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 160 | `create_apr_ngram_language_model` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 161 | `create_demo_model` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 162 | `distill_attention_transfer` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 163 | `distill_knowledge_transfer` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 164 | `distill_layer_matching` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 165 | `distill_quantization_aware` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 166 | `distill_self_distillation` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 167 | `distributed_gossip_protocol` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 168 | `distributed_inference` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 169 | `distributed_model_sharding` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 170 | `distributed_pipeline_parallel` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 171 | `distributed_ring_allreduce` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 172 | `format_batch_export` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 173 | `format_convert_quantize` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 174 | `format_export_gguf` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 175 | `format_export_safetensors` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 176 | `format_import_hf` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 177 | `format_migration_pipeline` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 178 | `format_publish` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 179 | `format_pull_cache` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 180 | `format_rosetta_chain` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 181 | `format_rosetta_convert` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 182 | `format_rosetta_verify` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 183 | `import_hf_cache` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 184 | `import_multi_format` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 185 | `publish_dry_run` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 186 | `publish_multi_registry` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 187 | `pull_resume` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 188 | `pull_verify_decompress` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 189 | `validate_manifest_happy` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 190 | `validate_manifest_live_check` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 191 | `validate_manifest_sha_mismatch` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 192 | `flash_attention_inference` | gpu | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 193 | `gpu_capability_detect` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 194 | `gpu_cuda_inference` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 195 | `gpu_memory_management` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 196 | `gpu_memory_planner` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 197 | `gpu_memory_pool` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 198 | `gpu_multi_gpu_inference` | gpu | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 199 | `gpu_ptx_analysis` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 200 | `gpu_tensor_core_optimization` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 201 | `gpu_vulkan_inference` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 202 | `ptx_disassembly` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 203 | `ptx_map_hot_regions` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 204 | `ptx_map_sass_to_ptx` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 205 | `ptx_register_usage` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 206 | `adaptive_batch_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 207 | `chat_kv_cache` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 208 | `chat_multiturn` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 209 | `chat_tool_use` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 210 | `dynamic_batch_with_sla` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 211 | `ensemble_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 212 | `inference_apr_run` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 213 | `inference_mmap_lazy_load` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 214 | `inference_run_temperature_sweep` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 215 | `model_pipeline` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 216 | `pipeline_3stage` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 217 | `pipeline_resilient` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 218 | `quantized_inference_comparison` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 219 | `simple_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 220 | `speculative_decode` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 221 | `streaming_token_generator` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 222 | `awq_lint_batch` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 223 | `awq_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 224 | `awq_lint_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 225 | `dry_sampling_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 226 | `dry_sampling_lint_pipeline` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 227 | `dry_sampling_lint_repetition` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 228 | `gbnf_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 229 | `gbnf_lint_malformed` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 230 | `gbnf_lint_pipeline` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 231 | `ollama_chat_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 232 | `ollama_chat_lint_schema_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 233 | `ollama_chat_lint_stream` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 234 | `oom_lint_allocation_trace` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 235 | `oom_lint_happy` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 236 | `oom_lint_missing_breadcrumb` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 237 | `tool_use_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 238 | `tool_use_lint_invalid_args` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 239 | `tool_use_lint_streaming` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 240 | `mcp_client_simulation` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 241 | `mcp_stdio_server` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 242 | `mcp_tool_discovery` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 243 | `cbtop_headless` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 244 | `cbtop_histogram` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 245 | `cbtop_streaming` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 246 | `hash_chain_audit` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 247 | `inference_cost_tracking` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 248 | `inference_explainability` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 249 | `latency_histogram` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 250 | `model_drift_detection` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 251 | `monitor_alerting` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 252 | `monitor_realtime` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 253 | `monitoring_energy_estimation` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 254 | `monitoring_memory_profiler` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 255 | `distill_checkpoint` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 256 | `distill_ensemble` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 257 | `distill_progressive` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 258 | `distill_standard_kl` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 259 | `finetune_lora` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 260 | `finetune_merge_adapter` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 261 | `finetune_plan_vram` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 262 | `finetune_qlora` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 263 | `merge_average` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 264 | `merge_dare` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 265 | `merge_hierarchical` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 266 | `merge_slerp` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 267 | `merge_ties` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 268 | `merge_weighted` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 269 | `optimize_full_pipeline` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 270 | `optimize_tune` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 271 | `prune_depth` | optimize | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 272 | `prune_gradual_schedule` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 273 | `prune_magnitude` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 274 | `prune_structured` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 275 | `prune_wanda` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 276 | `quantize_4bit` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 277 | `quantize_fake_qat` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 278 | `quantize_gptq` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 279 | `quantize_mixed_precision` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 280 | `tune_bayesian` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 281 | `tune_grid_early_stop` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 282 | `registry_aliases_diff` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 283 | `registry_aliases_list` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 284 | `registry_aliases_resolve` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 285 | `registry_model_comparison` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 286 | `registry_model_lineage` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 287 | `registry_model_rollback` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 288 | `registry_model_versioning` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 289 | `registry_register_apr` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 290 | `http_model_server` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 291 | `model_ab_testing` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 292 | `model_canary_deploy` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 293 | `model_rate_limiter` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 294 | `model_selection_router` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 295 | `serve_grpc_stream` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 296 | `serve_rate_limited` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 297 | `serverless_cold_start_optimization` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 298 | `serverless_container_image` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 299 | `serverless_edge_function` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 300 | `serverless_lambda_inference` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 301 | `serverless_model_warmup` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 302 | `simd_auto_vectorization` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 303 | `simd_avx_vnni_int8_inference` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 304 | `simd_matrix_ops` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 305 | `simd_quantized_operations` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 306 | `simd_vectorized_inference` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 307 | `trueno_simd_ops` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 308 | `speech_diarization` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 309 | `speech_multilingual` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 310 | `speech_vad` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 311 | `whisper_streaming` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 312 | `whisper_transcribe` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 313 | `autograd_backprop_viz` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 314 | `autograd_custom_ops` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 315 | `autograd_gradient_clipping` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 316 | `checkpoint_resume` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 317 | `continuous_train_curriculum` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 318 | `continuous_train_federated_simulation` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 319 | `continuous_train_incremental` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 320 | `continuous_train_online_learning` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 321 | `data_preprocessing` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 322 | `data_sharded_shuffle` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 323 | `data_streaming_tokens` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 324 | `entrenar_autograd_training` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 325 | `entrenar_eval_metrics` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 326 | `few_shot_finetune` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 327 | `gradient_accumulation` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 328 | `hyperparameter_sweep` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 329 | `learning_rate_schedule` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 330 | `mixed_precision_training` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 331 | `pretrain_checkpoint_resume` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 332 | `pretrain_nan_guard` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 333 | `pretrain_synthetic_decreasing` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 334 | `train_distributed_sim` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 335 | `train_grad_accum` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 336 | `wasm_browser_inference` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 337 | `wasm_model_loader` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 338 | `wasm_progressive_loading` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 339 | `wasm_streaming_compilation` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 340 | `wasm_web_worker` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 341 | `wasm_webgpu_acceleration` | wasm | ![wgpu](https://img.shields.io/badge/-wgpu-green) ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |

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
