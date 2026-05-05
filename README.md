<p align="center">
  <img src=".github/apr-cookbook-hero.svg" width="800" alt="apr-cookbook">
</p>

<h1 align="center">apr-cookbook</h1>

<p align="center">
  <strong>The umbrella cookbook for the PAIML sovereign AI stack — model bundling, data loading, deployment-as-recipe, and visualization, all in pure Rust.</strong>
</p>

<p align="center">
  <em>v6.0.0 (2026-05-05) consolidates sovereign-ai-cookbook, alimentar examples, and presentar examples into this repository per <a href="docs/specifications/centralize-cookbooks.md">centralize-cookbooks</a>. Source repositories archived 2026-05-05 (PMAT-070): <s><a href="https://github.com/paiml/sovereign-ai-cookbook">sovereign-ai-cookbook</a></s>, <s><a href="https://github.com/paiml/alimentar">alimentar</a></s>, <s><a href="https://github.com/paiml/presentar">presentar</a></s>. The alimentar and presentar Rust crates remain published on crates.io as <code>aprender-data</code> and <code>presentar</code> respectively.</em>
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

**411 recipes** | Build: [![CI](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml)

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
| 36 | `analysis_cpu_vs_gpu_parity_gate` | analysis | ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 37 | `analysis_debug` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 38 | `analysis_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 39 | `analysis_eval` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 40 | `analysis_explain` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 41 | `analysis_flow` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 42 | `analysis_hex` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 43 | `analysis_inspect` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 44 | `analysis_lint` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 45 | `analysis_model_fingerprint` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 46 | `analysis_oracle` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 47 | `analysis_parity` | analysis | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 48 | `analysis_probar` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 49 | `analysis_profile` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 50 | `analysis_qa_capability` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 51 | `analysis_qa_gates` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 52 | `analysis_qa_report` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 53 | `analysis_qualify` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 54 | `analysis_slice` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 55 | `analysis_tensors` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 56 | `analysis_tensors_stats` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 57 | `analysis_trace` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 58 | `analysis_tree` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 59 | `analysis_validate` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 60 | `bench_batch_sweep` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 61 | `bench_quantization_compare` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 62 | `canary_regression` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 63 | `canary_rolling_window` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 64 | `check_batch` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 65 | `check_json_report` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 66 | `debug_activation_dist` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 67 | `debug_nan_trace` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 68 | `eval_benchmark_suite` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 69 | `eval_pass_at_k` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 70 | `experiment_ab_test` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 71 | `experiment_multi_seed` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 72 | `explain_error_codes` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 73 | `explain_shape_mismatch` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 74 | `flow_arch_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 75 | `flow_depth_sweep` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 76 | `hex_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 77 | `hex_pattern_search` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 78 | `inspect_layer_params` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 79 | `inspect_quantization_stats` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 80 | `lint_naming_rules` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 81 | `lint_suppression` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 82 | `oracle_classifier` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 83 | `oracle_ensemble_vote` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 84 | `parity_format` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 85 | `parity_quantization` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 86 | `probar_regression_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 87 | `probar_suite_runner` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 88 | `profile_memory_layers` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 89 | `profile_roofline` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 90 | `qualify_remediation` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 91 | `qualify_scorecard` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 92 | `trace_per_op_latency` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 93 | `trace_run_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 94 | `tree_arch_diff` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 95 | `tree_param_rollup` | analysis | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 96 | `api_auth_middleware` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 97 | `api_batch_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 98 | `api_call_model_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 99 | `api_model_health_check` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 100 | `api_streaming_inference` | api | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 101 | `bundle_apr_lambda_package` | bundling | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 102 | `bundle_apr_quantized_q4` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 103 | `bundle_apr_signed` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 104 | `bundle_apr_static_binary` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 105 | `bundle_encrypted_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 106 | `bundle_quantized_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 107 | `bundle_static_model` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 108 | `encrypt_kdf_sweep` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 109 | `encrypt_signed` | bundling | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 110 | `cgp_regression_detector_baseline_vs_current` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 111 | `cgp_roofline_classify_kernel` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 112 | `cgp_roofline_ridge_point_per_precision` | cgp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 113 | `chat_chatml` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 114 | `chat_injection_defense` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 115 | `chat_llama2` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 116 | `chat_mistral` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 117 | `chat_multi_format` | chat | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 118 | `apr_bench` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 119 | `apr_info` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 120 | `cli_apr_bench` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 121 | `cli_apr_compile` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 122 | `cli_apr_convert` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 123 | `cli_apr_decrypt` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 124 | `cli_apr_diagnose` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 125 | `cli_apr_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 126 | `cli_apr_info` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 127 | `cli_apr_list` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 128 | `cli_apr_ptx_map` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 129 | `cli_apr_rm` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 130 | `cli_apr_runs` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 131 | `cli_apr_serve` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 132 | `cli_apr_tokenize` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 133 | `cli_apr_tui` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 134 | `cli_diff_values_aprt_stage` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 135 | `cli_publish_manifest_full` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 136 | `cli_publish_parent_chain_termination` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 137 | `cli_trace_save_tensor_layer0` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 138 | `cli_validate_manifest_safetensors_dtype` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 139 | `compile_ptx` | cli | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 140 | `compile_size_optimized` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 141 | `decrypt_batch` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 142 | `decrypt_key_rotation` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 143 | `diagnose_hardware` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 144 | `diagnose_multi_model` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 145 | `diff_quantization` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 146 | `diff_topology` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 147 | `list_json_export` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 148 | `list_size_filter` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 149 | `rm_dry_run` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 150 | `rm_retention_policy` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 151 | `runs_diff` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 152 | `runs_filter_sort` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 153 | `tokenize_bpe_trace` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 154 | `tokenize_compare_vocabs` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 155 | `tui_health_dashboard` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 156 | `tui_log_tail` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 157 | `validate_batch` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 158 | `validate_fix_suggestions` | cli | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 159 | `code_custom_agent_definition` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 160 | `code_hook_session_start` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 161 | `code_mcp_client_config` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 162 | `code_skill_discovery` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 163 | `code_slash_command_extension` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 164 | `code_subagent_spawn_payload` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 165 | `code_worktree_isolation_permission_mode` | code | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 166 | `contracts_macros_attribute_basic` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 167 | `contracts_macros_env_key_convention` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 168 | `contracts_macros_runtime_validator_bridge` | contracts-macros | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 169 | `convert_apr_to_gguf` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 170 | `convert_gguf_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 171 | `convert_onnx_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 172 | `convert_phi_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 173 | `convert_safetensors_to_apr` | conversion | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 174 | `create_apr_decision_tree` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 175 | `create_apr_from_scratch` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 176 | `create_apr_kmeans_clustering` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 177 | `create_apr_linear_regression` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 178 | `create_apr_neural_network` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 179 | `create_apr_ngram_language_model` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 180 | `create_demo_model` | creation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 181 | `basic_loading` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 182 | `cli_batch_commands` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 183 | `dataloader_batching` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 184 | `doctest_extraction` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 185 | `drift_detection` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 186 | `federated_split` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 187 | `hub_publishing` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 188 | `prose_detection` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 189 | `quality_check` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 190 | `registry_publish` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 191 | `repl_commands` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 192 | `repl_completer` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 193 | `repl_display_config` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 194 | `repl_health_status` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 195 | `repl_session` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 196 | `streaming_large` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 197 | `transforms_pipeline` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 198 | `tui_viewer` | data-loading | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 199 | `alimentar_ingest` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 200 | `apr_inference_server` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 201 | `batuta_agent` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 202 | `entrenar_train` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 203 | `jetson_edge_base` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 204 | `pacha_registry` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 205 | `pepita_sandbox` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 206 | `realizar_serve` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 207 | `renacer_observability` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 208 | `repartir_worker` | deployment-stacks | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 209 | `sovereign_ai_stack` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 210 | `trueno_db_analytics` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 211 | `trueno_rag_pipeline` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 212 | `whisper_apr_asr` | deployment-stacks | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 213 | `distill_attention_transfer` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 214 | `distill_knowledge_transfer` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 215 | `distill_layer_matching` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 216 | `distill_quantization_aware` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 217 | `distill_self_distillation` | distillation | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 218 | `distributed_gossip_protocol` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 219 | `distributed_inference` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 220 | `distributed_model_sharding` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 221 | `distributed_pipeline_parallel` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 222 | `distributed_ring_allreduce` | distributed | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 223 | `format_batch_export` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 224 | `format_convert_quantize` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 225 | `format_export_gguf` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 226 | `format_export_safetensors` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 227 | `format_import_hf` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 228 | `format_migration_pipeline` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 229 | `format_publish` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 230 | `format_pull_cache` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 231 | `format_rosetta_chain` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 232 | `format_rosetta_convert` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 233 | `format_rosetta_verify` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 234 | `import_hf_cache` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 235 | `import_multi_format` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 236 | `publish_dry_run` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 237 | `publish_multi_registry` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 238 | `pull_resume` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 239 | `pull_verify_decompress` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 240 | `validate_manifest_happy` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 241 | `validate_manifest_live_check` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 242 | `validate_manifest_sha_mismatch` | format | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 243 | `flash_attention_inference` | gpu | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 244 | `gpu_capability_detect` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 245 | `gpu_cuda_inference` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 246 | `gpu_memory_management` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 247 | `gpu_memory_planner` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 248 | `gpu_memory_pool` | gpu | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 249 | `gpu_multi_gpu_inference` | gpu | ![distributed](https://img.shields.io/badge/-distributed-red) | ✅ |
| 250 | `gpu_ptx_analysis` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 251 | `gpu_tensor_core_optimization` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 252 | `gpu_vulkan_inference` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green) | ✅ |
| 253 | `ptx_disassembly` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 254 | `ptx_map_hot_regions` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 255 | `ptx_map_sass_to_ptx` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 256 | `ptx_register_usage` | gpu | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 257 | `adaptive_batch_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 258 | `chat_kv_cache` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 259 | `chat_multiturn` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 260 | `chat_tool_use` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 261 | `dynamic_batch_with_sla` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 262 | `ensemble_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 263 | `inference_apr_run` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 264 | `inference_mmap_lazy_load` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 265 | `inference_run_temperature_sweep` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 266 | `model_pipeline` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 267 | `pipeline_3stage` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 268 | `pipeline_resilient` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 269 | `quantized_inference_comparison` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 270 | `simple_inference` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 271 | `speculative_decode` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 272 | `streaming_token_generator` | inference | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 273 | `awq_lint_batch` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 274 | `awq_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 275 | `awq_lint_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 276 | `dry_sampling_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 277 | `dry_sampling_lint_pipeline` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 278 | `dry_sampling_lint_repetition` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 279 | `gbnf_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 280 | `gbnf_lint_malformed` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 281 | `gbnf_lint_pipeline` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 282 | `ollama_chat_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 283 | `ollama_chat_lint_schema_violation` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 284 | `ollama_chat_lint_stream` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 285 | `oom_lint_allocation_trace` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 286 | `oom_lint_happy` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 287 | `oom_lint_missing_breadcrumb` | lint | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 288 | `tool_use_lint_happy` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 289 | `tool_use_lint_invalid_args` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 290 | `tool_use_lint_streaming` | lint | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 291 | `mcp_byte_parity_dispatcher_swap` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 292 | `mcp_client_simulation` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 293 | `mcp_embedded_initialize_handshake` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 294 | `mcp_embedded_protocol_invariants` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 295 | `mcp_embedded_tools_list_discovery` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 296 | `mcp_notification_progress_token` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 297 | `mcp_sse_event_envelope` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 298 | `mcp_stdio_server` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 299 | `mcp_tool_discovery` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 300 | `mcp_websocket_frame_envelope` | mcp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 301 | `cbtop_headless` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 302 | `cbtop_histogram` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 303 | `cbtop_streaming` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 304 | `hash_chain_audit` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 305 | `inference_cost_tracking` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 306 | `inference_explainability` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 307 | `latency_histogram` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 308 | `model_drift_detection` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 309 | `monitor_alerting` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 310 | `monitor_realtime` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 311 | `monitoring_energy_estimation` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 312 | `monitoring_memory_profiler` | monitoring | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 313 | `mc_business_revenue_forecast` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 314 | `mc_stock_price_simulation_gbm` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 315 | `mc_value_at_risk_historical_vs_parametric` | monte-carlo | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 316 | `distill_checkpoint` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 317 | `distill_ensemble` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 318 | `distill_progressive` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 319 | `distill_standard_kl` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 320 | `finetune_lora` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 321 | `finetune_merge_adapter` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 322 | `finetune_plan_vram` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 323 | `finetune_qlora` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 324 | `merge_average` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 325 | `merge_dare` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 326 | `merge_hierarchical` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 327 | `merge_slerp` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 328 | `merge_ties` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 329 | `merge_weighted` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 330 | `optimize_full_pipeline` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 331 | `optimize_tune` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 332 | `prune_depth` | optimize | ![cuda](https://img.shields.io/badge/-cuda-76b900) | ✅ |
| 333 | `prune_gradual_schedule` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 334 | `prune_magnitude` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 335 | `prune_structured` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 336 | `prune_wanda` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 337 | `quantize_4bit` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 338 | `quantize_fake_qat` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 339 | `quantize_gptq` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 340 | `quantize_mixed_precision` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 341 | `tune_bayesian` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 342 | `tune_grid_early_stop` | optimize | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 343 | `registry_aliases_diff` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 344 | `registry_aliases_list` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 345 | `registry_aliases_resolve` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 346 | `registry_model_comparison` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 347 | `registry_model_lineage` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 348 | `registry_model_rollback` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 349 | `registry_model_versioning` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 350 | `registry_register_apr` | registry | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 351 | `http_model_server` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 352 | `model_ab_testing` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 353 | `model_canary_deploy` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 354 | `model_rate_limiter` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 355 | `model_selection_router` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 356 | `serve_anthropic_messages_api_drop_in` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 357 | `serve_grpc_stream` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 358 | `serve_plan_hf_dryrun_no_weights` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 359 | `serve_rate_limited` | serve | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 360 | `serverless_cold_start_optimization` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 361 | `serverless_container_image` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 362 | `serverless_edge_function` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 363 | `serverless_lambda_inference` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 364 | `serverless_model_warmup` | serverless | ![serverless](https://img.shields.io/badge/-serverless-yellow) | ✅ |
| 365 | `shell_corpus_from_string` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 366 | `shell_history_parse_zsh` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 367 | `shell_trie_prefix_completion` | shell | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 368 | `simd_auto_vectorization` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 369 | `simd_avx_vnni_int8_inference` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 370 | `simd_matrix_ops` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 371 | `simd_quantized_operations` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 372 | `simd_vectorized_inference` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) | ✅ |
| 373 | `trueno_simd_ops` | simd | ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue) | ✅ |
| 374 | `speech_diarization` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 375 | `speech_multilingual` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 376 | `speech_vad` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 377 | `whisper_streaming` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 378 | `whisper_transcribe` | speech | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 379 | `autograd_backprop_viz` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 380 | `autograd_custom_ops` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 381 | `autograd_gradient_clipping` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 382 | `checkpoint_resume` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 383 | `continuous_train_curriculum` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 384 | `continuous_train_federated_simulation` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 385 | `continuous_train_incremental` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 386 | `continuous_train_online_learning` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 387 | `data_preprocessing` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 388 | `data_sharded_shuffle` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 389 | `data_streaming_tokens` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 390 | `entrenar_autograd_training` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 391 | `entrenar_eval_metrics` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 392 | `few_shot_finetune` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 393 | `gradient_accumulation` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 394 | `hyperparameter_sweep` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 395 | `learning_rate_schedule` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 396 | `mixed_precision_training` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 397 | `pretrain_checkpoint_resume` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 398 | `pretrain_nan_guard` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 399 | `pretrain_synthetic_decreasing` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 400 | `train_distributed_sim` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 401 | `train_grad_accum` | training | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 402 | `tsp_compare_tabu_vs_genetic` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 403 | `tsp_distance_matrix_explicit` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 404 | `tsp_solve_with_tabu` | tsp | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 405 | `load_visualization` | visualization | ![cpu](https://img.shields.io/badge/-cpu-lightgrey) | ✅ |
| 406 | `wasm_browser_inference` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 407 | `wasm_model_loader` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 408 | `wasm_progressive_loading` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 409 | `wasm_streaming_compilation` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 410 | `wasm_web_worker` | wasm | ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |
| 411 | `wasm_webgpu_acceleration` | wasm | ![wgpu](https://img.shields.io/badge/-wgpu-green) ![wasm](https://img.shields.io/badge/-wasm-purple) | ✅ |

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
