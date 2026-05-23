# Recipe Catalog (91 IIUR Recipes)

All recipes follow IIUR principles. See [principles.md](principles.md) for structure and test requirements.

---

## Category A: Model Creation (7 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| A.1 | `create_apr_from_scratch` | Create `.apr` model from raw tensors without external dependencies |
| A.2 | `create_demo_model` | Create a minimal demo model for testing and examples |
| A.3 | `create_apr_linear_regression` | Train linear regression model and save as `.apr` |
| A.4 | `create_apr_decision_tree` | Train decision tree classifier and save as `.apr` |
| A.5 | `create_apr_kmeans_clustering` | Train KMeans model on synthetic data |
| A.6 | `create_apr_ngram_language_model` | Build N-gram language model from text corpus |
| A.7 | `create_apr_neural_network` | Build and train a multi-layer neural network |

---

## Category B: Binary Bundling & Deployment (7 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| B.1 | `bundle_static_model` | Embed `.apr` into binary via `include_bytes!()` |
| B.2 | `bundle_quantized_model` | Bundle quantized model for reduced binary size |
| B.3 | `bundle_encrypted_model` | Bundle AES-256-GCM encrypted model (requires `encryption` feature) |
| B.4 | `bundle_apr_static_binary` | Embed `.apr` into Rust binary for zero-dependency deployment |
| B.5 | `bundle_apr_quantized_q4` | Bundle Q4_0 quantized model (75% size reduction) |
| B.6 | `bundle_apr_signed` | Bundle Ed25519 signed model with verification |
| B.7 | `bundle_apr_lambda_package` | Create AWS Lambda deployment package with bundled model |

---

## Category C: Continuous Training (16 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| C.1 | `continuous_train_incremental` | Update existing `.apr` model with new training data |
| C.2 | `continuous_train_online_learning` | Online learning with single-sample updates |
| C.3 | `continuous_train_federated_simulation` | Simulate federated learning with model averaging |
| C.4 | `continuous_train_curriculum` | Curriculum learning with progressive difficulty |
| C.5 | `entrenar_autograd_training` | Autograd-based training with entrenar |
| C.6 | `entrenar_eval_metrics` | Evaluation metrics (confusion matrix, F1, etc.) |
| C.7 | `hyperparameter_sweep` | Grid/random hyperparameter search |
| C.8 | `checkpoint_resume` | Save/restore training checkpoints |
| C.9 | `mixed_precision_training` | FP16/FP32 mixed precision training |
| C.10 | `few_shot_finetune` | Few-shot finetuning with small datasets |
| C.11 | `data_preprocessing` | Data preprocessing pipelines for training |
| C.12 | `learning_rate_schedule` | Learning rate schedulers (cosine, warmup, etc.) |
| C.13 | `gradient_accumulation` | Gradient accumulation for large effective batch sizes |
| C.14 | `autograd_custom_ops` | Custom autograd operations and backward passes |
| C.15 | `autograd_backprop_viz` | Backpropagation visualization and computation graphs |
| C.16 | `autograd_gradient_clipping` | Gradient clipping strategies for stable training |

---

## Category D: Format Conversion (5 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| D.1 | `convert_phi_to_apr` | Convert Microsoft Phi-3 Mini to `.apr` format |
| D.2 | `convert_safetensors_to_apr` | Convert SafeTensors format to `.apr` |
| D.3 | `convert_apr_to_gguf` | Export `.apr` to GGUF v3 format |
| D.4 | `convert_gguf_to_apr` | Import GGUF format to `.apr` |
| D.5 | `convert_onnx_to_apr` | Convert ONNX model to `.apr` format |

---

## Category E: Model Registry (5 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| E.1 | `registry_register_apr` | Register `.apr` model in registry with versioning |
| E.2 | `registry_model_lineage` | Track full model lineage (data -> recipe -> model -> deployment) |
| E.3 | `registry_model_comparison` | Compare model versions and metrics |
| E.4 | `registry_model_rollback` | Rollback to previous model version |
| E.5 | `registry_model_versioning` | Semantic versioning and model lifecycle management |

---

## Category F: API Integration (5 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| F.1 | `api_call_model_inference` | Call model inference via REST API |
| F.2 | `api_streaming_inference` | Streaming token generation via Server-Sent Events |
| F.3 | `api_batch_inference` | Batch inference for high throughput |
| F.4 | `api_model_health_check` | Health check and metrics endpoint usage |
| F.5 | `api_auth_middleware` | Authentication middleware for inference APIs |

---

## Category G: Serverless Deployment (5 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| G.1 | `serverless_lambda_inference` | Deploy `.apr` model to AWS Lambda |
| G.2 | `serverless_cold_start_optimization` | Minimize cold start with pre-warming |
| G.3 | `serverless_edge_function` | Edge function inference (CloudFront/Lambda@Edge) |
| G.4 | `serverless_container_image` | Deploy bundled `.apr` as container image for Lambda |
| G.5 | `serverless_model_warmup` | Model warmup strategies for serverless |

---

## Category H: WASM & Browser (6 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| H.1 | `wasm_browser_inference` | Run `.apr` inference in browser via WASM |
| H.2 | `wasm_web_worker` | Offload inference to Web Worker for responsive UI |
| H.3 | `wasm_progressive_loading` | Progressive model loading with streaming compilation |
| H.4 | `wasm_webgpu_acceleration` | WebGPU-accelerated inference in browser |
| H.5 | `wasm_streaming_compilation` | WASM streaming compilation pipeline |
| H.6 | `wasm_model_loader` | Model loader with format detection and validation |

---

## Category I: GPU Acceleration (8 recipes)

| # | Recipe | Objective | Falsifiable Claim |
|---|--------|-----------|-------------------|
| I.1 | `flash_attention_inference` | FlashAttention implementation for long sequences | (F6 deleted in v5.0 — CPU-tiled proxy, no measured speedup; keep as architecture demo) |
| I.2 | `gpu_cuda_inference` | CUDA-style GPU inference | — |
| I.3 | `gpu_tensor_core_optimization` | Tensor core utilization patterns | — |
| I.4 | `gpu_multi_gpu_inference` | Multi-GPU model parallel inference | — |
| I.5 | `gpu_memory_management` | GPU memory management and allocation | — |
| I.6 | `gpu_memory_pool` | Pool-based GPU memory allocator | — |
| I.7 | `gpu_ptx_analysis` | PTX kernel analysis and profiling | — |
| I.8 | `gpu_vulkan_inference` | Vulkan/wgpu inference on non-NVIDIA GPUs | — |

---

## Category J: SIMD Acceleration (6 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| J.1 | `trueno_simd_ops` | trueno SIMD operations demonstration |
| J.2 | `simd_matrix_ops` | SIMD-accelerated matrix operations |
| J.3 | `simd_vectorized_inference` | Vectorized inference pipeline |
| J.4 | `simd_quantized_operations` | SIMD quantized arithmetic (Int8/Int4) |
| J.5 | `simd_auto_vectorization` | Compiler auto-vectorization patterns |
| J.6 | `simd_avx_vnni_int8_inference` | AVX-VNNI Int8 dot product inference (Intel Meteor Lake+) |

---

## Category K: Model Distillation (5 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| K.1 | `distill_knowledge_transfer` | Knowledge distillation with soft targets |
| K.2 | `distill_layer_matching` | Layer-wise feature matching distillation |
| K.3 | `distill_quantization_aware` | Quantization-aware distillation |
| K.4 | `distill_attention_transfer` | Attention transfer between teacher and student |
| K.5 | `distill_self_distillation` | Self-distillation (born-again networks) |

---

## Category L: CLI Tools (16 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| L.1 | `apr_info` | Inspect `.apr` model metadata |
| L.2 | `apr_bench` | Benchmark inference performance |
| L.3 | `cli_apr_info` | Extended model inspection with JSON output |
| L.4 | `cli_apr_bench` | Extended benchmarking with statistical analysis |
| L.5 | `cli_apr_convert` | Format conversion CLI |
| L.6 | `cli_apr_serve` | Model serving CLI |
| L.7 | `cli_apr_diff` | Model comparison and diff |
| L.8 | `cli_apr_compile` | Model compilation and optimization |
| L.9 | `cli_apr_tui` | Terminal UI for model exploration |
| L.10 | `cli_apr_decrypt` | Model decryption CLI |
| L.11 | `cli_apr_diagnose` | Model diagnosis (5-whys analysis) |
| L.12 | `cli_apr_list` | List registered models |
| L.13 | `cli_apr_rm` | Remove models from registry |
| L.14 | `cli_apr_runs` | Training run management |
| L.15 | `cli_apr_tokenize` | Tokenization inspection |
| L.16 | `cli_apr_ptx_map` | PTX source mapping for GPU kernels |

---

## Recipe Dependency Summary

All 91 IIUR recipes depend on `aprender` (core ML library). Additional dependencies:

| Category | aprender | trueno | entrenar | ndarray |
|----------|----------|--------|----------|---------|
| A (Creation) | Required | — | — | — |
| B (Bundling) | Required | — | — | — |
| C (Training) | Required | — | Required | Required |
| D (Conversion) | Required | — | — | — |
| E (Registry) | Required | — | — | — |
| F (API) | Required | — | — | — |
| G (Serverless) | Required | — | — | — |
| H (WASM) | Required | — | — | — |
| I (GPU) | Required | Required | — | — |
| J (SIMD) | Required | Required | — | — |
| K (Distillation) | Required | — | — | — |
| L (CLI) | Required | — | — | — |

GPU (realizar), distributed (repartir), and speech (whisper-apr) patterns are **simulated** in cookbook examples without requiring these crates as compile-time dependencies.

---

## Centralize-Cookbooks Categories (v6.0, 2026-05-04)

Four new categories added by [centralize-cookbooks](../centralize-cookbooks.md) PMAT-065..067. These do NOT count toward the 91 IIUR recipe denominator; they are graded against either `recipe-iiur-v1.yaml` (Rust examples) or `recipe-iiur-config-v1.yaml` (declarative-config wrappers).

### Category: deployment-stacks (ex-sovereign-ai-cookbook, PMAT-065)

| # | Recipe | Wrapper |
|---|--------|---------|
| DS.1 | `alimentar-ingest` | `cargo run --example alimentar_ingest` |
| DS.2 | `apr-inference-server` | `cargo run --example apr_inference_server` |
| DS.3 | `batuta-agent` | `cargo run --example batuta_agent` |
| DS.4 | `entrenar-train` | `cargo run --example entrenar_train` |
| DS.5 | `jetson-edge-base` | `cargo run --example jetson_edge_base` |
| DS.6 | `pacha-registry` | `cargo run --example pacha_registry` |
| DS.7 | `pepita-sandbox` | `cargo run --example pepita_sandbox` |
| DS.8 | `realizar-serve` | `cargo run --example realizar_serve` |
| DS.9 | `renacer-observability` | `cargo run --example renacer_observability` |
| DS.10 | `repartir-worker` | `cargo run --example repartir_worker` |
| DS.11 | `sovereign-ai-stack` | `cargo run --example sovereign_ai_stack` |
| DS.12 | `trueno-db-analytics` | `cargo run --example trueno_db_analytics` |
| DS.13 | `trueno-rag-pipeline` | `cargo run --example trueno_rag_pipeline` |
| DS.14 | `whisper-apr-asr` | `cargo run --example whisper_apr_asr` |

Plus 10 multi-recipe stack compositions under `examples/deployment-stacks/stacks/`.

### Category: data-loading (ex-alimentar, PMAT-066)

| # | Recipe | Topic |
|---|--------|-------|
| DL.1 | `basic_loading` | CSV/JSON/Parquet via Arrow |
| DL.2 | `cli_batch_commands` | CLI batch operations |
| DL.3 | `dataloader_batching` | DataLoader batching |
| DL.4 | `doctest_extraction` | Doctest extraction |
| DL.5 | `drift_detection` | Dataset drift |
| DL.6 | `federated_split` | Federated learning split |
| DL.7 | `hub_publishing` | HuggingFace Hub publish |
| DL.8 | `prose_detection` | Prose vs code |
| DL.9 | `quality_check` | Dataset QA |
| DL.10 | `registry_publish` | Registry publish |
| DL.11..15 | `repl_*` | REPL components |
| DL.16 | `streaming_large` | Streaming datasets |
| DL.17 | `transforms_pipeline` | Transform composition |
| DL.18 | `tui_viewer` | TUI dataset viewer |

### Category: visualization (ex-presentar, PMAT-067)

| Subdir | Count | Format |
|--------|-------|--------|
| `ald/` | 6 | `.yaml` |
| `apr/` | 7 | `.yaml` |
| `charts/` | 3 | `.yaml` |
| `dashboards/` | 5 | `.yaml` |
| `edge_cases/` | 2 | `.yaml` |
| `prs/` | 5 | `.prs` |
| **Total** | **28** | All loaded by `load_visualization` validator |

### Category: machines (PMAT-065)

| Machine | Path |
|---------|------|
| `jetson` | `examples/machines/jetson/` |

---

## Six Coverage Invariants — v6.0 carve-outs (PMAT-068)

Per [centralize-cookbooks/iiur-conformance.md](../centralize-cookbooks/iiur-conformance.md) §"Coverage Invariants — Update":

- **A (CLI parity)** — UNCHANGED. Numerator stays scoped to `apr-cli` subcommands.
- **B (Contract grade)** — EXTENDED. Includes `recipe-iiur-config-v1.yaml`; denominator grows.
- **C (Format variants)** — CONDITIONAL. Applies to data-loading (CSV/JSON/Parquet) and visualization (YAML/.prs); not deployment-stacks.
- **D (arXiv citation)** — EXTENDED. Every new recipe header MUST cite. Gaps marked `Citation: N/A — see PMAT-066/067`.
- **E (Docs contract coverage)** — EXTENDED. Book chapters under `data-loading/` and `visualization/` count toward the docs ratio.
- **F (Variant depth)** — CARVED OUT. Does not apply to deployment-stacks (one config per service) or machines (one config per platform). Applies to data-loading and visualization at threshold ≥1.

## Expand-Cookbooks Categories (v6.1, 2026-05-05)

Six new categories and 8 extended categories added by [expand-cookbooks](../expand-cookbooks.md) PMAT-073..087. 44 new recipes / 167 new tests across the initiative; full per-recipe catalog in [expand-cookbooks/recipe-catalog.md](../expand-cookbooks/recipe-catalog.md).

### Category: code (apr code agentic surface, PMAT-074)

| # | Recipe | Parity row |
|---|--------|------------|
| C.1 | `code_mcp_client_config` | PMAT-CODE-MCP-CLIENT-001 |
| C.2 | `code_slash_command_extension` | PMAT-CODE-SLASH-PARITY-001 |
| C.3 | `code_hook_session_start` | PMAT-CODE-HOOKS-001 |
| C.4 | `code_subagent_spawn_payload` | PMAT-CODE-SPAWN-PARITY-001 |
| C.5 | `code_custom_agent_definition` | PMAT-CODE-CUSTOM-AGENTS-001 |
| C.6 | `code_skill_discovery` | PMAT-CODE-SKILLS-001 |
| C.7 | `code_worktree_isolation_permission_mode` | PMAT-CODE-WORKTREE-001 + PMAT-CODE-PERMISSIONS-001 |

### Category: tsp (aprender-tsp, PMAT-080)

| # | Recipe |
|---|--------|
| TSP.1 | `tsp_solve_with_tabu` |
| TSP.2 | `tsp_distance_matrix_explicit` |
| TSP.3 | `tsp_compare_tabu_vs_genetic` |

### Category: shell (aprender-shell, PMAT-081)

| # | Recipe |
|---|--------|
| SH.1 | `shell_history_parse_zsh` |
| SH.2 | `shell_corpus_from_string` |
| SH.3 | `shell_trie_prefix_completion` |

### Category: monte-carlo (aprender-monte-carlo, PMAT-082)

| # | Recipe |
|---|--------|
| MC.1 | `mc_stock_price_simulation_gbm` |
| MC.2 | `mc_business_revenue_forecast` |
| MC.3 | `mc_value_at_risk_historical_vs_parametric` |

### Category: cgp (aprender-cgp, PMAT-083)

| # | Recipe |
|---|--------|
| CGP.1 | `cgp_regression_detector_baseline_vs_current` |
| CGP.2 | `cgp_roofline_classify_kernel` |
| CGP.3 | `cgp_roofline_ridge_point_per_precision` |

### Category: contracts-macros (aprender-contracts-macros, PMAT-084)

| # | Recipe |
|---|--------|
| CM.1 | `contracts_macros_attribute_basic` |
| CM.2 | `contracts_macros_env_key_convention` |
| CM.3 | `contracts_macros_runtime_validator_bridge` |

### Extended categories — Tier 1 + Tier 3 + Tier 4 additions

| Category | New recipes | Tickets |
|----------|-------------|---------|
| `cli/` | `cli_trace_save_tensor_layer0`, `cli_diff_values_aprt_stage`, `cli_publish_manifest_full`, `cli_validate_manifest_safetensors_dtype`, `cli_publish_parent_chain_termination` | PMAT-075 + PMAT-076 |
| `serve/` | `serve_anthropic_messages_api_drop_in`, `serve_plan_hf_dryrun_no_weights` | PMAT-077 |
| `mcp/` | `mcp_embedded_*` (3), `mcp_sse_event_envelope`, `mcp_websocket_frame_envelope`, `mcp_notification_progress_token`, `mcp_byte_parity_dispatcher_swap` | PMAT-078 + PMAT-079 |
| `analysis/` | `analysis_cpu_vs_gpu_parity_gate`, `analysis_contract_algorithm_binding_pattern`, `analysis_pv_check_parity_authoring`, `analysis_json_schema_draft7_meta_validation` | PMAT-075 + PMAT-086 |
| `acceleration/` | `acceleration_moe_rayon_dispatch_bench`, `acceleration_mmap_per_tensor_diff_bench` | PMAT-085 |
| `bundling/` | `bundle_streaming_q4k_large_model` | PMAT-085 |
| `conversion/` | `conversion_gguf_legacy_quant_fallback` | PMAT-085 |
| `distillation/` | `distill_against_contract_v1` | PMAT-086 |
| `inference/` | `inference_qwen3_moe_numerical_parity_smoke` | PMAT-085 |

## v6.1 spec acceptance summary

Per [expand-cookbooks.md](../expand-cookbooks.md) §"Acceptance Criteria":

- ✅ Recipe count: 44/44 specced delivered, 18/18 sister-crate ≥3 satisfied
- ✅ IIUR grade: every recipe carries `Contract: contracts/recipe-iiur-v1.yaml` header
- ✅ CLI parity (Invariant A) extended for new `apr` subcommands
- ✅ Cargo.toml: 6 sister-crate dev-deps declared
- ✅ README recipe table regenerated to 420 recipes
- ✅ mdBook: 6 new category overview chapters + SUMMARY.md wiring
- ✅ Recipe-catalog spec extended with 6 new category sections (this section)
