# Recipe Catalog (60 IIUR Recipes)

All recipes follow IIUR principles. See [principles.md](principles.md) for structure and test requirements.

---

## Category A: Model Creation (5 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| A.1 | `create_apr_from_scratch` | Create `.apr` model from raw tensors without external dependencies |
| A.2 | `create_apr_linear_regression` | Train linear regression model and save as `.apr` |
| A.3 | `create_apr_decision_tree` | Train decision tree classifier and save as `.apr` |
| A.4 | `create_apr_kmeans_clustering` | Train KMeans model on synthetic data |
| A.5 | `create_apr_ngram_language_model` | Build N-gram language model from text corpus |

### A.1 Key Example

```rust
fn main() -> Result<()> {
    let ctx = RecipeContext::new("create_apr_from_scratch")?;
    let weights = Tensor::randn(&mut ctx.rng, &[768, 768]);
    let biases = Tensor::zeros(&[768]);
    let model = LinearModel::from_tensors(weights, biases)?;
    let apr_path = ctx.path("custom_model.apr");
    aprender::format::save(&model, ModelType::Linear, &apr_path, SaveOptions::default())?;
    let loaded: LinearModel = aprender::format::load(&apr_path, ModelType::Linear)?;
    assert_eq!(model.num_params(), loaded.num_params());
    Ok(())
}
```

---

## Category B: Binary Bundling & Deployment (5 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| B.1 | `bundle_apr_static_binary` | Embed `.apr` into Rust binary for zero-dependency deployment |
| B.2 | `bundle_apr_quantized_q4` | Bundle Q4_0 quantized model (75% size reduction) |
| B.3 | `bundle_apr_encrypted` | Bundle AES-256-GCM encrypted model with Argon2id KDF |
| B.4 | `bundle_apr_signed` | Bundle Ed25519 signed model with verification |
| B.5 | `bundle_apr_lambda_package` | Create AWS Lambda deployment package with bundled model |

---

## Category C: Continuous Training (4 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| C.1 | `continuous_train_incremental` | Update existing `.apr` model with new training data |
| C.2 | `continuous_train_online_learning` | Online learning with single-sample updates |
| C.3 | `continuous_train_federated_simulation` | Simulate federated learning with model averaging |
| C.4 | `continuous_train_curriculum` | Curriculum learning with progressive difficulty |

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

## Category E: Model Registry — Pacha Integration (4 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| E.1 | `registry_register_apr` | Register `.apr` model in Pacha registry with versioning |
| E.2 | `registry_model_lineage` | Track full model lineage (data -> recipe -> model -> deployment) |
| E.3 | `registry_model_comparison` | Compare model versions and metrics |
| E.4 | `registry_model_rollback` | Rollback to previous model version |

---

## Category F: API Integration — Realizar (4 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| F.1 | `api_call_model_inference` | Call model inference via REST API |
| F.2 | `api_streaming_inference` | Streaming token generation via Server-Sent Events |
| F.3 | `api_batch_inference` | Batch inference for high throughput |
| F.4 | `api_model_health_check` | Health check and metrics endpoint usage |

---

## Category G: Serverless Deployment (4 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| G.1 | `deploy_lambda_inference` | Deploy `.apr` model to AWS Lambda |
| G.2 | `deploy_lambda_batch` | Lambda batch processing with SQS integration |
| G.3 | `deploy_lambda_edge` | Lambda@Edge for global inference |
| G.4 | `deploy_lambda_container` | Deploy bundled `.apr` as container image for Lambda |

---

## Category H: WASM & Browser — Presentar (5 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| H.1 | `wasm_model_inference` | Run `.apr` inference in browser via WASM |
| H.2 | `wasm_interactive_demo` | Interactive model demo with Presentar widgets |
| H.3 | `wasm_visualization_dashboard` | Model metrics visualization dashboard |
| H.4 | `wasm_autocomplete_demo` | N-gram autocomplete (batuta showcase) |
| H.5 | `wasm_web_worker` | Offload inference to Web Worker for responsive UI |

---

## Category I: GPU Acceleration (5 recipes)

| # | Recipe | Objective | Falsifiable Claim |
|---|--------|-----------|-------------------|
| I.1 | `gpu_matrix_operations` | GPU-accelerated matrix operations via trueno | F7: AVX-512 >= 80 GFLOPS (1024x1024) |
| I.2 | `gpu_model_inference` | Full model inference on GPU | F6: FlashAttention >= 2x speedup (seq>=1024) |
| I.3 | `gpu_batch_inference` | Batched GPU inference for throughput | — |
| I.4 | `gpu_webgpu_fallback` | WebGPU fallback for browser GPU | — |
| I.5 | `gpu_vulkan_inference` | Vulkan/wgpu inference on Intel Arc (non-NVIDIA) | — |

---

## Category J: SIMD Acceleration (5 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| J.1 | `simd_vector_operations` | SIMD-accelerated vector operations |
| J.2 | `simd_matrix_multiply` | SIMD matrix multiplication |
| J.3 | `simd_convolution` | SIMD convolution operations |
| J.4 | `simd_softmax` | SIMD softmax with numerical stability |
| J.5 | `simd_avx_vnni_int8_inference` | AVX-VNNI Int8 dot product inference (Intel Meteor Lake+) |

---

## Category K: Model Distillation & HuggingFace (4 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| K.1 | `distill_hf_to_apr` | Distill HuggingFace model to compact `.apr` |
| K.2 | `distill_knowledge_transfer` | Knowledge distillation with soft targets |
| K.3 | `distill_layer_pruning` | Layer pruning for model compression |
| K.4 | `distill_quantization_aware` | Quantization-aware distillation |

---

## Category L: CLI Tools (4 recipes)

| # | Recipe | Objective |
|---|--------|-----------|
| L.1 | `cli_apr_info` | Inspect `.apr` model metadata |
| L.2 | `cli_apr_bench` | Benchmark inference performance |
| L.3 | `cli_apr_convert` | Format conversion CLI |
| L.4 | `cli_apr_validate` | Validate `.apr` integrity and signatures |

---

## Additional Recipes (Cross-Category)

| # | Recipe | Category | Device Tier | Objective |
|---|--------|----------|-------------|-----------|
| M.1 | `inference_mmap_lazy_load` | Inference | T0 cpu | Memory-mapped lazy loading for models approaching RAM limits |
| M.2 | `monitoring_energy_estimation` | Monitoring | T1a x86_64 | RAPL energy estimation (joules/inference, CO2) on Intel CPUs |
| M.3 | `acceleration_compression_benchmark` | Acceleration | T0 cpu | LZ4 vs ZSTD vs none: throughput (GB/s), ratio, decompression latency |
| M.4 | `acceleration_cache_tiling` | Acceleration | T1a x86_64 | Cache-oblivious vs tiled matmul, tile size sweep for L1/L2/L3 |
| M.5 | `monitoring_memory_profiler` | Monitoring | T0 cpu | Peak RSS tracking during model load + inference, container sizing |
| M.6 | `analysis_model_fingerprint` | Analysis | T0 cpu | blake3 content-addressable hashing + ed25519 signing, tamper detection |

---

## Recipe Dependency Summary

| Category | aprender | trueno | pacha | realizar | presentar |
|----------|----------|--------|-------|----------|-----------|
| A (Creation) | Required | - | - | - | - |
| B (Bundling) | Required | - | - | Optional | - |
| C (Training) | Required | - | - | - | - |
| D (Conversion) | Required | - | - | - | - |
| E (Registry) | Required | - | Required | - | - |
| F (API) | Required | - | Optional | Required | - |
| G (Serverless) | Required | - | - | Required | - |
| H (WASM) | Required | Optional | - | - | Required |
| I (GPU) | Required | Required | - | - | - |
| J (SIMD) | Required | Required | - | - | - |
| K (Distillation) | Required | - | - | - | - |
| L (CLI) | Required | - | Optional | - | - |
