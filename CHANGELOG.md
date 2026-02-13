# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Upgraded sovereign stack: aprender 0.25, trueno 0.14, entrenar 0.5
- Total recipe count increased from 9 to 121 across 20 categories

#### Category A: Model Creation (6 recipes)
- `create_apr_decision_tree` - Decision tree model creation
- `create_apr_kmeans_clustering` - K-means clustering model
- `create_apr_ngram_language_model` - N-gram language model
- `create_apr_neural_network` - Neural network from scratch

#### Category B: Binary Bundling (7 recipes)
- `bundle_apr_static_binary` - Static binary embedding
- `bundle_apr_quantized_q4` - Q4 quantization bundling
- `bundle_apr_signed` - Ed25519-signed model bundles
- `bundle_apr_lambda_package` - Lambda deployment package

#### Category C: Training (17 recipes)
- `entrenar_autograd_training` - Autograd-based training with entrenar
- `entrenar_lora_finetune` - LoRA fine-tuning
- `entrenar_qlora_finetune` - QLoRA fine-tuning
- `entrenar_distillation` - Knowledge distillation
- `entrenar_model_merge` - TIES/DARE/SLERP model merging
- `entrenar_eval_metrics` - Evaluation metrics and confusion matrices
- `hyperparameter_sweep` - Hyperparameter search
- `checkpoint_resume` - Training checkpoint and resume
- `continuous_train_incremental` - Incremental training
- `continuous_train_online_learning` - Online learning
- `continuous_train_federated_simulation` - Federated learning simulation
- `continuous_train_curriculum` - Curriculum learning
- `mixed_precision_training` - Mixed-precision training
- `few_shot_finetune` - Few-shot fine-tuning
- `gradient_accumulation` - Gradient accumulation for large batches
- `learning_rate_schedule` - Learning rate schedulers
- `data_preprocessing` - Data preprocessing pipelines

#### Category D: Format Conversion (5 recipes)
- `convert_phi_to_apr` - Phi model conversion
- `convert_onnx_to_apr` - ONNX format conversion

#### Category E: Model Registry (5 recipes)
- `registry_register_apr` - Model registration
- `registry_model_lineage` - Model lineage tracking
- `registry_model_comparison` - Model comparison
- `registry_model_rollback` - Model rollback
- `registry_model_versioning` - Semantic model versioning

#### Category F: API Integration (5 recipes)
- `api_call_model_inference` - REST inference endpoint
- `api_streaming_inference` - Streaming inference
- `api_batch_inference` - Batch inference
- `api_model_health_check` - Health check endpoint
- `api_auth_middleware` - Authentication middleware

#### Category G: Serverless (5 recipes)
- `serverless_lambda_inference` - AWS Lambda inference
- `serverless_cold_start_optimization` - Cold start optimization
- `serverless_edge_function` - Edge function deployment
- `serverless_container_image` - Container image packaging
- `serverless_model_warmup` - Model warmup strategies

#### Category H: WASM/Browser (6 recipes)
- `wasm_browser_inference` - Browser inference
- `wasm_web_worker` - Web Worker offloading
- `wasm_progressive_loading` - Progressive model loading
- `wasm_webgpu_acceleration` - WebGPU acceleration
- `wasm_streaming_compilation` - Streaming WASM compilation
- `wasm_model_loader` - WASM model loader

#### Category I: GPU Acceleration (6 recipes)
- `flash_attention_inference` - FlashAttention inference
- `gpu_cuda_inference` - CUDA inference
- `gpu_tensor_core_optimization` - Tensor core optimization
- `gpu_multi_gpu_inference` - Multi-GPU inference
- `gpu_memory_management` - GPU memory management
- `gpu_memory_pool` - GPU memory pool allocator

#### Category J: SIMD Acceleration (5 recipes)
- `trueno_simd_ops` - trueno SIMD operations
- `simd_matrix_operations` - SIMD matrix operations
- `simd_vectorized_inference` - Vectorized inference
- `simd_quantized_operations` - Quantized SIMD operations
- `simd_auto_vectorization` - Auto-vectorization

#### Category K: Model Distillation (5 recipes)
- `distill_knowledge_transfer` - Knowledge transfer
- `distill_layer_matching` - Layer matching distillation
- `distill_pruning_aware` - Pruning-aware distillation
- `distill_quantization_aware` - Quantization-aware distillation
- `distill_structured_pruning` - Structured pruning

#### Category L: CLI Tools (7 recipes)
- `cli_apr_info` - Model inspector
- `cli_apr_bench` - Benchmark tool
- `cli_apr_convert` - Format converter
- `cli_apr_serve` - Model server
- `cli_apr_diff` - Model diff tool

#### Category M: Monitoring (5 recipes)
- `inference_explainability` - Inference explainability
- `hash_chain_audit` - Hash chain audit trail
- `inference_cost_tracking` - Inference cost tracking
- `latency_histogram` - Latency histogram metrics
- `model_drift_detection` - Model drift detection

#### Category N: Speech Recognition (2 recipes)
- `whisper_transcribe` - whisper.apr transcription
- `whisper_streaming` - Streaming ASR

#### Category O: Distributed (2 recipes)
- `distributed_inference` - repartir multi-node inference
- `distributed_model_sharding` - Model sharding across nodes

#### Category P: Inference Patterns (11 recipes)
- `simple_inference` - Basic inference
- `speculative_decode` - Speculative decoding
- `chat_kv_cache` - KV-cache for chat
- `chat_multiturn` - Multi-turn conversation
- `chat_tool_use` - Tool use in chat
- `streaming_token_generator` - Streaming token generation
- `adaptive_batch_inference` - Adaptive batching
- `dynamic_batch_with_sla` - Dynamic batching with SLA
- `ensemble_inference` - Ensemble inference
- `model_pipeline` - Model pipeline composition
- `quantized_inference_comparison` - Quantized inference comparison

#### Category Q: Model Serving (5 recipes)
- `http_model_server` - HTTP REST model server
- `model_ab_testing` - A/B testing for models
- `model_canary_deploy` - Canary deployment
- `model_rate_limiter` - Rate limiting
- `model_selection_router` - Model selection router

#### Advanced Demos (16 recipes)
- End-to-end applications: RAG pipeline, Spanish tutor, style transfer, voice recognition, and more

## [0.1.0] - 2024-12-02

### Added
- Core library with bundle and convert modules
- `BundledModel` for loading APR models from bytes
- `ModelBundle` builder for creating APR bundles
- `AprConverter` for format conversion (SafeTensors, GGUF)
- Integration with `aprender` format module
- Property-based testing suite with proptest
- Examples:
  - `bundle_static_model` - Static model embedding
  - `bundle_quantized_model` - Quantized model bundling
  - `bundle_encrypted_model` - AES-256-GCM encryption (requires `encryption` feature)
  - `convert_safetensors_to_apr` - SafeTensors conversion
  - `convert_apr_to_gguf` - GGUF export
  - `convert_gguf_to_apr` - GGUF import
  - `simd_matrix_operations` - SIMD acceleration demo
  - `apr_info` - CLI model inspector
  - `apr_bench` - CLI benchmark tool

### Security
- AES-256-GCM authenticated encryption support
- Argon2id key derivation for password-based encryption

[Unreleased]: https://github.com/paiml/apr-cookbook/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/paiml/apr-cookbook/releases/tag/v0.1.0
