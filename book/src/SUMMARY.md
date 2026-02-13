# Summary

[Introduction](./introduction.md)

# Getting Started

- [Installation](./getting-started/installation.md)
- [Quick Start](./getting-started/quick-start.md)
- [Project Structure](./getting-started/structure.md)

# Core Concepts

- [The APR Format](./concepts/apr-format.md)
- [Model Bundling](./concepts/bundling.md)
- [Format Conversion](./concepts/conversion.md)
- [Zero-Copy Loading](./concepts/zero-copy.md)

---

# Category A: Model Creation

- [Overview](./recipes/a-creation/index.md)
- [Create APR from Scratch](./recipes/a-creation/create-apr-from-scratch.md)
- [Linear Regression Model](./recipes/a-creation/linear-regression.md)
- [Decision Tree Model](./recipes/a-creation/decision-tree.md)
- [K-Means Clustering](./recipes/a-creation/kmeans-clustering.md)
- [N-gram Language Model](./recipes/a-creation/ngram-language-model.md)
- [Neural Network](./recipes/a-creation/neural-network.md)

# Category B: Binary Bundling

- [Overview](./recipes/b-bundling/index.md)
- [Bundle Static Model](./recipes/b-bundling/bundle-static.md)
- [Bundle Quantized Model](./recipes/b-bundling/bundle-quantized.md)
- [Bundle Encrypted Model](./recipes/b-bundling/bundle-encrypted.md)
- [Static Binary Embedding](./recipes/b-bundling/static-binary.md)
- [Q4 Quantization](./recipes/b-bundling/quantized-q4.md)
- [Signed Models](./recipes/b-bundling/signed.md)
- [Lambda Package](./recipes/b-bundling/lambda-package.md)

# Category C: Training

- [Overview](./recipes/c-training/index.md)
- [Incremental Training](./recipes/c-training/incremental.md)
- [Online Learning](./recipes/c-training/online-learning.md)
- [Federated Simulation](./recipes/c-training/federated-simulation.md)
- [Curriculum Learning](./recipes/c-training/curriculum.md)
- [Autograd Training](./recipes/c-training/autograd.md)
- [LoRA Fine-tuning](./recipes/c-training/lora.md)
- [QLoRA Fine-tuning](./recipes/c-training/qlora.md)
- [Knowledge Distillation](./recipes/c-training/distillation.md)
- [Model Merge](./recipes/c-training/model-merge.md)
- [Evaluation Metrics](./recipes/c-training/eval-metrics.md)
- [Hyperparameter Sweep](./recipes/c-training/hyperparameter-sweep.md)
- [Checkpoint Resume](./recipes/c-training/checkpoint-resume.md)
- [Mixed-Precision Training](./recipes/c-training/mixed-precision.md)
- [Few-Shot Fine-tuning](./recipes/c-training/few-shot.md)
- [Gradient Accumulation](./recipes/c-training/gradient-accumulation.md)
- [Learning Rate Schedules](./recipes/c-training/lr-schedule.md)
- [Data Preprocessing](./recipes/c-training/data-preprocessing.md)

# Category D: Format Conversion

- [Overview](./recipes/d-conversion/index.md)
- [SafeTensors to APR](./recipes/d-conversion/safetensors-to-apr.md)
- [APR to GGUF](./recipes/d-conversion/apr-to-gguf.md)
- [GGUF to APR](./recipes/d-conversion/gguf-to-apr.md)
- [Phi Model to APR](./recipes/d-conversion/phi-to-apr.md)
- [ONNX to APR](./recipes/d-conversion/onnx-to-apr.md)

# Category E: Model Registry

- [Overview](./recipes/e-registry/index.md)
- [Register APR Model](./recipes/e-registry/register-apr.md)
- [Model Lineage](./recipes/e-registry/model-lineage.md)
- [Model Comparison](./recipes/e-registry/model-comparison.md)
- [Model Rollback](./recipes/e-registry/model-rollback.md)
- [Model Versioning](./recipes/e-registry/model-versioning.md)

# Category F: API Integration

- [Overview](./recipes/f-api/index.md)
- [Model Inference](./recipes/f-api/model-inference.md)
- [Streaming Inference](./recipes/f-api/streaming-inference.md)
- [Batch Inference](./recipes/f-api/batch-inference.md)
- [Health Check](./recipes/f-api/health-check.md)
- [Auth Middleware](./recipes/f-api/auth-middleware.md)

# Category G: Serverless

- [Overview](./recipes/g-serverless/index.md)
- [Lambda Inference](./recipes/g-serverless/lambda-inference.md)
- [Cold Start Optimization](./recipes/g-serverless/cold-start.md)
- [Edge Functions](./recipes/g-serverless/edge-function.md)
- [Container Image](./recipes/g-serverless/container-image.md)
- [Model Warmup](./recipes/g-serverless/model-warmup.md)

# Category H: WASM/Browser

- [Overview](./recipes/h-wasm/index.md)
- [Browser Inference](./recipes/h-wasm/browser-inference.md)
- [Web Workers](./recipes/h-wasm/web-worker.md)
- [Progressive Loading](./recipes/h-wasm/progressive-loading.md)
- [WebGPU Acceleration](./recipes/h-wasm/webgpu-acceleration.md)
- [Streaming Compilation](./recipes/h-wasm/streaming-compilation.md)
- [Model Loader](./recipes/h-wasm/model-loader.md)

# Category I: GPU Acceleration

- [Overview](./recipes/i-gpu/index.md)
- [FlashAttention](./recipes/i-gpu/flash-attention.md)
- [CUDA Inference](./recipes/i-gpu/cuda-inference.md)
- [Tensor Core Optimization](./recipes/i-gpu/tensor-core.md)
- [Multi-GPU Inference](./recipes/i-gpu/multi-gpu.md)
- [Memory Management](./recipes/i-gpu/memory-management.md)
- [Memory Pool](./recipes/i-gpu/memory-pool.md)

# Category J: SIMD Acceleration

- [Overview](./recipes/j-simd/index.md)
- [Matrix Operations](./recipes/j-simd/matrix-operations.md)
- [Vectorized Inference](./recipes/j-simd/vectorized-inference.md)
- [Quantized Operations](./recipes/j-simd/quantized-operations.md)
- [Auto-Vectorization](./recipes/j-simd/auto-vectorization.md)

# Category K: Model Distillation

- [Overview](./recipes/k-distillation/index.md)
- [Knowledge Transfer](./recipes/k-distillation/knowledge-transfer.md)
- [Layer Matching](./recipes/k-distillation/layer-matching.md)
- [Pruning-Aware Distillation](./recipes/k-distillation/pruning-aware.md)
- [Quantization-Aware Distillation](./recipes/k-distillation/quantization-aware.md)
- [Structured Pruning](./recipes/k-distillation/structured-pruning.md)

# Category L: CLI Tools

- [Overview](./recipes/l-cli/index.md)
- [apr-info](./recipes/l-cli/apr-info.md)
- [apr-bench](./recipes/l-cli/apr-bench.md)
- [apr-convert](./recipes/l-cli/apr-convert.md)
- [apr-serve](./recipes/l-cli/apr-serve.md)
- [apr-diff](./recipes/l-cli/apr-diff.md)

# Category M: Inference Monitoring

- [Overview](./recipes/m-monitoring/index.md)
- [Inference Explainability](./recipes/m-monitoring/explainability.md)
- [Hash Chain Audit](./recipes/m-monitoring/hash-chain-audit.md)
- [Cost Tracking](./recipes/m-monitoring/cost-tracking.md)
- [Latency Histogram](./recipes/m-monitoring/latency-histogram.md)
- [Drift Detection](./recipes/m-monitoring/drift-detection.md)

# Category N: Speech Recognition

- [Overview](./recipes/n-speech/index.md)
- [Whisper Transcription](./recipes/n-speech/whisper-transcribe.md)
- [Streaming ASR](./recipes/n-speech/whisper-streaming.md)

# Category O: Distributed Computing

- [Overview](./recipes/o-distributed/index.md)
- [Distributed Inference](./recipes/o-distributed/distributed-inference.md)
- [Model Sharding](./recipes/o-distributed/model-sharding.md)

# Category P: Inference Patterns

- [Overview](./recipes/p-inference/index.md)
- [Simple Inference](./recipes/p-inference/simple.md)
- [Speculative Decoding](./recipes/p-inference/speculative-decode.md)
- [KV-Cache Chat](./recipes/p-inference/kv-cache.md)
- [Multi-turn Chat](./recipes/p-inference/multi-turn.md)
- [Tool Use](./recipes/p-inference/tool-use.md)
- [Streaming Tokens](./recipes/p-inference/streaming.md)
- [Adaptive Batching](./recipes/p-inference/adaptive-batch.md)
- [Dynamic Batch SLA](./recipes/p-inference/dynamic-batch-sla.md)
- [Ensemble Inference](./recipes/p-inference/ensemble.md)
- [Model Pipeline](./recipes/p-inference/pipeline.md)
- [Quantized Comparison](./recipes/p-inference/quantized-comparison.md)

# Category Q: Model Serving

- [Overview](./recipes/q-serving/index.md)
- [HTTP Model Server](./recipes/q-serving/http-server.md)
- [A/B Testing](./recipes/q-serving/ab-testing.md)
- [Canary Deploy](./recipes/q-serving/canary-deploy.md)
- [Rate Limiter](./recipes/q-serving/rate-limiter.md)
- [Selection Router](./recipes/q-serving/selection-router.md)

---

# Reference

- [API Documentation](./reference/api.md)
- [Error Handling](./reference/errors.md)
- [Feature Flags](./reference/features.md)

# Appendix

- [Toyota Way Principles](./appendix/toyota-way.md)
- [Recipe QA Checklist](./appendix/qa-checklist.md)
