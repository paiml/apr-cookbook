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

# Category B: Binary Bundling

- [Overview](./recipes/b-bundling/index.md)
- [Bundle Static Model](./recipes/b-bundling/bundle-static.md)
- [Bundle Quantized Model](./recipes/b-bundling/bundle-quantized.md)
- [Bundle Encrypted Model](./recipes/b-bundling/bundle-encrypted.md)
- [Static Binary Embedding](./recipes/b-bundling/static-binary.md)
- [Q4 Quantization](./recipes/b-bundling/quantized-q4.md)
- [Signed Models](./recipes/b-bundling/signed.md)
- [Lambda Package](./recipes/b-bundling/lambda-package.md)

# Category C: Continuous Training

- [Overview](./recipes/c-training/index.md)
- [Incremental Training](./recipes/c-training/incremental.md)
- [Online Learning](./recipes/c-training/online-learning.md)
- [Federated Simulation](./recipes/c-training/federated-simulation.md)
- [Curriculum Learning](./recipes/c-training/curriculum.md)

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

# Category F: API Integration

- [Overview](./recipes/f-api/index.md)
- [Model Inference](./recipes/f-api/model-inference.md)
- [Streaming Inference](./recipes/f-api/streaming-inference.md)
- [Batch Inference](./recipes/f-api/batch-inference.md)
- [Health Check](./recipes/f-api/health-check.md)

# Category G: Serverless

- [Overview](./recipes/g-serverless/index.md)
- [Lambda Inference](./recipes/g-serverless/lambda-inference.md)
- [Cold Start Optimization](./recipes/g-serverless/cold-start.md)
- [Edge Functions](./recipes/g-serverless/edge-function.md)
- [Container Image](./recipes/g-serverless/container-image.md)

# Category H: WASM/Browser

- [Overview](./recipes/h-wasm/index.md)
- [Browser Inference](./recipes/h-wasm/browser-inference.md)
- [Web Workers](./recipes/h-wasm/web-worker.md)
- [Progressive Loading](./recipes/h-wasm/progressive-loading.md)
- [WebGPU Acceleration](./recipes/h-wasm/webgpu-acceleration.md)
- [Streaming Compilation](./recipes/h-wasm/streaming-compilation.md)

# Category I: GPU Acceleration

- [Overview](./recipes/i-gpu/index.md)
- [CUDA Inference](./recipes/i-gpu/cuda-inference.md)
- [Tensor Core Optimization](./recipes/i-gpu/tensor-core.md)
- [Multi-GPU Inference](./recipes/i-gpu/multi-gpu.md)
- [Memory Management](./recipes/i-gpu/memory-management.md)

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

# Category L: CLI Tools

- [Overview](./recipes/l-cli/index.md)
- [apr-info](./recipes/l-cli/apr-info.md)
- [apr-bench](./recipes/l-cli/apr-bench.md)
- [apr-convert](./recipes/l-cli/apr-convert.md)
- [apr-serve](./recipes/l-cli/apr-serve.md)

---

# Reference

- [API Documentation](./reference/api.md)
- [Error Handling](./reference/errors.md)
- [Feature Flags](./reference/features.md)

# Appendix

- [Toyota Way Principles](./appendix/toyota-way.md)
- [Recipe QA Checklist](./appendix/qa-checklist.md)
