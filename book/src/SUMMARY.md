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
- [Custom Autograd Ops](./recipes/c-training/autograd-custom-ops.md)
- [Gradient Clipping](./recipes/c-training/autograd-gradient-clipping.md)
- [Backprop Visualization](./recipes/c-training/autograd-backprop-viz.md)

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
- [PTX Analysis](./recipes/i-gpu/ptx-analysis.md)
- [Vulkan Inference (Intel Arc)](./recipes/i-gpu/vulkan-inference.md)

# Category J: SIMD Acceleration

- [Overview](./recipes/j-simd/index.md)
- [Matrix Operations](./recipes/j-simd/matrix-operations.md)
- [Vectorized Inference](./recipes/j-simd/vectorized-inference.md)
- [Quantized Operations](./recipes/j-simd/quantized-operations.md)
- [Auto-Vectorization](./recipes/j-simd/auto-vectorization.md)
- [AVX-VNNI Int8 Inference](./recipes/j-simd/avx-vnni-int8.md)

# Category K: Model Distillation

- [Overview](./recipes/k-distillation/index.md)
- [Knowledge Transfer](./recipes/k-distillation/knowledge-transfer.md)
- [Layer Matching](./recipes/k-distillation/layer-matching.md)
- [Pruning-Aware Distillation](./recipes/k-distillation/pruning-aware.md)
- [Quantization-Aware Distillation](./recipes/k-distillation/quantization-aware.md)
- [Structured Pruning](./recipes/k-distillation/structured-pruning.md)
- [Attention Transfer](./recipes/k-distillation/attention-transfer.md)
- [Self-Distillation](./recipes/k-distillation/self-distillation.md)

# Category L: CLI Tools

- [Overview](./recipes/l-cli/index.md)
- [apr-info](./recipes/l-cli/apr-info.md)
- [apr-bench](./recipes/l-cli/apr-bench.md)
- [apr-convert](./recipes/l-cli/apr-convert.md)
- [apr-serve](./recipes/l-cli/apr-serve.md)
- [apr-diff](./recipes/l-cli/apr-diff.md)
- [apr-tui](./recipes/l-cli/apr-tui.md)
- [apr-decrypt](./recipes/l-cli/apr-decrypt.md)
- [apr-diagnose](./recipes/l-cli/apr-diagnose.md)
- [apr-list](./recipes/l-cli/apr-list.md)
- [apr-rm](./recipes/l-cli/apr-rm.md)
- [apr-runs](./recipes/l-cli/apr-runs.md)
- [apr-tokenize](./recipes/l-cli/apr-tokenize.md)
- [apr-ptx-map](./recipes/l-cli/apr-ptx-map.md)

# Category M: Inference Monitoring

- [Overview](./recipes/m-monitoring/index.md)
- [Inference Explainability](./recipes/m-monitoring/explainability.md)
- [Hash Chain Audit](./recipes/m-monitoring/hash-chain-audit.md)
- [Cost Tracking](./recipes/m-monitoring/cost-tracking.md)
- [Latency Histogram](./recipes/m-monitoring/latency-histogram.md)
- [Drift Detection](./recipes/m-monitoring/drift-detection.md)
- [Headless cbtop](./recipes/m-monitoring/cbtop-headless.md)
- [Energy Estimation](./recipes/m-monitoring/energy-estimation.md)
- [Memory Profiler](./recipes/m-monitoring/memory-profiler.md)

# Category N: Speech Recognition

- [Overview](./recipes/n-speech/index.md)
- [Whisper Transcription](./recipes/n-speech/whisper-transcribe.md)
- [Streaming ASR](./recipes/n-speech/whisper-streaming.md)
- [Voice Activity Detection](./recipes/n-speech/vad.md)
- [Speaker Diarization](./recipes/n-speech/diarization.md)
- [Multilingual Identification](./recipes/n-speech/multilingual.md)

# Category O: Distributed Computing

- [Overview](./recipes/o-distributed/index.md)
- [Distributed Inference](./recipes/o-distributed/distributed-inference.md)
- [Model Sharding](./recipes/o-distributed/model-sharding.md)
- [Ring AllReduce](./recipes/o-distributed/ring-allreduce.md)
- [Pipeline Parallelism](./recipes/o-distributed/pipeline-parallel.md)
- [Gossip Protocol](./recipes/o-distributed/gossip-protocol.md)

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
- [APR Run](./recipes/p-inference/apr-run.md)
- [Mmap Lazy Loading](./recipes/p-inference/mmap-lazy-load.md)

# Category Q: Model Serving

- [Overview](./recipes/q-serving/index.md)
- [HTTP Model Server](./recipes/q-serving/http-server.md)
- [A/B Testing](./recipes/q-serving/ab-testing.md)
- [Canary Deploy](./recipes/q-serving/canary-deploy.md)
- [Rate Limiter](./recipes/q-serving/rate-limiter.md)
- [Selection Router](./recipes/q-serving/selection-router.md)

# Category R: Model Optimization

- [Overview](./recipes/r-optimize/index.md)
- [Full Pipeline](./recipes/r-optimize/full-pipeline.md)
- [LoRA Fine-tuning](./recipes/r-optimize/finetune-lora.md)
- [QLoRA Fine-tuning](./recipes/r-optimize/finetune-qlora.md)
- [Adapter Merge](./recipes/r-optimize/finetune-merge-adapter.md)
- [VRAM Planning](./recipes/r-optimize/finetune-plan-vram.md)
- [Magnitude Pruning](./recipes/r-optimize/prune-magnitude.md)
- [Structured Pruning](./recipes/r-optimize/prune-structured.md)
- [Depth Pruning](./recipes/r-optimize/prune-depth.md)
- [Wanda Pruning](./recipes/r-optimize/prune-wanda.md)
- [Gradual Schedule](./recipes/r-optimize/prune-gradual-schedule.md)
- [Standard KL Distillation](./recipes/r-optimize/distill-standard-kl.md)
- [Progressive Distillation](./recipes/r-optimize/distill-progressive.md)
- [Ensemble Distillation](./recipes/r-optimize/distill-ensemble.md)
- [Distillation Checkpoint](./recipes/r-optimize/distill-checkpoint.md)
- [Average Merge](./recipes/r-optimize/merge-average.md)
- [Weighted Merge](./recipes/r-optimize/merge-weighted.md)
- [SLERP Merge](./recipes/r-optimize/merge-slerp.md)
- [TIES Merge](./recipes/r-optimize/merge-ties.md)
- [DARE Merge](./recipes/r-optimize/merge-dare.md)
- [Hierarchical Merge](./recipes/r-optimize/merge-hierarchical.md)
- [Int4 Quantization](./recipes/r-optimize/quantize-4bit.md)
- [Fake QAT](./recipes/r-optimize/quantize-fake-qat.md)
- [Tune](./recipes/r-optimize/tune.md)

# Category S: Chat Templates

- [Overview](./recipes/s-chat/index.md)
- [ChatML Format](./recipes/s-chat/chatml.md)
- [LLaMA 2 Format](./recipes/s-chat/llama2.md)
- [Mistral Format](./recipes/s-chat/mistral.md)
- [Multi-Format Detection](./recipes/s-chat/multi-format.md)
- [Injection Defense](./recipes/s-chat/injection-defense.md)

# Category T: Model Analysis

- [Overview](./recipes/t-analysis/index.md)
- [Inspect](./recipes/t-analysis/inspect.md)
- [Validate](./recipes/t-analysis/validate.md)
- [Diff](./recipes/t-analysis/diff.md)
- [Bench](./recipes/t-analysis/bench.md)
- [Profile](./recipes/t-analysis/profile.md)
- [QA Gates](./recipes/t-analysis/qa-gates.md)
- [Oracle](./recipes/t-analysis/oracle.md)
- [Canary](./recipes/t-analysis/canary.md)
- [Tree](./recipes/t-analysis/tree.md)
- [Hex](./recipes/t-analysis/hex.md)
- [Explain](./recipes/t-analysis/explain.md)
- [Trace](./recipes/t-analysis/trace.md)
- [Eval](./recipes/t-analysis/eval.md)
- [Flow](./recipes/t-analysis/flow.md)
- [Lint](./recipes/t-analysis/lint.md)
- [Check](./recipes/t-analysis/check.md)
- [Debug](./recipes/t-analysis/debug.md)
- [Parity](./recipes/t-analysis/parity.md)
- [Qualify](./recipes/t-analysis/qualify.md)
- [Compare HuggingFace](./recipes/t-analysis/compare-hf.md)
- [Probar](./recipes/t-analysis/probar.md)
- [Tensors](./recipes/t-analysis/tensors.md)
- [Slice](./recipes/t-analysis/slice.md)
- [QA Capability](./recipes/t-analysis/qa-capability.md)
- [Model Fingerprint](./recipes/t-analysis/model-fingerprint.md)

# Category U: Format Operations

- [Overview](./recipes/u-format/index.md)
- [Import from HuggingFace](./recipes/u-format/import-hf.md)
- [Export SafeTensors](./recipes/u-format/export-safetensors.md)
- [Export GGUF](./recipes/u-format/export-gguf.md)
- [Rosetta Convert](./recipes/u-format/rosetta-convert.md)
- [Rosetta Chain](./recipes/u-format/rosetta-chain.md)
- [Rosetta Verify](./recipes/u-format/rosetta-verify.md)
- [Convert + Quantize](./recipes/u-format/convert-quantize.md)
- [Publish](./recipes/u-format/publish.md)
- [Pull + Cache](./recipes/u-format/pull-cache.md)
- [Batch Export](./recipes/u-format/batch-export.md)
- [Migration Pipeline](./recipes/u-format/migration-pipeline.md)

# Category V: Advanced Pipelines

- [Overview](./recipes/v-advanced/index.md)
- [Model Showcase](./recipes/v-advanced/showcase.md)
- [CI/CD Pipeline](./recipes/v-advanced/cicd-pipeline.md)
- [A/B Experiment](./recipes/v-advanced/ab-experiment.md)
- [Debug-Fix Loop](./recipes/v-advanced/debug-fix-loop.md)
- [Compliance Audit](./recipes/v-advanced/compliance-audit.md)

# Category Y: Acceleration

- [Overview](./recipes/y-acceleration/index.md)
- [Autotuner](./recipes/y-acceleration/autotuner.md)
- [Kernel Fusion](./recipes/y-acceleration/kernel-fusion.md)
- [Memory-Mapped Inference](./recipes/y-acceleration/mmap-inference.md)
- [Quantized MatMul](./recipes/y-acceleration/quantized-matmul.md)
- [Compression Benchmark](./recipes/y-acceleration/compression-benchmark.md)
- [Cache Tiling](./recipes/y-acceleration/cache-tiling.md)

---

# Deployment Stacks

- [Overview](./deployment-stacks/overview.md)
- [Recipes](./deployment-stacks/recipes/index.md)
  - [alimentar-ingest](./deployment-stacks/recipes/alimentar-ingest.md)
  - [apr-inference-server](./deployment-stacks/recipes/apr-inference-server.md)
  - [batuta-agent](./deployment-stacks/recipes/batuta-agent.md)
  - [entrenar-train](./deployment-stacks/recipes/entrenar-train.md)
  - [jetson-edge-base](./deployment-stacks/recipes/jetson-edge-base.md)
  - [pacha-registry](./deployment-stacks/recipes/pacha-registry.md)
  - [pepita-sandbox](./deployment-stacks/recipes/pepita-sandbox.md)
  - [realizar-serve](./deployment-stacks/recipes/realizar-serve.md)
  - [renacer-observability](./deployment-stacks/recipes/renacer-observability.md)
  - [repartir-worker](./deployment-stacks/recipes/repartir-worker.md)
  - [sovereign-ai-stack](./deployment-stacks/recipes/sovereign-ai-stack.md)
  - [trueno-db-analytics](./deployment-stacks/recipes/trueno-db-analytics.md)
  - [trueno-rag-pipeline](./deployment-stacks/recipes/trueno-rag-pipeline.md)
  - [whisper-apr-asr](./deployment-stacks/recipes/whisper-apr-asr.md)
- [Stacks](./deployment-stacks/stacks/index.md)
  - [01 Inference](./deployment-stacks/stacks/01-inference.md)
  - [02 Training](./deployment-stacks/stacks/02-training.md)
  - [03 RAG](./deployment-stacks/stacks/03-rag.md)
  - [04 Speech](./deployment-stacks/stacks/04-speech.md)
  - [05 Distributed Inference](./deployment-stacks/stacks/05-distributed-inference.md)
  - [06 Full Stack](./deployment-stacks/stacks/06-full-stack.md)
  - [07 Data Pipeline](./deployment-stacks/stacks/07-data-pipeline.md)
  - [08 Observability](./deployment-stacks/stacks/08-observability.md)
  - [09 Edge Inference](./deployment-stacks/stacks/09-edge-inference.md)
  - [10 Qwen-Coder](./deployment-stacks/stacks/10-qwen-coder.md)
- [Machines](./deployment-stacks/machines/index.md)
  - [Jetson](./deployment-stacks/machines/jetson.md)
- [forjar Integration](./deployment-stacks/forjar-integration.md)

---

# Reference

- [API Documentation](./reference/api.md)
- [Error Handling](./reference/errors.md)
- [Feature Flags](./reference/features.md)

# Appendix

- [Toyota Way Principles](./appendix/toyota-way.md)
- [Recipe QA Checklist](./appendix/qa-checklist.md)
