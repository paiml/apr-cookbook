# Vulkan Inference (Intel Arc)

Demonstrate wgpu/Vulkan inference on non-NVIDIA hardware (Intel Arc iGPU). Detects GPU backend via platform probing, benchmarks CPU vs simulated Vulkan matmul, and identifies the crossover point where GPU beats CPU.

**Device**: ![cuda](https://img.shields.io/badge/-cuda-76b900) ![wgpu](https://img.shields.io/badge/-wgpu-green)

```bash
cargo run --example gpu_vulkan_inference
```

**Key concepts**: wgpu backend detection, Intel Arc device probing, Vulkan pipeline configuration, CPU/GPU crossover analysis.
