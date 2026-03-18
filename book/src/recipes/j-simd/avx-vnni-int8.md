# AVX-VNNI Int8 Inference

Demonstrate AVX-VNNI (VPDPBUSD) for Int8 inference acceleration on Intel Meteor Lake+ CPUs. Detects VNNI capability at runtime, compares Int8 vs FP32 throughput, and measures quantization error.

**Device**: ![x86_64](https://img.shields.io/badge/-x86__64-blue) ![aarch64](https://img.shields.io/badge/-aarch64-blue)

```bash
cargo run --example simd_avx_vnni_int8_inference --release
```

**Key concepts**: AVX-VNNI detection, symmetric int8 quantization, quantization error analysis, GOPS benchmarking.
