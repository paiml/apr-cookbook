# Mmap Lazy Loading

Memory-mapped lazy loading for models approaching RAM limits. Creates synthetic models (10-100MB), benchmarks eager vs lazy loading, and shows 80% memory savings by loading only the tensors needed for inference.

**Device**: ![cpu](https://img.shields.io/badge/-cpu-lightgrey)

```bash
cargo run --example inference_mmap_lazy_load
```

**Key concepts**: mmap simulation, lazy tensor loading, memory budgeting for 16GB machines, eager vs lazy throughput tradeoff.
