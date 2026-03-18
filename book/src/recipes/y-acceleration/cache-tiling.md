# Cache Tiling

Cache-oblivious vs tiled matrix multiplication. Sweeps tile sizes (8-256) to find optimal for L1d/L2/L3 cache hierarchy, compares against trueno SIMD matmul, and shows which cache level dominates at each tile size.

**Device**: ![x86_64](https://img.shields.io/badge/-x86__64-blue)

```bash
cargo run --example acceleration_cache_tiling
```

**Key concepts**: Cache hierarchy (L1d=32KB, L2=2MB, L3=24MB), tiled matmul (6-loop), optimal tile size calculation, trueno comparison.
