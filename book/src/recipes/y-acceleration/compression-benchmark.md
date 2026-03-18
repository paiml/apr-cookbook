# Compression Benchmark

Benchmark LZ4 vs ZSTD (levels 1/3/9) vs uncompressed on model-like data. Measures compression ratio, throughput (GB/s), and decompression latency across random, structured, and sparse payloads.

**Device**: ![cpu](https://img.shields.io/badge/-cpu-lightgrey)

```bash
cargo run --example acceleration_compression_benchmark
```

**Key concepts**: LZ4 vs ZSTD tradeoffs, data-dependent compression ratios, decompression-optimized selection, F9 falsification claim.
