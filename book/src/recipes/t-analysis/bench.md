# Throughput Benchmarking

**CLI Equivalent:** `apr bench model.apr --batch-sizes 1,4,16,64`

## What This Demonstrates

Throughput benchmarking for APR model inference across multiple batch sizes. Measures latency, throughput (samples/sec), and memory scaling to identify optimal deployment configurations. Produces a batch-size scaling table and ASCII throughput chart.

## Run

```bash
cargo run --example analysis_bench
```

## Key APIs

- `bench_inference(&model_bytes, batch_size, iterations)` -- timed inference with warmup, returns `BenchResult`
- `BenchResult::new(batch_size, latency_ms, memory_bytes)` -- compute throughput from latency
- `simulate_matmul(&weights, &input, rows, cols)` -- simulated matrix multiplication for benchmarking
- `throughput_bar(value, max_value, width)` -- ASCII bar chart rendering

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_bench.rs}}
```

## Source

[`examples/analysis/analysis_bench.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_bench.rs)
