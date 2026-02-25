# Headless Performance Monitor (cbtop)

Headless inference monitoring: per-brick (layer) timing collection, hardware inventory, throughput computation, latency percentiles, and performance budget utilization -- all without a TUI.

## CLI Equivalent
```bash
apr cbtop --headless --json
```

## Key Concepts
- Per-brick timing and budget utilization analysis
- Throughput and latency percentile computation (p50, p95, p99)
- Hardware inventory detection (CPU, memory, GPU)

## Run
```bash
cargo run --example cbtop_headless
```

## Source
[`examples/monitoring/cbtop_headless.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/monitoring/cbtop_headless.rs)
