# apr-bench

Benchmark APR model inference performance.

## Usage

```bash
cargo run --example apr_bench --release -- [OPTIONS] [FILE]
```

## Options

| Option | Description | Default |
|--------|-------------|---------|
| `FILE` | Path to APR model file | - |
| `--demo` | Use demo mode | false |
| `-i, --iterations` | Number of iterations | 1000 |
| `-w, --warmup` | Warmup iterations | 100 |
| `-b, --batch-size` | Batch size | 1 |

## Examples

```bash
# Benchmark with defaults
cargo run --example apr_bench --release -- model.apr

# Demo mode
cargo run --example apr_bench --release -- --demo

# Custom iterations
cargo run --example apr_bench --release -- --iterations 10000 model.apr
```

## Output

```
=== APR Cookbook: Model Benchmark ===

Model: sentiment-classifier.apr

Model size:    1048576 bytes
Batch size:    1
Warmup:        100 iterations
Benchmark:     1000 iterations

Running benchmark...

=== Benchmark Results ===

Iterations:    1000
Total time:    45.23ms

Latency:
  Mean:        45.23μs
  Min:         42.10μs
  Max:         112.50μs
  P50:         44.00μs
  P99:         89.00μs

Throughput:    22109.22 inferences/sec
```

## Interpreting Results

| Metric | Good | Acceptable | Needs Work |
|--------|------|------------|------------|
| P50 latency | <1ms | <10ms | >10ms |
| P99 latency | <5ms | <50ms | >50ms |
| Throughput | >1000/s | >100/s | <100/s |
