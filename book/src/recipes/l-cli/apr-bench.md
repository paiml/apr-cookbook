# apr-bench

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Benchmark model inference performance.

## Run Command

```bash
cargo run --example cli_apr_bench -- --demo
```

## Code

```rust,ignore
{{#include ../../../../examples/cli/cli_apr_bench.rs}}
```

## Usage

```bash
apr-bench model.apr              # Run benchmark
apr-bench -n 1000 model.apr      # 1000 iterations
apr-bench --batch 32 model.apr   # Batch size 32
```
