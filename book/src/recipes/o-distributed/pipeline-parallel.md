# Pipeline Parallelism

Splits model layers across devices and processes micro-batches through a staged pipeline. Compares pipelined vs sequential execution and visualizes the schedule as an ASCII Gantt chart.

## CLI Equivalent
```bash
N/A
```

## Key Concepts
- Layer partitioning across multiple devices
- Micro-batch scheduling through pipeline stages
- Pipelined vs sequential throughput comparison

## Run
```bash
cargo run --example distributed_pipeline_parallel
```

## Source
[`examples/distributed/distributed_pipeline_parallel.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/distributed/distributed_pipeline_parallel.rs)
