# Ring-Allreduce

Demonstrates the ring-allreduce algorithm for distributed gradient aggregation across worker nodes. Proceeds in two phases (scatter-reduce and allgather) over a logical ring with optimal bandwidth utilization.

## CLI Equivalent
```bash
N/A
```

## Key Concepts
- Scatter-reduce phase: partial gradient accumulation around the ring
- Allgather phase: broadcast fully-reduced chunks to all workers
- Bandwidth-optimal O(N) communication pattern

## Run
```bash
cargo run --example distributed_ring_allreduce
```

## Source
[`examples/distributed/distributed_ring_allreduce/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/distributed/distributed_ring_allreduce/main.rs)
