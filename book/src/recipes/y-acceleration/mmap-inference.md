# Memory-Mapped Inference

Memory-mapped model loading vs eager loading. Memory-mapped access provides near-instant file open, demand-paged reads, and reduced resident memory when only a subset of tensors is accessed during inference.

## CLI Equivalent
```bash
N/A
```

## Key Concepts
- Memory-mapped vs eager model loading comparison
- Demand paging for reduced resident memory
- Page fault tracking to verify access patterns

## Run
```bash
cargo run --example acceleration_mmap_inference
```

## Source
[`examples/acceleration/acceleration_mmap_inference/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/acceleration/acceleration_mmap_inference/main.rs)
