# PTX Kernel Analysis

Maps a 7B model inference to its 12-step CUDA PTX kernel execution sequence, computes roofline analysis per kernel, and detects performance issues (low occupancy, excessive shared memory, uncoalesced access patterns).

## CLI Equivalent
```bash
apr ptx_map model.apr && apr ptx_explain model.apr
```

## Key Concepts
- CUDA PTX kernel mapping for transformer inference
- Roofline analysis per kernel (compute vs memory bound)
- Performance issue detection: occupancy, shared memory, coalescing

## Run
```bash
cargo run --example gpu_ptx_analysis
```

## Source
[`examples/gpu/gpu_ptx_analysis.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/gpu/gpu_ptx_analysis.rs)
