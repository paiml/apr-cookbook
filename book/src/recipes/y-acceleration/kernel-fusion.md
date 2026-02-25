# Kernel Fusion

Combines multiple transformer block operations into a single pass to reduce memory traffic. Models a computation graph, analyzes fusibility, applies fusion rules, and quantifies memory savings.

## CLI Equivalent
```bash
N/A
```

## Key Concepts
- Operator fusion to reduce memory round-trips
- Computation graph analysis for fusibility detection
- Memory traffic savings quantification

## Run
```bash
cargo run --example acceleration_kernel_fusion
```

## Source
[`examples/acceleration/acceleration_kernel_fusion.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/acceleration/acceleration_kernel_fusion.rs)
