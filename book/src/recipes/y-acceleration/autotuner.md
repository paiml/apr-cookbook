# Autotuner

Searches for optimal kernel configurations (tile size, unroll factor, vectorization width) for matrix multiply on a given hardware target using exhaustive, random, and Bayesian-inspired search strategies.

## CLI Equivalent
```bash
N/A
```

## Key Concepts
- Hardware-aware kernel configuration search
- Exhaustive, random, and Bayesian search strategies
- Tile size, unroll factor, and vectorization width tuning

## Run
```bash
cargo run --example acceleration_autotuner
```

## Source
[`examples/acceleration/acceleration_autotuner/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/acceleration/acceleration_autotuner/main.rs)
