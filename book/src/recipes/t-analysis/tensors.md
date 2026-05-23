# Tensor Listing

Lists all tensors in a model file with shape, dtype, size, and optional statistics (mean, std, min, max, NaN count, sparsity). Prints a compact table sorted by size with a total summary and dtype breakdown.

## CLI Equivalent
```bash
apr tensors model.apr --stats
```

## Key Concepts
- Tensor enumeration with shape, dtype, and size metadata
- Per-tensor descriptive statistics and sparsity analysis
- Size-sorted tabular display with dtype breakdown

## Run
```bash
cargo run --example analysis_tensors
```

## Source
[`examples/analysis/analysis_tensors/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_tensors/main.rs)
