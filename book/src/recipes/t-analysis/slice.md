# Tensor Slice

Extracts and decodes a range of elements from tensor data. Demonstrates index-range slicing, row/column extraction, strided access, hex dumping, dtype conversion with precision-loss analysis, and per-slice statistics.

## CLI Equivalent
```bash
apr tensors model.apr --slice weights --range 10..20
```

## Key Concepts
- Index-range, row, column, and strided tensor slicing
- f32 to f16 conversion with precision loss measurement
- Per-slice descriptive statistics (mean, min, max, sum)

## Run
```bash
cargo run --example analysis_slice
```

## Source
[`examples/analysis/analysis_slice.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_slice.rs)
