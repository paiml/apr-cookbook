# SLERP Merge

**CLI Equivalent:** `apr merge --method slerp --t 0.5 model1.apr model2.apr -o merged.apr`

## What This Demonstrates

Spherical Linear Interpolation (SLERP) merge between two models. Unlike linear interpolation, SLERP traverses the shortest arc on the hypersphere, preserving weight vector norms and producing smoother interpolations.

## Run

```bash
cargo run --example merge_slerp
```

## Key APIs

- `slerp_merge(&m1, &m2, &SlerpConfig::new(t))` -- spherical interpolation at parameter t
- `SlerpConfig::new(t)` -- interpolation factor (0.0 = model1, 1.0 = model2)

## Source

[`examples/optimize/merge_slerp.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/merge_slerp.rs)
