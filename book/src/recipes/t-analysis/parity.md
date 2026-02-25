# CPU vs GPU Parity

Compares CPU and GPU logit outputs using statistical process control metrics: cosine similarity, KL divergence, RMSE, max absolute error, and sigma level. Classifies each comparison as Pass, WarnArgmax, FailDivergent, or FailNan.

## CLI Equivalent
```bash
apr parity model.apr --device cpu,cuda
```

## Key Concepts
- Statistical process control for numerical reproducibility
- Cosine similarity and KL divergence computation
- Sigma-level classification for manufacturing-style quality gates

## Run
```bash
cargo run --example analysis_parity
```

## Source
[`examples/analysis/analysis_parity.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_parity.rs)
