# Activation Trace

Layer-by-layer statistical analysis of model tensor activations, computing per-layer statistics (mean, std, L2 norm, min, max, NaN/Inf counts) and detecting anomalies such as high-variance spikes, dead layers, and gradient explosion.

## CLI Equivalent
```bash
apr trace model.apr --stats --anomalies
```

## Key Concepts
- Per-layer activation statistics (mean, std, L2 norm)
- Anomaly detection: dead layers, NaN/Inf, gradient explosion
- Statistical process control for model health

## Run
```bash
cargo run --example analysis_trace
```

## Source
[`examples/analysis/analysis_trace.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_trace.rs)
