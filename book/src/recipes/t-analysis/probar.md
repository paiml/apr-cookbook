# Activation Probar

Exports per-layer activation statistics (histogram, mean, std, min, max, kurtosis) and compares two snapshots to detect regressions. Regression criteria: mean shift > 0.1, std change > 20%, or histogram KL divergence > 0.5.

## CLI Equivalent
```bash
apr probar model.apr --layers all --compare baseline.json
```

## Key Concepts
- Per-layer activation histogram and statistical snapshots
- KL divergence for distribution comparison
- Regression detection between model versions

## Run
```bash
cargo run --example analysis_probar
```

## Source
[`examples/analysis/analysis_probar.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_probar.rs)
