# Model Lint

Runs static quality checks on model metadata for best practices. Each lint rule checks a specific aspect of the model (compression, quantization, naming conventions, dtype consistency) and reports findings with severity and actionable suggestions.

## CLI Equivalent
```bash
apr lint model.apr
```

## Key Concepts
- Static quality analysis of model metadata
- Severity-based lint reporting (info, warn, error)
- Best-practice enforcement for compression, naming, dtypes

## Run
```bash
cargo run --example analysis_lint
```

## Source
[`examples/analysis/analysis_lint.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_lint.rs)
