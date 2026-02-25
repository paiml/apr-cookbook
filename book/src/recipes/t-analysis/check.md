# Pre-Flight Check

Runs a 10-stage sequential pre-flight health check pipeline on an APR model file. Each stage produces a pass/fail/skip result with detail. The final report summarizes overall model readiness for deployment.

## CLI Equivalent
```bash
apr check model.apr
```

## Key Concepts
- Multi-stage deployment readiness pipeline
- Pass/fail/skip health check stages
- Aggregate readiness scoring

## Run
```bash
cargo run --example analysis_check
```

## Source
[`examples/analysis/analysis_check.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_check.rs)
