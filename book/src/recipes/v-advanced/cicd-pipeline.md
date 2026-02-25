# CI/CD Model Pipeline

Simulates a full CI/CD pipeline for model deployment composing six stages: build, validate, QA gates, benchmark, publish, and report. Demonstrates how to enforce quality gates, latency budgets, and size budgets before promoting a model to production.

## CLI Equivalent
```bash
N/A (composes apr qa + apr bench + apr publish)
```

## Key Concepts
- Six-stage deployment pipeline with fail-fast semantics
- Quality gates: latency budget, size budget, accuracy threshold
- Structured pipeline reporting with pass/fail summary

## Run
```bash
cargo run --example cicd_model_pipeline
```

## Source
[`examples/advanced/cicd_model_pipeline.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/advanced/cicd_model_pipeline.rs)
