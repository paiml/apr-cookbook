# Model Qualification

Runs 11 diagnostic gates (smoke tests) to qualify a model for deployment. Each gate produces a Pass/Fail/Skip result with timing. The final report assigns a qualification tier: Smoke (all pass), Qualified (8+ pass), or Rejected.

## CLI Equivalent
```bash
apr qualify model.apr
```

## Key Concepts
- Multi-gate qualification pipeline with timing
- Tiered qualification: Smoke, Qualified, Rejected
- Deployment readiness scoring

## Run
```bash
cargo run --example analysis_qualify
```

## Source
[`examples/analysis/analysis_qualify/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_qualify/main.rs)
