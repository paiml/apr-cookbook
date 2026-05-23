# A/B Experiment

Controlled A/B experiment comparing two model versions end-to-end: run model A (baseline) and model B (candidate), diff outputs, evaluate metrics, and produce a promotion verdict with statistical significance.

## CLI Equivalent
```bash
N/A (composes apr run + apr diff + apr eval)
```

## Key Concepts
- Baseline vs candidate model comparison
- Statistical significance gating for promotion decisions
- Structured experiment reporting with verdict

## Run
```bash
cargo run --example ab_experiment
```

## Source
[`examples/advanced/ab_experiment/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/advanced/ab_experiment/main.rs)
