# Model Evaluation

Evaluates an APR language model by computing perplexity and cross-entropy on synthetic test data. Uses the log-sum-exp trick for numerical stability.

## CLI Equivalent
```bash
apr eval model.apr --dataset test.jsonl
```

## Key Concepts
- Perplexity and cross-entropy computation
- Log-sum-exp trick for numerical stability
- Pass/fail threshold gating on perplexity

## Run
```bash
cargo run --example analysis_eval
```

## Source
[`examples/analysis/analysis_eval.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_eval.rs)
