# QA Capability Check

Gate 0 pre-flight check: validates that hardware supports a model's required operations before loading weights. Prevents wasted time loading large models onto hardware that cannot run them.

## CLI Equivalent
```bash
apr qa_capability model.apr
```

## Key Concepts
- Hardware capability detection and op-set matching
- Pre-load validation to avoid wasted resource allocation
- Pass/fail/partial capability classification

## Run
```bash
cargo run --example analysis_qa_capability
```

## Source
[`examples/analysis/analysis_qa_capability.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_qa_capability.rs)
