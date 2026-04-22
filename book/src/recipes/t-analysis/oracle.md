# Model Family Identification

**CLI Equivalent:** `apr oracle model.apr`

## What This Demonstrates

Identifies model architecture family (Transformer, CNN, RNN, MLP) from weight tensor names and shapes using heuristic pattern matching. Scores confidence by counting pattern hits across tensor naming conventions (e.g., `attn`, `q_proj` for Transformer; `conv`, `bn` for CNN) and reports evidence for each classification signal.

## Run

```bash
cargo run --example analysis_oracle
```

## Key APIs

- `identify_family(&tensor_names, &shapes)` -- classify model into Transformer/CNN/RNN/MLP/Unknown with confidence
- `score_family(&tensor_names, &shapes, patterns)` -- count pattern matches and compute confidence score
- `TRANSFORMER_PATTERNS` / `CNN_PATTERNS` / `RNN_PATTERNS` / `MLP_PATTERNS` -- heuristic pattern lists
- `OracleResult { family, confidence, evidence }` -- classification result with evidence accumulation

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_oracle/main.rs}}
```

## Source

[`examples/analysis/analysis_oracle/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_oracle/main.rs)
