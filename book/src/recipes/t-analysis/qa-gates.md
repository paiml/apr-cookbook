# 6-Gate Falsifiable QA

**CLI Equivalent:** `apr qa model.apr`

## What This Demonstrates

Runs 6 falsifiable quality gates on an APR model for CI/CD pipelines: Format validation, Integrity (NaN/Inf), Performance (inference time budget), Size (file size budget), Accuracy (simulated evaluation), and Security (suspicious pattern detection). Each gate reports pass/fail with metric and threshold values.

## Run

```bash
cargo run --example analysis_qa_gates
```

## Key APIs

- `run_qa_gates(&model_bytes)` -- run all 6 gates with default config, returns `Vec<GateResult>`
- `run_qa_gates_with_config(&model_bytes, &QaConfig)` -- custom thresholds for inference time, size, accuracy
- `gate_format(&bytes)` -- APR2 magic bytes and minimum header size
- `gate_integrity(&bytes)` -- NaN/Inf scan of tensor payload
- `gate_performance(&bytes, max_ms)` -- simulated inference under time budget
- `gate_security(&bytes)` -- detect ELF/PE signatures, script shebangs, embedded URLs

## Code

```rust,ignore
{{#include ../../../../examples/analysis/analysis_qa_gates/main.rs}}
```

## Source

[`examples/analysis/analysis_qa_gates/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_qa_gates/main.rs)
