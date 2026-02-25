# Unified Model Run

Mirrors `apr run`: tokenize input, run a tiny 2-layer transformer forward pass, sample tokens autoregressively, decode output, and optionally benchmark throughput.

## CLI Equivalent
```bash
apr run model.apr --prompt "hello" --max-tokens 50
```

## Key Concepts
- End-to-end inference pipeline: tokenize, forward, sample, decode
- Autoregressive token generation with temperature sampling
- Optional throughput benchmarking (tokens/sec, latency)

## Run
```bash
cargo run --example inference_apr_run
```

## Source
[`examples/inference/inference_apr_run.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/inference/inference_apr_run.rs)
