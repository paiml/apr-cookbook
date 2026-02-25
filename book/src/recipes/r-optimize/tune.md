# Memory Planning (Tune)

Plans LoRA/QLoRA fine-tuning configurations by computing optimal rank given a VRAM budget. Compares Full, LoRA, and QLoRA methods across model sizes (1B, 7B, 13B), showing trainable parameters, memory estimates, and speedup.

## CLI Equivalent
```bash
apr tune
```

## Key Concepts
- VRAM budget planning for LoRA/QLoRA fine-tuning
- Trainable parameter count estimation across model sizes
- Method comparison: Full vs LoRA vs QLoRA memory and speedup

## Run
```bash
cargo run --example optimize_tune
```

## Source
[`examples/optimize/optimize_tune.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/optimize_tune.rs)
