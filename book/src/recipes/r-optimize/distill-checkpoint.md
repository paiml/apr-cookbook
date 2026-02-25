# Distillation with Checkpointing

**CLI Equivalent:** `apr distill --method kl --checkpoint-dir ./checkpoints --save-every 500 teacher.apr student.apr`

## What This Demonstrates

Distillation training loop with periodic checkpoint saving and resume capability. Enables long-running distillation jobs to survive interruptions and resume from the last saved state.

## Run

```bash
cargo run --example distill_checkpoint
```

## Key APIs

- `DistillationLoss::new(temperature, alpha)` -- configure distillation loss
- `Checkpoint::save(path, model, optimizer, step)` -- serialize training state
- `Checkpoint::load(path)` -- restore model, optimizer, and step counter
- `CheckpointSchedule::every(n_steps)` -- configure save frequency

## Source

[`examples/optimize/distill_checkpoint.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/distill_checkpoint.rs)
