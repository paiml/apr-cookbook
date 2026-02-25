# Gradual Pruning Schedule

**CLI Equivalent:** `apr prune --method magnitude --schedule cubic --target-sparsity 0.8 --steps 100 model.apr`

## What This Demonstrates

Gradual and cubic pruning schedules that incrementally increase sparsity over training steps. Avoids the accuracy shock of one-shot pruning by allowing the model to adapt between pruning rounds.

## Run

```bash
cargo run --example prune_gradual_schedule
```

## Key APIs

- `CubicSchedule::new(initial, target, start_step, end_step)` -- cubic sparsity ramp
- `.sparsity_at(step)` -- get target sparsity for current training step
- `GradualPruner::new(schedule)` -- pruner that follows the schedule
- `.prune_step(model, step)` -- apply pruning at current step

## Source

[`examples/optimize/prune_gradual_schedule.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/prune_gradual_schedule.rs)
