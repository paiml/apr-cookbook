# Ensemble Distillation

**CLI Equivalent:** `apr distill --method ensemble --teachers t1.apr,t2.apr,t3.apr --temperature 4.0 student.apr`

## What This Demonstrates

Multi-teacher ensemble distillation that combines knowledge from multiple teacher models into a single student. Teachers contribute uniformly or with custom weights to produce a blended soft-label target.

## Run

```bash
cargo run --example distill_ensemble
```

## Key APIs

- `EnsembleDistiller::uniform(num_teachers, temperature)` -- equal-weight ensemble
- `.combine_teachers(&[teacher_logits])` -- blend teacher outputs into single target
- `.distillation_loss(&student_logits, &combined_target, &labels)` -- compute loss against ensemble

## Source

[`examples/optimize/distill_ensemble.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/distill_ensemble.rs)
