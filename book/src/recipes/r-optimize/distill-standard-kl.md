# Standard KL Distillation

**CLI Equivalent:** `apr distill --method kl --temperature 4.0 --alpha 0.7 teacher.apr student.apr`

## What This Demonstrates

Standard knowledge distillation using KL divergence to transfer knowledge from a large teacher model to a smaller student. Balances soft-label loss (teacher logits) with hard-label loss (ground truth).

## Run

```bash
cargo run --example distill_standard_kl
```

## Key APIs

- `DistillationLoss::new(temperature, alpha)` -- configure temperature scaling and loss weighting
- `.forward(&student_logits, &teacher_logits, &labels)` -- compute combined distillation loss
- `softmax_with_temperature(logits, temp)` -- temperature-scaled softmax

## Source

[`examples/optimize/distill_standard_kl.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/distill_standard_kl.rs)
