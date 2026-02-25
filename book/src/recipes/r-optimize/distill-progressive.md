# Progressive Distillation

**CLI Equivalent:** `apr distill --method progressive --layers 12 --temperature 4.0 teacher.apr student.apr`

## What This Demonstrates

Progressive layer-wise distillation that transfers knowledge one layer at a time from teacher to student. Combines layer-wise MSE loss on hidden states with final KL divergence loss for more faithful knowledge transfer.

## Run

```bash
cargo run --example distill_progressive
```

## Key APIs

- `ProgressiveDistiller::uniform(num_layers, temperature)` -- create layer-aligned distiller
- `.layer_wise_mse_loss(&student_hidden, &teacher_hidden)` -- intermediate representation matching
- `.combined_loss(mse_loss, kl_loss)` -- weighted combination of layer and output losses

## Source

[`examples/optimize/distill_progressive.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/distill_progressive.rs)
