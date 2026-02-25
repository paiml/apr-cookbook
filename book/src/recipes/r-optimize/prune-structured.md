# Structured Width Pruning

**CLI Equivalent:** `apr prune --method structured --width-ratio 0.75 model.apr`

## What This Demonstrates

Structured width pruning (Minitron-style) that removes entire neurons/channels rather than individual weights. Produces genuinely smaller weight matrices that run faster without sparse tensor support.

## Run

```bash
cargo run --example prune_structured
```

## Key APIs

- `prune_structured(tensor, width_ratio)` -- remove lowest-importance columns
- `importance_score(tensor, axis)` -- rank neurons by L2 norm
- `reshape_layers(model, new_width)` -- adjust downstream layers for reduced width

## Source

[`examples/optimize/prune_structured.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/optimize/prune_structured.rs)
