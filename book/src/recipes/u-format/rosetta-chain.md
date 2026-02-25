# Rosetta Multi-Step Conversion Chain

**CLI Equivalent:** `apr rosetta chain`

## What This Demonstrates

Builds a multi-step conversion pipeline through intermediate formats. Useful when a direct conversion path does not exist or when you need to apply transformations (quantization, pruning) at specific intermediate stages.

## Run

```bash
cargo run --example format_rosetta_chain
```

## Key APIs

- `RosettaChain::new()` — Create an empty conversion chain
- `.add_step(source_fmt, target_fmt)` — Append a conversion step to the chain
- `.with_transform(stage, transform_fn)` — Insert a tensor transformation at a given stage
- `.execute(input_path, output_path)` — Run the full chain, writing only the final output

## Source

[`examples/format/format_rosetta_chain.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_rosetta_chain.rs)
