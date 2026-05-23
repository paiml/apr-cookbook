# Model Showcase

End-to-end model lifecycle demo: create a model from scratch, inspect its internals, validate integrity, benchmark throughput, convert formats, and compare tensors.

## CLI Equivalent
```bash
N/A (composes apr inspect + apr bench + apr convert)
```

## Key Concepts
- Full model lifecycle: create, inspect, validate, benchmark, convert
- Quality validation at every pipeline stage
- Format conversion with tensor comparison

## Run
```bash
cargo run --example model_showcase
```

## Source
[`examples/advanced/model_showcase/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/advanced/model_showcase/main.rs)
