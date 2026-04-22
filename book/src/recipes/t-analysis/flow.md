# Tensor Flow Visualization

Renders a model's tensor transformation flow as an ASCII pipeline diagram, showing data path through architecture components with parameter counts.

## CLI Equivalent
```bash
apr flow model.apr
```

## Key Concepts
- Parsing tensor names into architecture components
- Building a flow graph from flat tensor metadata
- ASCII flow diagram rendering for architecture visualization

## Run
```bash
cargo run --example analysis_flow
```

## Source
[`examples/analysis/analysis_flow/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/analysis/analysis_flow/main.rs)
