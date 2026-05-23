# jetson-edge-base

> Jetson Orin Nano base: strip bloat, CUDA apt, Rust, sovereign tools

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/jetson-edge-base.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/jetson-edge-base.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/jetson_edge_base.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/jetson_edge_base.rs)

## Run the wrapper

```bash
cargo run --example jetson_edge_base
cargo test --example jetson_edge_base
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/jetson-edge-base.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/jetson-edge-base.yaml` by PMAT-065 (centralize-cookbooks).
