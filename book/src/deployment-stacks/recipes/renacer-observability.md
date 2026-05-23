# renacer-observability

> Renacer observability stack — syscall tracing, Jaeger, Grafana

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/renacer-observability.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/renacer-observability.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/renacer_observability.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/renacer_observability.rs)

## Run the wrapper

```bash
cargo run --example renacer_observability
cargo test --example renacer_observability
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/renacer-observability.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/renacer-observability.yaml` by PMAT-065 (centralize-cookbooks).
