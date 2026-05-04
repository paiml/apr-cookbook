# repartir-worker

> Repartir distributed execution worker — TCP/TLS task executor

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/repartir-worker.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/repartir-worker.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/repartir_worker.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/repartir_worker.rs)

## Run the wrapper

```bash
cargo run --example repartir_worker
cargo test --example repartir_worker
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/repartir-worker.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/repartir-worker.yaml` by PMAT-065 (centralize-cookbooks).
