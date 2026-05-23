# trueno-db-analytics

> Trueno-DB analytics database — columnar storage with vector support

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/trueno-db-analytics.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/trueno-db-analytics.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/trueno_db_analytics.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/trueno_db_analytics.rs)

## Run the wrapper

```bash
cargo run --example trueno_db_analytics
cargo test --example trueno_db_analytics
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/trueno-db-analytics.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/trueno-db-analytics.yaml` by PMAT-065 (centralize-cookbooks).
