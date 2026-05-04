# alimentar-ingest

> Alimentar data pipeline — ingestion, preprocessing, distribution

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/alimentar-ingest.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/alimentar-ingest.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/alimentar_ingest.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/alimentar_ingest.rs)

## Run the wrapper

```bash
cargo run --example alimentar_ingest
cargo test --example alimentar_ingest
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/alimentar-ingest.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/alimentar-ingest.yaml` by PMAT-065 (centralize-cookbooks).
