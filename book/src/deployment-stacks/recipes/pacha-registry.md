# pacha-registry

> Pacha model/data registry — artifact versioning and distribution

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/pacha-registry.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/pacha-registry.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/pacha_registry.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/pacha_registry.rs)

## Run the wrapper

```bash
cargo run --example pacha_registry
cargo test --example pacha_registry
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/pacha-registry.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/pacha-registry.yaml` by PMAT-065 (centralize-cookbooks).
