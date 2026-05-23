# pepita-sandbox

> Pepita kernel sandbox — io_uring-based process isolation

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/pepita-sandbox.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/pepita-sandbox.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/pepita_sandbox.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/pepita_sandbox.rs)

## Run the wrapper

```bash
cargo run --example pepita_sandbox
cargo test --example pepita_sandbox
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/pepita-sandbox.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/pepita-sandbox.yaml` by PMAT-065 (centralize-cookbooks).
