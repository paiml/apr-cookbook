# realizar-serve

> Realizar model server — GGUF/safetensors serving with GPU acceleration

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/realizar-serve.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/realizar-serve.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/realizar_serve.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/realizar_serve.rs)

## Run the wrapper

```bash
cargo run --example realizar_serve
cargo test --example realizar_serve
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/realizar-serve.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/realizar-serve.yaml` by PMAT-065 (centralize-cookbooks).
