# sovereign-ai-stack

> Sovereign AI lab — GPU inference + distributed workers + observability

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/sovereign-ai-stack.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/sovereign-ai-stack.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/sovereign_ai_stack.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/sovereign_ai_stack.rs)

## Run the wrapper

```bash
cargo run --example sovereign_ai_stack
cargo test --example sovereign_ai_stack
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/sovereign-ai-stack.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/sovereign-ai-stack.yaml` by PMAT-065 (centralize-cookbooks).
