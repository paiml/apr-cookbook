# batuta-agent

> Batuta autonomous agent — Perceive/Reason/Act loop with Jidoka safety

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/batuta-agent.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/batuta-agent.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/batuta_agent.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/batuta_agent.rs)

## Run the wrapper

```bash
cargo run --example batuta_agent
cargo test --example batuta_agent
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/batuta-agent.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/batuta-agent.yaml` by PMAT-065 (centralize-cookbooks).
