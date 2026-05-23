# apr-inference-server

> Aprender inference server — GPU model serving with health checks

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/apr-inference-server.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/apr-inference-server.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/apr_inference_server.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/apr_inference_server.rs)

## Run the wrapper

```bash
cargo run --example apr_inference_server
cargo test --example apr_inference_server
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/apr-inference-server.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/apr-inference-server.yaml` by PMAT-065 (centralize-cookbooks).
