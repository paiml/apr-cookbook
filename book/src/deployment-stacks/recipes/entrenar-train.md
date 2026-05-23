# entrenar-train

> Entrenar training pipeline — LoRA, quantization, model merging

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/entrenar-train.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/entrenar-train.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/entrenar_train.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/entrenar_train.rs)

## Run the wrapper

```bash
cargo run --example entrenar_train
cargo test --example entrenar_train
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/entrenar-train.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/entrenar-train.yaml` by PMAT-065 (centralize-cookbooks).
