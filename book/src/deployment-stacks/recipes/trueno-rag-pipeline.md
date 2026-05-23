# trueno-rag-pipeline

> Trueno RAG pipeline — embedding, retrieval, and vector storage

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/trueno-rag-pipeline.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/trueno-rag-pipeline.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/trueno_rag_pipeline.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/trueno_rag_pipeline.rs)

## Run the wrapper

```bash
cargo run --example trueno_rag_pipeline
cargo test --example trueno_rag_pipeline
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/trueno-rag-pipeline.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/trueno-rag-pipeline.yaml` by PMAT-065 (centralize-cookbooks).
