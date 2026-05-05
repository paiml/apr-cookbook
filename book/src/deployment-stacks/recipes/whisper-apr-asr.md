# whisper-apr-asr

> Whisper-APR speech recognition — real-time ASR with GPU acceleration

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/whisper-apr-asr.yaml`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/recipes/whisper-apr-asr.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/whisper_apr_asr.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/deployment-stacks/whisper_apr_asr.rs)

## Run the wrapper

```bash
cargo run --example whisper_apr_asr
cargo test --example whisper_apr_asr
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/whisper-apr-asr.yaml \
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/whisper-apr-asr.yaml` by PMAT-065 (centralize-cookbooks).
