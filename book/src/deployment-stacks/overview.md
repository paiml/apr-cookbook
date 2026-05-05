# Deployment Stacks

The **Deployment Stacks** category contains declarative recipes for provisioning the sovereign AI stack on real machines. Each recipe is a YAML file consumed by [forjar](https://github.com/paiml/forjar) (a Rust-native infrastructure-as-code tool); cookbook users get a Rust loader/validator wrapper that exercises the YAML's schema without running real provisioning.

This category was migrated into the cookbook from the now-archived `sovereign-ai-cookbook` repository as part of the [centralize-cookbooks spec](https://github.com/paiml/apr-cookbook/blob/main/docs/specifications/centralize-cookbooks.md) (PMAT-065).

## Layout

```
examples/deployment-stacks/
├── recipes/                  # 14 YAML deployment recipes
│   ├── apr-inference-server.yaml
│   ├── entrenar-train.yaml
│   └── ... (12 more)
├── stacks/                   # 10 multi-recipe compositions
│   ├── 01-inference/
│   ├── 02-training/
│   └── ... (8 more)
└── *.rs                      # 14 Rust loader wrappers (one per recipe)

examples/machines/
└── jetson/                   # Edge machine provisioning configs
```

## Why Rust wrappers?

Cookbook policy requires every example to be runnable and testable. Sovereign recipes are declarative configs — they don't execute on their own, and full execution requires a real target machine plus root privileges. The Rust wrapper bridges the gap:

- It loads the YAML via `include_str!` (no runtime I/O dependency)
- It parses with `serde_yaml` and validates required fields via the shared helper at `src/deployment_stack.rs`
- It exits 0 if the recipe is well-formed, prints the recipe name + version + input count
- Its `#[test]` block asserts schema invariants — so a sovereign-side schema break trips a cookbook test

The wrapper is graded against the new [`recipe-iiur-config-v1.yaml`](https://github.com/paiml/apr-cookbook/blob/main/contracts/recipe-iiur-config-v1.yaml) contract, a sibling to the standard IIUR contract that relaxes the runtime obligations for declarative-config recipes.

## Recipe inventory

| Recipe | Purpose | Wrapper example |
|--------|---------|-----------------|
| [alimentar-ingest](recipes/alimentar-ingest.md) | Data ingestion via alimentar | `cargo run --example alimentar_ingest` |
| [apr-inference-server](recipes/apr-inference-server.md) | GPU model serving | `cargo run --example apr_inference_server` |
| [batuta-agent](recipes/batuta-agent.md) | Batuta agent service | `cargo run --example batuta_agent` |
| [entrenar-train](recipes/entrenar-train.md) | Training run via entrenar | `cargo run --example entrenar_train` |
| [jetson-edge-base](recipes/jetson-edge-base.md) | Jetson edge node base image | `cargo run --example jetson_edge_base` |
| [pacha-registry](recipes/pacha-registry.md) | Model registry service | `cargo run --example pacha_registry` |
| [pepita-sandbox](recipes/pepita-sandbox.md) | Sandbox runtime | `cargo run --example pepita_sandbox` |
| [realizar-serve](recipes/realizar-serve.md) | HTTP inference server | `cargo run --example realizar_serve` |
| [renacer-observability](recipes/renacer-observability.md) | Observability stack | `cargo run --example renacer_observability` |
| [repartir-worker](recipes/repartir-worker.md) | Distributed worker | `cargo run --example repartir_worker` |
| [sovereign-ai-stack](recipes/sovereign-ai-stack.md) | Full-stack composition | `cargo run --example sovereign_ai_stack` |
| [trueno-db-analytics](recipes/trueno-db-analytics.md) | trueno-db analytics | `cargo run --example trueno_db_analytics` |
| [trueno-rag-pipeline](recipes/trueno-rag-pipeline.md) | trueno RAG pipeline | `cargo run --example trueno_rag_pipeline` |
| [whisper-apr-asr](recipes/whisper-apr-asr.md) | Whisper.apr ASR service | `cargo run --example whisper_apr_asr` |

## Stack inventory

Stacks are multi-recipe compositions that wire several deployment recipes together onto one or more machines:

- [01 — Single-Machine Inference](stacks/01-inference.md)
- [02 — Single-Machine Training](stacks/02-training.md)
- [03 — RAG Pipeline](stacks/03-rag.md)
- [04 — Speech (Whisper)](stacks/04-speech.md)
- [05 — Distributed Inference](stacks/05-distributed-inference.md)
- [06 — Full Stack](stacks/06-full-stack.md)
- [07 — Data Pipeline](stacks/07-data-pipeline.md)
- [08 — Observability](stacks/08-observability.md)
- [09 — Edge Inference](stacks/09-edge-inference.md)
- [10 — Qwen-Coder](stacks/10-qwen-coder.md)

## Machines

- [Jetson](machines/jetson.md) — NVIDIA Jetson edge provisioning

## forjar integration

These recipes are consumed by `forjar` for actual deployment:

```bash
forjar apply examples/deployment-stacks/recipes/apr-inference-server.yaml \
  --inputs model_source=TheBloke/Llama-2-7B-GGUF \
  --inputs port=8080
```

See [forjar Integration](forjar-integration.md) for the full execution model.
