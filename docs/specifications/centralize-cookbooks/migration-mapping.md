# Migration Mapping

Source artifact → destination path. This is the authoritative manifest used by `scripts/centralize-verify.sh`. Edits here REQUIRE a corresponding update to that script.

Conventions:
- `→` indicates the file is moved verbatim
- `↦` indicates the file is moved AND a Rust wrapper is generated alongside it
- Paths under `apr-cookbook/` are relative to the repo root

---

## sovereign-ai-cookbook

### Recipes (14 files, each gets a Rust loader wrapper)

```
sovereign-ai-cookbook/recipes/alimentar-ingest.yaml
  ↦ examples/deployment-stacks/recipes/alimentar-ingest.yaml
  ↦ examples/deployment-stacks/alimentar_ingest.rs              [generated wrapper]

sovereign-ai-cookbook/recipes/apr-inference-server.yaml
  ↦ examples/deployment-stacks/recipes/apr-inference-server.yaml
  ↦ examples/deployment-stacks/apr_inference_server.rs

sovereign-ai-cookbook/recipes/batuta-agent.yaml
  ↦ examples/deployment-stacks/recipes/batuta-agent.yaml
  ↦ examples/deployment-stacks/batuta_agent.rs

sovereign-ai-cookbook/recipes/entrenar-train.yaml
  ↦ examples/deployment-stacks/recipes/entrenar-train.yaml
  ↦ examples/deployment-stacks/entrenar_train.rs

sovereign-ai-cookbook/recipes/jetson-edge-base.yaml
  ↦ examples/deployment-stacks/recipes/jetson-edge-base.yaml
  ↦ examples/deployment-stacks/jetson_edge_base.rs

sovereign-ai-cookbook/recipes/pacha-registry.yaml
  ↦ examples/deployment-stacks/recipes/pacha-registry.yaml
  ↦ examples/deployment-stacks/pacha_registry.rs

sovereign-ai-cookbook/recipes/pepita-sandbox.yaml
  ↦ examples/deployment-stacks/recipes/pepita-sandbox.yaml
  ↦ examples/deployment-stacks/pepita_sandbox.rs

sovereign-ai-cookbook/recipes/realizar-serve.yaml
  ↦ examples/deployment-stacks/recipes/realizar-serve.yaml
  ↦ examples/deployment-stacks/realizar_serve.rs

sovereign-ai-cookbook/recipes/renacer-observability.yaml
  ↦ examples/deployment-stacks/recipes/renacer-observability.yaml
  ↦ examples/deployment-stacks/renacer_observability.rs

sovereign-ai-cookbook/recipes/repartir-worker.yaml
  ↦ examples/deployment-stacks/recipes/repartir-worker.yaml
  ↦ examples/deployment-stacks/repartir_worker.rs

sovereign-ai-cookbook/recipes/sovereign-ai-stack.yaml
  ↦ examples/deployment-stacks/recipes/sovereign-ai-stack.yaml
  ↦ examples/deployment-stacks/sovereign_ai_stack.rs

sovereign-ai-cookbook/recipes/trueno-db-analytics.yaml
  ↦ examples/deployment-stacks/recipes/trueno-db-analytics.yaml
  ↦ examples/deployment-stacks/trueno_db_analytics.rs

sovereign-ai-cookbook/recipes/trueno-rag-pipeline.yaml
  ↦ examples/deployment-stacks/recipes/trueno-rag-pipeline.yaml
  ↦ examples/deployment-stacks/trueno_rag_pipeline.rs

sovereign-ai-cookbook/recipes/whisper-apr-asr.yaml
  ↦ examples/deployment-stacks/recipes/whisper-apr-asr.yaml
  ↦ examples/deployment-stacks/whisper_apr_asr.rs
```

### Stacks (10 directories, verbatim)

```
sovereign-ai-cookbook/stacks/01-inference/
  → examples/deployment-stacks/stacks/01-inference/

sovereign-ai-cookbook/stacks/02-training/
  → examples/deployment-stacks/stacks/02-training/

sovereign-ai-cookbook/stacks/03-rag/
  → examples/deployment-stacks/stacks/03-rag/

sovereign-ai-cookbook/stacks/04-speech/
  → examples/deployment-stacks/stacks/04-speech/

sovereign-ai-cookbook/stacks/05-distributed-inference/
  → examples/deployment-stacks/stacks/05-distributed-inference/

sovereign-ai-cookbook/stacks/06-full-stack/
  → examples/deployment-stacks/stacks/06-full-stack/

sovereign-ai-cookbook/stacks/07-data-pipeline/
  → examples/deployment-stacks/stacks/07-data-pipeline/

sovereign-ai-cookbook/stacks/08-observability/
  → examples/deployment-stacks/stacks/08-observability/

sovereign-ai-cookbook/stacks/09-edge-inference/
  → examples/deployment-stacks/stacks/09-edge-inference/

sovereign-ai-cookbook/stacks/09-qwen-coder/
  → examples/deployment-stacks/stacks/10-qwen-coder/   [RENAMED to resolve dup `09-` prefix]
```

### Machines (1 tree)

```
sovereign-ai-cookbook/machines/jetson/
  → examples/machines/jetson/
```

---

## alimentar

### Examples (18 files, verbatim)

```
alimentar/examples/basic_loading.rs           → examples/data-loading/basic_loading.rs
alimentar/examples/cli_batch_commands.rs      → examples/data-loading/cli_batch_commands.rs
alimentar/examples/dataloader_batching.rs     → examples/data-loading/dataloader_batching.rs
alimentar/examples/doctest_extraction.rs      → examples/data-loading/doctest_extraction.rs
alimentar/examples/drift_detection.rs         → examples/data-loading/drift_detection.rs
alimentar/examples/federated_split.rs         → examples/data-loading/federated_split.rs
alimentar/examples/hub_publishing.rs          → examples/data-loading/hub_publishing.rs
alimentar/examples/prose_detection.rs         → examples/data-loading/prose_detection.rs
alimentar/examples/quality_check.rs           → examples/data-loading/quality_check.rs
alimentar/examples/registry_publish.rs        → examples/data-loading/registry_publish.rs
alimentar/examples/repl_commands.rs           → examples/data-loading/repl_commands.rs
alimentar/examples/repl_completer.rs          → examples/data-loading/repl_completer.rs
alimentar/examples/repl_display_config.rs     → examples/data-loading/repl_display_config.rs
alimentar/examples/repl_health_status.rs      → examples/data-loading/repl_health_status.rs
alimentar/examples/repl_session.rs            → examples/data-loading/repl_session.rs
alimentar/examples/streaming_large.rs         → examples/data-loading/streaming_large.rs
alimentar/examples/transforms_pipeline.rs     → examples/data-loading/transforms_pipeline.rs
alimentar/examples/tui_viewer.rs              → examples/data-loading/tui_viewer.rs
```

Each file gets an **IIUR retrofit pass**: add the `RecipeContext::new(...)` boilerplate, prepend the IIUR doc header with `Contract: contracts/recipe-iiur-v1.yaml`, add an arXiv/DOI citation in the doc header (use the originating alimentar paper for now: arXiv:2502.xxxxx — placeholder, ticket PMAT-066 will resolve), and append a `#[cfg(test)] mod tests` block. See [iiur-conformance.md](iiur-conformance.md) for the retrofit recipe.

### Book chapters (alimentar/book/src/)

```
alimentar/book/src/introduction.md             → book/src/data-loading/introduction.md
alimentar/book/src/100-examples/               → book/src/data-loading/100-examples/
alimentar/book/src/architecture/               → book/src/data-loading/architecture/
alimentar/book/src/backends/                   → book/src/data-loading/backends/
alimentar/book/src/cli/                        → book/src/data-loading/cli/
alimentar/book/src/dataloader/                 → book/src/data-loading/dataloader/
alimentar/book/src/dataset/                    → book/src/data-loading/dataset/
alimentar/book/src/datasets/                   → book/src/data-loading/datasets/
alimentar/book/src/hf-hub/                     → book/src/data-loading/hf-hub/
alimentar/book/src/transforms/                 → book/src/data-loading/transforms/
alimentar/book/src/appendix/                   → book/src/data-loading/appendix/
```

Dropped (overlap with apr-cookbook equivalents):
```
alimentar/book/src/development/   [dropped]
alimentar/book/src/ecosystem/     [dropped]
```

---

## presentar

### Declarative examples (28 files, verbatim subdir layout)

```
presentar/examples/ald/                → examples/visualization/ald/         (6 .yaml)
presentar/examples/apr/                → examples/visualization/apr/         (7 .yaml)
presentar/examples/charts/             → examples/visualization/charts/      (3 .yaml)
presentar/examples/dashboards/         → examples/visualization/dashboards/  (5 .yaml)
presentar/examples/edge_cases/         → examples/visualization/edge_cases/  (2 .yaml)
presentar/examples/prs/                → examples/visualization/prs/         (5 .prs)
```

Plus **one Rust validator wrapper** that exercises every yaml/prs file:
```
[NEW] examples/visualization/load_visualization.rs
```
This single binary loads each declarative file, parses it via `presentar` (dev-dep), asserts schema validity, and serves as the IIUR-graded entry point for the entire `visualization/` category. Per-file Rust wrappers are NOT generated — the visualization corpus is too large and the validation logic is uniform.

### Book chapters (presentar/book/src/)

```
presentar/book/src/introduction.md         → book/src/visualization/introduction.md
presentar/book/src/getting-started/        → book/src/visualization/getting-started/
presentar/book/src/architecture/           → book/src/visualization/architecture/
presentar/book/src/examples/               → book/src/visualization/examples/
presentar/book/src/layout/                 → book/src/visualization/layout/
presentar/book/src/quality/                → book/src/visualization/quality/
presentar/book/src/advanced/               → book/src/visualization/advanced/
presentar/book/src/appendix/               → book/src/visualization/appendix/
```

Dropped:
```
presentar/book/src/development/   [dropped]
presentar/book/src/ecosystem/     [dropped]
```

---

## Rename Ledger

The mapping above is the source of truth, but a few entries differ from a pure name-preserving migration. They are listed here for the verifier:

| Original | New | Reason |
|----------|-----|--------|
| `sovereign-ai-cookbook/stacks/09-qwen-coder/` | `examples/deployment-stacks/stacks/10-qwen-coder/` | Original repo had two `09-` prefixed stacks; renumbered to keep ordering stable |
| `alimentar/book/src/development/*` | _(dropped)_ | Overlaps with apr-cookbook's existing development docs |
| `alimentar/book/src/ecosystem/*` | _(dropped)_ | Will be regenerated as part of post-merge ecosystem update |
| `presentar/book/src/development/*` | _(dropped)_ | Same as above |
| `presentar/book/src/ecosystem/*` | _(dropped)_ | Same as above |

All other entries are name-preserving moves under a new prefix.

---

## Verification Script Contract

`scripts/centralize-verify.sh` MUST:

1. Read this file's mapping (parse the `→` and `↦` lines from the code blocks above)
2. For each `→` entry, assert the destination file exists and its sha256 matches a recorded checksum
3. For each `↦` entry, assert (a) the destination YAML matches the source sha256, AND (b) a sibling `.rs` file exists in the destination and includes the IIUR header with `Contract: contracts/recipe-iiur-v1.yaml` (or `-config-v1.yaml`)
4. For dropped entries, assert no destination exists (catches accidental migration of dropped chapters)
5. Exit nonzero if any assertion fails

The script is part of CI: a PR touching `examples/{deployment-stacks,data-loading,visualization,machines}/` runs it as a required check.
