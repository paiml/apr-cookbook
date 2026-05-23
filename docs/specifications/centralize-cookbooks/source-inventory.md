# Source Inventory

Snapshot date: **2026-05-04**. Counts taken via `find` against `~/src/{sovereign-ai-cookbook,alimentar,presentar}` at the spec authoring moment. Re-verify before executing migration with `scripts/centralize-verify.sh --inventory-only`.

---

## sovereign-ai-cookbook

**Repository**: github.com/paiml/sovereign-ai-cookbook
**Local**: `~/src/sovereign-ai-cookbook`
**Character**: Recipe-as-deployment-config. YAMLs are consumed by `forjar` to provision real machines.

### Recipes (14 files, all `recipes/*.yaml`)

| File | Description (from header) |
|------|---------------------------|
| `alimentar-ingest.yaml` | Data ingestion pipeline (alimentar-driven) |
| `apr-inference-server.yaml` | Aprender inference server on GPU machines |
| `batuta-agent.yaml` | Batuta agent service deployment |
| `entrenar-train.yaml` | Training run with entrenar (aprender-train) |
| `jetson-edge-base.yaml` | Jetson edge node base image |
| `pacha-registry.yaml` | Pacha model registry |
| `pepita-sandbox.yaml` | Pepita sandbox runtime |
| `realizar-serve.yaml` | Realizar serve (HTTP inference) |
| `renacer-observability.yaml` | Renacer observability stack |
| `repartir-worker.yaml` | Repartir distributed worker |
| `sovereign-ai-stack.yaml` | Full-stack composition recipe |
| `trueno-db-analytics.yaml` | trueno-db analytics |
| `trueno-rag-pipeline.yaml` | trueno RAG pipeline |
| `whisper-apr-asr.yaml` | Whisper.apr ASR service |

### Stacks (10 directories, each with `forjar.yaml` + `recipes/` + `README.md`)

| Stack | Purpose |
|-------|---------|
| `01-inference` | Single-machine GPU inference |
| `02-training` | Single-machine training |
| `03-rag` | RAG pipeline composition |
| `04-speech` | Speech (Whisper) deployment |
| `05-distributed-inference` | Multi-node inference (repartir) |
| `06-full-stack` | Full sovereign stack on one box |
| `07-data-pipeline` | Data ingestion + transform pipeline |
| `08-observability` | Renacer + log aggregation |
| `09-edge-inference` | Edge (Jetson) inference |
| `09-qwen-coder` | Qwen-coder specific composition (note: dup `09-` prefix to resolve in migration) |

### Machines (1 directory)

| Machine | Files |
|---------|-------|
| `jetson` | systemd units, sysctl conf, base image manifest |

### Other (NOT migrated)

- `certs/` — TLS fixtures for tests; cookbook uses inline self-signed
- `scripts/generate-readme.sh`, `scripts/sync-version.sh` — generation tools tied to source repo layout
- `Makefile` — references stack/recipe paths that change during migration
- `deny.toml`, `CLAUDE.md` — repo-local config

---

## alimentar

**Repository (archived)**: github.com/paiml/alimentar — read-only as of 2026-05-05; `REDIRECT.md` points to apr-cookbook.
**Canonical source (post-2026-05-06)**: `aprender/crates/aprender-data/` (package `aprender-data`, lib `alimentar`, v0.31.2). All future development of the library happens in the APR-MONO monorepo.
**Local (legacy)**: `~/src/alimentar` (still on disk; matches the archived `pre-archive-2026-05` tag).
**Local (canonical)**: `~/src/aprender/crates/aprender-data/`.
**Character**: Library repo with examples + book. Cookbook absorbed only `examples/` and `book/src/`. The library `src/` lives in aprender-data and is consumed via `alimentar = { version = "0.31.2", package = "aprender-data" }` in `apr-cookbook/Cargo.toml`. **`apr-cookbook/examples/data-loading/` is the canonical recipe gallery and MUST be expanded as aprender-data adds APIs** (see Amendment §3 in the parent spec).

### Examples (18 files, all `examples/*.rs`)

| File | Topic |
|------|-------|
| `basic_loading.rs` | Load CSV/JSON/Parquet via Arrow |
| `cli_batch_commands.rs` | CLI batch operations |
| `dataloader_batching.rs` | DataLoader batching patterns |
| `doctest_extraction.rs` | Doctest extraction utility |
| `drift_detection.rs` | Dataset drift detection |
| `federated_split.rs` | Federated learning data split |
| `hub_publishing.rs` | Publish to HuggingFace Hub |
| `prose_detection.rs` | Prose vs code classification |
| `quality_check.rs` | Dataset quality checks |
| `registry_publish.rs` | Publish to alimentar registry |
| `repl_commands.rs` | REPL command set |
| `repl_completer.rs` | REPL tab completion |
| `repl_display_config.rs` | REPL display config |
| `repl_health_status.rs` | REPL health status |
| `repl_session.rs` | REPL session lifecycle |
| `streaming_large.rs` | Streaming large datasets |
| `transforms_pipeline.rs` | Transform pipeline composition |
| `tui_viewer.rs` | TUI dataset viewer |

### Book (103 chapters, `book/src/**/*.md`)

Top-level sections (under `book/src/`):
- `introduction.md` (1 file)
- `100-examples/` — example explanations
- `appendix/` — references
- `architecture/` — design docs
- `backends/` — backend (Arrow, S3, HF) reference
- `cli/` — 7 chapters (overview, convert, schema, head, registry, view, info)
- `dataloader/` — DataLoader semantics
- `dataset/` — Dataset trait & implementations
- `datasets/` — Built-in dataset catalog
- `development/` — 5 chapters (code-review, extreme-tdd, quality-gates, testing, contributing)
- `ecosystem/` — relationships to other paiml crates
- `hf-hub/` — HuggingFace Hub integration (5 chapters)
- `transforms/` — transform reference (filter, drop, cast, normalize, custom, ...)

Migration consolidates the `development/` chapters out — apr-cookbook has its own development/quality-gates docs that supersede these. The other ~95 chapters land under `book/src/data-loading/`.

### Other (NOT migrated)

- `src/`, `Cargo.toml`, `Cargo.lock` — alimentar crate stays published
- `benches/`, `tests/` — exercise the library, not cookbook material
- `pkg/`, `lcov.info`, `mutants.out` — build artifacts
- `docker-compose.yml`, `justfile`, `Makefile` — repo-local

---

## presentar

**Repository**: github.com/paiml/presentar
**Local**: `~/src/presentar`
**Character**: Library + sub-crates (apr-widgets, ald-widgets, prs-runtime). Cookbook absorbs only `examples/` and `book/src/`.

### Examples (28 declarative files across 7 subdirs)

| Subdir | Count | Format | Purpose |
|--------|-------|--------|---------|
| `ald/` | 6 | `.yaml` | Alimentar dataset widgets (data_distribution, data_timeseries, class_balance, data_scatter, data_table_virtualized, data_card_basic) |
| `apr/` | 7 | `.yaml` | APR model widgets (model_card_basic, model_comparison, model_metrics_chart, model_export_preview, shell_autocomplete, model_inference_demo, model_gradient_flow) |
| `charts/` | 3 | `.yaml` | Chart primitives (line_chart_basic, pie_chart_basic, bar_chart_grouped) |
| `dashboards/` | 5 | `.yaml` | Composite dashboards (dataset_explorer, model_comparison_dashboard, experiment_tracker, training_dashboard, confusion_matrix) |
| `edge_cases/` | 2 | `.yaml` | Edge cases (empty_dataset, large_dataset) |
| `prs/` | 5 | `.prs` | Declarative presentar scenes (sentiment-demo, parameter-tuner, data-explorer, image-classifier, minimal) |
| `models/`, `data/` | 0 | — | Empty placeholder dirs (not migrated) |

### Book (121 chapters, `book/src/**/*.md`)

Top-level sections:
- `introduction.md`
- `getting-started/`
- `architecture/`
- `examples/` — chapter-per-example walkthroughs
- `layout/` — layout system
- `quality/` — visual QA, snapshot testing
- `advanced/` — advanced topics (custom widgets, theming)
- `appendix/`
- `development/`
- `ecosystem/`

`development/` and `ecosystem/` overlap with apr-cookbook equivalents and are dropped during migration. Remaining ~110 chapters land under `book/src/visualization/`.

### Other (NOT migrated)

- `src/`, `crates/{apr-widgets,ald-widgets,prs-runtime,...}` — presentar + sub-crates stay published
- `models.dvc`, `dvc.yaml`, `mlflow.yaml` — DVC/MLflow orchestration
- `flake.nix`, `Brewfile`, `docker-bake.hcl`, `Dockerfile` — packaging
- `data/` — fixture data (large)
- `coverage.json`, `BUILD_MANIFEST.json`, `mutants.out` — build artifacts

---

## Migration Volumes

| Source | Recipe-class artifacts | Book chapters | Machine configs |
|--------|------------------------|---------------|-----------------|
| sovereign-ai-cookbook | 14 recipes + 10 stacks | 0 | 1 (jetson) |
| alimentar | 18 examples | ~95 (after dropping dev/ecosystem) | 0 |
| presentar | 28 declarative configs | ~110 (after dropping dev/ecosystem) | 0 |
| **Total** | **70 artifacts** | **~205 chapters** | **1 machine config tree** |

Each total is a **lower bound** for verification: `scripts/centralize-verify.sh` enumerates files under the source paths above and asserts ≥ this count exists in the destination.
