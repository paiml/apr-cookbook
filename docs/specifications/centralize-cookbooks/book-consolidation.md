# Book Consolidation

apr-cookbook ships an mdBook (`book/`) that mirrors the recipe catalog. Migration adds two top-level sections — **Data Loading** and **Visualization** — built from the alimentar and presentar books. Sovereign machine configs and deployment stacks are documented in apr-cookbook's existing book under a new **Deployment Stacks** chapter; they don't bring their own mdBook (sovereign-ai-cookbook only had a generated README, not a book).

---

## Current apr-cookbook book structure

(verbatim from `book/src/SUMMARY.md` at spec time; counts approximate)

```
- Introduction
- Quickstart
- Recipe Catalog (24 categories)
- Format Reference (.apr v2)
- Falsification Discipline
- Quality Gates
- References
- Appendices
```

After migration, two new top-level sections slot in after **Recipe Catalog** and before **Format Reference**:

```
- Recipe Catalog
- Data Loading              [NEW — ex-alimentar/book/src]
- Visualization             [NEW — ex-presentar/book/src]
- Deployment Stacks         [NEW — written from scratch, references sovereign YAMLs]
- Format Reference (.apr v2)
- Falsification Discipline
- Quality Gates
- References
- Appendices
```

---

## SUMMARY.md merge plan

### New Data Loading section

Source: `alimentar/book/src/SUMMARY.md` (drops `development/` and `ecosystem/`).

```markdown
# Data Loading

- [Introduction](data-loading/introduction.md)
- [Architecture](data-loading/architecture/index.md)
  - ... (alimentar architecture chapters verbatim, paths re-prefixed)
- [DataLoader](data-loading/dataloader/overview.md)
  - [Batching](data-loading/dataloader/batching.md)
  - [Shuffling](data-loading/dataloader/shuffling.md)
  - [Drop-Last](data-loading/dataloader/drop-last.md)
  - [Iteration Patterns](data-loading/dataloader/iteration-patterns.md)
- [Dataset](data-loading/dataset/overview.md)
- [Datasets Catalog](data-loading/datasets/index.md)
- [Backends](data-loading/backends/overview.md)
- [Transforms](data-loading/transforms/overview.md)
  - [Filter](data-loading/transforms/filter.md)
  - [Drop](data-loading/transforms/drop.md)
  - [Cast](data-loading/transforms/cast.md)
  - [Normalize](data-loading/transforms/normalize.md)
  - [Custom Transforms](data-loading/transforms/custom.md)
- [HuggingFace Hub](data-loading/hf-hub/overview.md)
  - [Importing](data-loading/hf-hub/importing.md)
  - [Publishing](data-loading/hf-hub/publishing.md)
  - [Cache](data-loading/hf-hub/cache.md)
  - [API Reference](data-loading/hf-hub/api-reference.md)
- [CLI Reference](data-loading/cli/overview.md)
  - [convert](data-loading/cli/convert.md)
  - [schema](data-loading/cli/schema.md)
  - [head](data-loading/cli/head.md)
  - [view](data-loading/cli/view.md)
  - [info](data-loading/cli/info.md)
  - [registry](data-loading/cli/registry.md)
- [Examples](data-loading/100-examples/index.md)
- [Appendix](data-loading/appendix/index.md)
```

### New Visualization section

Source: `presentar/book/src/SUMMARY.md` (drops `development/` and `ecosystem/`).

```markdown
# Visualization

- [Introduction](visualization/introduction.md)
- [Getting Started](visualization/getting-started/index.md)
- [Architecture](visualization/architecture/index.md)
- [Layout System](visualization/layout/overview.md)
- [Examples](visualization/examples/index.md)
  - [Charts](visualization/examples/charts.md)
  - [Dashboards](visualization/examples/dashboards.md)
  - [ALD Widgets](visualization/examples/ald.md)
  - [APR Widgets](visualization/examples/apr.md)
  - [PRS Scenes](visualization/examples/prs.md)
  - [Edge Cases](visualization/examples/edge-cases.md)
- [Quality (Visual QA, Snapshots)](visualization/quality/index.md)
- [Advanced](visualization/advanced/index.md)
- [Appendix](visualization/appendix/index.md)
```

### New Deployment Stacks section (written from scratch)

This section does NOT come from a source book — sovereign-ai-cookbook had only a generated README. The chapter is written during migration (PMAT-065) using the recipe YAMLs and stack compositions as reference material.

```markdown
# Deployment Stacks

- [Overview](deployment-stacks/overview.md)
- [Recipes](deployment-stacks/recipes/index.md)
  - [APR Inference Server](deployment-stacks/recipes/apr-inference-server.md)
  - [Entrenar Training](deployment-stacks/recipes/entrenar-train.md)
  - [Realizar Serve](deployment-stacks/recipes/realizar-serve.md)
  - ... (one chapter per migrated recipe)
- [Stacks](deployment-stacks/stacks/index.md)
  - [01 Single-Machine Inference](deployment-stacks/stacks/01-inference.md)
  - [02 Single-Machine Training](deployment-stacks/stacks/02-training.md)
  - [03 RAG Pipeline](deployment-stacks/stacks/03-rag.md)
  - [04 Speech (Whisper)](deployment-stacks/stacks/04-speech.md)
  - [05 Distributed Inference](deployment-stacks/stacks/05-distributed-inference.md)
  - [06 Full Stack](deployment-stacks/stacks/06-full-stack.md)
  - [07 Data Pipeline](deployment-stacks/stacks/07-data-pipeline.md)
  - [08 Observability](deployment-stacks/stacks/08-observability.md)
  - [09 Edge Inference](deployment-stacks/stacks/09-edge-inference.md)
  - [10 Qwen-Coder](deployment-stacks/stacks/10-qwen-coder.md)
- [Machines](deployment-stacks/machines/index.md)
  - [Jetson](deployment-stacks/machines/jetson.md)
- [forjar Integration](deployment-stacks/forjar-integration.md)
```

Each chapter is short (<200 lines): summary + recipe inputs table + a representative invocation + cross-link to the YAML and the Rust wrapper.

---

## Cross-Reference Hygiene

After consolidation, links in the migrated chapters fall into three categories. All must be resolved before `mdbook build` ships clean:

| Link Class | Source pattern | Action |
|------------|----------------|--------|
| Intra-section (same book section) | `[Foo](../bar/baz.md)` within `data-loading/` | No change; relative paths still resolve |
| Cross-section (now intra-book) | `presentar/book` linking to `alimentar/book` cross-references (e.g., visualization examples that reference data-loading concepts) | Rewrite to `[ALD](../data-loading/...)` |
| External (was inter-repo) | `https://github.com/paiml/alimentar/blob/main/...` | Rewrite to relative path inside apr-cookbook OR keep as historic external link (acceptable for archived-source references) |

A migration script `scripts/rewrite-book-links.sh` handles classes 2 and 3 mechanically; class-1 needs no work. The script's logic:

1. Scan migrated `book/src/data-loading/**/*.md` and `book/src/visualization/**/*.md`
2. For each markdown link target:
   - If target starts with `https://github.com/paiml/{alimentar,presentar}/blob/main/`, rewrite to relative path inside `book/src/` if the corresponding chapter was migrated; otherwise leave as historic
   - If target starts with `../`, validate it resolves; warn if it doesn't

CI gate: `mdbook build book/` MUST exit 0 with zero "Link points to file that does not exist" warnings.

---

## Frontmatter and Code Snippet Cleanup

Two known content-level cleanups during migration:

### 1. Code snippet imports

Many alimentar book chapters open with:

```rust
use alimentar::Dataset;
```

After migration, the cookbook example workspace pulls alimentar from crates.io. Imports stay valid (the alimentar crate still exists), but cited paths change from `src/dataset.rs:NN` to `crates.io: alimentar v<version>`. The script `scripts/rewrite-book-links.sh` rewrites cited file paths only when the target file no longer exists in the destination tree.

### 2. References to in-repo benches/tests

Phrases like "see `tests/integration_test.rs` for usage" reference files that don't migrate. Two acceptable rewrites:

- If the test demonstrated a recipe-level concept: rewrite to point at the migrated `examples/data-loading/<example>.rs`
- Otherwise: drop the reference, replace with a one-line note "(integration test in alimentar crate; see crates.io)"

This is manual editorial work — the migration script flags candidate references for human review but does not auto-rewrite.

---

## SUMMARY.md ordering policy

The merged `SUMMARY.md` orders sections **by stack layer**, not by repo origin:

```
1. Introduction / Quickstart        (cookbook meta)
2. Recipe Catalog                   (existing apr-cookbook recipes — model layer)
3. Data Loading                     (alimentar — data layer feeds models)
4. Visualization                    (presentar — UI layer presents results)
5. Deployment Stacks                (sovereign — infra layer hosts everything)
6. Format Reference                 (cookbook meta — .apr details)
7. Falsification / Quality / Refs   (cookbook meta — methodology)
```

Rationale: a reader navigating the book follows the data flow (data → model → output → infra), not the historical accident of where each chapter originated.
