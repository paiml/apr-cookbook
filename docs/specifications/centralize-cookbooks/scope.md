# Scope & Charter Expansion

## Decision

**apr-cookbook becomes the umbrella sovereign-stack cookbook.** The repository slug, crate name, and Pages URL are preserved; the README, book hero text, and `docs/specifications/apr-cookbook.md` Executive Summary are updated to reflect expanded scope.

This is decided despite the slug ambiguity (`apr-` originally referred to the .apr model format) because the cost of a rename — invalidating crates.io references, breaking the `interactive.paiml.com`-adjacent Pages URL, and requiring downstream CI badge fixes — exceeds the cost of a one-line scope re-statement in the README.

## Before / After Scope

### Before (apr-cookbook v5.0)

> The APR Cookbook is the technical manual for production ML deployment using the `.apr` format.

24 categories: creation, bundling, training, conversion, registry, api, serverless, wasm, gpu, simd, distillation, cli, monitoring, speech, distributed, advanced (+ 8 sub-buckets).

### After (apr-cookbook v6.0, post-centralization)

> The APR Cookbook is the umbrella technical manual for the PAIML sovereign AI stack: model bundling and deployment (`.apr`), data loading (`alimentar`), declarative visualization (`presentar`), and infrastructure-as-recipes (`sovereign`/`forjar`). All examples are idiomatic Rust, all artifacts are IIUR-graded, all performance claims are falsifiable.

28 categories: 24 prior + `deployment-stacks`, `data-loading`, `visualization`, `machines`.

## What Migrates

| From | What | Where it lands |
|------|------|----------------|
| `sovereign-ai-cookbook/recipes/*.yaml` | 14 deployment recipes | `examples/deployment-stacks/recipes/` + Rust loader wrappers in `examples/deployment-stacks/` |
| `sovereign-ai-cookbook/stacks/NN-*/` | 10 multi-recipe stack compositions | `examples/deployment-stacks/stacks/` |
| `sovereign-ai-cookbook/machines/jetson/` | Edge machine config | `examples/machines/jetson/` |
| `alimentar/examples/*.rs` | 18 Rust data-loading examples | `examples/data-loading/` |
| `alimentar/book/src/**/*.md` | 103 mdBook chapters | `book/src/data-loading/` (new top-level section) |
| `presentar/examples/{charts,dashboards,ald,apr,prs,...}/` | 28 `.yaml` and `.prs` declarative configs | `examples/visualization/` (preserved subdir layout) |
| `presentar/book/src/**/*.md` | 121 mdBook chapters | `book/src/visualization/` (new top-level section) |

## What Does NOT Migrate

- `alimentar/src/` — the alimentar crate itself stays published on crates.io
- `presentar/src/` — same; presentar crate stays separate
- `presentar/crates/` — sub-crates (apr-widgets, ald-widgets, prs-runtime) stay in presentar
- `sovereign-ai-cookbook/scripts/` — generation/lint scripts that target the source repo's structure; apr-cookbook has its own equivalents
- Any source-repo `.github/workflows/` — apr-cookbook has its own CI; we don't merge workflows
- `sovereign-ai-cookbook/certs/` and `presentar/data/` — large binary fixtures stay out; recipes that need them switch to inline fixtures or `tempfile::tempdir()` per IIUR
- `*.dvc`, `mlflow.yaml`, `flake.nix`, `Brewfile` from presentar — orchestration config not relevant to cookbook examples

## Naming Conventions Inside apr-cookbook

After migration the `examples/` tree gains:

```
examples/
├── deployment-stacks/
│   ├── recipes/                  # sovereign YAMLs verbatim
│   ├── stacks/                   # sovereign stack compositions verbatim
│   ├── apr_inference_server.rs   # Rust loader/validator wrapping recipes/apr-inference-server.yaml
│   └── ... (14 wrappers, one per recipe)
├── data-loading/
│   ├── basic_loading.rs          # ex-alimentar
│   ├── dataloader_batching.rs
│   └── ... (18 examples)
├── visualization/
│   ├── charts/                   # ex-presentar/examples/charts
│   ├── dashboards/
│   ├── ald/
│   ├── apr/
│   ├── prs/
│   └── load_chart.rs             # Rust validator that loads any visualization YAML
└── machines/
    └── jetson/
        └── *.toml, *.service     # verbatim
```

Naming rules:
- Migrated Rust example file names are **preserved** (ex-`basic_loading.rs` stays `basic_loading.rs`)
- Sovereign YAML recipe file names are **preserved** (ex-`apr-inference-server.yaml` stays the same; the Rust wrapper takes the same stem with `.rs`)
- Presentar example subdirectories are **preserved** (`charts/`, `dashboards/`, etc.)

## Charter Boundaries

The umbrella cookbook covers:
- ✅ Model creation, bundling, conversion, deployment (existing)
- ✅ Data loading and dataset management (new, ex-alimentar)
- ✅ Declarative visualization and dashboards (new, ex-presentar)
- ✅ Deployment-as-recipe configurations (new, ex-sovereign)
- ✅ Edge machine provisioning (new, ex-sovereign machines/)

The umbrella cookbook does **not** cover:
- ❌ Library implementation (alimentar, presentar, forjar, realizar internals)
- ❌ The `aprender` monorepo crate sources (covered in their own docs.rs/spec)
- ❌ Tutorial-style "intro to ML" content (cookbook is reference, not pedagogy)
- ❌ Live production deployment runbooks for paiml.com infrastructure (separate ops repo)

## Versioning

Post-merge, `docs/specifications/apr-cookbook.md` bumps from v5.0.0 → v6.0.0 (major: scope expansion, breaking re-categorization). The centralize-cookbooks spec itself stays at v1.0.0 — it's a one-shot.
