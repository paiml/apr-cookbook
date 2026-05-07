# Architecture Demos Specification

**Version**: 1.0.0
**Status**: ACTIVE (PMAT-300..307 implemented; CI gate enforced)
**MSRV**: 1.89 (inherits from apr-cookbook v6.0)
**Date**: 2026-05-07
**Repository**: [github.com/paiml/apr-cookbook](https://github.com/paiml/apr-cookbook)
**Sovereign Stack**: APR-MONO v0.31.2

---

## Executive Summary

This spec proposes systematic coverage of **Hugging Face model architectures** in apr-cookbook, mirroring the [apr-model-qa-playbook](https://github.com/paiml/apr-model-qa-playbook) per-model YAML playbook style but lifted from individual checkpoints to **architectures** (`config.json#architectures` keys: `LlamaForCausalLM`, `MistralForCausalLM`, etc.).

The unit is the architecture, not the model. HF hosts millions of model repos but Transformers exposes ~300 distinct `architectures` keys. apr-model-qa-playbook tracks 256 individual playbooks across 43 distinct model families (`bloom`, `bloomz`, `codegemma`, `codellama`, `codestral`, `deepseek`, `falcon`, `gemma`, `granite`, `internlm2.5`, `llama`, `mistral`, `phi`, `qwen2`, `qwen3`, …). The 43-family resolution is the natural target — every family gets one upstream `aprender::rosetta` loader + one cookbook recipe demonstrating load-and-forward, with the per-checkpoint matrix continuing to live in apr-model-qa-playbook as before.

apr-cookbook today implements **Qwen2 only** (in `aprender-core/src/models/qwen2/`), with one-off recipes (`convert_phi_to_apr.rs`, `inference_qwen3_moe_numerical_parity_smoke.rs`) for two more. A user landing on the cookbook learns nothing about whether Llama, Mistral, Gemma, Phi, DeepSeek, Falcon, BLOOM, MAMBA, or any of the 35+ remaining families are supported. This spec proposes ~43 new recipes (one per family) plus the upstream `aprender::rosetta` loader registry that backs them.

**Net additions:** 43 recipe artifacts split between `examples/inference/` (smoke loads) and `examples/conversion/` (HF-format converters where non-trivial), plus a manifest-driven coverage gate that forces the cookbook to track upstream architecture support.

---

## Component Documents

| Document | Purpose |
|----------|---------|
| [scope.md](architecture-demos/scope.md) | Charter, naming decision (architecture vs model vs checkpoint), non-goals |
| [manifest.yaml](architecture-demos/manifest.yaml) | Single source of truth — one entry per architecture family, status: certified / in-progress / blocked |
| [manifest.schema.yaml](architecture-demos/manifest.schema.yaml) | JSON Schema for the manifest, ported from apr-model-qa-playbook/playbook.schema.yaml |
| [coverage-matrix.md](architecture-demos/coverage-matrix.md) | Per-family table: upstream rosetta loader path, recipe path, formats, quantizations, status |
| [recipe-template.md](architecture-demos/recipe-template.md) | Standard `inference_<family>_smoke.rs` template with verdict enum, IIUR header, citations |
| [generator.md](architecture-demos/generator.md) | `scripts/architecture-demos-gen.sh` walks the manifest and emits recipe stubs + Cargo.toml entries |
| [tickets.md](architecture-demos/tickets.md) | PMAT ticket breakdown — one per family family-batch (PMAT-300..3??) |

---

## Acceptance Criteria

The architecture-demos initiative is **done** when, and only when:

1. **Manifest parity**: `docs/specifications/architecture-demos/manifest.yaml` lists every architecture family supported in upstream `aprender::rosetta`, and every entry has either status `certified` (recipe lands and passes) or `in-progress` (loader landed upstream, recipe pending) or `blocked` (loader not yet implemented upstream, with link to aprender ticket).
2. **Recipe-per-family**: every `status: certified` entry has a recipe at `examples/inference/inference_<family>_smoke.rs` that loads a small reference checkpoint and produces a deterministic forward-pass verdict.
3. **IIUR grade**: every recipe satisfies `contracts/recipe-iiur-v1.yaml` — `RecipeContext::new`, deterministic output, no network, ≥1 arXiv/DOI citation, full unit-test block.
4. **Provable-contract grade A (Invariant B)**: every `status: certified` family ships a per-family provable-contract at `contracts/inference-<family>-smoke-v1.yaml` with `kernel_structure.phases:`, per-equation `preconditions:` / `postconditions:` / `lean_theorem:`, per-obligation `tolerance:` + `lean: {theorem, status}` block. `status: wip` is the honest default at landing; `proved` requires a real `.lean` file under `lean/` (per MEMORY.md `pv lint / score` discipline). `pv lint contracts/inference-<family>-smoke-v1.yaml` must pass; `pv score --summary` must report grade A.
5. **Format coverage (Invariant C extended)**: where a family ships in multiple HF formats, the recipe demonstrates each: `<family>.safetensors` → `<family>.apr` → `<family>.gguf`. Single-format families (e.g., MAMBA SafeTensors-only) are exempt and noted in the manifest.
6. **CI gate**: `make architecture-demos-coverage` walks the manifest and fails the build if any `certified` entry lacks a corresponding recipe, contract, or vice versa. Wired into the unified gate as a non-advisory required check, alongside the existing `make contract-grade` Invariant B gate.
7. **Generator**: `scripts/architecture-demos-gen.sh --check` runs in CI and is no-op when manifest, on-disk recipes, and contract YAMLs are in sync; running with `--update` emits stub recipes + Cargo.toml `[[example]]` entries + provable-contract YAML stubs for new families.
8. **Coverage matrix table**: [coverage-matrix.md](architecture-demos/coverage-matrix.md) is regenerated from the manifest on every PR; staleness fails the deterministic-table gate.

---

## Non-Goals

- **No HF Hub network calls at recipe runtime.** Recipes must use bundled small test fixtures or `include_bytes!`-embedded micro-checkpoints. The Hub is only referenced in the manifest's `hf_repo` field as documentation for which checkpoint a real-world user would pass.
- **No coverage of vision / audio / multimodal architectures in v1.** Scope is decoder-only and encoder-decoder text models. Vision (`ViTForImageClassification`, `LlavaForConditionalGeneration`, …) is a v2 expansion tracked separately.
- **No re-implementation of upstream loaders.** When `aprender::rosetta` lacks a family, the manifest entry is `status: blocked` with an upstream aprender ticket link. apr-cookbook does not implement loaders.
- **No per-checkpoint coverage.** The 256 individual playbooks in apr-model-qa-playbook stay where they are. One recipe demonstrates the family; specific quantization × backend × scenario combinations remain that repo's job.
- **No GPU-required recipes.** All recipes run CPU-only by default. GPU paths use `#[cfg_attr(not(feature = "cuda"), ignore)]` per the existing cookbook convention.
- **No reciprocal coupling with apr-model-qa-playbook.** That repo continues to operate against published `aprender` releases; this spec only adds a `references` link both ways.

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| HF `architectures` namespace is unbounded — new families ship every week in `transformers` | Manifest is append-only; `make architecture-demos-coverage` doesn't fail when upstream `transformers` adds a family that aprender hasn't implemented (only when aprender ships a loader without a recipe) |
| Family aliasing — `qwen2` vs `qwen2.5` vs `qwen3` are technically distinct architectures with shared loader code | Manifest entries are normalized to the upstream `aprender::rosetta` loader name, not the marketing version. Aliasing is a documentation concern in coverage-matrix.md, not a recipe-explosion concern |
| Bundled test fixtures inflate repo size | Use the smallest available reference checkpoint (1B–3B family); for very large families, generate a synthetic 2-layer micro-config with random weights and demonstrate load-without-forward as a fallback |
| Format-coverage matrix (3 formats × 43 families = 129 cells) becomes 129 recipes if applied naively | Per-family recipe demonstrates **all formats it supports** in one file, not three; format coverage is per-recipe internal, not per-recipe-multiplication |
| Recipe drift when upstream `aprender::rosetta` API evolves | Recipes pin to `^0.31`; an upstream breaking change surfaces as a cookbook test fail (the canary we want), and the manifest entry flips to `in-progress` until the recipe is updated |
| 43 recipes is a big sprint | Decompose into PMAT tickets by family batch (≤5 families per ticket), shippable independently. Status `blocked` entries land first as documentation; recipes follow as upstream loaders land |
| apr-model-qa-playbook divergence — schema or terminology drift between the two repos | Schema is **forked once** from apr-model-qa-playbook into `architecture-demos/manifest.schema.yaml`, not symlinked or imported. Drift is acceptable; the two repos serve different purposes (per-checkpoint QA vs per-family demonstration) and don't need lockstep evolution |

---

## Cross-References

- Parent spec: [apr-cookbook.md](apr-cookbook.md) — IIUR, falsification discipline, Six Coverage Invariants
- Predecessor: [expand-cookbooks.md](expand-cookbooks.md) — sister-crate ≥3-recipe coverage
- Companion repo: [apr-model-qa-playbook](https://github.com/paiml/apr-model-qa-playbook) — per-checkpoint Toyota/Popper QA framework
- Memory: `memory/MEMORY.md` — current 28-category structure post-centralize-cookbooks; aprender monorepo port map at [memory/aprender_monorepo_ports.md](../../.claude/projects/-home-noah-src-apr-cookbook/memory/aprender_monorepo_ports.md)
- Upstream entry point: `aprender::rosetta` — architecture loader registry at `../aprender/crates/aprender-core/src/format/`

---

## Approval

This spec moves to `Status: ACTIVE` after:
1. Repository owner approval (Noah Gift)
2. Component documents under [architecture-demos/](architecture-demos/) drafted (scope, manifest schema, coverage matrix seed, recipe template, generator script, ticket breakdown)
3. PMAT-300 created and assigned for the first family batch (Llama / Mistral / Gemma / Phi / Qwen — the families with existing fixtures or upstream loaders)

Until then, no new examples land and no Cargo.toml changes are made.
