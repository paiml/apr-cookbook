# Scope & Charter

## Decision

Add **one apr-cookbook recipe per Hugging Face model architecture family** that ships an upstream `aprender::rosetta` loader. The unit of coverage is the architecture family (`Llama`, `Mistral`, `Phi`, …), not the individual checkpoint. Per-checkpoint qualification stays in [apr-model-qa-playbook](https://github.com/paiml/apr-model-qa-playbook); per-family demonstration lands here.

## Before / After Scope

### Before (apr-cookbook v6.1.0, 2026-05-07)

> Upstream `aprender` ships 18 model-family descriptors at `aprender/contracts/model-families/` (bert, deepseek, falcon_h1, gemma, gpt2, gptneox, llama, mamba, mistral, moonshine, openelm, opt, phi, qwen2, qwen3, qwen3_5, rwkv7, whisper). Cookbook demonstrates **2** of them (`convert_phi_to_apr.rs`, `inference_qwen3_moe_numerical_parity_smoke.rs`) plus implicit Qwen2 coverage in monorepo `aprender-core/src/models/qwen2/`. Coverage gap: 16 of 18 upstream-supported families have zero recipe.

### After (apr-cookbook v6.2.0 post-architecture-demos)

> Every upstream-supported family has a recipe at `examples/inference/inference_<family>_smoke.rs` that loads a synthetic micro-checkpoint, runs a deterministic forward pass, and emits a `Verdict::Ok { … }` value. Manifest-driven CI gate fails when upstream lands a new family loader without a corresponding cookbook recipe.

29 categories. ~440 → ~458 recipes. No new top-level categories — recipes land in existing `examples/inference/` and `examples/conversion/`.

## What Migrates From Where

Nothing migrates. This is **net-new** content driven by manifest reconciliation. The schema is **forked** (not symlinked) from `apr-model-qa-playbook/playbooks/playbook.schema.yaml`, simplified to drop per-checkpoint matrix concerns (modalities, backends, scenarios, oracles, profile_ci) that don't apply at the family level.

## Coverage Unit Decision

We considered three resolutions:

| Resolution | Count | Pros | Cons |
|------------|-------|------|------|
| HF `architectures[]` keys | ~300 | finest grain, matches `transformers` exactly | recipe explosion; many keys differ only by head config |
| Family (`apr-model-qa-playbook` style) | ~43 | tractable, matches upstream loader naming | one family can span multiple `architectures[]` keys |
| Vendor (Alibaba, Meta, Mistral AI) | ~12 | very compact | hides architectural variation (Llama vs Llama-MoE) |

**Decided: family**, normalized to `aprender::rosetta` loader names. The 43 figure tracks apr-model-qa-playbook's playbook count; the 18 figure tracks what aprender currently implements; the gap (25 families with HF QA playbooks but no upstream loader) becomes the upstream backlog. Family aliasing (qwen2 vs qwen2.5 vs qwen3) is documented in [coverage-matrix.md](coverage-matrix.md), not multiplied across recipes.

## Naming Conventions

- Recipe filename: `examples/inference/inference_<family>_smoke.rs` (e.g., `inference_llama_smoke.rs`, `inference_mistral_smoke.rs`).
- Optional companion converter: `examples/conversion/convert_<family>_to_apr.rs` when the family ships a non-trivial HF→APR mapping (Phi already exists; Llama/Mistral/Gemma will follow).
- Family name in code matches upstream `family:` field in `aprender/contracts/model-families/<name>.yaml` exactly. No spaces, no version suffix in the recipe name (use `qwen3` not `qwen3.0` or `qwen3-0.6b`).
- Synthetic micro-checkpoint fixtures live in `tests/fixtures/architectures/<family>/` and are < 1 MB each (2-layer reduced-config configs with random weights).

## Charter Boundaries

The architecture-demos initiative covers:
- ✅ One smoke recipe per family with `aprender::rosetta` loader support
- ✅ Multi-format demonstration where the family supports it (SafeTensors → APR → GGUF in one recipe)
- ✅ Manifest-as-source-of-truth with CI-enforced reconciliation
- ✅ Upstream-backlog visibility — `status: blocked` entries with aprender ticket links

The architecture-demos initiative does **not** cover:
- ❌ Per-checkpoint qualification (stays in apr-model-qa-playbook)
- ❌ Vision / audio / multimodal (text decoder + encoder-decoder only in v1; whisper/moonshine are speech-only and remain in `examples/speech/`)
- ❌ Live HF Hub network calls — fixtures are bundled or synthetic
- ❌ GPU-required CI gates — CPU smoke is the floor, GPU is `#[cfg_attr(not(feature = "cuda"), ignore)]`
- ❌ Implementing upstream loaders — `status: blocked` entries are tracked, not resolved here

## Upstream Contribution Discipline

Added in v1.1 (PMAT-313). When a cookbook meta-recipe surfaces a primitive that genuinely belongs upstream — not a one-off demo, but a reusable API surface that other consumers would want — it gets lifted into `aprender-core` and the cookbook recipe becomes the falsification suite for the upstream API.

Concrete instance: PMAT-309's `inference_arch_detector.rs` reverse-engineered a discriminator-dispatch table from the 18 family-smoke recipes. PMAT-313 lifted that dispatch table into `aprender::format::FamilyRegistry::detect_from_config_str` ([aprender#1562](https://github.com/paiml/aprender/pull/1562)) plus added a `register_alias` mechanism that unblocks 16 of the 25 `status: blocked` manifest entries.

Pattern when this applies:
1. A cookbook recipe demonstrates a primitive that maps cleanly to upstream's domain (architecture inspection, loader dispatch, …)
2. The recipe's tests serve as a falsification contract on the upstream behavior
3. Upstream PR adds the public API; cookbook recipe doc-header documents the upstream destination
4. After the upstream PR ships in a release, cookbook bumps its aprender pin and the recipe refactors to call the upstream API directly (the tests become integration tests for upstream)

Pattern when this does NOT apply:
- Recipe is genuinely cookbook-specific (depends on `RecipeContext`, IIUR plumbing, etc.)
- Recipe is a thin wrapper over an existing upstream API (no new primitive)
- Recipe is exploratory — not yet stabilized enough to bake into upstream

## Versioning

apr-cookbook spec bumps from v6.1.0 → **v6.2.0** after architecture-demos v1 lands (PMAT-308). v1.1 (PMAT-309..313) is additive within v6.2.x — the architecture-demos spec bumps to **1.1.0** to reflect the cross-family meta-recipe expansion + upstream-contribution discipline.
