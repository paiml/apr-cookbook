# Fine-Tuning Cookbook Specification

**Version**: 1.2.0
**Status**: PROPOSED (PMAT-330..361 planned; spec for review)
**Manifest count**: 155 recipes (was 100 in v1.0.0 → 128 in v1.1.0 after Ludwig deep-dive → 155 in v1.2.0 after TRL + LLaMA-Factory + Axolotl surveys)
**MSRV**: 1.89 (inherits from apr-cookbook v6.0)
**Date**: 2026-05-10
**Repository**: [github.com/paiml/apr-cookbook](https://github.com/paiml/apr-cookbook)
**Sovereign Stack**: APR-MONO v0.31.2

---

## Executive Summary

apr-cookbook hosts ~1670 recipes across 28 categories, but **fine-tuning coverage is shallow**. Today the cookbook ships 4 LoRA-related recipes plus 7 entrenar autograd/training examples. The two most-cited fine-tuning toolchains — Ludwig (Uber, declarative YAML configs) and Unsloth (2x faster LoRA / QLoRA / GRPO) — have ~50 and ~60 recipe surfaces respectively. This spec proposes ≥100 cookbook recipes that mirror Ludwig + Unsloth idiomatically against the `apr` CLI, organized **simple → hard** so a learner can walk the surface from "fine-tune Llama on a CSV" to "GRPO long-context reasoning on Qwen3.5-MoE."

The reference assets exist locally:

| Repo | Purpose | Recipes |
|------|---------|---------|
| `~/src/ground-truth-apr-ludwig` | Rust port of Ludwig examples; CLI recipe shape mirrors `apr finetune` semantics | 21 (recipes 01-16 mirror Ludwig, 17-21 apr-native) |
| `~/src/unsloth` | Python source of truth; feature/notebook catalog | ~60 notebook surfaces |
| `~/src/huggingface-fine-tuning` | HF curriculum; tabular + text + image | ~30 recipes/labs |
| `~/src/HF-Advanced-Fine-Tuning` | HF advanced track; DPO, RLHF, multi-modal | ~20 recipes |

Cookbook integration is straightforward — the `apr` CLI already ships `apr finetune`, `apr eval`, `apr quantize`, `apr serve`, `apr merge`, `apr distill`, `apr prune`, `apr chat`. This spec adds 100 recipes that exercise these subcommands across the four-tier curriculum below.

---

## Component Documents

| Document | Purpose |
|----------|---------|
| [scope.md](fine-tuning-cookbook/scope.md) | Charter, four-tier curriculum (simple → hard), Ludwig + Unsloth feature mapping, non-goals |
| [manifest.yaml](fine-tuning-cookbook/manifest.yaml) | Single source of truth — one entry per recipe with tier, technique, base family, status |
| [manifest.schema.yaml](fine-tuning-cookbook/manifest.schema.yaml) | JSON Schema for manifest entries; CI validates |
| [recipe-template.md](fine-tuning-cookbook/recipe-template.md) | Canonical recipe shape (IIUR + provable contract + dataset fixture) |
| [coverage-matrix.md](fine-tuning-cookbook/coverage-matrix.md) | Auto-regenerated matrix: tier × technique × base family; tracks coverage gaps |
| [tickets.md](fine-tuning-cookbook/tickets.md) | PMAT-330..355 ticket breakdown (~4 recipes per ticket, ~25 dev-days) |

---

## Invariants

The cookbook enforces these invariants at PR time. The fine-tuning spec extends the existing six (architecture-demos):

1. **Recipe-per-tier coverage**: every tier has at least its target recipe count (T1=25, T2=25, T3=25, T4=25). Manifest tracks tier counts; CI fails if a recipe-tier mismatch lands.
2. **Mirror parity (Invariant L)**: every recipe is *inspired by* Ludwig, Unsloth, or HuggingFace TRL — but only ≥35% set an explicit `ludwig_mirror:` or `unsloth_mirror:` field (mostly Tier 1+2 where Ludwig has direct examples and Unsloth notebooks are 1:1 mappable). Tier 3 calibration/imbalance also map cleanly. Tier 4 RL recipes derive from TRL/Unsloth and OpenAI/Anthropic literature rather than a single canonical mirror. The 5 `apr_native: true` recipes (smoke + bench) wrap apr-only flags. CI counts: ≥35% explicit mirror, the rest documented inline.
3. **CLI parity (Invariant A — extended)**: every recipe exercises at least one `apr` subcommand (`apr finetune`, `apr eval`, `apr quantize`, `apr serve`, `apr merge`, `apr distill`, `apr prune`, `apr chat`). New `--method`, `--dataset-format`, `--reward-model` flags surface at least once.
4. **Dataset fixture discipline**: every recipe ships a synthetic, deterministic dataset fixture under `tests/fixtures/finetune/<recipe-name>/` ≤ 1 MB. No live HF Hub network calls in CI; production-scale datasets get a separate non-CI smoke recipe.
5. **Provable-contract grade A (Invariant B — extended)**: every certified recipe ships `contracts/finetune-<recipe>-v1.yaml` with proof_obligations covering totality, determinism, and convergence. Lean theorems may be `not-applicable` for stochastic-gradient obligations (documented).
6. **Falsification per recipe**: each recipe ships a falsifier — typically "training loss decreases monotonically over the first N steps on a convex / well-conditioned subproblem" — with a concrete `cargo test` that fails red if the property breaks.

---

## Success Criteria

| Criterion | Target | How measured |
|-----------|--------|--------------|
| Recipe count | ≥ 155 | `wc -l` over manifest entries with `status: certified` |
| Tier coverage | T1=25, T2=45, T3=48, T4=37 | manifest tier histogram (revised in v1.2.0 after TRL/LLaMA-Factory/Axolotl) |
| Ludwig mirror coverage | ≥ 55 | manifest entries with `ludwig_mirror:` set (concrete file paths in `~/src/ludwig/examples/`) |
| TRL mirror coverage | ≥ 12 | `trl_mirror:` set — Tier 4 RL grounded in `~/src/trl/examples/scripts/` |
| LLaMA-Factory mirror coverage | ≥ 10 | `llamafactory_mirror:` set — GaLore/BAdam/Apollo/DoRA/NEFTune/quantized-LoRA |
| Axolotl mirror coverage | ≥ 6 | `axolotl_mirror:` set — ReLoRA/LISA/QAT/sample-packing/FSDP-LoRA |
| Unsloth mirror coverage | ≥ 16 | `unsloth_mirror:` set (Colab notebook refs) |
| Distinct techniques | ≥ 63 | manifest `technique:` enum cardinality |
| Provable-contract grade A | 100% of certified | `pv score contracts/finetune-*.yaml --binding` aggregate ≥ 0.93 |
| CLI subcommand exercise | every `apr finetune`/`apr eval`/`apr quantize`/`apr serve`/`apr merge` flag | recipe coverage table |
| CI runtime | < 5 min on PR | `cargo test --lib --tests --features finetune-fixtures` |
| Spec gate | `make fine-tuning-coverage` green | new Makefile target + `.github/workflows/fine-tuning.yml` |

---

## Non-Goals

- ❌ **Production training runs** — recipes use synthetic fixtures, not real datasets. Production runs are documented in `apr-model-qa-playbook` (per-checkpoint qualification), out of cookbook scope.
- ❌ **Multi-GPU distributed training** — single-GPU SFT/LoRA only in v1; DDP/FSDP tracked in a separate v2 spec.
- ❌ **Training-time CI on actual GPUs** — recipes must run on CPU smoke fixtures; GPU-only paths are `#[cfg_attr(not(feature = "cuda"), ignore)]`.
- ❌ **Re-implementing Ludwig or Unsloth** — recipes wrap existing aprender APIs (`entrenar`, `aprender::format`); they do NOT port Ludwig's `Trainer` or Unsloth's Triton kernels.
- ❌ **Live HF Hub or W&B network calls** — fixtures are bundled or synthetic; observability is local file-based (`tests/fixtures/finetune-results/`).

---

## Versioning

apr-cookbook spec bump from v6.2.0 → **v6.3.0** when fine-tuning-cookbook v1.0 lands (PMAT-355). Subsequent expansions (DDP, multi-modal, RLHF-with-reward-model) are additive minors within v6.3.x.

---

## Status & Implementation Plan

| Phase | Milestone | Tickets | Estimate |
|-------|-----------|---------|----------|
| 0 | Spec acceptance + scaffolding | PMAT-330 | 1 day |
| 1 | Tier 1 (SFT + eval + tabular, 25 recipes) | PMAT-331..337 | 7 days |
| 2 | Tier 2 (LoRA + QLoRA + continued pretraining + 9 PEFT variants, 34 recipes) | PMAT-338..346 | 9 days |
| 3 | Tier 3 (instruction/hyperopt/calibration/imbalance/multimodal + 19 Ludwig categories, 44 recipes) | PMAT-347..357 | 11 days |
| 4 | Tier 4 (DPO/ORPO/KTO/GRPO/RLHF, 25 recipes) | PMAT-358..360 | 5 days |
| 5 | CI gate + book chapter + closeout | PMAT-361 | 1 day |
| **Total** | | **PMAT-330..361 (32 tickets)** | **~34 dev-days** |

Sister-ticket batches within a phase are pure parallel (no inter-ticket deps after PMAT-330's scaffolding), so realistic calendar is ~12 days with parallelism.

---

## See Also

- [`docs/specifications/architecture-demos.md`](architecture-demos.md) — model-coverage spec; this fine-tuning spec is its training-side counterpart
- [`~/src/ground-truth-apr-ludwig`](https://github.com/paiml/ground-truth-apr-ludwig) — Rust port of Ludwig examples (21 recipes)
- [Ludwig](https://ludwig.ai) — Uber's declarative ML framework
- [Unsloth](https://unsloth.ai) — 2x faster LLM fine-tuning
- [HuggingFace Trainer / TRL](https://huggingface.co/docs/trl) — DPO, ORPO, GRPO upstream implementations
