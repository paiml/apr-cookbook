# PMAT Ticket Breakdown

The 100-recipe curriculum lands in 26 tickets (PMAT-330..355) over ~28 dev-days. Sister-ticket batches within a tier are pure parallel (no inter-ticket deps after PMAT-330's scaffolding), so realistic calendar is **~12 days with parallelism**.

| Ticket | Scope | Priority | Estimate | Depends on |
|--------|-------|----------|----------|------------|
| PMAT-330 | Generator + manifest + contract scaffolding | P1 | 1 day | spec acceptance |
| PMAT-331 | Tier 1.1 — SFT minimal × 5 families (5 recipes) | P1 | 1 day | PMAT-330 |
| PMAT-332 | Tier 1.2 — Eval primitives (5 recipes) | P1 | 1 day | PMAT-330 |
| PMAT-333 | Tier 1.3 — Tabular regression (5 recipes) | P1 | 1 day | PMAT-330 |
| PMAT-334 | Tier 1.4 — Tabular classification (5 recipes) | P1 | 1 day | PMAT-330 |
| PMAT-335 | Tier 1.5 — Smoke + bench (5 recipes) | P2 | 1 day | PMAT-330 |
| PMAT-336 | Tier 1 closeout — invariant L baseline + book chapter | P2 | 0.5 day | PMAT-331..335 |
| PMAT-337 | (reserved for Tier 1 fixup) | P3 | 0.5 day | — |
| PMAT-338 | Tier 2.1a — LoRA rank 8 × 5 families (5 recipes) | P1 | 1 day | PMAT-330 |
| PMAT-339 | Tier 2.1b — LoRA rank 32 × 5 families (5 recipes) | P1 | 1 day | PMAT-330 |
| PMAT-340 | Tier 2.2 — QLoRA × 5 (5 recipes) | P1 | 1 day | PMAT-330 |
| PMAT-341 | Tier 2.3 — Continued pretraining × 5 (5 recipes) | P1 | 1 day | PMAT-330 |
| PMAT-342 | Tier 2.4 — Adapter merge × 5 (5 recipes) | P2 | 1 day | PMAT-330 |
| PMAT-343 | Tier 2 closeout | P2 | 0.5 day | PMAT-338..342 |
| PMAT-344 | (reserved for Tier 2 fixup) | P3 | 0.5 day | — |
| PMAT-345 | Tier 3.1 — Instruction tuning × 5 (5 recipes) | P1 | 1 day | PMAT-330 |
| PMAT-346 | Tier 3.2 — Hyperopt × 5 (5 recipes) | P1 | 1 day | PMAT-330 |
| PMAT-347 | Tier 3.3 — Calibration × 5 (5 recipes) | P2 | 1 day | PMAT-330 |
| PMAT-348 | Tier 3.4 — Class imbalance × 5 (5 recipes) | P2 | 1 day | PMAT-330 |
| PMAT-349 | Tier 3.5 — Multimodal + multitask + kfold (5 recipes) | P2 | 1 day | PMAT-330 |
| PMAT-350 | Tier 3 closeout | P2 | 0.5 day | PMAT-345..349 |
| PMAT-351 | (reserved for Tier 3 fixup) | P3 | 0.5 day | — |
| PMAT-352 | Tier 4.1+4.2 — DPO × 5 + ORPO × 3 (8 recipes) | P1 | 1.5 days | PMAT-330 |
| PMAT-353 | Tier 4.3+4.4 — KTO × 3 + GRPO × 5 (8 recipes) | P1 | 1.5 days | PMAT-330 |
| PMAT-354 | Tier 4.5+4.6+4.7 — RLHF × 3 + RLAIF × 3 + reward × 3 (9 recipes) | P2 | 1.5 days | PMAT-330 |
| PMAT-355 | CI gate + book chapter + spec bump v6.2 → v6.3 | P3 | 0.5 day | PMAT-331..354 |

**Total: 26 tickets, ~28 dev-days; ~12 days with parallelism.**

---

## PMAT-330 — Generator + manifest + contract scaffolding (prerequisite)

**Priority**: P1
**Estimate**: 1 day
**Depends on**: spec acceptance

### Scope

1. Create `scripts/finetune-gen.sh` mirroring `scripts/architecture-demos-gen.sh`:
   - `--check` (CI gate, read-only)
   - `--update` (write recipe stubs, contract stubs, fixture stubs)
   - `--diff` (preview)
   - `--target` ∈ {recipes, contracts, fixtures, cargo, coverage-matrix, all}
2. Implement contract-stub generation: walk manifest, emit `contracts/finetune-<id>-v1.yaml` per recipe-template.md.
3. Implement fixture-stub generation: walk manifest, emit `tests/fixtures/finetune/<id>/{data.jsonl,expected.json,README.md}`.
4. Wire `--check` into `.github/workflows/fine-tuning.yml` as a required check.
5. Add `make fine-tuning-coverage` target invoking `--check`.
6. Manifest already drafted (manifest.yaml, 100 entries planned).

### Definition of Done

- `bash scripts/finetune-gen.sh --check` returns 0 on a fresh checkout
- `--update --target contracts` emits 100 `contracts/finetune-*-v1.yaml` stubs that all pass `pv lint`
- `--update --target fixtures` emits 100 fixture directories
- CI workflow committed and runs on PRs touching `docs/specifications/fine-tuning-cookbook/**`

---

## PMAT-331..335 — Tier 1 (Foundations, 25 recipes)

**Priority**: P1 (PMAT-331..334), P2 (PMAT-335)
**Estimate**: 1 day each
**Depends on**: PMAT-330

### Scope (per ticket)

Five recipes per ticket, following recipe-template.md. Each recipe ships:

- `examples/finetune/<id>.rs` (≤ 200 LOC)
- `tests/fixtures/finetune/<id>/data.{jsonl,csv}` (≤ 1 MB)
- `contracts/finetune-<id>-v1.yaml` (target Grade A)
- `lean/ProvableContracts/Finetune/<Id>.lean`
- 4-test `#[cfg(test)] mod tests` block (recipe_runs, falsifier_holds, falsifier_breaks_on_perturbed, deterministic)

### Definition of Done (per ticket)

- All 5 recipes pass `cargo test --example <id>`
- All 5 contracts pass `pv lint` and report Grade B+ minimum (Grade A target by tier closeout)
- Manifest entries flipped from `planned` → `certified`
- `make fine-tuning-coverage` green

---

## PMAT-336, 343, 350 — Tier closeout tickets

**Priority**: P2
**Estimate**: 0.5 day

### Scope

For each tier (1, 2, 3):

1. Promote all in-tier contracts to Grade A (Spec 1.0 / Falsify 1.0 / Kani 0.9 / Lean 1.0 / Bind 1.0).
2. Verify mirror parity: ≥ tier-target Ludwig/Unsloth refs explicitly set in manifest.
3. Add tier-specific book chapter under `book/src/finetune/tier-<n>/`.
4. Update tier-section of `coverage-matrix.md` (auto-regenerated; sanity-check).

---

## PMAT-338..342 — Tier 2 (Adaptive Methods, 25 recipes)

**Priority**: P1 / P2
**Estimate**: 1 day each
**Depends on**: PMAT-330

Same recipe-per-ticket cadence as Tier 1.

LoRA tickets (PMAT-338, PMAT-339) are the highest-traffic — most cookbook users want LoRA fine-tuning more than anything else. Tier 2.1a (rank-8) lands first as the canonical pattern; Tier 2.1b (rank-32) follows with parameter-count and convergence assertions that compare against rank-8.

---

## PMAT-345..349 — Tier 3 (Specialization, 25 recipes)

**Priority**: P1 / P2
**Estimate**: 1 day each
**Depends on**: PMAT-330

PMAT-345 (instruction tuning) is the gateway to Tier 4 — instruction-tuned base models are the input to DPO/GRPO. Land it first.

PMAT-346 (hyperopt) requires `apr tune` to be wired up (it's part of `apr-cli` 0.31.2 already; verify before landing).

PMAT-347..348 (calibration + imbalance) are pure tabular and can land in parallel.

PMAT-349 (multimodal + kfold) has 4 multimodal stub recipes + 1 kfold recipe; vision recipes use 32×32 image fixtures (no full-resolution).

---

## PMAT-352..354 — Tier 4 (Reinforcement, 25 recipes)

**Priority**: P1 (PMAT-352, PMAT-353), P2 (PMAT-354)
**Estimate**: 1.5 days each
**Depends on**: PMAT-330

Tier 4 is the technically-deepest tier — DPO/ORPO/KTO/GRPO/PPO require gradient-flow correctness across a reference model + policy + (sometimes) reward model.

Each ticket bundles 8-9 recipes because the per-recipe surface is similar within a technique (e.g., 5 DPO recipes share most code; varying base family).

PMAT-353 (GRPO) is the most algorithmically complex — it requires:
- Group sampling (sample N completions per prompt)
- Reward computation (verifiable reward; math/code/format/classification/length)
- Group-relative advantage normalization
- KL penalty against reference

The 5 GRPO recipes vary the reward function only; the GRPO loop itself is shared.

PMAT-354 (RLHF/RLAIF/reward modeling) ships the most experimental code — full PPO with reference model, reward model, advantage estimation, GAE. May require workspace-feature gating (`--features rlhf-ppo`).

---

## PMAT-355 — CI gate + book chapter + spec bump

**Priority**: P3
**Estimate**: 0.5 day
**Depends on**: PMAT-331..354

### Scope

1. Make `fine-tuning-coverage` a required (non-advisory) status check on `main`.
2. Add `book/src/finetune/` chapter with TOC linking each tier and recipe.
3. Update `README.md` with a "Fine-Tuning Cookbook" matrix linking to coverage-matrix.md.
4. Bump apr-cookbook spec to v6.3.0 (fine-tuning additive minor).
5. Update `tests/contracts.rs` CONTRACT_FILES with the 100 new contract entries.
6. Update `contracts/binding.yaml` with the 100 new binding entries.

### Definition of Done

- All 100 fine-tuning contracts pass `pv lint`
- All 100 contracts report ≥ Grade A under `pv score --summary`
- `make fine-tuning-coverage` enforced; PMAT bumps required-checks list
- `book/src/finetune/` chapter built and pushed to GitHub Pages
- v6.3.0 tag pushed

---

## Score progression target (per tier)

| Stage | Spec | Falsify | Kani | Lean | Bind | Aggregate | Grade |
|-------|------|---------|------|------|------|-----------|-------|
| PMAT-330 (stubs) | 1.0 | 1.0 | 0.6 | 0.0 | 0.0 | 0.65 | C |
| Each tier landing | 1.0 | 1.0 | 0.9 | 0.7 | 1.0 | 0.91 | A- |
| Tier closeout | 1.0 | 1.0 | 0.9 | 1.0 | 1.0 | 0.98 | A |
| **PMAT-355 (final)** | **1.0** | **1.0** | **0.9** | **1.0** | **1.0** | **0.98** | **A (100/100)** |

Same target as architecture-demos. Lean theorems for SGD-convergence claims may carry `status: not-applicable` (stochastic gradient descent doesn't admit a closed-form Lean proof of monotone-decreasing loss); convergence falsification then becomes a runtime-only invariant via the falsifier test.

---

## Backlog (post-fine-tuning-cookbook v1)

- **Multi-GPU distributed training** (DDP, FSDP, ZeRO-3) — separate v2 spec; depends on `aprender-train` shipping a distributed backend
- **Production-scale dataset preprocessing** — handled by `alimentar` (data-loading recipes already in cookbook); fine-tuning recipes reference but don't re-implement
- **Vision/audio/multimodal at full resolution** — v2; v1 ships 32×32 image stubs only
- **Live HF Hub integration** — explicitly out of scope (CI fixtures only); can be a v2 follow-up if there's a consumer demand
- **Reward-model-as-API serving** (`apr serve --reward-model`) — Tier 4 RLHF recipes may need this; if not shipped in `apr-cli` v0.31.2, file an upstream ticket
