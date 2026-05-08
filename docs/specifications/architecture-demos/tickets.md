# PMAT Ticket Breakdown

Architecture-demos work has shipped two waves: **v1 (PMAT-300..308, 9 tickets)** built one smoke recipe per family with a provable-contract; **v1.1 (PMAT-309..314, 6 tickets)** added cross-family meta-recipes plus the upstream-bridge to `aprender::format::FamilyRegistry` ([aprender#1562](https://github.com/paiml/aprender/pull/1562)).

Each ticket authors recipe(s) + provable-contract YAML + (where applicable) fixture + tests, gating manifest-flip on `pv lint` + `pv score` grade C minimum. PMAT numbers are real ticket IDs assigned at landing time.

| Ticket | Scope | Priority | Estimate | Depends on |
|--------|-------|----------|----------|------------|
| PMAT-300 | Generator + manifest + contract-stub scaffolding | P1 | 1.5 days | spec acceptance |
| PMAT-301 | Llama family (recipe + contract) + 10 alias unblock prep | P1 | 2.5 days | PMAT-300 |
| PMAT-302 | Mistral family (recipe + contract) + 6 alias unblock prep | P1 | 2.5 days | PMAT-300 |
| PMAT-303 | Qwen family batch (recipes + 3 contracts) | P1 | 3 days | PMAT-300 |
| PMAT-304 | Phi family (recipe + contract) + companion_converter wiring | P2 | 1.5 days | PMAT-300 |
| PMAT-305 | Gemma + GPT-2 + GPT-NeoX (recipes + 3 contracts) | P2 | 3 days | PMAT-300 |
| PMAT-306 | DeepSeek + Falcon-H1 + RWKV-7 (recipes + 3 contracts) | P2 | 3 days | PMAT-300 |
| PMAT-307 | OpenELM + OPT + MAMBA + BERT (recipes + 4 contracts) | P2 | 3 days | PMAT-300 |
| PMAT-308 | Coverage-matrix CI gate + Invariant B baseline bump + book | P3 | 0.5 day | 301..307 |

Total: ~20.5 dev-days (vs 14.5 IIUR-only — contract authoring adds ~0.5 day per family). Sister-ticket batches are pure parallel (no inter-ticket deps after PMAT-300), so realistic calendar is ~6 days with parallelism.

---

## PMAT-300 — Generator + manifest + contract scaffolding (prerequisite)

**Priority**: P1
**Estimate**: 1.5 days (extended for contract-stub generation)
**Depends on**: spec acceptance

### Scope

1. Create `scripts/architecture-demos-gen.sh` per [generator.md](generator.md), with five `--target` modes including `contracts`.
2. Create `scripts/architecture-demos-gen-fixture.py` for synthetic micro-checkpoint generation.
3. Implement provable-contract stub generation: walk the manifest, emit `contracts/inference-<family>-smoke-v1.yaml` from the skeleton in [recipe-template.md § Per-family contract skeleton](recipe-template.md). Stubs all start at `lean.status: wip`.
4. Wire `--check` into `.github/workflows/architecture-demos.yml` as a required check. The check runs `pv lint` against every `contracts/inference-*-smoke-v1.yaml` file.
5. Add `make architecture-demos-coverage` target invoking `--check`.
6. First-pass manifest (already drafted in [manifest.yaml](manifest.yaml)).
7. First-pass coverage matrix (already drafted in [coverage-matrix.md](coverage-matrix.md)).

### Definition of Done

- `bash scripts/architecture-demos-gen.sh --check` returns 0 on a fresh checkout (manifest is internally consistent even with zero recipes — `in-progress` is a legal state pre-implementation).
- `--update --target contracts` emits 16 `contracts/inference-*-smoke-v1.yaml` stubs that all pass `pv lint`.
- `--update` writes to disk; `git diff` shows expected files.
- `pv lint manifest.schema.yaml` passes.
- CI workflow committed and runs on PR touching `docs/specifications/architecture-demos/**`.

---

## PMAT-301 — Llama family + alias backlog ticket

**Priority**: P1
**Estimate**: 2.5 days (extended for contract authoring)
**Depends on**: PMAT-300

### Scope

1. Generate fixture `tests/fixtures/architectures/llama/` (2-layer Llama, ~256 MB → compress to <1 MB via aggressive synthetic seeding).
2. Implement `examples/inference/inference_llama_smoke.rs` per [recipe-template.md](recipe-template.md).
3. Demonstrate all three formats: safetensors, apr, gguf.
4. Author `contracts/inference-llama-smoke-v1.yaml` — fill in concrete pre/postconditions for `loader_dispatch`, `tensor_validation`, `forward_determinism`. Reference Lean theorems (`Theorems.Llama.LoaderDispatch`, etc.) at `lean.status: wip` until proofs land.
5. Flip manifest entry `llama: status: in-progress` → `certified` only after `pv lint` and `pv score --summary` (grade C minimum) both pass.
6. Open upstream aprender ticket `aprender#1562` (alias mechanism resolved) for codellama/dolphin/hermes/openchat/smollm/smollm2/tinyllama/vicuna/wizardcoder/yi (10 aliases unblock as a batch when this lands).
7. Update [coverage-matrix.md](coverage-matrix.md) via generator.

### Definition of Done

- `cargo test --example inference_llama_smoke` passes.
- Fixture committed and < 1 MB.
- `pv lint contracts/inference-llama-smoke-v1.yaml` exits 0.
- `pv score contracts/inference-llama-smoke-v1.yaml --summary` reports grade C minimum.
- Manifest entry flipped to `certified` with `lean_status: wip`.
- Upstream ticket opened and linked in the manifest's `notes` field.

---

## PMAT-302 — Mistral family + alias backlog ticket

**Priority**: P1
**Estimate**: 2.5 days (extended for contract authoring)
**Depends on**: PMAT-300

### Scope

Same shape as PMAT-301 for `mistral` — recipe + provable-contract + Lean theorem refs. Open upstream ticket for codestral/dolphin/hermes/openchat/wizardcoder/zephyr alias batch (6 aliases unblock together).

### Definition of Done

- `cargo test --example inference_mistral_smoke` passes.
- `pv lint contracts/inference-mistral-smoke-v1.yaml` exits 0.
- `pv score contracts/inference-mistral-smoke-v1.yaml --summary` reports grade C minimum.
- Manifest entry flipped to `certified`.
- Upstream alias ticket opened.

---

## PMAT-303 — Qwen family batch

**Priority**: P1
**Estimate**: 2 days
**Depends on**: PMAT-300

### Scope

Three recipes in one ticket because the Qwen family shares fixture-generation tooling:
- `examples/inference/inference_qwen2_smoke.rs`
- `examples/inference/inference_qwen3_smoke.rs`
- `examples/inference/inference_qwen3_5_smoke.rs`

The existing `inference_qwen3_moe_numerical_parity_smoke.rs` stays — it's a deeper test recorded as `companion_smoke` in the manifest.

### Definition of Done

- All 3 recipes pass tests.
- Manifest entries for qwen2, qwen3, qwen3_5 all flipped to `certified`.
- Companion smoke recipe linkage preserved.

---

## PMAT-304 — Phi family + companion converter wiring

**Priority**: P2
**Estimate**: 1 day
**Depends on**: PMAT-300

### Scope

1. Implement `examples/inference/inference_phi_smoke.rs`.
2. Verify the existing `examples/conversion/convert_phi_to_apr.rs` still works against the updated `aprender::rosetta` Phi loader; update the manifest's `companion_converter` field if the path changed.
3. Link the two recipes in book chapter mark-up.

### Definition of Done

- Both recipes (smoke + converter) pass tests.
- Manifest entry flipped to `certified`.
- Book TOC under `book/src/inference/` links smoke + converter.

---

## PMAT-305 — Gemma + GPT-2 + GPT-NeoX

**Priority**: P2
**Estimate**: 2 days
**Depends on**: PMAT-300

### Scope

Three recipes covering the Google/EleutherAI/OpenAI legacy decoder cluster:
- `inference_gemma_smoke.rs`
- `inference_gpt2_smoke.rs`
- `inference_gptneox_smoke.rs`

Gemma demonstrates GGUF; GPT-2 demonstrates f32 (smallest models still ship full-precision); GPT-NeoX is large-only (size_category: large) so the fixture uses a 2-layer minimal config rather than scaled-down weights.

### Definition of Done

- All 3 recipes pass tests.
- Manifest entries flipped to `certified`.
- Open `aprender#1562` (alias mechanism unblocks codegemma + distilgpt2 + pythia) (3 aliases unblock).

---

## PMAT-306 — DeepSeek + Falcon-H1 + RWKV-7

**Priority**: P2
**Estimate**: 2 days
**Depends on**: PMAT-300

### Scope

Three "newer non-Llama" families:
- `inference_deepseek_smoke.rs` (V2/V3 architectures, MoE-aware)
- `inference_falcon_h1_smoke.rs` (hybrid SSM-transformer)
- `inference_rwkv7_smoke.rs` (linear-attention)

Each recipe demonstrates the specific architectural twist (MoE routing, SSM state, RWKV time-mix) in a comment block alongside the smoke pass.

### Definition of Done

- All 3 recipes pass tests.
- Manifest entries flipped to `certified`.

---

## PMAT-307 — OpenELM + OPT + MAMBA + BERT

**Priority**: P2
**Estimate**: 2 days
**Depends on**: PMAT-300

### Scope

Four recipes — closes the in-progress backlog:
- `inference_openelm_smoke.rs` (Apple, layer-wise scaling)
- `inference_opt_smoke.rs` (Meta, classic decoder)
- `inference_mamba_smoke.rs` (state-space)
- `inference_bert_smoke.rs` (encoder-only, MLM head — NOT decoder)

BERT is the only encoder-only family in v1; the verdict shape may need a fourth `MaskedLMOk` arm to capture the difference. Recipe-template doc updated accordingly.

### Definition of Done

- All 4 recipes pass tests.
- Manifest entries flipped to `certified`.
- Open `aprender#1562` (alias mechanism unblocks galactica) (OPT-shared).
- Recipe-template note added for encoder-only verdict shape.

---

## PMAT-308 — Coverage-matrix CI gate + book chapter

**Priority**: P3
**Estimate**: 0.5 day
**Depends on**: PMAT-301..307

### Scope

1. Make `architecture-demos-coverage` a required (non-advisory) status check, including `pv lint` over all 16 family contracts.
2. Extend Invariant B (Recipe Contract Grade A) baseline to include `contracts/inference-*-smoke-v1.yaml` — should add 16 grade-A contracts to the existing 341/341 baseline (357/357 post-architecture-demos).
3. Add `book/src/architecture-demos/` chapter with TOC linking each smoke recipe + its contract.
4. Update `README.md` with a "Supported Architectures" matrix linking to coverage-matrix.md.
5. Bump apr-cookbook spec to v6.2.0 (architecture-demos additive minor).

### Definition of Done

- All 16 in-progress families now `certified` in the manifest, all with `lean_status: wip` minimum.
- All 16 contracts pass `pv lint` and report grade C minimum under `pv score --summary`.
- CI gate enforced on main; Invariant B baseline extended.
- mdbook builds with new architecture-demos section.
- README badge updated.
- v6.2.0 tag pushed.

---

## v1.1 Expansion (PMAT-309..318) — Cross-Family Meta + Upstream Bridge + Grade A Sweep

After v1 closeout (PMAT-308), the spec was extended with cross-family meta-recipes, an upstream-contribution bridge, and a contract-quality sweep that lifted **all 23 architecture-demos contracts to Grade A 0.98**.

| Ticket | Scope | PR |
|--------|-------|----|
| PMAT-309 | Cross-family detector — discriminator-dispatch from raw config.json | #413 |
| PMAT-310 | Family summary — discriminator catalog across 16 families | #413 |
| PMAT-311 | Cross-family compare — diff two configs, classify FamilyRelation | #414 |
| PMAT-312 | Quirk audit — flag configs matching >1 family discriminator | #414 |
| PMAT-313 | Upstream bridge — alias-resolver demo + [aprender#1562](https://github.com/paiml/aprender/pull/1562) | #415 + aprender#1562 |
| PMAT-314 | Spec retrofit — bump status, document v1.1, refresh upstream-ticket links | #416 |
| PMAT-315 | Contract bindings — D5 Binding 0.0→1.0 (51 binding-registry entries) | #417 |
| PMAT-316 | 1:1 Kani harness coverage — D3 0.6→0.9 (3rd harness per contract) | #417 |
| PMAT-317 | Real Lean proofs — D4 0.0→1.0 (23 modules, zero `sorry`) | #418 |
| PMAT-318 | Moonshine binding — final 23/23 contract at Grade A 0.98 | #419 |

Each v1.1 ticket follows the same IIUR + provable-contract discipline as v1. The 5 meta-recipes ship 68 unit tests across the family suite. The contract-quality sweep (PMAT-315..318) lifted aggregate from 0.65 Grade C to **0.98 Grade A** across all 23 contracts.

## v1.2 Forward-Bridge (PMAT-320) — Composed Resolution Pipeline

aprender#1562 is open but not yet merged to aprender main as of 2026-05-08. PMAT-320 delivers consumer-side progress that doesn't depend on the upstream merge: a composed `(hf_repo, body) → DetectedFamily` recipe that exercises the future `FamilyRegistry::resolve_alias` + `detect_from_config_str` shape against the cookbook's reverse implementation.

| Ticket | Scope | Status |
|--------|-------|--------|
| PMAT-320 | Forward-bridge resolution pipeline — `inference_arch_resolution_pipeline` + provable-contract + Lean module + 3 Kani harnesses | landed (this PR) |

The recipe has 10 new unit tests; one of them (`all_alias_eligible_resolve_to_parent`) is the falsification claim that all 16 alias-eligible blocked manifest entries resolve to a known parent — proving the alias table stays in sync with the manifest. When aprender#1562 ships, the pipeline body becomes a thin wrapper over the upstream API and the 46 inherited tests become integration tests for upstream behavior.

Total architecture-demos contracts after PMAT-320: **24**, all at Grade A 0.98.

## Score progression (architecture-demos contracts)

| Stage | Spec | Falsify | Kani | Lean | Bind | Aggregate | Grade |
|-------|------|---------|------|------|------|-----------|-------|
| PMAT-300 (stubs) | 1.0 | 1.0 | 0.6 | 0.0 | 0.0 | 0.65 | C |
| PMAT-315 (bindings) | 1.0 | 1.0 | 0.6 | 0.0 | 1.0 | 0.85 | B |
| PMAT-316 (kani 3rd) | 1.0 | 1.0 | 0.83 | 0.0 | 1.0 | 0.93 | A |
| PMAT-317 (Lean) | 1.0 | 1.0 | 0.9 | 1.0 | 1.0 | 0.98 | A (22/23) |
| **PMAT-318 (moonshine)** | **1.0** | **1.0** | **0.9** | **1.0** | **1.0** | **0.98** | **A (23/23)** |

## Backlog (post-architecture-demos v1.2)

- **Refactor cookbook detector to consume upstream API** — when aprender ships a release containing #1562, bump the cookbook's aprender pin and replace `inference_arch_detector.rs` body with a thin call to `FamilyRegistry::detect_from_config_str`. The 22 detector tests become integration tests for the upstream API. PMAT-320 already shipped the consumer-side composed pipeline (`inference_arch_resolution_pipeline`) so when upstream merges, the swap is a one-line body change inside that recipe plus the detector. **Status: blocked on external aprender release.** Last verified upstream state (2026-05-08): aprender#1562 is open on a parallel branch, 10 commits behind aprender main; not in published 0.32.0.
- **Resolution of `status: blocked` entries** — 16 of 25 are alias-eligible (codellama, tinyllama, vicuna, yi, smollm, smollm2, dolphin, hermes, openchat, wizardcoder, codestral, zephyr, distilgpt2, pythia, galactica, codegemma) and unblock via aprender#1562's `register_alias` once it ships. The cookbook-side alias table is already proved in `inference_arch_resolution_pipeline::all_alias_eligible_resolve_to_parent` so flipping the manifest entries from `blocked` → `aliased` is a one-shot manifest edit once upstream merges. The remaining 9 (bloom, falcon-classic, granite, internlm2_5, nemotron, olmo, stablelm, starcoder2, tiny_starcoder_py) need new upstream loaders, deferred per scope non-goal. **Status: blocked on external aprender work.**
- **D3 Kani 0.9 → 1.0** — would require actual `#[kani::proof]` Rust harness function implementations for the 72 declared harness names (currently declared in YAML but no Rust impl beyond the ones already added in PMAT-320). Substantial separate effort; current 0.9 already exceeds every other flagship contract's Kani score (mmap 0.83, avx512 not measured).
- **Vision/audio/multimodal expansion** (LLaVA, SDXL, ViT) — separate v2 spec, not in current scope.
- **ONNX format coverage** — none of the 18 in-progress loaders ship ONNX; awaits upstream support.

## Lifted from backlog (resolved in v1.1 / v1.2)

- ~~**Lift contracts C→A**~~ — **DONE** in PMAT-315..318. All 23 architecture-demos contracts now at 0.98 Grade A (Spec 1.0 / Falsify 1.0 / Kani 0.9 / Lean 1.0 / Bind 1.0). Exceeds every prior cookbook flagship: mmap 0.91, whisper 0.94, avx512 0.73, flash-attention 0.71. PMAT-320 added a 24th contract at the same grade.
- ~~**Stale coverage-matrix.md**~~ — **DONE** in PMAT-320. Matrix body now reflects 18 certified (was claiming 2 certified / 16 in-progress / 25 blocked) and includes a cross-family meta-recipes section enumerating PMAT-309..313 + PMAT-320 deliverables.
