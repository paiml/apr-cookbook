# Recipe Template

Every fine-tuning-cookbook recipe follows the canonical apr-cookbook IIUR shape (idempotent / isolated / unobservable / repeatable) and ships a per-recipe provable contract. This document is the load-bearing template — every PMAT-330..354 recipe must match this shape exactly.

## File layout per recipe

```
examples/finetune/<id>.rs               # the recipe (~150 LOC)
contracts/finetune-<id>-v1.yaml         # provable contract (Spec/Falsify/Kani/Lean/Bind axes)
lean/ProvableContracts/Finetune/<Id>.lean  # Lean theorems (status: proved or not-applicable)
tests/fixtures/finetune/<id>/           # bundled deterministic dataset (≤ 1 MB)
  data.jsonl                            # (or data.csv for tabular)
  expected.json                         # what the recipe should produce
  README.md                             # one-line origin + license
```

The `<id>` here is the manifest's `id` field — e.g. `t1_sft_minimal_llama`, `t2_lora_rank8_mistral`.

## Recipe doc-header (required)

```rust
//! # <Tier N — Section> — <Title>
//!
//! <One-paragraph description: what's fine-tuned, on what data, what's the
//! observable outcome.>
//!
//! Demonstrates the **<TECHNIQUE>** recipe per
//! `docs/specifications/fine-tuning-cookbook.md` v1.0 (PMAT-NNN).
//!
//! ## Mirror
//!
//! - Ludwig: <path/name or "n/a">
//! - Unsloth: <path/name or "n/a">
//! - apr-native: <true|false — and why if true>
//!
//! ## Falsifiable claim
//!
//! <The one falsifiable property this recipe asserts. Should match
//! manifest.yaml's `falsifier:` field.>
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/finetune-<id>-v1.yaml (target Grade A; lean_status: proved)
//! Citation: <arXiv / blog / HF doc / Ludwig docs URL>
//!
//! Run with: cargo run --example <id>
//!
//! Added by PMAT-NNN.
```

## Recipe `main()` shape

```rust
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn main() -> Result<()> {
    let ctx = RecipeContext::new("<id>")?;

    // ---- 1. Load synthetic fixture (≤ 1 MB, bundled) ----
    let data_path = "tests/fixtures/finetune/<id>/data.jsonl";

    // ---- 2. Run the apr workflow under test ----
    //   For SFT recipes: invoke aprender::format::apr_finetune
    //   For LoRA: aprender::format::apr_lora_apply
    //   For DPO/ORPO/KTO/GRPO: aprender-train (entrenar)::rl::*
    //   etc. See manifest.yaml apr_subcommand[] for which.
    let result = run_workflow(data_path)?;

    // ---- 3. Assert the falsifiable claim ----
    //   The single most important line. Must be a concrete numeric or
    //   structural assertion, not a smoke check.
    assert_falsifier(&result)?;

    // ---- 4. Emit a human-readable verdict ----
    println!("✓ <id> passed: <falsifier-text> (final value: {})", result.metric);
    Ok(())
}

fn run_workflow(data_path: &str) -> Result<Workflow Result> { /* ... */ }
fn assert_falsifier(r: &WorkflowResult) -> Result<()> { /* ... */ }
```

Constraints:

- ≤ 200 LOC per recipe (file-size invariant)
- Cyclomatic complexity ≤ 20 per function
- No `unwrap()` in the main path
- Deterministic (fixed seed `42`)
- CPU-only path runs in ≤ 60 s on a stable laptop; GPU-only paths gated `#[cfg(feature = "cuda")]`

## Test block (mandatory)

Every recipe ships a `#[cfg(test)] mod tests` block with at minimum:

1. **`recipe_runs`** — invokes `main()` and asserts no panic.
2. **`falsifier_holds_on_fixture`** — re-asserts the manifest's falsifier on the bundled fixture.
3. **`falsifier_breaks_on_perturbed_input`** — feeds a deliberately-broken input and asserts the falsifier *fails* (Popperian discipline).
4. **`deterministic_across_runs`** — two consecutive `main()` calls produce equal outputs.

Tests run as part of `cargo test --lib --tests` (the post-PMAT-328 scoped Quality Gate).

## Contract template

Every certified recipe ships `contracts/finetune-<id>-v1.yaml` modeled on `contracts/inference-arch-resolution-pipeline-v1.yaml`:

```yaml
metadata:
  version: "1.0.0"
  created: "2026-MM-DD"
  author: "PAIML Engineering"
  description: "<technique> on <base_family> — <one-line>"
  references:
    - "docs/specifications/fine-tuning-cookbook.md"
    - "docs/specifications/fine-tuning-cookbook/manifest.yaml#<id>"
    - "<upstream Ludwig/Unsloth/TRL link>"
  depends_on: ["recipe-iiur-v1"]
  tags: [finetune, t<N>, <technique>, <base_family>]

kernel_structure:
  phases:
    - { name: setup,    description: "...", invariant: "..." }
    - { name: train,    description: "...", invariant: "..." }
    - { name: eval,     description: "...", invariant: "..." }
    - { name: verify,   description: "...", invariant: "..." }
    - { name: teardown, description: "...", invariant: "..." }

equations:
  convergence:
    formula: "loss(t) is non-increasing on the convex sub-objective"
    domain: "step ∈ {0..T}"
    codomain: "ℝ_{≥0}"
    invariants:
      - "training is deterministic for fixed seed"
      - "<technique-specific invariant 1>"
    preconditions:
      - "fixture is bounded, valid UTF-8 JSONL"
    postconditions:
      - "loss[T] ≤ loss[0]"
    lean_theorem: "ProvableContracts.Finetune.<Id>.Convergence"
    tolerance: 0.0
    lean:
      theorem: "ProvableContracts.Finetune.<Id>.Convergence"
      status: not-applicable    # SGD convergence is stochastic; honest default
      module: "lean/ProvableContracts/Finetune/<Id>.lean"

  determinism:
    formula: "main() with seed=42 produces equal output across runs"
    # Same shape as architecture-demos contracts.
    lean:
      theorem: "ProvableContracts.Finetune.<Id>.Determinism"
      status: proved
      module: "lean/ProvableContracts/Finetune/<Id>.lean"

  totality:
    formula: "main() returns Result<()> for any well-formed fixture (no panic)"
    lean:
      theorem: "ProvableContracts.Finetune.<Id>.Totality"
      status: proved
      module: "lean/ProvableContracts/Finetune/<Id>.lean"

falsification_tests:
  - id: FALSIFY-FT-<ID>-001
    rule: "<falsifier text from manifest>"
    test: "cargo test --example <id> -- falsifier_holds_on_fixture"
    if_fails: "<technique> regression — investigate hyperparams or fixture"
  - id: FALSIFY-FT-<ID>-002
    rule: "Perturbed input breaks the falsifier"
    test: "cargo test --example <id> -- falsifier_breaks_on_perturbed_input"
    if_fails: "Falsifier is too weak — accepts inputs it should reject"
  - id: FALSIFY-FT-<ID>-003
    rule: "Deterministic across runs"
    test: "cargo test --example <id> -- deterministic_across_runs"
    if_fails: "Non-determinism leaked into the workflow (clock, RNG, threading)"

kani_harnesses:
  # 3 stub harnesses following the architecture-demos pattern.
  - id: KANI-FT-<ID>-001
    obligation: "<convergence-style claim, bounded-int proxy>"
    bound: 4
    strategy: bounded_int
    harness: "kani_harnesses::ft_<id>_convergence"
  # ... etc

qa_gate:
  id: F-FT-<ID>-001
  name: "Fine-tuning <id> contract"
  description: "<technique> on <base_family> with falsifier <id-fragment>"
  checks:
    - convergence_falsifier_holds
    - perturbed_input_falsifier_fails
    - deterministic_across_runs
  pass_criteria: "All 3 falsification tests pass on bundled fixture"
```

## Lean module template

```lean
-- Theorems for `contracts/finetune-<id>-v1.yaml`.
namespace ProvableContracts.Finetune.<Id>

/-- main() returns Result<()> on any well-formed fixture (totality). -/
theorem Totality (run : Nat → Option Nat) (input : Nat) :
    ∃ v : Option Nat, run input = v := ⟨run input, rfl⟩

/-- Two consecutive runs over the same input produce equal output (determinism). -/
theorem Determinism (run : Nat → Option Nat) (input : Nat) :
    run input = run input := rfl

/-- SGD convergence is stochastic; modeled as decreasing on convex
    sub-objective. Lean status: not-applicable for stochastic obligations.
    Recipes that exercise convex sub-problems (linear regression head, …)
    can promote this to `proved`. -/
theorem Convergence : True := trivial

end ProvableContracts.Finetune.<Id>
```

## Fixture template

```
tests/fixtures/finetune/<id>/
├── README.md                    # one-line origin + license + size
├── data.jsonl                   # (or data.csv for tabular)
└── expected.json                # what the recipe should produce
```

`README.md`:

```
# <id> fixture

Synthetic deterministic dataset for the <id> recipe.

- Generated by: scripts/finetune-fixtures.py --recipe=<id> --seed=42
- Size: ~<NN>KB (target ≤ 1MB; CI fails otherwise)
- License: PAIML / public-domain
- Last regenerated: 2026-MM-DD
```

Fixtures are committed (small) so CI doesn't need network access. Regen script is committed too so reviewers can verify reproducibility.

## CI integration

PMAT-355 adds `.github/workflows/fine-tuning.yml` with:

1. `make fine-tuning-coverage` — runs `bash scripts/finetune-gen.sh --check` (manifest reconciliation, mirror parity, fixture-size budget)
2. `cargo test --lib --tests --features finetune-fixtures` — runs the recipe test suites
3. Optional `cargo build --examples=t1_*,t2_*,t3_*,t4_*` (cron only) — verifies all 100 examples compile

## Authoring checklist

When landing a new recipe (PMAT-331..354):

- [ ] `examples/finetune/<id>.rs` follows the doc-header + main() shape
- [ ] `tests/fixtures/finetune/<id>/data.{jsonl,csv}` exists, ≤ 1 MB, deterministic
- [ ] `contracts/finetune-<id>-v1.yaml` validates with `pv lint`
- [ ] `lean/ProvableContracts/Finetune/<Id>.lean` compiles with `lake build`
- [ ] Recipe added to `Cargo.toml` `[[example]]` block
- [ ] Contract added to `tests/contracts.rs` CONTRACT_FILES
- [ ] Manifest entry status flipped from `planned` → `certified`
- [ ] `cargo test --lib --tests` green
- [ ] `pv score contracts/finetune-<id>-v1.yaml --binding contracts/binding.yaml --summary` reports Grade A (≥ 0.93)
- [ ] `make fine-tuning-coverage` green
