# Fine-Tuning Cookbook — 155 Recipes

This directory ships **155 runnable Rust recipes** that mirror the canonical
fine-tuning surface from Ludwig, Unsloth, TRL, LLaMA-Factory, and Axolotl
against the `apr` CLI and APR-MONO sovereign stack.

> **Status**: 100% complete (Tier 1: 25 ✓, Tier 2: 45 ✓, Tier 3: 48 ✓, Tier 4: 37 ✓).
> Every recipe is verified to run via `cargo run --example <name>` and ships 4
> tests (`recipe_runs`, `falsifier_holds_on_fixture`, `falsifier_breaks_on_perturbed_input`,
> `deterministic_across_runs`). See [`docs/specifications/fine-tuning-cookbook.md`](../../docs/specifications/fine-tuning-cookbook.md)
> for the full spec.

## What was built

| Tier | Theme | Recipes | Sample techniques |
|------|-------|--------:|-------------------|
| **Tier 1** | Foundation | 25 | SFT, eval primitives (perplexity, BLEU, ROUGE-L, F1, accuracy), tabular regression/classification, smoke + bench |
| **Tier 2** | Adaptation | 45 | LoRA r8/r32, QLoRA, continued pretraining, adapter merge (TIES/DARE/SLERP), PEFT variants (CorDA, EVA, PiSSA, LoftQ, OFT, LN-tuning, TinyLoRA, V-Bank, regex-freeze), GaLore, BAdam, Apollo, DoRA, AQLM/AWQ/GPTQ, ReLoRA, LISA, NEFTune |
| **Tier 3** | Specialization | 48 | Instruction tuning (Alpaca/ShareGPT/OpenAssistant/chat templates), hyperopt (grid/random/TPE/ASHA/Hyperband), calibration (temperature/Platt/isotonic/conformal/ensemble), class imbalance (weighted/focal/SMOTE/threshold/cost), multimodal (text+image/text+tabular/multitask/zero-shot/k-fold), anomaly (Deep SAD/SVDD/DROCC), open-set (max-softmax/entropic/objectosphere), uncertainty (MC-dropout/calibrated CI), image encoders (CLIP/DINOv2/SigLIP), optimizers (Muon/schedule-free/L-BFGS), FAMO, SegFormer, JSON-schema decode, Mamba, hypernet, FP8/MXFP4 QAT, sample packing, FSDP-LoRA |
| **Tier 4** | Alignment | 37 | DPO × 5 families, ORPO × 3, KTO × 3, GRPO (math/code-exec/format/classification/length-budget) × 5, RLHF/PPO × 3, RLAIF/Constitutional × 3, reward modeling (pairwise/scalar/ensemble), online preference (online-DPO/XPO/Nash-MD/RLOO/async-GRPO), BCO/CPO/SimPO, PRM, GKD, GSPO, MPO |
| **Total** | | **155** | |

### How the recipes are organized

Each recipe ID is `t<tier>_<technique>[_<variant>].rs`. For example:

```
examples/finetune/
├── t1_sft_minimal_llama.rs            # Tier 1.1 SFT — llama family
├── t1_eval_perplexity.rs              # Tier 1.2 eval primitive
├── t2_lora_rank8_mistral.rs           # Tier 2.1 LoRA
├── t2_qlora_4bit_rank32_phi.rs        # Tier 2.2 QLoRA
├── t3_calibration_temperature.rs      # Tier 3.3 calibration
├── t3_optimizer_muon.rs               # Tier 3.10 optimizer
├── t4_dpo_llama.rs                    # Tier 4.1 DPO
├── t4_grpo_math.rs                    # Tier 4.4 GRPO
└── ...                                # 155 total
```

Logic that's shared across recipes lives in `src/finetune/<topic>.rs` (21 helper
modules, ~3,400 LOC, 130+ unit tests). Each recipe file is intentionally thin
(~50–100 LOC) so a reader can scan it in one screen and see exactly what
fixture, parameters, and falsifier the technique uses.

## How to use these recipes

### 1. Run any single recipe

```bash
cargo run --example t1_sft_minimal_llama
```

Each recipe prints a single `✓` line summarizing the result. Example:

```
✓ llama SFT minimal: loss 21.3601 → 0.7374 (100 steps, ratio 0.035)
✓ DPO β=0.1: correct loss = 0.6444, swapped = 0.7444
✓ GRPO math: 50 steps, reward 0.000 → 0.980
✓ MXFP4 block_size=32: 4.250 bits/elem (target 4.25)
```

If a falsifier asserts cleanly, the binary exits with status 0. If the
assertion fires, the recipe panics with the failing condition — exactly the
same contract the test suite uses.

### 2. Run the test suite for a recipe

Each recipe ships **4 standard tests**. Run them with:

```bash
cargo test --example t2_dora
```

```
running 4 tests
test tests::recipe_runs                       ... ok
test tests::falsifier_holds_on_fixture        ... ok
test tests::falsifier_breaks_on_perturbed_input ... ok
test tests::deterministic_across_runs         ... ok
```

The four tests cover:

| Test | What it proves |
|------|----------------|
| `recipe_runs` | `main()` executes without panicking — the recipe's whole pipeline works end-to-end |
| `falsifier_holds_on_fixture` | The closed-form falsifier passes on the canonical fixture (positive control) |
| `falsifier_breaks_on_perturbed_input` | The falsifier *fails* on a deliberately bad input, proving it has discriminating power (negative control) |
| `deterministic_across_runs` | Two back-to-back invocations produce bit-identical output (no hidden randomness) |

### 3. Run all 155 recipes at once

```bash
# All as binaries (fast: each recipe is closed-form)
for name in $(grep -E '^name = "t[1-4]_' Cargo.toml | sed 's/name = "//; s/"//' | sort -u); do
  cargo run --quiet --example "$name" || echo "FAIL: $name"
done

# Or all 620 recipe tests via cargo test:
cargo test --tests 'finetune::'    # helper unit tests
cargo test --examples              # recipe tests (recipe_runs + 3 falsifier tests each)
```

### 4. Browse by tier or technique

```bash
ls examples/finetune/t1_*.rs   # 25 Tier 1 (foundation)
ls examples/finetune/t2_*.rs   # 45 Tier 2 (adaptation)
ls examples/finetune/t3_*.rs   # 48 Tier 3 (specialization)
ls examples/finetune/t4_*.rs   # 37 Tier 4 (alignment)

ls examples/finetune/t4_dpo_*.rs       # all DPO variants
ls examples/finetune/t2_lora_rank32_*.rs  # all rank-32 LoRA families
```

### 5. Use a recipe as a starting point for your own work

Each recipe is a **standalone, self-contained Rust binary**. To build something
on top:

1. Copy `examples/finetune/<closest_recipe>.rs` to your project.
2. Replace the helper import (`use apr_cookbook::finetune::lora as l;`) with
   the corresponding crate (`use entrenar::*;` for real training, or import
   the helper source directly).
3. Replace the synthetic fixture with your dataset.
4. Tighten or relax the falsifier to match your acceptance criteria.

Because each recipe has a closed-form falsifier (no stochastic eval-uplift
claims), the assertion translates directly into a CI gate for your downstream
fine-tuning pipeline.

## The recipe pattern (IIUR + provable falsifier)

Every recipe satisfies four invariants from the apr-cookbook spec:

| Invariant | What it means |
|-----------|---------------|
| **Idempotent** | Running the recipe twice produces the same output (`deterministic_across_runs` test). |
| **Isolated** | Operates inside a `RecipeContext` with its own seeded RNG and tempdir. |
| **Unobservable** | No network, no global state mutation; safe to run in CI. |
| **Repeatable** | Bit-identical output on cold cache. |

In addition, every recipe carries a **falsifier** — a closed-form invariant that
the technique *must* satisfy. This is what makes the cookbook genuinely useful
as a reference: each recipe's assertion is a property a real implementation
must also satisfy. Examples:

| Technique | Falsifier |
|-----------|-----------|
| LoRA | merge → unmerge round-trip is bit-identical when α/r=1 |
| QLoRA | 4-bit base + LoRA storage ≤ 0.4× of FP16 baseline |
| OFT | R^T R = I within ε=1e-4 |
| TinyLoRA | trainable ratio ≤ 0.06% on a 4096×4096 base |
| DPO | `dpo_loss(chosen, rejected) < dpo_loss(rejected, chosen)` |
| GRPO advantages | sum to zero (mean-centered) |
| PPO clipping | clipped ratio ∈ [1−ε, 1+ε] |
| Hyperband | R=81, η=3 → exactly 5 brackets |
| MXFP4 | 4·N + 8 bits → 4.25 bits/elem at N=32 |
| Conformal at α=0.1 | empirical coverage ≈ 0.9 |

## Helper modules

Mathematical primitives are in `src/finetune/` so the recipe files can stay
thin. Each helper has its own unit-test suite (130+ tests total).

| Module | Topic | Recipes consuming it |
|--------|-------|---------------------|
| `sft_minimal` | Linear regression SGD | 5 SFT family recipes |
| `eval_primitives` | perplexity, BLEU, ROUGE-L, F1 | 5 eval recipes |
| `tabular_regression` | OLS via Gaussian elimination | 5 regression recipes |
| `tabular_classification` | NCM, top-k, macro-F1 | 5 classification recipes |
| `smoke` | plan/resume/early-stop/dry-run/bench | 5 smoke recipes |
| `lora` | LoRA layer + merge/unmerge | 10 LoRA recipes |
| `qlora` | 4-bit absmax-quantization + double-quant | 5 QLoRA recipes |
| `continued_pretrain` | UnigramLM + CP report | 5 CP recipes |
| `adapter_merge` | average / SLERP / DARE / TIES / multi-LoRA | 5 merge recipes |
| `peft_variants` | CorDA / EVA / PiSSA / LoftQ + OFT / LN / TinyLoRA / V-Bank / regex-freeze | 9 PEFT recipes |
| `memory_optimizers` | GaLore / BAdam / Apollo / DoRA / freeze | 5 optimizer recipes |
| `quantized_base` | AQLM / AWQ / GPTQ / ReLoRA / LISA / NEFTune | 6 quant-base + alignment recipes |
| `instruction_tuning` | Alpaca / ShareGPT / OpenAssistant / chat templates | 5 instruction recipes |
| `hyperopt` | grid / random / TPE / ASHA / Hyperband | 5 hyperopt recipes |
| `calibration` | temperature / Platt / isotonic / conformal / ensemble | 5 calibration recipes |
| `imbalance` | weighted / focal / SMOTE / threshold / cost-sensitive | 5 imbalance recipes |
| `multimodal` | concat / gated fusion + multitask + zero-shot + k-fold | 5 multimodal recipes |
| `anomaly_open_uncertainty` | Deep SAD/SVDD/DROCC + open-set + MC-dropout + CI | 8 specialty recipes |
| `encoders_optimizers` | linear probe + cosine sim + Muon + schedule-free | 5 encoder/optimizer recipes |
| `specialty` | L-BFGS, FAMO, SegFormer, JSON, Mamba, hypernet | 6 single-recipe sub-sections |
| `tier3_closeout` | FP8 / MXFP4 / sample packing / FSDP-LoRA | 4 closeout recipes |
| `preference` | DPO / ORPO / KTO loss + implicit reward | 11 preference recipes |
| `rl_alignment` | GRPO / PPO clip / adaptive KL | 8 RL alignment recipes |
| `rlaif_reward` | Pearson / refusal-uplift / pairwise / R² / ensemble | 6 RLAIF recipes |
| `online_alt` | online-DPO / XPO / Nash-MD / RLOO / BCO / CPO / SimPO | 7 online + alt-loss recipes |
| `tier4_closeout` | async-GRPO / PRM / GKD / GSPO / MPO | 5 closeout recipes |

## Provable contracts

Every recipe has a matching YAML contract under `contracts/finetune-t<n>-<id>-v1.yaml`
and a Lean module under `lean/ProvableContracts/Finetune/T<N><Name>.lean`. CI
gates these via:

```bash
cargo test --test contracts                   # validates all 155 stubs parse
cargo run -p aprender-contracts-cli -- lint contracts/      # spec-depth scoring
make fine-tuning-coverage                     # manifest ↔ disk reconciliation
```

The Lean modules ship the standard three-theorem skeleton (`Totality`,
`Determinism`, plus a per-recipe `Property` placeholder). All 155 modules
import in `lean/ProvableContracts.lean` and compile under Lean 4.

## What this is *not*

- **Not a benchmark suite.** The recipes use small synthetic fixtures so they
  run in milliseconds. They prove the *math* of each technique, not its
  performance on a leaderboard.
- **Not a replacement for `apr` CLI.** The `apr finetune`, `apr merge`,
  `apr distill`, `apr quantize` CLIs are the real entry points. The cookbook
  recipes are reference implementations of the algorithms behind those
  subcommands.
- **Not a Python wrapper.** Every recipe is pure Rust. No PyO3, no
  CUDA-Python, no `transformers` import. The fixtures and helpers are
  closed-form so the cookbook stays portable, deterministic, and under 60s
  to run end-to-end on a laptop.

## See also

- [`docs/specifications/fine-tuning-cookbook.md`](../../docs/specifications/fine-tuning-cookbook.md) — full v1.2.0 spec
- [`docs/specifications/fine-tuning-cookbook/manifest.yaml`](../../docs/specifications/fine-tuning-cookbook/manifest.yaml) — single source of truth (155 entries)
- [`docs/specifications/fine-tuning-cookbook/recipe-template.md`](../../docs/specifications/fine-tuning-cookbook/recipe-template.md) — canonical recipe shape
- [`scripts/finetune-gen.sh`](../../scripts/finetune-gen.sh) — auto-regenerates contract stubs and fixture dirs
- `make fine-tuning-coverage` — CI gate reconciling manifest ↔ disk
