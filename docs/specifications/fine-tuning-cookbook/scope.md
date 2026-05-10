# Scope & Charter

## Decision

Add **≥ 100 fine-tuning recipes** to apr-cookbook organized into a four-tier curriculum (simple → hard) that mirrors Ludwig (declarative tabular + LLM) and Unsloth (LoRA / QLoRA / GRPO). Every recipe wraps the existing `apr` CLI surface — no new training engines, no new optimizers; just curated workflows that exercise `apr finetune`, `apr eval`, `apr quantize`, `apr serve`, `apr merge`, `apr distill`, `apr prune`, `apr chat` against the 18 certified architecture families.

## Why Now

Cookbook v6.2.0 closed architecture-demos: 18 families × 3 obligations × Grade A. Coverage of *what models* is comprehensive. Coverage of *how to train them* is shallow:

| Cookbook today | Count | Notes |
|---------------|-------|-------|
| `apr finetune` recipes | 4 | apr_finetune_lora_apply, qlora_minimal, etc. |
| `entrenar` autograd examples | 7 | tensor ops, optimizer steps |
| Distillation recipes | 14 | knowledge transfer, but not SFT-style |
| RLHF / DPO / GRPO | 0 | not represented |
| Tabular fine-tuning | 0 | Ludwig's flagship |
| Hyperparameter optimization | 0 | despite `apr tune` shipping |

Versus Ludwig + Unsloth:

| Source | Recipe shape | Count |
|--------|-------------|-------|
| Ludwig examples (`ludwig.ai/examples/`) | Tabular + text + multimodal + hyperopt | ~50 published |
| Unsloth notebooks (`unslothai/notebooks`) | LoRA + GRPO + DPO + Vision + TTS | ~60 |
| HuggingFace TRL | DPO + ORPO + KTO + PPO | ~20 |
| Total reference surface | | **~130** |

The cookbook should sit in the middle: 100 curated recipes, each exercising one falsifiable claim, organized so a reader walks simple → hard.

## Four-Tier Curriculum

### Tier 1 — Foundations (25 recipes, simple)

Goal: someone who has never run `apr` end-to-end can complete a finetune+eval cycle inside 5 minutes against a synthetic fixture.

- **1.1 SFT minimal (5 recipes)** — single epoch, 100-row JSONL, one base family per recipe (llama, mistral, phi, qwen, gemma). Each lands `apr finetune --base <family> --data fixtures/sft-100/data.jsonl --epochs 1` then `apr eval --metric loss`.
- **1.2 Eval primitives (5 recipes)** — perplexity, accuracy, F1, ROUGE-L, BLEU; each on the same 100-sample fixture, no training.
- **1.3 Tabular regression (5 recipes)** — `apr finetune --task regression` on synthetic numeric CSVs (housing-prices, energy-consumption, time-series, multi-target, with-missing-values).
- **1.4 Tabular classification (5 recipes)** — binary, multi-class (3, 7, 100 classes), with-class-imbalance.
- **1.5 Smoke + bench (5 recipes)** — `apr finetune --plan` (no actual training), `apr finetune --resume`, `apr finetune --early-stop`, `apr bench finetune`, `apr finetune --dry-run`.

Mirrors: Ludwig `getting_started/`, `titanic/`, `wine_quality/`, `synthetic/`. Unsloth: minimal SFT notebooks.

### Tier 2 — Adaptive Methods (25 recipes, intermediate)

Goal: parameter-efficient fine-tuning at every standard rank/quantization combination.

- **2.1 LoRA on each family (10 recipes)** — rank ∈ {4, 8, 16, 32, 64} × {llama, mistral, phi, qwen, gemma}; each recipe pins a specific (family, rank) pair, asserts merge round-trips bit-identically.
- **2.2 QLoRA (5 recipes)** — 4-bit base + LoRA at rank ∈ {8, 16, 32}, two of which compare double-quantization on/off.
- **2.3 Continued pretraining (5 recipes)** — domain-adapted CP on raw text fixture (legal-corpus-mini, code-corpus-mini, medical-corpus-mini, code-switching, scientific).
- **2.4 Adapter composition (5 recipes)** — multi-LoRA loading (`apr finetune --merge-loras=a,b,c`), TIES merge of two LoRAs, DARE merge, SLERP merge, average merge.

Mirrors: Ludwig `lora_adaptation/`, Unsloth `Llama3.1_(8B)-Alpaca.ipynb`, `Qwen3_(4B)-GRPO.ipynb` (LoRA portion).

### Tier 3 — Specialization (25 recipes, applied)

Goal: real-world fine-tuning patterns: instruction tuning, calibration, hyperopt, imbalance handling, multimodal.

- **3.1 Instruction tuning (5 recipes)** — Alpaca format, ShareGPT format, OpenAssistant format, custom chat-template, system-prompt prefix.
- **3.2 Hyperparameter optimization (5 recipes)** — grid search (`apr tune --strategy grid`), random search, TPE, ASHA, Hyperband; each on a 4-dim hyperparam space with budget=10 trials.
- **3.3 Calibration (5 recipes)** — temperature scaling, Platt scaling, isotonic regression, conformal prediction, ensemble averaging.
- **3.4 Class imbalance (5 recipes)** — weighted sampling, focal loss, SMOTE on tabular, threshold tuning, cost-sensitive training.
- **3.5 Multimodal + multitask (5 recipes)** — text+image (vision-LM stub), text+tabular fusion, multi-task SFT (3 tasks shared encoder), zero-shot eval, k-fold CV (k=5).

Mirrors: Ludwig `hyperopt/`, `calibration/`, `class_imbalance/`, `multimodal/`, `kfold_cv/`. Unsloth: vision notebooks (Llama 3.2 Vision, Gemma 3 Vision).

### Tier 4 — Reinforcement (25 recipes, advanced)

Goal: preference-learning, RL, and reasoning-style fine-tuning.

- **4.1 DPO (5 recipes)** — Direct Preference Optimization on each family with a 50-pair preference dataset; one recipe demonstrates the implicit reward model.
- **4.2 ORPO (3 recipes)** — Odds-Ratio Preference Optimization; reference-model-free DPO variant.
- **4.3 KTO (3 recipes)** — Kahneman-Tversky Optimization; binary feedback (helpful / unhelpful).
- **4.4 GRPO (5 recipes)** — Group Relative Policy Optimization; reasoning-style RL with verifiable rewards (math, code-exec, format-match, classification, length-budget).
- **4.5 RLHF / PPO (3 recipes)** — full RLHF pipeline: SFT → reward model → PPO; uses synthetic preference data.
- **4.6 RLAIF / Constitutional (3 recipes)** — RL from AI feedback; constitutional principles → judge → policy update.
- **4.7 Reward modeling (3 recipes)** — reward-model SFT, pairwise reward, scalar regression head.

Mirrors: Unsloth GRPO notebooks, HuggingFace TRL DPO/ORPO/KTO examples. Ludwig has no direct RL surface — Tier 4 is mostly Unsloth + TRL territory.

## What Migrates From Where

- **Recipe shape**: idiomatic apr-cookbook (`RecipeContext`, `apr_cookbook::prelude::*`, IIUR doc-header). Not Ludwig's YAML configs; not Unsloth's notebook shape.
- **CLI surface**: every recipe lands one or more `apr` subcommand invocations. We do NOT introduce new subcommands; every flag/method already ships in `apr` v0.31.2.
- **Datasets**: bundled synthetic JSONL/CSV under `tests/fixtures/finetune/`; no live HF Hub. Each fixture ≤ 1 MB, deterministic seed.
- **Provable contracts**: per recipe at `contracts/finetune-<recipe>-v1.yaml`; mirrors architecture-demos shape (Spec / Falsify / Kani / Lean / Bind axes).

## Naming Conventions

- Recipe filename: `examples/finetune/<tier-prefix>_<technique>_<base-family>.rs`
- Tier prefix: `t1_` / `t2_` / `t3_` / `t4_`
- Examples:
  - `t1_sft_minimal_llama.rs`
  - `t2_lora_rank8_mistral.rs`
  - `t3_calibration_temperature_scaling.rs`
  - `t4_dpo_qwen3.rs`
- Contract: `contracts/finetune-<tier-prefix>-<technique>-<base-family>-v1.yaml`

## Charter Boundaries

The fine-tuning-cookbook initiative covers:

- ✅ One recipe per (technique, base family) pair, with a falsifiable convergence claim
- ✅ CPU smoke fixtures bundled in-tree
- ✅ Provable-contract per recipe at Grade A target
- ✅ Ludwig + Unsloth mirror parity (≥80% of recipes mirror an upstream)
- ✅ Single-GPU LoRA/QLoRA stub paths gated behind `--features cuda`

The fine-tuning-cookbook initiative does **not** cover:

- ❌ Multi-GPU distributed training (DDP, FSDP) — separate v2 spec
- ❌ Production-scale dataset preprocessing — `alimentar` (data-loading) and apr-model-qa-playbook own that
- ❌ Live HF Hub / W&B / WandB integration — fixtures are bundled
- ❌ New `apr` subcommands — recipes wrap existing CLI surface only
- ❌ Vision/audio multimodal at full resolution — stub fixtures (32×32 image; 16-sample audio)

## Coverage Unit Decision

We considered three resolutions:

| Resolution | Count | Pros | Cons |
|------------|-------|------|------|
| One recipe per Ludwig example + per Unsloth notebook | ~130 | exhaustive 1:1 mirror | massive redundancy; many notebooks are the same technique on different families |
| One recipe per (technique × base family) pair | ~100 | covers every interesting cell of the matrix | some cells less interesting (e.g., GRPO on bert) |
| One recipe per technique only | ~30 | minimal | undersells base-family discriminator handling |

**Decided: (technique × base family)** with selective pruning to land at exactly 100. Tier 1+2 prioritize family coverage (each technique × 5 families); Tier 3+4 prioritize technique coverage (one or two families per technique because the family axis matters less when the focus is the algorithm).

Each recipe is the smallest end-to-end unit that exercises one technique against one base family with one falsifiable claim.
