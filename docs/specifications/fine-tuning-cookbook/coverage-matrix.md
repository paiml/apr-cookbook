# Coverage Matrix

Auto-regenerated from [manifest.yaml](manifest.yaml) by `scripts/finetune-gen.sh --update`.
Hand-edits will be overwritten. Edit the manifest, not this file.

**Last regenerated:** 2026-05-10 (initial spec landing — all 100 entries planned)
**Totals:** 0 certified · 100 planned · **100 total** across 4 tiers

## Tier × Technique × Base Family Matrix

100 recipes split into 22 distinct techniques across 4 tiers.

### Tier 1 — Foundations (25 recipes)

| Technique | Recipe IDs | Base families covered |
|-----------|-----------|----------------------|
| sft (5) | t1_sft_minimal_{llama,mistral,phi,qwen,gemma} | 5 architecture families |
| eval (5) | t1_eval_{perplexity,accuracy,f1,rouge_l,bleu} | tabular + llama |
| tabular_regression (5) | t1_tabular_regression_{housing,energy,timeseries,multitarget,missing} | tabular-only |
| tabular_classification (5) | t1_tabular_{binary,3class,7class,100class,imbalanced} | tabular-only |
| smoke (5) | t1_smoke_{plan,resume,early_stop,dry_run,bench} | cross-family + llama |

### Tier 2 — Adaptive Methods (25 recipes)

| Technique | Recipe IDs | Base families covered |
|-----------|-----------|----------------------|
| lora (10) | t2_lora_{rank8,rank32}_{llama,mistral,phi,qwen,gemma} | 5 families × 2 ranks |
| qlora (5) | t2_qlora_{4bit_rank8_llama,4bit_rank16_mistral,4bit_rank32_phi,double_quant_qwen,double_quant_off_gemma} | 5 families |
| continued_pretrain (5) | t2_continued_pretrain_{legal,code,medical,codeswitch,scientific} | llama, mistral, phi, qwen3, gemma |
| adapter_merge (5) | t2_adapter_merge_{ties,dare,slerp,average,multilora} | 5 families |

### Tier 3 — Specialization (25 recipes)

| Technique | Recipe IDs | Base families covered |
|-----------|-----------|----------------------|
| instruction_tune (5) | t3_instruction_{alpaca,sharegpt,openassistant,chat_template,system_prompt} | 5 families |
| hyperopt (5) | t3_hyperopt_{grid,random,tpe,asha,hyperband} | tabular-only |
| calibration (5) | t3_calibration_{temperature,platt,isotonic,conformal,ensemble} | tabular-only |
| imbalance (5) | t3_imbalance_{weighted,focal,smote,threshold,costsensitive} | tabular-only |
| multimodal (4) + kfold_cv (1) | t3_multimodal_{text_image,text_tabular,multitask,zero_shot}, t3_kfold_cv | gemma, cross-family, llama, tabular |

### Tier 4 — Reinforcement (25 recipes)

| Technique | Recipe IDs | Base families covered |
|-----------|-----------|----------------------|
| dpo (5) | t4_dpo_{llama,mistral,phi,qwen,gemma} | 5 families |
| orpo (3) | t4_orpo_{llama,mistral,qwen} | 3 families |
| kto (3) | t4_kto_{llama,phi,gemma} | 3 families |
| grpo (5) | t4_grpo_{math,code_exec,format_match,classification,length_budget} | qwen3, phi, llama, mistral, gemma |
| rlhf_ppo (3) | t4_rlhf_ppo_{llama,mistral,qwen} | 3 families |
| rlaif (3) | t4_rlaif_{judge,constitutional,self_critique} | llama, phi, gemma |
| reward_modeling (3) | t4_reward_{pairwise,scalar,ensemble} | llama, mistral, qwen |

## Mirror Coverage

| Source | Entries with explicit mirror | Notes |
|--------|------------------------------|-------|
| Ludwig | 21 | Tier 1 + Tier 3 calibration/imbalance/hyperopt; mostly tabular and getting-started |
| Unsloth | 16 | Tier 1+2 LoRA notebooks + Tier 4 DPO/GRPO references |
| HF TRL | implicit | Tier 4 DPO/ORPO/KTO/GRPO are TRL-style; not explicitly mirrored in manifest field |
| apr-native | 5 | t1_smoke_{plan,resume,early_stop,dry_run,bench} — apr-only flags |

The 100 recipes are not 1:1 ports — they're idiomatic apr-cookbook recipes inspired by upstream sources. The mirror fields name a canonical reference, not a code-clone target.

## CLI Subcommand Coverage

Per-recipe `apr_subcommand[]` distribution (from manifest):

| Subcommand | Recipes |
|-----------|---------|
| finetune | 89 (every training recipe) |
| eval | 73 (every recipe with a metric) |
| chat | 4 (Tier 3 instruction tuning) |
| merge | 12 (LoRA merge + adapter_merge tier) |
| quantize | 5 (QLoRA tier) |
| tune | 5 (Tier 3 hyperopt) |
| serve | 0 in v1 — see backlog (multi-recipe `apr serve` smoke deferred) |
| distill | 0 in v1 — already covered by 14 distillation/ recipes pre-spec |
| prune | 0 in v1 — already covered by optimize/ recipes |
| bench | 1 (t1_smoke_bench) |

## Build Steps

To regenerate this matrix from the manifest:

```bash
bash scripts/finetune-gen.sh --update --target coverage-matrix
```

To validate manifest schema:

```bash
cargo run -p aprender-contracts-cli -- lint docs/specifications/fine-tuning-cookbook/manifest.yaml
```

## Status Legend

- **planned** — manifest entry exists, no code yet (all 100 entries, initial state)
- **in-progress** — recipe stub or fixture committed; not yet passing falsifier
- **certified** — recipe + contract + fixture + tests all green
- **blocked** — depends on upstream feature not yet shipped (e.g., `apr serve --reward-model` for some Tier 4 RLHF recipes)
