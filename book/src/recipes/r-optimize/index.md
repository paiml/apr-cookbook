# Category R: Optimize

Model optimization recipes covering the full `apr` CLI optimization surface: fine-tuning, pruning, distillation, merging, and quantization. These examples mirror the subcommands available in `apr finetune`, `apr prune`, `apr distill`, `apr merge`, and `apr quantize`.

## Full Pipeline

| Recipe | Example | Description |
|--------|---------|-------------|
| [Full Pipeline](full-pipeline.md) | `optimize_full_pipeline` | Composed finetune, prune, distill, merge, quantize pipeline |

## Fine-Tuning (`apr finetune`)

| Recipe | Example | Description |
|--------|---------|-------------|
| [LoRA Fine-Tuning](finetune-lora.md) | `finetune_lora` | LoRA adapter training with rank/alpha control |
| [QLoRA Fine-Tuning](finetune-qlora.md) | `finetune_qlora` | Quantized LoRA for memory-efficient fine-tuning |
| [Merge Adapter](finetune-merge-adapter.md) | `finetune_merge_adapter` | Merge and unmerge LoRA adapters with base model |
| [Plan VRAM](finetune-plan-vram.md) | `finetune_plan_vram` | VRAM estimation and memory planning |

## Pruning (`apr prune`)

| Recipe | Example | Description |
|--------|---------|-------------|
| [Magnitude Pruning](prune-magnitude.md) | `prune_magnitude` | Weight magnitude-based unstructured pruning |
| [Structured Pruning](prune-structured.md) | `prune_structured` | Width pruning (Minitron-style) |
| [Depth Pruning](prune-depth.md) | `prune_depth` | Layer removal (Minitron-style) |
| [Wanda Pruning](prune-wanda.md) | `prune_wanda` | Pruning with calibration data (Wanda method) |
| [Gradual Schedule](prune-gradual-schedule.md) | `prune_gradual_schedule` | Cubic and gradual pruning schedules |

## Distillation (`apr distill`)

| Recipe | Example | Description |
|--------|---------|-------------|
| [Standard KL](distill-standard-kl.md) | `distill_standard_kl` | Standard KL divergence knowledge distillation |
| [Progressive](distill-progressive.md) | `distill_progressive` | Layer-wise progressive distillation |
| [Ensemble](distill-ensemble.md) | `distill_ensemble` | Multi-teacher ensemble distillation |
| [Checkpoint](distill-checkpoint.md) | `distill_checkpoint` | Distillation with checkpoint saving/resuming |

## Merging (`apr merge`)

| Recipe | Example | Description |
|--------|---------|-------------|
| [Average Merge](merge-average.md) | `merge_average` | Uniform average of model weights |
| [Weighted Merge](merge-weighted.md) | `merge_weighted` | Weighted average merge with custom ratios |
| [SLERP Merge](merge-slerp.md) | `merge_slerp` | Spherical linear interpolation merge |
| [TIES Merge](merge-ties.md) | `merge_ties` | TIES merge with density parameter |
| [DARE Merge](merge-dare.md) | `merge_dare` | DARE merge with drop probability |
| [Hierarchical Merge](merge-hierarchical.md) | `merge_hierarchical` | Multi-model hierarchical merge strategy |

## Quantization (`apr quantize`)

| Recipe | Example | Description |
|--------|---------|-------------|
| [4-bit Quantization](quantize-4bit.md) | `quantize_4bit` | Int4 weight quantization |
| [Fake QAT](quantize-fake-qat.md) | `quantize_fake_qat` | Fake quantization-aware training |
