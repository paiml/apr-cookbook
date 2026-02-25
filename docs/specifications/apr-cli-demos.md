# APR CLI Demos — Specification

## Overview

This specification defines 48 new cookbook examples that mirror the `apr` CLI's 44 subcommands. Each example demonstrates a real CLI workflow using entrenar/aprender library APIs directly, teaching users how the CLI composes these primitives.

## Architecture

```
apr CLI (44 subcommands)
    ↓ composes
entrenar APIs (merge, distill, prune, finetune, quant)
aprender APIs (format, chat templates, model inspection)
    ↓ demonstrated by
Cookbook examples (48 new + 121 existing)
```

## New Example Categories

### `examples/optimize/` — Model Optimization Pipeline (22 examples)

| # | File | CLI Equivalent | entrenar API |
|---|------|---------------|-------------|
| 1 | `optimize_full_pipeline.rs` | composed: finetune→prune→distill→merge→quantize | All below |
| 2 | `finetune_lora.rs` | `apr finetune --method lora` | `LoRALayer::new()`, `AdamW`, `.merge()` |
| 3 | `finetune_qlora.rs` | `apr finetune --method qlora` | `QLoRALayer` simulation, `.memory_stats()` |
| 4 | `finetune_merge_adapter.rs` | `apr finetune --merge --adapter` | `LoRALayer::merge()`/`.unmerge()` |
| 5 | `finetune_plan_vram.rs` | `apr finetune --plan` | VRAM planner, `OptimalConfig` |
| 6 | `prune_magnitude.rs` | `apr prune --method magnitude` | Magnitude-based weight pruning |
| 7 | `prune_structured.rs` | `apr prune --method structured` | Neuron/channel removal |
| 8 | `prune_depth.rs` | `apr prune --method depth` | Layer removal (Minitron-style) |
| 9 | `prune_wanda.rs` | `apr prune --method wanda` | Wanda pruning with calibration |
| 10 | `prune_gradual_schedule.rs` | `apr prune` with schedules | Gradual, cubic, cosine schedules |
| 11 | `distill_standard_kl.rs` | `apr distill --strategy standard` | `DistillationLoss::new().forward()` |
| 12 | `distill_progressive.rs` | `apr distill --strategy progressive` | `ProgressiveDistiller::uniform()` |
| 13 | `distill_ensemble.rs` | `apr distill --strategy ensemble` | `EnsembleDistiller::uniform()` |
| 14 | `distill_checkpoint.rs` | `apr distill` + save/resume | Checkpoint save/resume pattern |
| 15 | `merge_average.rs` | `apr merge --strategy average` | Uniform average merge |
| 16 | `merge_weighted.rs` | `apr merge --strategy weighted` | Weighted average merge |
| 17 | `merge_slerp.rs` | `apr merge --strategy slerp` | `slerp_merge()`, `SlerpConfig` |
| 18 | `merge_ties.rs` | `apr merge --strategy ties` | `ties_merge()`, `TiesConfig` |
| 19 | `merge_dare.rs` | `apr merge --strategy dare` | `dare_merge()`, `DareConfig` |
| 20 | `merge_hierarchical.rs` | composed multi-model merge | Iterative SLERP + TIES |
| 21 | `quantize_4bit.rs` | `apr quantize --scheme int4` | 4-bit quantize/dequantize |
| 22 | `quantize_fake_qat.rs` | QAT training-aware quantize | Fake quantize + STE backward |

### `examples/chat/` — Chat Templates (5 examples)

| # | File | CLI Equivalent | API |
|---|------|---------------|-----|
| 23 | `chat_chatml.rs` | `apr chat` (ChatML) | ChatML template formatting |
| 24 | `chat_llama2.rs` | `apr chat` (LLaMA 2) | LLaMA 2 template formatting |
| 25 | `chat_mistral.rs` | `apr chat` (Mistral) | Mistral template formatting |
| 26 | `chat_multi_format.rs` | `apr chat` (all formats) | Format detection and routing |
| 27 | `chat_injection_defense.rs` | security invariants | Input sanitization |

### `examples/analysis/` — Model Analysis (11 examples)

| # | File | CLI Equivalent |
|---|------|---------------|
| 28 | `analysis_inspect.rs` | `apr inspect` — metadata, architecture, tensor list |
| 29 | `analysis_validate.rs` | `apr validate` — 100-point integrity check |
| 30 | `analysis_diff.rs` | `apr diff --weights --values` |
| 31 | `analysis_bench.rs` | `apr bench` — throughput benchmarking |
| 32 | `analysis_profile.rs` | `apr profile --granular` — roofline analysis |
| 33 | `analysis_qa_gates.rs` | `apr qa` — 6-gate falsifiable QA |
| 34 | `analysis_oracle.rs` | `apr oracle` — model family identification |
| 35 | `analysis_canary.rs` | `apr canary create/check` — regression testing |
| 36 | `analysis_tree.rs` | `apr tree` — architecture visualization |
| 37 | `analysis_hex.rs` | `apr hex` — format-aware binary forensics |
| 38 | `analysis_explain.rs` | `apr explain` — error code explanations |

### `examples/format/` — Format Operations (10 examples)

| # | File | CLI Equivalent |
|---|------|---------------|
| 39 | `format_import_hf.rs` | `apr import hf://org/repo` |
| 40 | `format_export_safetensors.rs` | `apr export --format safetensors` |
| 41 | `format_export_gguf.rs` | `apr export --format gguf` |
| 42 | `format_rosetta_convert.rs` | `apr rosetta convert` — cross-format |
| 43 | `format_rosetta_chain.rs` | `apr rosetta chain` — multi-step conversion |
| 44 | `format_rosetta_verify.rs` | `apr rosetta verify` — round-trip check |
| 45 | `format_convert_quantize.rs` | `apr convert --quantize --compress` |
| 46 | `format_publish.rs` | `apr publish` — HuggingFace upload |
| 47 | `format_pull_cache.rs` | `apr pull` — download and cache |
| 48 | `format_batch_export.rs` | `apr export --batch gguf,mlx,safetensors` |

## Example Template

Every new example follows this structure:

```rust
//! # Recipe: [Title]
//!
//! **Category**: [optimize|chat|analysis|format]
//! **CLI Equivalent**: `apr [command] [flags]`
//!
//! ## QA Checklist
//! 1. [x] `cargo run` succeeds (Exit Code 0)
//! 2. [x] `cargo test` passes
//! 3. [x] Deterministic output (Verified)
//! 4. [x] No temp files leaked
//! 5. [x] Clippy clean
//! 6. [x] No `unwrap()` in logic
//!
//! ## Learning Objective
//! [What this teaches]
//!
//! ## Run Command
//! ```bash
//! cargo run --example [name]
//! ```

use apr_cookbook::prelude::*;

fn main() -> Result<()> {
    let mut ctx = RecipeContext::new("example_name")?;
    // ... sections
    ctx.report()?;
    Ok(())
}

#[cfg(test)]
mod tests {
    // 8-15 unit tests
}
```

## Refactoring Strategy

Deprecate-and-redirect existing overlapping examples:
- `training/entrenar_lora_finetune.rs` → `optimize/finetune_lora.rs`
- `training/entrenar_qlora_finetune.rs` → `optimize/finetune_qlora.rs`
- `training/entrenar_model_merge.rs` → `optimize/merge_*.rs`
- `training/entrenar_distillation.rs` → `optimize/distill_*.rs`
- `distillation/distill_pruning_aware.rs` → `optimize/prune_*.rs`
- `distillation/distill_structured_pruning.rs` → `optimize/prune_structured.rs`

Keep unchanged: creation/, bundling/, api/, serverless/, wasm/, gpu/, simd/, registry/, monitoring/, speech/, distributed/, serve/, advanced/

## Build Order

1. **Phase 1**: `optimize_full_pipeline.rs` — flagship composed pipeline
2. **Phase 2**: `optimize/` individual steps (22 examples)
3. **Phase 3**: `chat/` (5 examples)
4. **Phase 4**: `analysis/` (11 examples)
5. **Phase 5**: `format/` (10 examples)
6. **Phase 6**: Cargo.toml + deprecation pass

## Verification

After each phase:
1. `cargo build --examples` — all compile
2. `cargo test --all-features` — all pass
3. `cargo clippy --all-targets -- -D warnings` — zero warnings
