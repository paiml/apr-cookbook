# CLI Demo Recipes — 1:1 Parity with apr-cli

These examples mirror the `apr` CLI's **57 subcommands** (excluding `help`) and their **~400 variants** (flag combinations). Each recipe demonstrates a real CLI workflow using entrenar/aprender library APIs directly, teaching users how the CLI composes these primitives.

```
apr CLI (57 subcommands, ~400 variants)
    | composes
entrenar APIs (merge, distill, prune, finetune, quant)
aprender APIs (format, chat templates, model inspection)
    | demonstrated by
Cookbook recipes (1 recipe per subcommand, ≥1 recipe per variant)
    | verified by 5 invariants
A: cli-parity        — every subcommand has a recipe
B: contract-grade    — every recipe has grade-A provable-contract
C: format-coverage   — APR + GGUF + SafeTensors variants where applicable
D: citation-check    — arXiv/DOI citation per recipe
E: docs-validate     — repo docs bound to provable-contracts
```

## Five Coverage Invariants

See [Quality Gates § Five Coverage Invariants](quality-gates.md#five-coverage-invariants) for formal definitions. Summary:

| Invariant | Gate | Baseline | Status |
|-----------|------|----------|--------|
| **A** CLI Recipe Parity | `make cli-parity` | 57/57 (100%) | **ENFORCED** |
| **B** Recipe Contract Grade A | `make contract-grade` | 219/219 (100%) | **ENFORCED** |
| **C** Model Format Coverage | `make format-coverage` | 219/219 (100%) | **ENFORCED** |
| **D** arXiv Citation | `make citation-check` | 219/219 (100%) | **ENFORCED** |
| **E** Docs Contract Coverage | `make docs-validate` | 13/268 (4.9%) | TARGET |

Every new `.rs` recipe under `examples/` **must** include a doc comment header of the form:

```rust
//! CLI Equivalent: `apr <sub> --flag value`
//! Demonstrates: <flag1>, <flag2>
//! Contract: contracts/<name>-v1.yaml
//! Lean proof: L2+
//!
//! ## References
//! - Author et al. (YEAR). *Title*. arXiv:NNNN.NNNNN
```

### Model Format Coverage (Invariant C)

Recipes that accept model files must demonstrate all applicable formats:

| Format | Extension | When required |
|--------|-----------|---------------|
| APR | `.apr` | Always (native format) |
| GGUF | `.gguf` | When subcommand accepts GGUF (run, inspect, bench, convert, etc.) |
| SafeTensors | `.safetensors` | When subcommand accepts SafeTensors (import, convert, compare-hf, etc.) |

Example: `apr run` accepts all three → recipe must show `apr run model.apr`, `apr run model.gguf`, `apr run model.safetensors`.

---

## `examples/optimize/` — Model Optimization Pipeline (22 examples)

### Full Pipeline

| # | File | CLI Equivalent | entrenar API |
|---|------|---------------|-------------|
| 1 | `optimize_full_pipeline.rs` | composed: finetune->prune->distill->merge->quantize | All below |

### Finetuning (4)

| # | File | CLI Equivalent | API |
|---|------|---------------|-----|
| 2 | `finetune_lora.rs` | `apr finetune --method lora` | `LoRALayer::new()`, `AdamW`, `.merge()` |
| 3 | `finetune_qlora.rs` | `apr finetune --method qlora` | `QLoRALayer` simulation, `.memory_stats()` |
| 4 | `finetune_merge_adapter.rs` | `apr finetune --merge --adapter` | `LoRALayer::merge()`/`.unmerge()` |
| 5 | `finetune_plan_vram.rs` | `apr finetune --plan` | VRAM planner, `OptimalConfig` |

### Pruning (5)

| # | File | CLI Equivalent | API |
|---|------|---------------|-----|
| 6 | `prune_magnitude.rs` | `apr prune --method magnitude` | Magnitude-based weight pruning |
| 7 | `prune_structured.rs` | `apr prune --method structured` | Neuron/channel removal |
| 8 | `prune_depth.rs` | `apr prune --method depth` | Layer removal (Minitron-style) |
| 9 | `prune_wanda.rs` | `apr prune --method wanda` | Wanda pruning with calibration |
| 10 | `prune_gradual_schedule.rs` | `apr prune` with schedules | Gradual, cubic, cosine schedules |

### Distillation (4)

| # | File | CLI Equivalent | API |
|---|------|---------------|-----|
| 11 | `distill_standard_kl.rs` | `apr distill --strategy standard` | `DistillationLoss::new().forward()` |
| 12 | `distill_progressive.rs` | `apr distill --strategy progressive` | `ProgressiveDistiller::uniform()` |
| 13 | `distill_ensemble.rs` | `apr distill --strategy ensemble` | `EnsembleDistiller::uniform()` |
| 14 | `distill_checkpoint.rs` | `apr distill` + save/resume | Checkpoint save/resume pattern |

### Merging (6)

| # | File | CLI Equivalent | API |
|---|------|---------------|-----|
| 15 | `merge_average.rs` | `apr merge --strategy average` | Uniform average merge |
| 16 | `merge_weighted.rs` | `apr merge --strategy weighted` | Weighted average merge |
| 17 | `merge_slerp.rs` | `apr merge --strategy slerp` | `slerp_merge()`, `SlerpConfig` |
| 18 | `merge_ties.rs` | `apr merge --strategy ties` | `ties_merge()`, `TiesConfig` |
| 19 | `merge_dare.rs` | `apr merge --strategy dare` | `dare_merge()`, `DareConfig` |
| 20 | `merge_hierarchical.rs` | composed multi-model merge | Iterative SLERP + TIES |

### Quantization (2)

| # | File | CLI Equivalent | API |
|---|------|---------------|-----|
| 21 | `quantize_4bit.rs` | `apr quantize --scheme int4` | 4-bit quantize/dequantize |
| 22 | `quantize_fake_qat.rs` | QAT training-aware quantize | Fake quantize + STE backward |

---

## `examples/chat/` — Chat Templates (5 examples)

| # | File | CLI Equivalent | API |
|---|------|---------------|-----|
| 23 | `chat_chatml.rs` | `apr chat` (ChatML) | ChatML template formatting |
| 24 | `chat_llama2.rs` | `apr chat` (LLaMA 2) | LLaMA 2 template formatting |
| 25 | `chat_mistral.rs` | `apr chat` (Mistral) | Mistral template formatting |
| 26 | `chat_multi_format.rs` | `apr chat` (all formats) | Format detection and routing |
| 27 | `chat_injection_defense.rs` | security invariants | Input sanitization |

---

## `examples/analysis/` — Model Analysis (11 examples)

| # | File | CLI Equivalent | Purpose |
|---|------|---------------|---------|
| 28 | `analysis_inspect.rs` | `apr inspect` | Metadata, architecture, tensor list |
| 29 | `analysis_validate.rs` | `apr validate` | 100-point integrity check |
| 30 | `analysis_diff.rs` | `apr diff --weights --values` | Model weight comparison |
| 31 | `analysis_bench.rs` | `apr bench` | Throughput benchmarking |
| 32 | `analysis_profile.rs` | `apr profile --granular` | Roofline analysis |
| 33 | `analysis_qa_gates.rs` | `apr qa` | 6-gate falsifiable QA |
| 34 | `analysis_oracle.rs` | `apr oracle` | Model family identification |
| 35 | `analysis_canary.rs` | `apr canary create/check` | Regression testing |
| 36 | `analysis_tree.rs` | `apr tree` | Architecture visualization |
| 37 | `analysis_hex.rs` | `apr hex` | Format-aware binary forensics |
| 38 | `analysis_explain.rs` | `apr explain` | Error code explanations |

---

## `examples/format/` — Format Operations (10 examples)

| # | File | CLI Equivalent | Purpose |
|---|------|---------------|---------|
| 39 | `format_import_hf.rs` | `apr import hf://org/repo` | HuggingFace import |
| 40 | `format_export_safetensors.rs` | `apr export --format safetensors` | SafeTensors export |
| 41 | `format_export_gguf.rs` | `apr export --format gguf` | GGUF export |
| 42 | `format_rosetta_convert.rs` | `apr rosetta convert` | Cross-format conversion |
| 43 | `format_rosetta_chain.rs` | `apr rosetta chain` | Multi-step conversion |
| 44 | `format_rosetta_verify.rs` | `apr rosetta verify` | Round-trip verification |
| 45 | `format_convert_quantize.rs` | `apr convert --quantize --compress` | Convert with quantization |
| 46 | `format_publish.rs` | `apr publish` | HuggingFace upload |
| 47 | `format_pull_cache.rs` | `apr pull` | Download and cache |
| 48 | `format_batch_export.rs` | `apr export --batch gguf,mlx,safetensors` | Batch multi-format export |

---

## Refactoring Strategy

Deprecate-and-redirect existing overlapping examples:

| Old Location | New Location |
|-------------|-------------|
| `training/entrenar_lora_finetune.rs` | `optimize/finetune_lora.rs` |
| `training/entrenar_qlora_finetune.rs` | `optimize/finetune_qlora.rs` |
| `training/entrenar_model_merge.rs` | `optimize/merge_*.rs` |
| `training/entrenar_distillation.rs` | `optimize/distill_*.rs` |
| `distillation/distill_pruning_aware.rs` | `optimize/prune_*.rs` |
| `distillation/distill_structured_pruning.rs` | `optimize/prune_structured.rs` |

**Keep unchanged**: creation/, bundling/, api/, serverless/, wasm/, gpu/, simd/, registry/, monitoring/, speech/, distributed/, serve/, advanced/

---

## `examples/cli/` — Missing Subcommand Coverage (7 examples)

| # | File | CLI Equivalent | Purpose |
|---|------|---------------|---------|
| 49 | `cli_apr_decrypt.rs` | `apr decrypt` | Decrypt model weights encrypted with `apr encrypt` |
| 50 | `cli_apr_diagnose.rs` | `apr diagnose` | Automated Five Whys diagnosis on training checkpoint |
| 51 | `cli_apr_list.rs` | `apr list` | List cached models (Ollama-like UX) |
| 52 | `cli_apr_rm.rs` | `apr rm` | Remove model from cache |
| 53 | `cli_apr_runs.rs` | `apr runs` | List, show, and compare training experiment runs |
| 54 | `cli_apr_tokenize.rs` | `apr tokenize` | BPE tokenizer training pipeline |
| 55 | `cli_apr_ptx_map.rs` | `apr ptx-map` | Model-to-PTX source mapping (GPU kernel visibility) |

---

## Parity Cross-References

High-traffic recipes should link to parity repos showing how the same workflow looks in competing runtimes. This grounds apr-cookbook in measurable, falsifiable comparisons.

### Key recipe → parity mappings

| Recipe workflow | CLI invocation | Competing equivalent | Parity repo |
|----------------|------------|---------------------|-------------|
| Model inference | `apr run model.gguf` | `ollama run`, `llama-cli -m`, `vllm serve` | [qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy) |
| GGUF inference (Rust) | `apr run --device cpu` | `candle-cli infer` | [candle-vs-apr](https://github.com/paiml/candle-vs-apr) |
| Training throughput | `apr finetune --method lora` | Ollama, vLLM training | [qwen-train-canary](https://github.com/paiml/qwen-train-canary) |
| Benchmark eval | `apr eval --perplexity` | HumanEval/MBPP Python suite | [apr-leaderboard](https://github.com/paiml/apr-leaderboard) |
| Format round-trip | `apr convert`, `apr rosetta` | `safetensors` Python API | [tiny-model-ground-truth](https://github.com/paiml/tiny-model-ground-truth) |
| Whisper transcription | `apr run whisper.apr` | `whisper.cpp`, Python Whisper | [whisper.apr](https://github.com/paiml/whisper.apr) |
| Full stack deployment | `apr serve`, `apr compile` | Docker + vLLM + Triton | [sovereign-ai-cookbook](https://github.com/paiml/sovereign-ai-cookbook) |
| Model validation | `apr validate`, `apr qualify` | manual QA checklist | [apr-model-qa-playbook](https://github.com/paiml/apr-model-qa-playbook) |

### Doc-comment convention

```rust
//! ## Parity
//! - Benchmark: [qwen-coder-deploy](https://github.com/paiml/qwen-coder-deploy)
//! - Competing: `ollama run qwen2.5-coder:1.5b` → 28 tok/s vs apr 45 tok/s
```

---

## Build Order

1. **Phase 1**: `optimize_full_pipeline.rs` — flagship composed pipeline
2. **Phase 2**: `optimize/` individual steps (23 examples)
3. **Phase 3**: `chat/` (5 examples)
4. **Phase 4**: `analysis/` (25 examples)
5. **Phase 5**: `format/` (11 examples)
6. **Phase 6**: CLI gap coverage (16 examples)
7. **Phase 7**: Parity cross-references + deprecation pass
