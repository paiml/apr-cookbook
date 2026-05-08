# Coverage Matrix

Auto-regenerated from [manifest.yaml](manifest.yaml) by `scripts/architecture-demos-gen.sh --update`.
Hand-edits will be overwritten. Edit the manifest, not this file.

**Last regenerated:** 2026-05-08 (post-PMAT-320 forward-bridge)
**Totals:** 18 certified · 0 in-progress · 25 blocked · **43 total**

## Certified (18 — text decoder/encoder + speech)

Every certified family has a recipe at `examples/inference/inference_<family>_smoke.rs` (or `examples/speech/` for whisper/moonshine), a per-family contract at `contracts/inference-<family>-smoke-v1.yaml`, and Lean theorem proofs at `lean/ProvableContracts/ArchitectureDemos/<Family>.lean`. All 18 contracts score **0.98 (Grade A)** under `pv score --summary` after PMAT-315..318.

| Family | Vendor | HF Architectures | Recipe | Contract | Lean status |
|--------|--------|-----------------|--------|----------|-------------|
| bert | Google | BertForMaskedLM, BertForSequenceClassification | `examples/inference/inference_bert_smoke.rs` | `contracts/inference-bert-smoke-v1.yaml` | proved |
| deepseek | DeepSeek-AI | DeepseekForCausalLM, DeepseekV2/V3ForCausalLM | `examples/inference/inference_deepseek_smoke.rs` | `contracts/inference-deepseek-smoke-v1.yaml` | proved |
| falcon_h1 | TII | FalconH1ForCausalLM | `examples/inference/inference_falcon_h1_smoke.rs` | `contracts/inference-falcon-h1-smoke-v1.yaml` | proved |
| gemma | Google | GemmaForCausalLM, Gemma2/3ForCausalLM | `examples/inference/inference_gemma_smoke.rs` | `contracts/inference-gemma-smoke-v1.yaml` | proved |
| gpt2 | OpenAI | GPT2LMHeadModel | `examples/inference/inference_gpt2_smoke.rs` | `contracts/inference-gpt2-smoke-v1.yaml` | proved |
| gptneox | EleutherAI | GPTNeoXForCausalLM | `examples/inference/inference_gptneox_smoke.rs` | `contracts/inference-gptneox-smoke-v1.yaml` | proved |
| llama | Meta | LlamaForCausalLM, Llama2/3ForCausalLM | `examples/inference/inference_llama_smoke.rs` | `contracts/inference-llama-smoke-v1.yaml` | proved |
| mamba | state-spaces | MambaForCausalLM, Mamba2ForCausalLM | `examples/inference/inference_mamba_smoke.rs` | `contracts/inference-mamba-smoke-v1.yaml` | proved |
| mistral | Mistral AI | MistralForCausalLM, MixtralForCausalLM | `examples/inference/inference_mistral_smoke.rs` | `contracts/inference-mistral-smoke-v1.yaml` | proved |
| moonshine | useful-sensors | MoonshineForConditionalGeneration | `examples/speech/` | `contracts/inference-moonshine-smoke-v1.yaml` | proved |
| openelm | Apple | OpenELMForCausalLM | `examples/inference/inference_openelm_smoke.rs` | `contracts/inference-openelm-smoke-v1.yaml` | proved |
| opt | Meta AI | OPTForCausalLM | `examples/inference/inference_opt_smoke.rs` | `contracts/inference-opt-smoke-v1.yaml` | proved |
| phi | Microsoft | PhiForCausalLM, Phi3ForCausalLM, Phi3SmallForCausalLM | `examples/inference/inference_phi_smoke.rs` (companion: `convert_phi_to_apr.rs`) | `contracts/inference-phi-smoke-v1.yaml` | proved |
| qwen2 | Alibaba | Qwen2ForCausalLM, Qwen2_5ForCausalLM | `examples/inference/inference_qwen2_smoke.rs` | `contracts/inference-qwen2-smoke-v1.yaml` | proved |
| qwen3 | Alibaba | Qwen3ForCausalLM, Qwen3MoeForCausalLM | `examples/inference/inference_qwen3_smoke.rs` (companion: `inference_qwen3_moe_numerical_parity_smoke.rs`) | `contracts/inference-qwen3-smoke-v1.yaml` | proved |
| qwen3_5 | Alibaba | Qwen3_5ForCausalLM | `examples/inference/inference_qwen3_5_smoke.rs` | `contracts/inference-qwen3-5-smoke-v1.yaml` | proved |
| rwkv7 | BlinkDL | RWKV7ForCausalLM | `examples/inference/inference_rwkv7_smoke.rs` | `contracts/inference-rwkv7-smoke-v1.yaml` | proved |
| whisper | OpenAI | WhisperForConditionalGeneration | `examples/speech/whisper_transcribe.rs` | `contracts/whisper-wer-v1.yaml` | proved |

## Cross-family meta-recipes (PMAT-309..313, 320)

Higher-level recipes that consume the 18 family fixtures rather than implementing a single family. Each ships its own provable contract.

| Recipe | Purpose | Contract | Added in |
|--------|---------|----------|----------|
| `inference_arch_detector` | Discriminator-based dispatch from raw `config.json` body to a family identifier | `contracts/inference-arch-detector-v1.yaml` | PMAT-309 |
| `inference_arch_summary` | Catalog the (family, discriminator) pairs across all 16 in-progress families | `contracts/inference-arch-summary-v1.yaml` | PMAT-310 |
| `inference_arch_compare` | Diff two configs and classify their `FamilyRelation` (same / sibling / distant) | `contracts/inference-arch-compare-v1.yaml` | PMAT-311 |
| `inference_arch_quirk_audit` | Flag configs matching multiple family discriminators | `contracts/inference-arch-quirk-audit-v1.yaml` | PMAT-312 |
| `inference_arch_alias_resolver` | Mirror upstream `FamilyRegistry::resolve_alias` semantics for derived models | `contracts/inference-arch-alias-resolver-v1.yaml` | PMAT-313 |
| `inference_arch_resolution_pipeline` | Compose alias-resolver + detector into a single (hf_repo, body) → family pipeline; forward-bridge to upstream API | `contracts/inference-arch-resolution-pipeline-v1.yaml` | PMAT-320 |

## Blocked (25)

apr-model-qa-playbook ships per-checkpoint coverage; awaits upstream `aprender::rosetta` loader. Recipe lands when the upstream ticket closes. **Status of [aprender#1562](https://github.com/paiml/aprender/pull/1562) (alias mechanism)**: open as of 2026-05-08, not yet merged to main, not yet shipped to crates.io. The 16 alias-eligible entries below unblock as a batch when that PR ships.

| Family | Vendor | HF Architectures | Aliases? | Upstream Ticket |
|--------|--------|-----------------|----------|-----------------|
| bloom | BigScience | BloomForCausalLM | — | aprender#TODO-bloom-loader |
| codegemma | Google | GemmaForCausalLM | gemma | aprender#1562 (alias) |
| codellama | Meta | LlamaForCausalLM | llama | aprender#1562 (alias) |
| codestral | Mistral AI | MistralForCausalLM | mistral | aprender#1562 (alias) |
| distilgpt2 | HuggingFace | GPT2LMHeadModel | gpt2 | aprender#1562 (alias) |
| dolphin | cognitivecomputations | LlamaForCausalLM, MistralForCausalLM | llama/mistral | aprender#1562 (alias) |
| falcon | TII | FalconForCausalLM | — (classic, not H1) | aprender#TODO-falcon-classic-loader |
| galactica | Meta AI | OPTForCausalLM | opt | aprender#1562 (alias) |
| granite | IBM | GraniteForCausalLM | — | aprender#TODO-granite-loader |
| hermes | NousResearch | LlamaForCausalLM, MistralForCausalLM | llama/mistral | aprender#1562 (alias) |
| internlm2_5 | InternLM | InternLM2ForCausalLM | — | aprender#TODO-internlm2-loader |
| nemotron | NVIDIA | NemotronForCausalLM | — | aprender#TODO-nemotron-loader |
| olmo | AllenAI | OlmoForCausalLM, Olmo2ForCausalLM | — | aprender#TODO-olmo-loader |
| openchat | openchat | LlamaForCausalLM, MistralForCausalLM | llama/mistral | aprender#1562 (alias) |
| pythia | EleutherAI | GPTNeoXForCausalLM | gptneox | aprender#1562 (alias) |
| smollm | HuggingFace | LlamaForCausalLM | llama | aprender#1562 (alias) |
| smollm2 | HuggingFace | LlamaForCausalLM | llama | aprender#1562 (alias) |
| stablelm | Stability AI | StableLmForCausalLM | — | aprender#TODO-stablelm-loader |
| starcoder2 | BigCode | Starcoder2ForCausalLM | — | aprender#TODO-starcoder2-loader |
| tinyllama | TinyLlama | LlamaForCausalLM | llama | aprender#1562 (alias) |
| tiny_starcoder_py | bigcode | GPTBigCodeForCausalLM | — | aprender#TODO-gptbigcode-loader |
| vicuna | lmsys | LlamaForCausalLM | llama | aprender#1562 (alias) |
| wizardcoder | WizardLM | LlamaForCausalLM, MistralForCausalLM | llama/mistral | aprender#1562 (alias) |
| yi | 01-ai | LlamaForCausalLM | llama | aprender#1562 (alias) |
| zephyr | HuggingFaceH4 | MistralForCausalLM | mistral | aprender#1562 (alias) |

## Aliasing Reference

16 of the 25 blocked entries are pure aliases — they share the same upstream loader and only need an `hf_pattern` registration. They unblock as a single batch when [aprender#1562](https://github.com/paiml/aprender/pull/1562) ships. The cookbook-side alias table is mirrored in `examples/inference/inference_arch_alias_resolver.rs::ALIASES` (16 entries) and exercised end-to-end by `examples/inference/inference_arch_resolution_pipeline.rs` (PMAT-320).

| Base loader | Aliases that unblock together |
|-------------|-------------------------------|
| llama | codellama, dolphin, hermes, openchat, smollm, smollm2, tinyllama, vicuna, wizardcoder, yi |
| mistral | codestral, dolphin, hermes, openchat, wizardcoder, zephyr |
| gemma | codegemma |
| gpt2 | distilgpt2 |
| gptneox | pythia |
| opt | galactica |

## Format Coverage Distribution

| Format | Family count | Note |
|--------|--------------|------|
| safetensors | 18 | Universal — every upstream loader |
| apr | 18 | Universal — every loader emits APR |
| gguf | 7 | llama, mistral, gemma, phi, deepseek, qwen2, qwen3 (post-quant pipeline subset) |
| onnx | 0 | Not yet supported by any upstream loader |

## Quantization Coverage

| Quant | Family count |
|-------|--------------|
| f32 | 2 (bert, gpt2) |
| f16 | 9 |
| bf16 | 0 |
| q8_0 | 7 |
| q5_k_m | 3 (llama, mistral, qwen2) |
| q4_k_m | 11 |
| Other (q3, q2, q5_0) | 0 |

## Build Steps

To regenerate this file from the manifest:

```bash
bash scripts/architecture-demos-gen.sh --update --target coverage-matrix
```

To validate manifest schema before regeneration:

```bash
cargo run -p aprender-contracts-cli -- lint docs/specifications/architecture-demos/manifest.yaml
```
