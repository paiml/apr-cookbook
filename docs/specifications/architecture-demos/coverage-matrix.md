# Coverage Matrix

Auto-regenerated from [manifest.yaml](manifest.yaml) by `scripts/architecture-demos-gen.sh --update`.
Hand-edits will be overwritten. Edit the manifest, not this file.

**Last regenerated:** 2026-05-07 (seed; CI will refresh)
**Totals:** 4 certified · 14 in-progress · 25 blocked · **43 total**

## Certified (2)

Speech-only families covered by `examples/speech/`. Listed for completeness only — they are not part of the `examples/inference/` smoke matrix.

| Family | Vendor | HF Architectures | Recipe |
|--------|--------|-----------------|--------|
| moonshine | useful-sensors | MoonshineForConditionalGeneration | `examples/speech/` |
| whisper | OpenAI | WhisperForConditionalGeneration | `examples/speech/whisper_transcribe.rs` |

## In-Progress (16 — text decoder/encoder)

Upstream loader exists in `aprender/contracts/model-families/`; recipe + provable-contract pending.

| Family | Vendor | Recipe (planned) | Provable Contract (planned) | Lean status |
|--------|--------|------------------|------------------------------|-------------|
| bert | Google | `examples/inference/inference_bert_smoke.rs` | `contracts/inference-bert-smoke-v1.yaml` | wip |
| deepseek | DeepSeek-AI | `examples/inference/inference_deepseek_smoke.rs` | `contracts/inference-deepseek-smoke-v1.yaml` | wip |
| falcon_h1 | TII | `examples/inference/inference_falcon_h1_smoke.rs` | `contracts/inference-falcon-h1-smoke-v1.yaml` | wip |
| gemma | Google | `examples/inference/inference_gemma_smoke.rs` | `contracts/inference-gemma-smoke-v1.yaml` | wip |
| gpt2 | OpenAI | `examples/inference/inference_gpt2_smoke.rs` | `contracts/inference-gpt2-smoke-v1.yaml` | wip |
| gptneox | EleutherAI | `examples/inference/inference_gptneox_smoke.rs` | `contracts/inference-gptneox-smoke-v1.yaml` | wip |
| llama | Meta | `examples/inference/inference_llama_smoke.rs` | `contracts/inference-llama-smoke-v1.yaml` | wip |
| mamba | state-spaces | `examples/inference/inference_mamba_smoke.rs` | `contracts/inference-mamba-smoke-v1.yaml` | wip |
| mistral | Mistral AI | `examples/inference/inference_mistral_smoke.rs` | `contracts/inference-mistral-smoke-v1.yaml` | wip |
| openelm | Apple | `examples/inference/inference_openelm_smoke.rs` | `contracts/inference-openelm-smoke-v1.yaml` | wip |
| opt | Meta AI | `examples/inference/inference_opt_smoke.rs` | `contracts/inference-opt-smoke-v1.yaml` | wip |
| phi | Microsoft | `examples/inference/inference_phi_smoke.rs` (companion: `convert_phi_to_apr.rs` ✓) | `contracts/inference-phi-smoke-v1.yaml` | wip |
| qwen2 | Alibaba | `examples/inference/inference_qwen2_smoke.rs` | `contracts/inference-qwen2-smoke-v1.yaml` | wip |
| qwen3 | Alibaba | `examples/inference/inference_qwen3_smoke.rs` (companion: `inference_qwen3_moe_numerical_parity_smoke.rs` ✓) | `contracts/inference-qwen3-smoke-v1.yaml` | wip |
| qwen3_5 | Alibaba | `examples/inference/inference_qwen3_5_smoke.rs` | `contracts/inference-qwen3-5-smoke-v1.yaml` | wip |
| rwkv7 | BlinkDL | `examples/inference/inference_rwkv7_smoke.rs` | `contracts/inference-rwkv7-smoke-v1.yaml` | wip |

## Blocked (25)

apr-model-qa-playbook ships per-checkpoint coverage; awaits upstream `aprender::rosetta` loader. Recipe lands when the upstream ticket closes.

| Family | Vendor | HF Architectures | Aliases? | Upstream Ticket |
|--------|--------|-----------------|----------|-----------------|
| bloom | BigScience | BloomForCausalLM | — | aprender#TODO-bloom-loader |
| codegemma | Google | GemmaForCausalLM | gemma | aprender#TODO-codegemma-alias |
| codellama | Meta | LlamaForCausalLM | llama | aprender#TODO-codellama-alias |
| codestral | Mistral AI | MistralForCausalLM | mistral | aprender#TODO-codestral-alias |
| distilgpt2 | HuggingFace | GPT2LMHeadModel | gpt2 | aprender#TODO-distilgpt2-alias |
| dolphin | cognitivecomputations | LlamaForCausalLM, MistralForCausalLM | llama/mistral | aprender#TODO-dolphin-alias |
| falcon | TII | FalconForCausalLM | — (classic, not H1) | aprender#TODO-falcon-classic-loader |
| galactica | Meta AI | OPTForCausalLM | opt | aprender#TODO-galactica-alias |
| granite | IBM | GraniteForCausalLM | — | aprender#TODO-granite-loader |
| hermes | NousResearch | LlamaForCausalLM, MistralForCausalLM | llama/mistral | aprender#TODO-hermes-alias |
| internlm2_5 | InternLM | InternLM2ForCausalLM | — | aprender#TODO-internlm2-loader |
| nemotron | NVIDIA | NemotronForCausalLM | — | aprender#TODO-nemotron-loader |
| olmo | AllenAI | OlmoForCausalLM, Olmo2ForCausalLM | — | aprender#TODO-olmo-loader |
| openchat | openchat | LlamaForCausalLM, MistralForCausalLM | llama/mistral | aprender#TODO-openchat-alias |
| pythia | EleutherAI | GPTNeoXForCausalLM | gptneox | aprender#TODO-pythia-alias |
| smollm | HuggingFace | LlamaForCausalLM | llama | aprender#TODO-smollm-alias |
| smollm2 | HuggingFace | LlamaForCausalLM | llama | aprender#TODO-smollm2-alias |
| stablelm | Stability AI | StableLmForCausalLM | — | aprender#TODO-stablelm-loader |
| starcoder2 | BigCode | Starcoder2ForCausalLM | — | aprender#TODO-starcoder2-loader |
| tinyllama | TinyLlama | LlamaForCausalLM | llama | aprender#TODO-tinyllama-alias |
| tiny_starcoder_py | bigcode | GPTBigCodeForCausalLM | — | aprender#TODO-gptbigcode-loader |
| vicuna | lmsys | LlamaForCausalLM | llama | aprender#TODO-vicuna-alias |
| wizardcoder | WizardLM | LlamaForCausalLM, MistralForCausalLM | llama/mistral | aprender#TODO-wizardcoder-alias |
| yi | 01-ai | LlamaForCausalLM | llama | aprender#TODO-yi-alias |
| zephyr | HuggingFaceH4 | MistralForCausalLM | mistral | aprender#TODO-zephyr-alias |

## Aliasing Reference

11 of the 25 blocked entries are pure aliases — they share the same upstream loader and only need an `hf_pattern` registration. Once aprender lands a generic alias mechanism, these unblock as a single batch.

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
