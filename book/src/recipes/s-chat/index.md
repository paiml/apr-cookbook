# Category S: Chat Templates

Chat template formatting for LLM inference, mirroring the `apr chat` CLI subcommand. Each recipe implements a specific template format from scratch, showing exact byte-level structure with special tokens.

## Recipes

| Recipe | Description | Status |
|--------|-------------|--------|
| [ChatML](./chatml.md) | ChatML template format (OpenAI, Qwen, Yi) | Verified |
| [LLaMA 2](./llama2.md) | LLaMA 2 chat template with `[INST]` delimiters | Verified |
| [Mistral](./mistral.md) | Mistral Instruct template (no native system role) | Verified |
| [Multi-Format](./multi-format.md) | Auto-detect and apply correct template by model name | Verified |
| [Injection Defense](./injection-defense.md) | Prompt injection detection and sanitization | Verified |
