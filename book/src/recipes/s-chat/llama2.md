# LLaMA 2 Chat Template

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

**CLI Equivalent:** `apr chat --format llama2`

## What This Demonstrates

LLaMA 2 uses a unique chat format with `[INST]` / `[/INST]` delimiters and a `<<SYS>>` block for system prompts. System prompts are embedded inside the first `[INST]` block only, and each complete turn is wrapped with `<s>` (BOS) and `</s>` (EOS) tokens.

## Run Command

```bash
cargo run --example chat_llama2
```

## Key APIs

- `format_system_block(&content)` -- Wrap system message in `<<SYS>>` delimiters
- `format_llama2(&messages, add_generation_prompt)` -- Format a full conversation with per-turn BOS/EOS wrapping

## Code

```rust,ignore
{{#include ../../../../examples/chat/chat_llama2.rs}}
```

## Source

[`examples/chat/chat_llama2.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/chat/chat_llama2.rs)
