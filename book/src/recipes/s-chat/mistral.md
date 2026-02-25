# Mistral Chat Template

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

**CLI Equivalent:** `apr chat --format mistral`

## What This Demonstrates

Mistral Instruct uses `[INST]` / `[/INST]` delimiters like LLaMA 2 but has no native system prompt role. System instructions are prepended to the first user message. A single BOS token appears at the start (not per-turn), producing a tighter format with fewer tokens.

## Run Command

```bash
cargo run --example chat_mistral
```

## Key APIs

- `format_mistral(&messages, add_generation_prompt)` -- Format conversation with system-as-prefix handling
- `has_native_system_support()` -- Returns `false`; documents the lack of a dedicated system role

## Code

```rust,ignore
{{#include ../../../../examples/chat/chat_mistral.rs}}
```

## Source

[`examples/chat/chat_mistral.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/chat/chat_mistral.rs)
