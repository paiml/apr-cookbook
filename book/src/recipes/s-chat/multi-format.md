# Multi-Format Auto-Detection

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

**CLI Equivalent:** `apr chat` (auto-detect format from model name)

## What This Demonstrates

A unified router that auto-detects the correct chat template format based on model name and applies the appropriate formatting. Supports ChatML, LLaMA 2, Mistral, Phi, and Alpaca templates with side-by-side output comparison and token count estimates.

## Run Command

```bash
cargo run --example chat_multi_format
```

## Key APIs

- `detect_format(&model_name)` -- Case-insensitive model name matching to `TemplateFormat` enum
- `format_messages(format, &messages, add_generation_prompt)` -- Dispatch to the correct formatter
- `estimate_tokens(&formatted)` -- Rough token count estimate (~4 chars per token)

## Code

```rust,ignore
{{#include ../../../../examples/chat/chat_multi_format.rs}}
```

## Source

[`examples/chat/chat_multi_format.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/chat/chat_multi_format.rs)
