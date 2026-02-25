# ChatML Template Format

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

**CLI Equivalent:** `apr chat --format chatml`

## What This Demonstrates

ChatML is the standard chat template used by OpenAI-compatible models, Qwen, Yi, and many fine-tuned variants. This example implements the ChatML format from scratch, showing exact byte-level structure with `<|im_start|>` and `<|im_end|>` special tokens, multi-turn conversations, and generation prompt toggling.

## Run Command

```bash
cargo run --example chat_chatml
```

## Key APIs

- `format_chatml_message(&msg)` -- Format a single message as `<|im_start|>role\ncontent<|im_end|>\n`
- `format_chatml(&messages, add_generation_prompt)` -- Format a full conversation with optional generation prompt
- `count_special_tokens(&formatted)` -- Count `<|im_start|>` and `<|im_end|>` occurrences

## Code

```rust,ignore
{{#include ../../../../examples/chat/chat_chatml.rs}}
```

## Source

[`examples/chat/chat_chatml.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/chat/chat_chatml.rs)
