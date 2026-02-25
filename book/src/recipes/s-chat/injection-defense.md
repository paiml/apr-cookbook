# Prompt Injection Defense

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

**CLI Equivalent:** security hardening for `apr chat`

## What This Demonstrates

Defense patterns against prompt injection attacks in chat template formatting. Covers role spoofing (injecting `<|im_start|>system`), instruction override phrases ("ignore previous instructions"), delimiter injection across all template formats, and encoded payloads including base64, zero-width Unicode characters, and homoglyphs.

## Run Command

```bash
cargo run --example chat_injection_defense
```

## Key APIs

- `contains_injection(&input)` -- Quick boolean check for known injection patterns
- `scan_for_injection(&input)` -- Detailed scan returning an `InjectionReport` with specific findings
- `sanitize_content(&input)` -- Escape dangerous template tokens and strip zero-width characters
- `defend_input(&input)` -- Combined detect-and-sanitize pipeline

## Code

```rust,ignore
{{#include ../../../../examples/chat/chat_injection_defense.rs}}
```

## Source

[`examples/chat/chat_injection_defense.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/chat/chat_injection_defense.rs)
