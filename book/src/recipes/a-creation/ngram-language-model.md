# N-gram Language Model

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Build a simple n-gram language model for text generation.

## Run Command

```bash
cargo run --example create_apr_ngram_language_model
```

## Code

```rust,ignore
{{#include ../../../../examples/creation/create_apr_ngram_language_model.rs}}
```

## Key Concepts

1. **N-gram Storage**: Context-to-next-word probability mappings
2. **Vocabulary**: Token-to-index mapping stored in model metadata
3. **Smoothing**: Handle unseen n-grams with backoff strategies
