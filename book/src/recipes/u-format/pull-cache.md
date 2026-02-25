# Pull and Cache Management

**CLI Equivalent:** `apr pull`

## What This Demonstrates

Downloads a model from a remote source (HuggingFace, HTTP, S3) into a local cache directory, with content-addressed deduplication and integrity verification via SHA-256 checksums.

## Run

```bash
cargo run --example format_pull_cache
```

## Key APIs

- `ModelCache::new(cache_dir)` — Initialize or open a local model cache
- `.pull(source_url)` — Download a model if not already cached, returning the local path
- `.verify(model_id)` — Re-check SHA-256 integrity of a cached model
- `.evict(policy)` — Remove cached models by LRU, age, or size policy

## Source

[`examples/format/format_pull_cache.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_pull_cache.rs)
