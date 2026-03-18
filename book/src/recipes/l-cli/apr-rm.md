# apr-rm

Remove a model from the local cache. Supports dry-run mode and force deletion. Shows bytes freed and remaining cache contents.

```bash
cargo run --example cli_apr_rm -- --demo
```

**CLI equivalent**: `apr rm whisper-tiny`, `apr rm --force llama-3.2-1b`, `apr rm --dry-run phi-3-mini`
