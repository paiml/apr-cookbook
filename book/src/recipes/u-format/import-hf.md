# Import from HuggingFace

**CLI Equivalent:** `apr import hf://org/repo`

## What This Demonstrates

Downloads a model from a HuggingFace repository, resolves its format (SafeTensors, PyTorch, GGUF), and converts it into a native `.apr` bundle with proper tensor layout and metadata.

## Run

```bash
cargo run --example format_import_hf
```

## Key APIs

- `HfImporter::new(repo_id)` — Create an importer targeting a HuggingFace repository
- `.resolve_format()` — Auto-detect the source format from repo contents
- `.import_to_apr(output_path)` — Download, convert, and write the `.apr` bundle
- `ImportConfig::default().with_cache(path)` — Configure local cache for downloaded blobs

## Source

[`examples/format/format_import_hf.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_import_hf.rs)
