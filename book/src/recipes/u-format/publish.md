# Publish to HuggingFace

**CLI Equivalent:** `apr publish`

## What This Demonstrates

Pushes a local `.apr` model to a HuggingFace repository, including model card generation, tensor upload with chunked streaming, and metadata attachment (architecture, quantization, license).

## Run

```bash
cargo run --example format_publish
```

## Key APIs

- `HfPublisher::new(repo_id, token)` — Create a publisher targeting a HuggingFace repository
- `.with_model_card(card)` — Attach a generated model card (README.md) to the upload
- `.upload(apr_path)` — Stream the `.apr` file to the repository with progress reporting
- `ModelCard::from_apr(model)` — Auto-generate a model card from APR metadata

## Source

[`examples/format/format_publish/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_publish/main.rs)
