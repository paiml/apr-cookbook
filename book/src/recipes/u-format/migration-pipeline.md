# Migration Pipeline

Complete model migration pipeline composing four stages: import, lint, convert, and export. This is the workflow used when migrating a HuggingFace SafeTensors model into the APR v2 format with quality checks and round-trip verification.

## CLI Equivalent
```bash
apr migrate model.safetensors --to apr2 --lint --verify
```

## Key Concepts
- Multi-stage migration pipeline (import, lint, convert, export)
- Round-trip verification with cosine similarity
- Checksum and manifest generation for exported bundles

## Run
```bash
cargo run --example format_migration_pipeline
```

## Source
[`examples/format/format_migration_pipeline.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_migration_pipeline.rs)
