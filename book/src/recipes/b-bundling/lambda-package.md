# Lambda Package

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Package APR models for AWS Lambda deployment.

## Run Command

```bash
cargo run --example bundle_apr_lambda_package
```

## Code

```rust,ignore
{{#include ../../../../examples/bundling/bundle_apr_lambda_package.rs}}
```

## Lambda Optimization

- Compressed binary (<50MB unzipped limit)
- Fast cold start via embedded model
- No S3 fetch at initialization
