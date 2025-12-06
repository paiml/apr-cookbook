# apr-serve

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Serve APR model via HTTP API.

## Run Command

```bash
cargo run --example cli_apr_serve -- --demo
```

## Code

```rust,ignore
{{#include ../../../../examples/cli/cli_apr_serve.rs}}
```

## Usage

```bash
apr-serve model.apr                    # Serve on :8080
apr-serve --port 9000 model.apr        # Custom port
apr-serve --workers 8 model.apr        # 8 worker threads
```
