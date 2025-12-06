# apr-info

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Inspect APR model metadata and structure.

## Run Command

```bash
cargo run --example cli_apr_info -- --demo
```

## Code

```rust,ignore
{{#include ../../../../examples/cli/cli_apr_info.rs}}
```

## Usage

```bash
apr-info model.apr           # Show model info
apr-info --verbose model.apr # Detailed output
apr-info --json model.apr    # JSON output
```
