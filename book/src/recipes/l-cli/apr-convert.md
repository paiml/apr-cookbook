# apr-convert

> **Status**: Verified | **Idempotent**: Yes | **Coverage**: 95%+

Convert between model formats.

## Run Command

```bash
cargo run --example cli_apr_convert -- --demo
```

## Code

```rust,ignore
{{#include ../../../../examples/cli/cli_apr_convert.rs}}
```

## Usage

```bash
apr-convert input.safetensors output.apr
apr-convert input.apr output.gguf
apr-convert --quantize q4 input.apr output.apr
```
