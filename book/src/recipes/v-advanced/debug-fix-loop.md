# Debug-Fix Loop

Iterative debug-fix loop composing: trace, debug, fix, check, validate. Each iteration identifies a model issue, diagnoses the root cause, applies a targeted fix, and verifies the repair until all issues are resolved.

## CLI Equivalent
```bash
N/A (composes apr trace + apr debug + apr check + apr validate)
```

## Key Concepts
- Iterative diagnosis and repair loop
- Root cause detection from layer-level traces
- Fix verification via check and validate stages

## Run
```bash
cargo run --example debug_fix_loop
```

## Source
[`examples/advanced/debug_fix_loop/main.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/advanced/debug_fix_loop/main.rs)
