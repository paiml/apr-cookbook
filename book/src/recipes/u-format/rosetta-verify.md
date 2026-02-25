# Rosetta Round-Trip Verification

**CLI Equivalent:** `apr rosetta verify`

## What This Demonstrates

Performs a round-trip conversion (A -> B -> A) and verifies that tensor data survives the journey within acceptable numerical tolerances. This validates format fidelity and catches lossy conversions before they reach production.

## Run

```bash
cargo run --example format_rosetta_verify
```

## Key APIs

- `RosettaVerifier::new(original_path)` — Create a verifier anchored to the original model
- `.round_trip(intermediate_fmt)` — Convert to the intermediate format and back
- `.verify(tolerance)` — Compare tensors element-wise against the original within the given tolerance
- `VerifyReport` — Contains per-tensor max absolute error, mean error, and pass/fail status

## Source

[`examples/format/format_rosetta_verify.rs`](https://github.com/paiml/apr-cookbook/blob/main/examples/format/format_rosetta_verify.rs)
