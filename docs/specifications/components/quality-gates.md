# Quality Gates & Falsification Testing

---

## Popperian Falsification

Following Karl Popper's criterion of demarcation, every performance or correctness claim must be:

1. **Specific**: Quantified with measurable thresholds
2. **Testable**: Executable via automated test
3. **Refutable**: Clear conditions for falsification

**Anti-pattern (unfalsifiable)**: "APR v2 is faster than alternatives."
**Pattern (falsifiable)**: "APR v2 LZ4 decompression achieves >= 3 GB/s on x86_64-AVX2."

### Falsifiable Claims Registry

| Code | Claim | Threshold | Refutation | Test |
|------|-------|-----------|------------|------|
| F1 | LZ4 decompression throughput | >= 3 GB/s (AVX2) | < 2.5 GB/s | `cargo bench --bench compression` |
| F2 | Zero-copy mmap latency (<=100MB) | < 1ms | p95 > 2ms | `cargo bench --bench loading` |
| F3 | Int4 quantization accuracy loss | < 2% | > 2.5% | `cargo test --test quantization_accuracy` |
| F4 | AES-256-GCM decrypt latency (100MB) | < 5ms | p95 > 10ms | `cargo bench --bench encryption` |
| F5 | whisper.apr WER (LibriSpeech) | < 10% | > 12% | `cargo test --test whisper_wer` |
| F6 | FlashAttention speedup (seq>=1024) | >= 2x | < 1.5x | `cargo bench --bench attention` |
| F7 | AVX-512 matmul GFLOPS (1024x1024) | >= 80 | < 60 | `cargo bench --bench matmul` |

### Falsification Test Suite

```rust
/// F1: LZ4 decompression >= 3 GB/s on AVX2
/// Refutation: measured < 2.5 GB/s
#[test]
fn f1_lz4_decompression_throughput() {
    let data = vec![0u8; 100_000_000]; // 100MB
    let compressed = Compression::Lz4.compress(&data).unwrap();
    let start = Instant::now();
    let _decompressed = Compression::Lz4.decompress(&compressed).unwrap();
    let elapsed = start.elapsed();
    let throughput_gbps = data.len() as f64 / elapsed.as_secs_f64() / 1e9;
    assert!(throughput_gbps >= 2.5,
        "FALSIFIED: LZ4 throughput {:.2} < 2.5 GB/s threshold", throughput_gbps);
}

/// F3: Int4 quantization accuracy loss < 2%
/// Refutation: measured loss > 2.5%
#[test]
fn f3_int4_quantization_accuracy() {
    let model_fp32 = AprModel::load("models/test-fp32.apr").unwrap();
    let model_int4 = AprModel::load("models/test-int4.apr").unwrap();
    let test_inputs = load_test_inputs();
    let mut total_diff = 0.0;
    for input in &test_inputs {
        let out_fp32 = model_fp32.predict(input).unwrap();
        let out_int4 = model_int4.predict(input).unwrap();
        total_diff += (out_fp32 - out_int4).abs();
    }
    let accuracy_loss = total_diff / test_inputs.len() as f64;
    assert!(accuracy_loss < 0.025,
        "FALSIFIED: accuracy loss {:.2}% > 2.5% threshold", accuracy_loss * 100.0);
}
```

---

## PMAT Integration

### Quality Configuration

```toml
# .pmat/tdg-rules.toml
[quality_gates]
rust_min_grade = "A"
max_score_drop = 3.0
mode = "strict"
block_on_regression = true

[thresholds]
test_coverage = 95
mutation_score = 80
cyclomatic_complexity = 10

[defects]
patterns = ["unwrap()", "expect(", "panic!", "todo!", "unimplemented!"]
exceptions = ["#[cfg(test)]", "#[test]"]

[recipes]
isolation_required = true
idempotency_required = true
proptest_min_cases = 100
```

### Automated 10-Point QA Checklist

`pmat` validates the following for every recipe:

1. **Execution Success**: `cargo run --example <name>` exits with code 0
2. **Test Pass Rate**: All unit and integration tests pass
3. **Lint Compliance**: `cargo clippy --example <name>` returns 0 warnings
4. **Style Compliance**: `cargo fmt --check` passes
5. **Deterministic Output**: Two sequential runs produce bitwise-identical output
6. **Resource Isolation**: Temp directory count unchanged before/after execution
7. **Proptest Coverage**: At least 3 distinct property tests executed
8. **Code Coverage**: Line coverage exceeds 95% (llvm-cov)
9. **Mutation Robustness**: `cargo mutants` score exceeds 80%
10. **Documentation Standards**: Doc comments contain "Run Command" and "Learning Objective"

---

## Coverage Requirements

| Metric | Target | Enforcement |
|--------|--------|-------------|
| Line Coverage | 95% | `cargo llvm-cov --fail-under 95` |
| Branch Coverage | 90% | `cargo llvm-cov --branch` |
| Mutation Score | 80% | `cargo mutants` |
| Property Tests | 3+ per recipe | proptest |

---

## Quality Commands

```bash
# Pre-commit (required)
pmat analyze defects --path .
pmat analyze tdg --path .
cargo clippy --all-targets -- -D warnings
cargo fmt --all -- --check
cargo test --all-features

# Falsification suite
cargo test --test falsification -- --nocapture

# Pre-release
pmat rust-project-score --full --verbose
cargo mutants --timeout 300
cargo bench --bench performance
cargo llvm-cov --min-coverage 95
```

Minimum grade: **A**. Coverage target: **95%**.

---

## CI Pipeline

```yaml
# .github/workflows/recipes.yml
name: Recipe Validation

on: [push, pull_request]

jobs:
  test-all-recipes:
    runs-on: ${{ matrix.os }}
    strategy:
      matrix:
        os: [ubuntu-latest, macos-latest]
        rust: [stable, 1.75.0]
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@master
        with:
          toolchain: ${{ matrix.rust }}

      - name: Run all recipe tests
        run: cargo test --all-features

      - name: Check coverage
        run: |
          cargo install cargo-llvm-cov
          cargo llvm-cov --all-features --fail-under 95

      - name: Run examples
        run: |
          for example in $(cargo build --examples 2>&1 | grep "Compiling" | awk '{print $2}'); do
            cargo run --example $example || exit 1
          done

      - name: Idempotency check
        run: |
          cargo run --example create_apr_from_scratch
          cargo run --example create_apr_from_scratch
```

### Continuous Falsification in CI

```yaml
# .github/workflows/falsification.yml
name: Popperian Falsification

on: [push, pull_request]

jobs:
  falsify:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable

      - name: Run Falsification Tests
        run: cargo test --test falsification -- --nocapture

      - name: Benchmark with Criterion
        run: cargo bench --bench performance

      - name: Upload Benchmark Results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: target/criterion
```

---

## Provable Contracts

All falsifiable claims (F1–F7) and structural invariants are formalized as YAML contracts in `contracts/`, following the [provable-contracts](https://github.com/paiml/provable-contracts) schema.

### Contract Schema

Each contract follows the chain: `metadata → equations → proof_obligations → falsification_tests → kani_harnesses → qa_gate`.

| Section | Purpose |
|---------|---------|
| `metadata` | Version, author, academic references, tags |
| `equations` | Formal domain/codomain/invariants for each property |
| `proof_obligations` | Typed obligations (bound, roundtrip, equivalence, invariant, etc.) |
| `falsification_tests` | FALSIFY-PREFIX-NNN tests with predictions and failure explanations |
| `kani_harnesses` | KANI-PREFIX-NNN formal verification harnesses |
| `qa_gate` | F-PREFIX-NNN quality gate tying checks to pass criteria |

### Contract Inventory

| Contract | F-Claim | Obligations | Tests |
|----------|---------|-------------|-------|
| `lz4-decompression-v1` | F1 | 3 | 3 |
| `mmap-inference-v1` | F2 | 3 | 3 |
| `int4-quantization-v1` | F3 | 3 | 3 |
| `aes256-gcm-decrypt-v1` | F4 | 3 | 3 |
| `whisper-wer-v1` | F5 | 3 | 3 |
| `flash-attention-v1` | F6 | 3 | 3 |
| `avx512-matmul-v1` | F7 | 3 | 3 |
| `recipe-iiur-v1` | IIUR | 4 | 4 |
| `apr-format-roundtrip-v1` | Conversion | 4 | 4 |

### Provability Invariant

Every non-registry contract must satisfy:

```
∀ contract C:
  |C.proof_obligations| > 0
  |C.falsification_tests| >= |C.proof_obligations|
  |C.kani_harnesses| > 0
  ∀ h ∈ C.kani_harnesses: h.obligation ∈ C.proof_obligations
```

---

## APR CLI QA Process

The installed `apr` binary is tested via the `/qa` Claude Code skill (`.claude/skills/qa/SKILL.md`). This is an exhaustive, model-in-the-loop QA process that exercises every subcommand against a real model.

### Process

1. **Pull** a test model (default: Qwen2.5-Coder-1.5B Q4_K_M, ~1 GB)
2. **Exercise** all 40+ subcommands across 7 categories
3. **Detect** bugs matching the defect taxonomy below
4. **File** GitHub issues via `gh` with reproduction steps

### Subcommand Test Matrix

| Category | Commands | Timeout |
|----------|----------|---------|
| Inspection | inspect, tensors, tree, flow, debug, hex, explain, oracle | 30s |
| Validation | validate, check, lint, qa, qualify | 90s |
| Inference | run, bench, eval, serve plan, chat | 60s |
| Transform | convert, import, export, quantize, merge, prune, compile, encrypt/decrypt | 30s |
| Training | finetune, distill, train plan, data audit, tokenize plan | 15s |
| Operational | list, rm, gpu, diff, trace, profile, parity, ptx-map, rosetta, compare-hf | 30s |
| Edge Cases | nonexistent file, empty file, invalid file, unknown subcommand, duplicate flags | 5s |

### Defect Taxonomy

| Type | Severity | Example |
|------|----------|---------|
| Panic/crash | P0 | `thread 'main' panicked` on any input |
| Exit code lie | P1 | Output says `error` or `failed` but exit code is 0 |
| Hang | P1 | Command does not complete within timeout |
| Data corruption | P0 | Encrypt/decrypt roundtrip loses data |
| Wrong output | P2 | Quantization shows `"0"` instead of `"Q4_K_M"` |
| No-op flag | P2 | `--vocab` accepted but produces identical output |
| Missing fallback | P1 | GPU failure with no CPU fallback path |
| Cache inconsistency | P1 | `pull` says cached but `list` shows empty |
| Misleading message | P3 | Valid model labeled "Garbage (model broken)" |

### Invocation

```bash
# From the apr-cookbook project directory:
/qa                           # Default model
/qa /path/to/model.gguf       # Specific model
```

### Issue Filing Convention

- One issue per distinct bug (group related exit-code bugs)
- Title: `<subcommand>: <concise description>`
- Body: Description, Reproduction (exact commands), Expected, Version
- Severity label in body: P0/P1/P2/P3
