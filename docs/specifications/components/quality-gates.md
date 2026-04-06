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

### Contract Inventory (measured 2026-04-06)

11 contracts exist in `contracts/`. All pass `pv lint` (0 errors, 99 warnings, mean score 0.54). No contract has a Lean proof yet.

#### Existing Contracts

| Contract | Covers | Type | `pv lint` | Lean proof | Equations |
|----------|--------|------|-----------|------------|-----------|
| `cli-parity-v1` | Invariant **A**: 57 subcommands have recipes | Structural | PASS | None | 5 |
| `docs-schema-v1` | Invariant **E**: 13 .md files validated | Structural | PASS | None | 5 |
| `recipe-iiur-v1` | IIUR compliance for all recipes | Structural | PASS | None | 4 |
| `apr-format-roundtrip-v1` | Format conversion round-trip correctness | Correctness | PASS | None | 4 |
| `lz4-decompression-v1` | F1: LZ4 >= 3 GB/s (AVX2) | Performance | PASS | None | 3 |
| `mmap-inference-v1` | F2: mmap latency < 1ms (<=100MB) | Performance | PASS | None | 3 |
| `int4-quantization-v1` | F3: Int4 accuracy loss < 2% | Accuracy | PASS | None | 3 |
| `aes256-gcm-decrypt-v1` | F4: AES-256-GCM decrypt < 5ms (100MB) | Performance | PASS | None | 3 |
| `whisper-wer-v1` | F5: whisper.apr WER < 10% | Accuracy | PASS | None | 3 |
| `flash-attention-v1` | F6: FlashAttention >= 2x speedup | Performance | PASS | None | 3 |
| `avx512-matmul-v1` | F7: AVX-512 matmul >= 80 GFLOPS | Performance | PASS | None | 3 |

#### Missing Contracts

| Contract needed | Invariant | Status | Priority |
|----------------|-----------|--------|----------|
| `contract-grade-v1` | **B**: recipe-to-contract binding | Not created | P2 |
| `format-coverage-v1` | **C**: APR/GGUF/SafeTensors variants | Not created | P2 |
| `arxiv-citation-v1` | **D**: arXiv/DOI per recipe | Not created | P3 |
| Per-subcommand contracts (×57) | **B**: one contract per subcommand | 0/57 created | P2 |

#### Contract Quality Gaps

All 11 existing contracts share the same warnings (99 total across 11 contracts, 9 per contract):

- Every equation missing `preconditions` and `postconditions`
- Every equation missing `lean_theorem` reference
- Mean `pv lint` score: **0.54** (target: grade A requires > 0.8)
- Gate 7 (reverse-coverage) skipped: no `--binding` or `--crate-dir` provided

#### What Has No Contract At All

| Asset class | Count | Contract coverage |
|-------------|-------|-------------------|
| Recipe `.rs` files with `Contract:` header | 0 / 219 | 0% — no recipe references any contract |
| `book/src/` .md files | 0 / 252 | 0% — not bound to `docs-schema-v1` or any contract |
| Lean proofs | 0 / 11 | 0% — no contract has `pv lean-status` >= L2 |
| Per-subcommand contracts | 0 / 57 | 0% — subcommands share `cli-parity-v1` but have no individual contracts |

### Provability Invariant

Every non-registry contract must satisfy:

```
∀ contract C:
  |C.proof_obligations| > 0
  |C.falsification_tests| >= |C.proof_obligations|
  |C.kani_harnesses| > 0
  ∀ h ∈ C.kani_harnesses: h.obligation ∈ C.proof_obligations
```

**Status**: All 11 contracts satisfy obligation/test counts. None satisfy `kani_harnesses > 0` (no Kani harnesses exist yet).

---

## Docs Schema Enforcement

All `*.md` files in the repository are enforced by **`pmat validate-readme`** (factual accuracy + hallucination detection) and carry a **provable-contract schema** (`contracts/docs-schema-v1.yaml`, F-DOCS-001).

### What gets validated

| Target | Tool | Enforcement |
|--------|------|-------------|
| `README.md` | `pmat validate-readme` | `--fail-on-contradiction --fail-on-unverified` |
| `CLAUDE.md` | `pmat validate-readme` | Same |
| `docs/specifications/**/*.md` | `pmat validate-readme` + `pv validate` | Same + schema frontmatter |
| All `*.md` links | `pmat validate-docs` | `--fail-on-error` |
| Every `apr <cmd>` reference in any `*.md` | CLI binding check | must exist in `apr --help` |

### Invariants (from `contracts/docs-schema-v1.yaml`)

```
∀ m ∈ *.md: pmat.validate_readme(m) ⊨ {unverified = 0, contradictions = 0}
∀ link ∈ *.md: link.target.exists
∀ `apr <cmd>` ∈ docs: cmd ∈ apr.subcommands
∀ m ∈ docs/specifications/**/*.md: validates(m.frontmatter, doc_schema)
```

### Verification

```bash
# Generate deep context once
pmat context --output deep-context.md

# Validate every *.md
pmat validate-readme \
    --targets README.md CLAUDE.md docs/specifications/**/*.md \
    --deep-context deep-context.md \
    --fail-on-contradiction \
    --fail-on-unverified

# Validate links + cross-references
pmat validate-docs --root . --fail-on-error

# Validate contract schema compliance
pv validate contracts/docs-schema-v1.yaml
pv lint contracts/
```

A `make docs-validate` target chains these. Docs validation is a **mandatory pre-commit gate**.

---

## Five Coverage Invariants

These are the **master quality gates** for the cookbook. Invariant A is enforced (blocks commits). Invariants B–E are measured and reported but do not yet block — they will become enforcing gates once baselines exceed 50%.

### Invariant A — CLI Recipe Parity (F-CLIPARITY-001) — ENFORCED

Every apr-cli subcommand (excluding `help`) must have ≥1 cookbook recipe.

```
∀ s ∈ apr.subcommands \ {help}: ∃ r ∈ recipes: r.cli_equivalent = s
```

**Baseline**: 57/57 = **100%**. **Gate**: `make cli-parity` (exits non-zero on regression).

### Invariant B — Recipe Contract Grade (F-CONTRACT-GRADE-001) — TARGET

Every recipe should reference a provable-contract YAML that passes `pv lint` at **grade A**.

```
∀ r ∈ recipes:
  ∃ c ∈ contracts/: r.contract = c
  pv grade(c) = A
  pv lean-status(c) ≥ L2
```

Grade A requires: complete `metadata` (incl. academic references), ≥3 `proof_obligations`, matching `falsification_tests`, ≥1 `kani_harness`, and a passing `qa_gate`.

**Baseline**: 0/219 recipes reference a contract (0%). 11 contracts exist, mean `pv lint` score 0.54. **Gate**: `make contract-grade` (reports; warns until > 50%).

### Invariant C — Model Format Coverage (F-FORMAT-COV-001) — TARGET

Every recipe that accepts a model file should demonstrate all three canonical formats where the subcommand supports them: **APR** (`.apr`), **GGUF** (`.gguf`), **SafeTensors** (`.safetensors`).

```
∀ r ∈ recipes where r.accepts_model_input:
  ∀ fmt ∈ {apr, gguf, safetensors} where r.supports(fmt):
    ∃ variant v ∈ r: v.format = fmt
```

Examples:
- `apr run` accepts all three → 3 format variants required
- `apr encrypt` is APR-only → 1 variant sufficient
- `apr import hf://…` outputs APR → 1 variant sufficient

**Baseline**: 3/219 recipes demonstrate all three formats (1.4%). Most recipes are APR-only. **Gate**: `make format-coverage` (reports; warns until > 50%).

### Invariant D — arXiv Citation (F-ARXIV-001) — TARGET

Every recipe should cite ≥1 arXiv paper or peer-reviewed reference linking the technique to the literature.

```
∀ r ∈ recipes: |r.citations ∩ (arXiv ∪ peer_reviewed)| ≥ 1
```

Doc comment format:

```rust
//! ## References
//! - Hu et al. (2021). *LoRA: Low-Rank Adaptation*. arXiv:2106.09685
```

**Baseline**: 219/219 = **100%**. **Gate**: `make citation-check` — **ENFORCED**.

### Invariant E — Docs Contract Coverage (F-DOCS-CONTRACT-001) — TARGET

Every documentation artifact — `README.md`, `CLAUDE.md`, mdbook chapters, spec components — should be bound to a provable-contract and pass `pmat validate-readme`.

```
∀ d ∈ {README.md, CLAUDE.md, book/src/**/*.md, docs/specifications/**/*.md}:
  ∃ c ∈ contracts/: d ∈ c.bindings
  pv lint(c) = PASS
  pmat validate-readme(d) ⊨ {unverified = 0, contradictions = 0}
```

**Baseline**: 13/268 .md files validated (4.9%). `make docs-validate` covers `README.md`, `CLAUDE.md`, `docs/specifications/**/*.md`. The `book/src/` tree (252 files) is not yet bound. **Gate**: `make docs-validate` (enforced for 13 bound files; book coverage is a target).

### Variant definition

A **variant** is a distinct (subcommand, flag, value) triple exposed in `apr <sub> --help`. For example, `apr run` has ~20 flags; `apr merge --strategy` has 5 values (average, weighted, slerp, ties, dare), yielding 5 variants just for `--strategy`. Over 57 subcommands, apr-cli exposes ~468 distinct flag variants.

### Recipe doc-comment contract

Every recipe `.rs` file **must** start with a doc comment containing the following fields (case-insensitive, `**` decoration allowed):

```rust
//! # <Recipe Name>
//!
//! CLI Equivalent: `apr <subcommand> [--flag value]`
//! Demonstrates: <flag1>, <flag2>
//! Contract: contracts/<name>-v1.yaml
//! Lean proof: L2+
//! Run Command: cargo run --example <name>
//! Learning Objective: <one-line summary>
//!
//! ## References
//! - Author et al. (YEAR). *Title*. arXiv:NNNN.NNNNN
```

The `make cli-parity` regex matches all of:
- `CLI Equivalent: apr <cmd>`
- `CLI equivalent: apr <cmd>`
- `**CLI Equivalent**: \`apr <cmd>\``
- `CLI Equivalent**: \`apr <cmd>\``

### Coverage dashboard

| Dimension | Baseline | Target | Gate | Status |
|-----------|----------|--------|------|--------|
| Subcommands with ≥1 recipe | 57/57 (100%) | 57/57 | `make cli-parity` | **ENFORCED** |
| Recipes with contract reference | 0/219 (0%) | 219/219 | `make contract-grade` | TARGET |
| Recipes with all format variants | 3/219 (1.4%) | 100% applicable | `make format-coverage` | TARGET |
| Recipes with arXiv/DOI citation | 219/219 (100%) | 219/219 | `make citation-check` | **ENFORCED** |
| Docs validated by contract | 13/268 (4.9%) | 268/268 | `make docs-validate` | TARGET |
| Flag variants | ~468 | ~468 | `make variant-coverage` | measured |
| Contracts with Lean proof ≥ L2 | 0/11 | 80%+ | `pv lean-status` | TARGET |

### Pre-commit enforcement

```bash
make cli-parity              # Invariant A: every subcommand has a recipe
make contract-grade          # Invariant B: every recipe has grade-A contract
make format-coverage         # Invariant C: APR/GGUF/SafeTensors variants
make citation-check          # Invariant D: arXiv citation per recipe
make docs-validate           # Invariant E: docs contract coverage
make variant-coverage        # Detailed per-subcommand flag matrix
```
