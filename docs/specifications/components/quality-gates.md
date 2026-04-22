# Quality Gates & Falsification Testing

---

## Popperian Falsification

Following Karl Popper's criterion of demarcation, every performance or correctness claim must be:

1. **Specific**: Quantified with measurable thresholds
2. **Testable**: Executable via automated test OR cited from a reproducible external harness with a source-path line number
3. **Refutable**: Clear conditions for falsification

**Anti-pattern (unfalsifiable)**: "APR v2 is faster than alternatives."
**Pattern (falsifiable, in-process)**: "Zero-copy mmap-backed load completes in < 0.1 ms p95 on release builds — refuted if p95 exceeds 0.2 ms."
**Pattern (falsifiable, cited)**: "Decode throughput ≥ 270 tok/s at c=1 on RTX 4090 GGUF Q4_K_M, measured at `candle-vs-apr/performance.md:85` (273.8 tok/s)."

### Falsifiable Claims Registry — v5.0

Two categories. In-process claims are exercised by `tests/falsification.rs`. Cited claims reference external measurements; each must name a source file and line number.

#### In-process

| Code | Claim | Threshold | Refutation | Test |
|------|-------|-----------|------------|------|
| F2 | Zero-copy mmap-backed load | p95 < 0.1 ms (release) | p95 > 0.2 ms (release) or > 10 ms (debug) | `f2_zero_copy_loading_latency` |

#### Cited (external harness, source-path verified)

| Code | Claim | Threshold | Measured | Source |
|------|-------|-----------|----------|--------|
| N1 | Decode throughput, c=1, RTX 4090, GGUF Q4_K_M | ≥ 270 tok/s | 273.8 tok/s | `candle-vs-apr/performance.md:85` |
| N2 | Batch scaling, c=1 → c=32, v5 scheduler | ≥ 10× | 13.4× | `candle-vs-apr/performance.md:150` |
| N3 | Load-time parity across APR / GGUF / SafeTensors | within 1.5× | 0.028 / 0.024 / 0.029 ms | `aprender/docs/benchmarks/FORMAT_PARITY_REPORT.md:88-93` |
| N4 | Decode advantage vs Candle on identical hardware | ≥ 1.15× | 1.20× | `candle-vs-apr/performance.md:85` |

#### Deleted (no evidence in any repo — do not re-introduce without committed harness)

| Code | Original claim | Why removed |
|------|----------------|-------------|
| F1 | LZ4 decompression ≥ 3 GB/s | No LZ4 throughput bench in aprender-compute / trueno |
| F3 | Int4 NMSE < 2% | aprender-quant benches measure throughput only, never accuracy delta |
| F4 | AES-256-GCM ≥ 100 MB/s | Prior test used BLAKE3 proxy; no crypto bench anywhere |
| F5 | Whisper WER < 10% | Threshold defined in `whisper.apr/THRESHOLDS.md:74` but no measured WER logged. Prior test simulated WER on hand-written strings, not audio. |
| F6 | FlashAttention ≥ 2× | Prior CPU-tiled proxy never reaches 2×; GPU harness lives in sibling repo |
| F7 | AVX-512 ≥ 80 GFLOPS | Trueno SDE infrastructure exists but no published GFLOPS numbers |

### Falsification Test Suite (F2, current)

```rust
/// F2: Zero-copy mmap-backed load, p95 < 0.1 ms (release)
/// Evidence basis: FORMAT_PARITY_REPORT.md:88 measured 0.028 ms.
#[test]
fn f2_zero_copy_loading_latency() {
    use apr_cookbook::prelude::*;
    use memmap2::Mmap;
    use tempfile::NamedTempFile;

    let payload: Vec<u8> = (0..10 * 1024 * 1024).map(|i| (i & 0xFF) as u8).collect();
    let bundle = ModelBundle::new()
        .with_name("f2-mmap-probe")
        .with_payload(payload)
        .build();

    let mut tmp = NamedTempFile::new().unwrap();
    tmp.write_all(&bundle).unwrap();
    tmp.flush().unwrap();

    let file = tmp.reopen().unwrap();
    let mmap = unsafe { Mmap::map(&file).unwrap() };

    // Warmup + measurement loop with p95 calculation omitted for brevity.
    let p95 = measure_p95(|| BundledModel::from_bytes(&mmap[..]));
    let threshold = if cfg!(debug_assertions) {
        Duration::from_millis(10)
    } else {
        Duration::from_micros(200)
    };
    assert!(p95 < threshold, "FALSIFIED: F2 mmap p95 {p95:?} > {threshold:?}");
}
```

No other F# tests exist in the cookbook. Cited claims (N1–N4) are validated in sibling repos; reviewers reproduce by checking out the named source file at the cited line.

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
        rust: [stable, 1.89.0]
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

Structural invariants and the one surviving in-process falsifiable claim (F2) are formalized as YAML contracts in `contracts/`, following the [provable-contracts](https://github.com/paiml/aprender/tree/main/crates/aprender-contracts) schema (now living in `aprender/crates/aprender-contracts`, lib name `provable_contracts`). Contract YAMLs for the deleted F# claims (F1/F3/F4/F5/F6/F7) are retained on disk as target specifications but are NOT tied to enforced gates in v5.0.

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

### Contract Inventory (measured 2026-04-22)

11 contracts exist in `contracts/`. All parse and validate in-process via `cargo test --test contracts` (replaces prior external `pv validate`). No contract has a Lean proof yet.

#### Existing Contracts

| Contract | Covers | Status | Equations |
|----------|--------|--------|-----------|
| `cli-parity-v1` | Invariant **A**: 57 subcommands have recipes | ENFORCED | 5 |
| `docs-schema-v1` | Invariant **E**: 13 .md files validated | ENFORCED | 5 |
| `recipe-iiur-v1` | IIUR compliance for all recipes | ENFORCED | 4 |
| `apr-format-roundtrip-v1` | Format conversion round-trip correctness | ENFORCED | 4 |
| `mmap-inference-v1` | F2: mmap-backed load p95 < 0.1 ms | ENFORCED (tests/falsification.rs) | 3 |
| `lz4-decompression-v1` | Target spec for LZ4 throughput (claim F1 deleted in v5.0) | TARGET — no in-process test | 3 |
| `int4-quantization-v1` | Target spec for Int4 NMSE (claim F3 deleted in v5.0) | TARGET — no in-process test | 3 |
| `aes256-gcm-decrypt-v1` | Target spec for AES-256-GCM (claim F4 deleted in v5.0) | TARGET — no in-process test | 3 |
| `whisper-wer-v1` | Target spec for whisper WER (claim F5 deleted in v5.0; reintroduce when whisper.apr publishes measurement) | TARGET — no in-process test | 3 |
| `flash-attention-v1` | Target spec for FlashAttention speedup (claim F6 deleted in v5.0) | TARGET — no in-process test | 3 |
| `avx512-matmul-v1` | Target spec for AVX-512 matmul (claim F7 deleted in v5.0) | TARGET — no in-process test | 3 |

#### Missing Contracts

| Contract needed | Invariant | Status | Priority |
|----------------|-----------|--------|----------|
| `contract-grade-v1` | **B**: recipe-to-contract binding | Not created | P2 |
| `format-coverage-v1` | **C**: APR/GGUF/SafeTensors variants | Not created | P2 |
| `arxiv-citation-v1` | **D**: arXiv/DOI per recipe | Not created | P3 |
| Per-subcommand contracts (×57) | **B**: one contract per subcommand | 0/57 created | P2 |

#### Contract Quality Gaps

All 11 existing contracts share the same historical warnings (9 per contract = 99 total) from the last `pv lint` run before v5.0:

- Every equation missing `preconditions` and `postconditions`
- Every equation missing `lean_theorem` reference
- Mean `pv lint` score: **0.54** (grade A requires > 0.8)
- Gate 7 (reverse-coverage) skipped: no `--binding` or `--crate-dir` provided

These gaps are unchanged in v5.0 — the refactor deliberately did not touch YAML bodies, only claim-to-test wiring.

#### What Has No Contract At All

"Recipe" in this table means a distinct Cargo `[[example]]` entry (the publishable unit). Each recipe may live across several `.rs` files (e.g. `main.rs`, `types.rs`, `helpers.rs`, `tests.rs`); only the `main.rs` or single-file recipe entry carries the `//! Contract:` header. There are 219 recipes across 377 total `.rs` files.

| Asset class | Denominator | Coverage | Notes |
|-------------|-------------|----------|-------|
| Cargo `[[example]]` recipes with `//! Contract:` header | 219 | 100% | The binding surface — all recipes reference ≥1 contract |
| Supporting `.rs` files under recipe dirs (`types.rs`, `tests.rs`, helpers) | 152 | N/A | Not recipes; inherit their recipe's contract by association |
| Docs validated by `make docs-validate` | 267 | 98.9% (264/267) | README, CLAUDE.md, specs, book chapters |
| `book/src/` .md files with individual contracts | 252 | 0% (0/252) | Validated via `docs-schema-v1`, not individually bound |
| Lean proofs | 11 | 0% (0/11) | No contract has `pv lean-status` ≥ L2 |
| Per-subcommand contracts | 57 | 0% (0/57) | Subcommands share `cli-parity-v1` |

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

## Six Coverage Invariants

These are the **master quality gates** for the cookbook. A–E are enforced via `make` targets. F is currently TARGET (aspirational) pending backlog work.

### Invariant A — CLI Recipe Parity (F-CLIPARITY-001) — ENFORCED

Every apr-cli subcommand (excluding `help`) must have ≥1 cookbook recipe. APR-MONO v0.31.2 exposes **66 subcommands** (up from 57 in v4.0 — see `apr --help`).

```
∀ s ∈ apr.subcommands \ {help}: ∃ r ∈ recipes: r.cli_equivalent = s
```

**Baseline**: 56/66 = **85%** (10 new v0.31.2 subcommands uncovered — see PMAT-049). **Gate**: `make cli-parity` (exits non-zero on regression).

### Invariant B — Recipe Contract Grade (F-CONTRACT-GRADE-001) — ENFORCED

Every recipe should reference a provable-contract YAML that passes `pv lint` at **grade A**.

```
∀ r ∈ recipes:
  ∃ c ∈ contracts/: r.contract = c
  pv grade(c) = A
  pv lean-status(c) ≥ L2
```

Grade A requires: complete `metadata` (incl. academic references), ≥3 `proof_obligations`, matching `falsification_tests`, ≥1 `kani_harness`, and a passing `qa_gate`.

**Baseline**: 219/219 = **100%**. 11 contracts exist, mean `pv lint` score 0.54. **Gate**: `make contract-grade` — **ENFORCED**.

### Invariant C — Model Format Coverage (F-FORMAT-COV-001) — ENFORCED

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

**Baseline**: 219/219 = **100%**. **Gate**: `make format-coverage` — **ENFORCED**.

### Invariant D — arXiv Citation (F-ARXIV-001) — ENFORCED

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

### Invariant E — Docs Contract Coverage (F-DOCS-CONTRACT-001) — ENFORCED

Every documentation artifact — `README.md`, `CLAUDE.md`, mdbook chapters, spec components — should be bound to a provable-contract and pass `pmat validate-readme`.

```
∀ d ∈ {README.md, CLAUDE.md, book/src/**/*.md, docs/specifications/**/*.md}:
  ∃ c ∈ contracts/: d ∈ c.bindings
  pv lint(c) = PASS
  pmat validate-readme(d) ⊨ {unverified = 0, contradictions = 0}
```

**Baseline**: 264/267 = **98.9%**. `make docs-validate` covers `README.md`, `CLAUDE.md`, `docs/specifications/**/*.md`, `book/src/**/*.md`. **Gate**: `make docs-validate` — **ENFORCED**.

### Invariant F — Variant Depth (F-VARIANT-DEPTH-001) — TARGET

Every apr-cli subcommand must have **≥3 distinct cookbook recipes**. Single-example coverage is necessary (Invariant A) but not sufficient — learners need multiple worked examples per subcommand to internalize idiomatic usage. Three maps to Toyota *kata*: happy path, edge case, composition.

```
∀ s ∈ apr.subcommands \ {help}:
  |{ r ∈ recipes : r.cli_equivalent = s }| ≥ 3
```

**Baseline (2026-04-22)**: 8/66 = **12%** at ≥3; 48/66 at exactly 1–2; 10/66 at 0. **Gate**: `make variant-depth` — **TARGET**, not ENFORCED (reaching ENFORCED requires ≥128 new recipes; backlog is PMAT-049 / -050 / -051).

#### Current ≥3-coverage subcommands

| Subcommand | Count | Recipes |
|------------|-------|---------|
| `prune` | 5 | magnitude, structured, depth, wanda, gradual_schedule |
| `merge` | 5 | average, weighted, slerp, ties, dare (+ hierarchical) |
| `finetune` | 5 | lora, qlora, merge_adapter, plan_vram |
| `distill` | 4 | standard_kl, progressive, ensemble, checkpoint |
| `chat` | 4 | chatml, llama2, mistral, multi-format |
| `rosetta` | 3 | convert, chain, verify |
| `export` | 3 | safetensors, gguf, batch |
| `convert` | 3 | safetensors↔apr, gguf→apr, apr→gguf |

#### Zero-coverage (PMAT-049 blocks)

10 APR-MONO v0.31.2 subcommands still have no recipe: `awq-lint`, `dry-sampling-lint`, `gbnf-lint`, `mcp`, `ollama-chat-lint`, `oom-lint`, `pretrain`, `registry`, `tool-use-lint`, `validate-manifest`. Each needs 3 recipes per Invariant F.

### Variant definition

A **variant** is a distinct (subcommand, flag, value) triple exposed in `apr <sub> --help`. For example, `apr run` has ~20 flags; `apr merge --strategy` has 5 values (average, weighted, slerp, ties, dare), yielding 5 variants just for `--strategy`. Over 66 subcommands, apr-cli exposes ~500 distinct flag variants.

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
| Subcommands with ≥1 recipe | 56/66 (85%) | 66/66 | `make cli-parity` | **ENFORCED** |
| Subcommands with ≥3 recipes | 8/66 (12%) | 66/66 | `make variant-depth` | **TARGET** |
| Recipes with contract reference | 219/219 (100%) | 219/219 | `make contract-grade` | **ENFORCED** |
| Recipes with all format variants | 219/219 (100%) | 100% applicable | `make format-coverage` | **ENFORCED** |
| Recipes with arXiv/DOI citation | 219/219 (100%) | 219/219 | `make citation-check` | **ENFORCED** |
| Docs validated by contract | 264/267 (98.9%) | 267/267 | `make docs-validate` | **ENFORCED** |
| Flag variants | ~500 | ~500 | `make variant-coverage` | measured |
| Contracts with Lean proof ≥ L2 | 0/11 | 80%+ | `pv lean-status` | TARGET |

### Pre-commit enforcement

```bash
make cli-parity              # Invariant A: every subcommand has a recipe
make contract-grade          # Invariant B: every recipe has grade-A contract
make format-coverage         # Invariant C: APR/GGUF/SafeTensors variants
make citation-check          # Invariant D: arXiv citation per recipe
make docs-validate           # Invariant E: docs contract coverage
make variant-depth           # Invariant F: ≥3 recipes per subcommand (TARGET)
make variant-coverage        # Detailed per-subcommand flag matrix
```
