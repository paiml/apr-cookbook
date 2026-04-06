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

| Contract | Invariant/F-Claim | Obligations | Tests |
|----------|-------------------|-------------|-------|
| `cli-parity-v1` | **A** F-CLIPARITY-001 | 5 | 5 |
| `contract-grade-v1` | **B** F-CONTRACT-GRADE-001 | TBD | TBD |
| `format-coverage-v1` | **C** F-FORMAT-COV-001 | TBD | TBD |
| `arxiv-citation-v1` | **D** F-ARXIV-001 | TBD | TBD |
| `docs-schema-v1` | **E** F-DOCS-CONTRACT-001 | 5 | 5 |
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

The installed `apr` binary is tested via the `/qa` Claude Code skill (`.claude/skills/qa/SKILL.md`). This is an exhaustive, **fleet-wide**, model-in-the-loop QA process that exercises every subcommand against a real model on **every reachable hardware target**, then audits subcommand coverage against the `provable-contracts` registry.

### Process

1. **Audit** contract coverage: every apr subcommand → YAML contract + Lean proof (Phase 0)
2. **Probe** all targets: intel, yoga, jetson, lambda-labs + local (Phase 1 setup)
3. **Pull** a test model (default: Qwen2.5-Coder-1.5B Q4_K_M, ~1 GB) on each target
4. **Exercise** all 40+ subcommands across 8 categories per target
5. **Detect** bugs matching the defect taxonomy below (tag with target)
6. **File** GitHub issues via `gh` with per-target reproduction steps

### Target Fleet

| Target | Access | Arch | Accelerator | Purpose |
|--------|--------|------|-------------|---------|
| `local` | (host) | discover | discover | Fast iteration, contract audit |
| `intel` | `ssh intel` | x86_64 | AVX2/AVX-512 CPU | x86_64 SIMD parity |
| `yoga` | `ssh yoga` | discover | CPU | Laptop-class baseline |
| `jetson` | `ssh jetson` | aarch64 | NVIDIA GPU | ARM + CUDA parity |
| `lambda-labs` | `ssh -p 2222 noahgift@localhost` | x86_64 | NVIDIA GPU | Cloud GPU parity |

Arch-divergence bugs (works on x86_64 but panics on aarch64) are a first-class defect category. Every bug report must list the affected targets as checkboxes.

### Subcommand Test Matrix

| Category | Commands | Timeout | Run on |
|----------|----------|---------|--------|
| Inspection | inspect, tensors, tree, flow, debug, hex, explain, oracle | 30s | all targets |
| Validation | validate, check, lint, qa, qualify | 90s | all targets |
| Inference | run, bench, eval, serve plan, chat | 60s | all targets |
| Transform | convert, import, export, quantize, merge, prune, compile, encrypt/decrypt | 30s | all targets |
| Training | finetune, distill, train plan, data audit, tokenize plan | 15s | all targets |
| Operational | list, rm, gpu, diff, trace, profile, parity, ptx-map, rosetta, compare-hf | 30s | all targets |
| GPU-specific | gpu, ptx-map, parity --gpu, bench --device gpu | 60s | jetson, lambda-labs |
| Edge Cases | nonexistent file, empty file, invalid file, unknown subcommand, duplicate flags | 5s | local only |

### Defect Taxonomy

Patterns distilled from 500 historical issues (paiml/aprender #24–#607, 500 issues analyzed 2026-04-05).

| Type | Severity | Historical Freq | Example | Ref |
|------|----------|-----------------|---------|-----|
| Panic/crash | P0 | 2.6% (13) | `thread 'main' panicked` | #598 WGSL -inf |
| Hang | P1 | 0.8% (4) | Spinner prints then blocks forever | #606 quantize/merge/prune/export/compile |
| Data corruption | P0 | — | Encrypt/decrypt roundtrip loses data | #580 merge drops tokenizer |
| NaN/Inf numerical | P0 | 7.0% (35) | Shader outputs NaN, training diverges | #563 CUDA NaN, #598 WGSL -inf |
| Exit code lie | P1 | 0.2%+ | Output says `error`/`failed`/`✗`, exit=0 | #601 lint/parity/rm |
| Silently ignored flag | P2 | 4.6% (23) | `--rank 16` → actually uses 256 | #568 --rank, #604 --vocab, #595 --quiet |
| Hardcoded value | P2 | 3.0% (15) | `num_classes: 5` regardless of input | #500, #605 DType IDs |
| Wrong output | P2 | 2.8% (14) | Q4_K_M → displays "0" | #603, #499 14.2B vs 1.8B |
| JSON output bug | P2 | 2.4% (12) | Schema drop, f32 precision artifact | #596, #510, #508 |
| Cross-subcmd divergence | P1 | — | `oracle` finds family, `serve plan` doesn't | #600, #605 |
| Perf regression | P1 | — | `run --gpu` 391× slower than `serve run --gpu` | #573 |
| Missing fallback | P1 | — | GPU failure with no CPU path | #598 + cascade |
| Cache inconsistency | P1 | 1.2% (6) | `pull` says cached, `list` shows empty | #602 |
| Misleading message | P3 | — | Valid model labeled "Garbage" | #599 |
| Phantom subcommand | P2 | — | `--help` lists subcommand that errors | #587 |
| Version sentinel bug | P3 | — | Version string shows `(unknown)` | #597 |
| Arch divergence | P0/P1 | 2.0% (10) | Passes x86_64, panics aarch64/Blackwell | #557 Jetson, #550 sm_121, #556 gx10 |
| GPU backend-specific | P0/P1 | 8.2% (41) | Works CPU+CUDA, panics WGPU | #598, #471, #573 |
| Div-by-zero/underflow | P0 | 1.2% (6) | Unsigned subtraction underflow | #492, #497, #498 |
| Build/CI breakage | P1 | 1.4% (7) | Nightly fails, check-cfg broken | #589, #590, #593 |
| Contract drift | P1 | — | Benchmark contradicts YAML obligation | — |
| Missing contract | P2 | — | apr subcommand with no YAML | — |
| Missing Lean proof | P3 | — | Contract has no `pv lean-status` proof | — |
| Contract schema | P1 | 8.2% (41) | YAML fails `pv lint` | #588 |

### QA Protocols (from historical patterns)

The skill MUST execute these **protocol-level** checks beyond the per-command grid, because bug patterns in the issue history map to systemic failures not caught by naive invocation.

1. **Silent-Flag Protocol** — For every accepted flag, run the command with AND without the flag; if output is byte-identical, the flag is a no-op (P2). Catches 23+ historical issues.
2. **Exit-Code Contradiction Protocol** — grep output for `\b(error|failed|FAIL|✗)\b`; if matched and exit code is 0, flag as exit-code lie (P1). Catches #601 family.
3. **Flag-Echo Protocol** — When the user passes `--rank 16`, parse the command's own output; if it reports a different value ("Rank: 256"), the flag is silently overridden (P2). Catches #568.
4. **Cross-Subcommand Consistency Protocol** — Run `{inspect, check, oracle, tensors, rosetta inspect, serve plan}` on the same model; diff detected `{family, dtype names, param count, tokenizer}`. Any mismatch is P1. Catches #600, #605.
5. **Cache Registry Integrity Protocol** — `pull X` → `list` must contain X → `rm X` → `list` must not contain X. Catches #602.
6. **GPU/CPU Parity Protocol** — Run same prompt on `--device cpu` vs `--device gpu`; output similarity ≥ 0.95 cosine AND tok/s ratio within 20×. Catches #573.
7. **NaN/Inf Sentinel Protocol** — Grep inference output for `\b(NaN|nan|[+-]?[Ii]nf|[+-]?[Ii]nfinity)\b` in tensor/metric values. Any match is P0. Catches #598, #563.
8. **Version Sanity Protocol** — `apr --version` must not contain `unknown`, `<empty>`, or `0000000`. Catches #597.
9. **Phantom Subcommand Protocol** — Every subcommand listed in `apr --help` must execute without returning "unknown subcommand" or "not yet implemented". Catches #587.
10. **JSON Schema Stability Protocol** — Every `--json` invocation must: (a) produce valid JSON, (b) not contain f32 precision artifacts on fields typed as integer/ratio, (c) preserve all CLI-output fields. Catches #596, #508, #510.
11. **Default-Defamation Protocol** — Never emit "Garbage", "broken", "corrupt" labels when running with default flags on a known-good model. If defaults produce insufficient samples, warn instead of defame. Catches #599.
12. **Hardware Cascade Protocol** — When GPU init fails, CPU fallback must engage silently AND correctness must be preserved. No CPU-fallback → NaN cascade (#568 → OOM → CPU fallback → NaN). P0 if cascade produces corrupt output.

### Contract Coverage Invariant

Every apr CLI subcommand must map to at least one provable-contract YAML in `contracts/`, and every contract must carry a Lean 4 proof verified by `pv lean-status`. This is subsumed by **Invariant B** (grade-A contract per recipe) but stated explicitly for the `/qa` skill:

```
∀ subcommand s ∈ apr.subcommands \ {help}:
  ∃ contract C ∈ contracts/: s ∈ C.bindings

∀ contract C ∈ contracts/:
  pv lint C = PASS
  pv lean-status C ≥ L2
```

**Status (2026-04-06)**: 11 contracts / 57 subcommands. CLI recipe parity: 57/57 = 100%. Fleet: yoga deployed (ca687120), intel/jetson pending. Gaps are filed as **P2 (missing contract)** or **P3 (missing Lean proof)** issues.

The audit uses the `pv` CLI from `../provable-contracts`:

```bash
pv lint contracts/              # 8-gate quality validation
pv lean-status contracts/       # Lean 4 proof status per contract
pv proof-status contracts/      # L1–L5 hierarchical proof levels
pv coverage contracts/          # cross-contract obligation coverage
pv audit contracts/             # traceability chain audit (paper → contract → test)
```

### Invocation

```bash
# From the apr-cookbook project directory:
/qa                                      # Default model, all reachable targets
/qa /path/to/model.gguf                  # Specific model, all targets
/qa --targets=intel,jetson               # Subset of targets
/qa --targets=local                      # Local-only (skip SSH)
```

### Issue Filing Convention

- One issue per distinct bug (group related exit-code bugs)
- Title: `<subcommand>: <concise description>`
- Body: Description, Reproduction (exact commands), Expected, Version
- Severity label in body: P0/P1/P2/P3

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

These invariants are the **master quality gates** for the cookbook. All five must pass before any release. See also the [root spec](../apr-cookbook.md#five-coverage-invariants) for the executive summary.

### Invariant A — CLI Recipe Parity (F-CLIPARITY-001)

Every apr-cli subcommand (excluding `help`) must have ≥1 cookbook recipe.

```
∀ s ∈ apr.subcommands \ {help}: ∃ r ∈ recipes: r.cli_equivalent = s
```

**Status (2026-04-06)**: 57/57 = **100%**. Enforced by `make cli-parity`.

### Invariant B — Recipe Contract Grade (F-CONTRACT-GRADE-001)

Every recipe must reference a provable-contract YAML that passes `pv lint` at **grade A**.

```
∀ r ∈ recipes:
  ∃ c ∈ contracts/: r.contract = c
  pv grade(c) = A
  pv lean-status(c) ≥ L2
```

Grade A requires: complete `metadata` (incl. academic references), ≥3 `proof_obligations`, matching `falsification_tests`, ≥1 `kani_harness`, and a passing `qa_gate`.

**Status (2026-04-06)**: 11 contracts passing `pv lint`. Target: 1 grade-A contract per recipe.

### Invariant C — Model Format Coverage (F-FORMAT-COV-001)

Every recipe that accepts a model file must demonstrate all three canonical formats where the subcommand supports them: **APR** (`.apr`), **GGUF** (`.gguf`), **SafeTensors** (`.safetensors`).

```
∀ r ∈ recipes where r.accepts_model_input:
  ∀ fmt ∈ {apr, gguf, safetensors} where r.supports(fmt):
    ∃ variant v ∈ r: v.format = fmt
```

Examples:
- `apr run` accepts all three → 3 format variants required
- `apr encrypt` is APR-only → 1 variant sufficient
- `apr import hf://…` outputs APR → 1 variant sufficient

Enforced by `make format-coverage`. Each format variant is either a separate recipe or a documented section within the recipe demonstrating that format path.

**Status (2026-04-06)**: Not yet measured. Target: 100% where applicable.

### Invariant D — arXiv Citation (F-ARXIV-001)

Every recipe must cite ≥1 arXiv paper or peer-reviewed reference linking the technique to the literature.

```
∀ r ∈ recipes: |r.citations ∩ (arXiv ∪ peer_reviewed)| ≥ 1
```

Doc comment format:

```rust
//! ## References
//! - Hu et al. (2021). *LoRA: Low-Rank Adaptation*. arXiv:2106.09685
```

Enforced by `make citation-check`.

**Status (2026-04-06)**: Not yet measured. Target: 100%.

### Invariant E — Docs Contract Coverage (F-DOCS-CONTRACT-001)

Every documentation artifact — `README.md`, `CLAUDE.md`, mdbook chapters, spec components — must be bound to a provable-contract and pass `pmat validate-readme`.

```
∀ d ∈ {README.md, CLAUDE.md, book/src/**/*.md, docs/specifications/**/*.md}:
  ∃ c ∈ contracts/: d ∈ c.bindings
  pv lint(c) = PASS
  pmat validate-readme(d) ⊨ {unverified = 0, contradictions = 0}
```

Enforced by `make docs-validate` + `pv validate contracts/docs-schema-v1.yaml`.

**Status (2026-04-06)**: `docs-schema-v1.yaml` covers spec components. Target: all repo `*.md` files bound.

### Variant definition

A **variant** is a distinct (subcommand, flag, value) triple exposed in `apr <sub> --help`. For example, `apr run` has ~20 flags; `apr merge --strategy` has 5 values (average, weighted, slerp, ties, dare), yielding 5 variants just for `--strategy`. Over 57 subcommands, apr-cli exposes ~400 distinct variants.

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

| Dimension | Actual | Target | Gate |
|-----------|--------|--------|------|
| Subcommands with ≥1 recipe | 57/57 | 57/57 | `make cli-parity` |
| Recipes with grade-A contract | 11/97+ | 97+/97+ | `make contract-grade` |
| Recipes with format variants (APR/GGUF/ST) | TBD | 100% applicable | `make format-coverage` |
| Recipes with arXiv citation | TBD | 97+/97+ | `make citation-check` |
| Docs with provable-contract | partial | 100% | `make docs-validate` |
| Estimated flag variants | ~400 | ~400 | `make variant-coverage` |
| Contracts with Lean proof ≥ L2 | 0 | 80%+ | `pv lean-status` |

### Pre-commit enforcement

```bash
make cli-parity              # Invariant A: every subcommand has a recipe
make contract-grade          # Invariant B: every recipe has grade-A contract
make format-coverage         # Invariant C: APR/GGUF/SafeTensors variants
make citation-check          # Invariant D: arXiv citation per recipe
make docs-validate           # Invariant E: docs contract coverage
make variant-coverage        # Detailed per-subcommand flag matrix
```
