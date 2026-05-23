# Kani CI Specification

**Version**: 1.0.1
**Status**: ACTIVE (PMAT-324 merged on main 2026-05-09 via PR #421; kani-gate + free-disk-space landed; runs on every PR)
**MSRV**: 1.89 (inherits from apr-cookbook v6.0)
**Date**: 2026-05-09
**Repository**: [github.com/paiml/apr-cookbook](https://github.com/paiml/apr-cookbook)

---

## Executive Summary

apr-cookbook ships **108 `#[kani::proof]` bounded-model-check harnesses** in the standalone `kani/` crate. Before PMAT-324 these harnesses were declared in YAML and implemented in Rust but never actually verified — `pv score` accepted the existence of the declared harness as evidence and capped the D3 (Kani) score at 0.9 for the `bounded_int` strategy. PMAT-324 closes that gap by adding a **`kani-gate` CI job** that installs Kani and runs `cargo kani` against the entire harness suite on every PR. The green badge is the runtime witness that all 108 contract obligations are actually proved (within their bounded domains), not just declared.

## Coverage

The `kani-gate` CI job verifies harnesses across **every cookbook contract** that ships a `kani_harnesses[]` block. As of v1.0.0 (PMAT-324 landing):

| Tier | Contracts | Harnesses | Notes |
|------|-----------|-----------|-------|
| Base contracts | 11 | 39 | aes256, apr-format-roundtrip, avx512-matmul, cli-parity, docs-schema, flash-attention, int4-quantization, lz4-decompression, mmap-inference, recipe-iiur, whisper-wer |
| Architecture-demos meta | 6 | 18 | detector, summary, compare, quirk-audit, alias-resolver (PMAT-309..313) + resolution-pipeline (PMAT-320) |
| Architecture-demos family-smoke | 17 | 51 | bert, deepseek, falcon-h1, gemma, gpt2, gptneox, llama, mamba, mistral, moonshine, openelm, opt, phi, qwen2, qwen3, qwen3_5, rwkv7 (PMAT-301..307; harnesses backfilled in PMAT-322..323) |
| **Total** | **34** | **108** | |

The `recipe-iiur-config-v1.yaml` contract registers 3 harnesses against the same `kani_harnesses::iiur_*` Rust functions as `recipe-iiur-v1.yaml` (intentional reuse), bringing the YAML-declared total to 111 obligations covered by 108 unique Rust functions.

## CI Workflow

`.github/workflows/ci.yml` adds a `kani-gate` job parallel to the existing `gate` (clippy/test/coverage/audit) and `lean-gate` (Lake-builds Lean proofs):

```yaml
kani-gate:
  name: Kani Gate
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
      with:
        ref: ${{ github.event.pull_request.head.sha || github.sha }}
    - uses: dtolnay/rust-toolchain@stable
    - uses: actions/cache@v4
      with:
        path: |
          ~/.cargo/bin/kani-verifier
          ~/.cargo/bin/cargo-kani
          ~/.kani
        key: kani-${{ runner.os }}-0.67.0
    - name: Install Kani
      run: |
        if ! command -v cargo-kani >/dev/null 2>&1; then
          cargo install --locked kani-verifier --version 0.67.0
          cargo kani setup
        fi
        cargo kani --version
    - name: Run Kani harnesses
      working-directory: kani
      run: cargo kani
```

The `ci-status` summary job lists `kani-gate` alongside `gate` and `lean-gate`; branch protection on `main` should require `CI Status` to be green.

## Runtime

Local timing on a stable laptop (rustc 1.95.0):

- **Kani install (cold)**: ~3-5 minutes (one-time cargo install + `cargo kani setup`)
- **Kani install (warm cache)**: ~10 seconds (cache hit on `~/.cargo/bin/kani-verifier`)
- **`cargo kani` over 108 harnesses**: ~14 seconds total

The cache key is pinned to the Kani toolchain version (`kani-${{ runner.os }}-0.67.0`) so a Kani upgrade triggers a clean install but day-to-day PRs benefit from warm cache.

## Strategy Field & Score Ceiling

`pv score` reads the YAML `kani_harnesses[].strategy` field to compute the D3 (Kani) axis score:

| Strategy | Score weight | Semantics |
|----------|--------------|-----------|
| `exhaustive` | 1.0 | enumerates the full bounded domain (small finite state space) |
| `bounded_int` | 0.9 | symbolic integer inputs with an upper bound |
| `stub_float` | 0.8 | f32/f64 properties stubbed via integer proxies |
| `compositional` | 0.7 | composes multiple smaller invariants |
| (none) | 0.5 | strategy unspecified |

108 of 111 declared harnesses use `bounded_int` so D3 caps at 0.9 for the corresponding contracts. **This 0.9 is the static-analysis ceiling**: it reports "harnesses are declared and implemented" but cannot, by itself, attest that they actually pass.

The runtime witness for D3 is the **green `kani-gate` badge** on the PR. When that badge is green, every harness has been symbolically executed by Kani and verified — the contract obligations are proved (within their bounded domains), not just declared.

This split is intentional and mirrors the Lean side: `pv score` reports D4 (Lean) score 1.0 when `lean.status: proved` is set, but only the green `lean-gate` badge attests that `lake build` actually compiles the proof. **Score is structural readiness; CI gate is runtime verification.**

## Falsification Discipline

The `kani-gate` job is a falsifier in the Popperian sense — its sole purpose is to break when a harness regression slips in:

1. **Harness bit-rot**: a harness compiles but no longer encodes its obligation correctly (e.g., overflow in `arch_quirk_audit_total` caught during PMAT-324 verification — fixed pre-merge).
2. **Toolchain drift**: a Kani upgrade changes verification semantics; the gate goes red until harnesses are updated.
3. **Implementation drift**: if a harness is rewritten to call into the actual cookbook code (rather than a stub), and the cookbook code violates the bounded property, the gate fails.

In v1.0.0 the harnesses are teaching-grade stubs (bounded_int proxies) that verify the *shape* of each obligation without invoking the real recipe code. Promoting harnesses to call the real code is tracked in a separate backlog item (Kani symbolic execution of String inputs is the gating constraint).

## Definition of Done

PMAT-324 ships when:

- [x] `cargo kani` runs cleanly against the 108-harness suite locally (verified 2026-05-09: 108/108 pass in 13.9s)
- [x] `arch_quirk_audit_total` overflow regression discovered and fixed during local verification
- [x] `.github/workflows/ci.yml` adds a `kani-gate` job with caching + install + run steps
- [x] `ci-status` summary job lists `kani-gate` alongside `gate` and `lean-gate`
- [x] `kani-ci.md` spec authored documenting the integration and the score-ceiling framing
- [x] Architecture-demos `tickets.md` backlog updated to point the "D3 0.9 → 1.0" item at the kani-gate runtime witness rather than at score-axis movement

## Backlog

- **Promote selected `bounded_int` harnesses to `exhaustive`** where the input domain is genuinely tiny (e.g. `arch_resolution_pipeline_total` covers 16 boolean cube cases). Would bump per-contract D3 score from 0.9 to 1.0 for those specific harnesses. Editorial decision; not blocking.
- **Replace stub-grade harnesses with calls to real cookbook code** where Kani's solver permits it (small bounded String inputs via byte arrays + length bounds). Substantial follow-up; tracked separately.
- **Add `actually_verified: true` field to YAML** sourced from a `kani-results.json` artifact emitted by CI. Would let `pv score` honestly bump D3 to 1.0 once the green `kani-gate` runs. Requires upstream `aprender-contracts` change (out of cookbook scope) — tracked in [aprender#1595](https://github.com/paiml/aprender/issues/1595).

---

## Component Documents

| Document | Purpose |
|----------|---------|
| [`.github/workflows/ci.yml`](../../.github/workflows/ci.yml) | The actual workflow definition — `kani-gate` job lives here |
| [`kani/src/lib.rs`](../../kani/src/lib.rs) | 1500+ lines containing all 108 `#[kani::proof]` Rust functions |
| [`kani/Cargo.toml`](../../kani/Cargo.toml) | Standalone crate; not a workspace member of apr-cookbook (kept independent so `cargo build` from repo root doesn't touch it) |
| [`docs/specifications/architecture-demos.md`](architecture-demos.md) | Spec where the architecture-demos harness obligations are defined |
| [`contracts/*.yaml`](../../contracts/) | The 34 YAML contracts that declare `kani_harnesses[]` blocks |

## See Also

- [Kani user guide](https://model-checking.github.io/kani/) (upstream documentation)
- [PMAT-046](../roadmaps/roadmap.yaml) — original Kani harness scaffolding ticket
- [PMAT-320..323](architecture-demos/tickets.md) — architecture-demos harness expansion
