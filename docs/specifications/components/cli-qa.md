# APR CLI QA Process

The installed `apr` binary is tested via the `/qa` Claude Code skill (`.claude/skills/qa/SKILL.md`). This is an exhaustive, **fleet-wide**, model-in-the-loop QA process that exercises every subcommand against a real model on **every reachable hardware target**, then audits subcommand coverage against the `provable-contracts` registry.

---

## Process

1. **Audit** contract coverage: every apr subcommand → YAML contract + Lean proof (Phase 0)
2. **Probe** all targets: intel, yoga, jetson, lambda-labs + local (Phase 1 setup)
3. **Pull** a test model (default: Qwen2.5-Coder-1.5B Q4_K_M, ~1 GB) on each target
4. **Exercise** all 57 subcommands across 8 categories per target
5. **Detect** bugs matching the defect taxonomy below (tag with target)
6. **File** GitHub issues via `gh` with per-target reproduction steps

---

## Target Fleet

| Target | Access | Arch | Accelerator | Purpose |
|--------|--------|------|-------------|---------|
| `local` | (host) | discover | discover | Fast iteration, contract audit |
| `intel` | `ssh intel` | x86_64 | AVX2/AVX-512 CPU | x86_64 SIMD parity |
| `yoga` | `ssh yoga` | discover | CPU | Laptop-class baseline |
| `jetson` | `ssh jetson` | aarch64 | NVIDIA GPU | ARM + CUDA parity |
| `lambda-labs` | `ssh -p 2222 noahgift@localhost` | x86_64 | NVIDIA GPU | Cloud GPU parity |

Arch-divergence bugs (works on x86_64 but panics on aarch64) are a first-class defect category. Every bug report must list the affected targets as checkboxes.

---

## Subcommand Test Matrix

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

---

## Defect Taxonomy

Patterns distilled from 500 historical issues (paiml/aprender #24–#607, 500 issues analyzed 2026-04-05).

| Type | Severity | Historical Freq | Example | Ref |
|------|----------|-----------------|---------|-----|
| Panic/crash | P0 | 2.6% (13) | `thread 'main' panicked` | #598 WGSL -inf |
| Hang | P1 | 0.8% (4) | Spinner prints then blocks forever | #606 |
| Data corruption | P0 | — | Encrypt/decrypt roundtrip loses data | #580 |
| NaN/Inf numerical | P0 | 7.0% (35) | Shader outputs NaN, training diverges | #563, #598 |
| Exit code lie | P1 | 0.2%+ | Output says `error`/`failed`/`✗`, exit=0 | #601 |
| Silently ignored flag | P2 | 4.6% (23) | `--rank 16` → actually uses 256 | #568, #604, #595 |
| Hardcoded value | P2 | 3.0% (15) | `num_classes: 5` regardless of input | #500, #605 |
| Wrong output | P2 | 2.8% (14) | Q4_K_M → displays "0" | #603, #499 |
| JSON output bug | P2 | 2.4% (12) | Schema drop, f32 precision artifact | #596, #510, #508 |
| Cross-subcmd divergence | P1 | — | `oracle` finds family, `serve plan` doesn't | #600, #605 |
| Perf regression | P1 | — | `run --gpu` 391x slower than `serve run --gpu` | #573 |
| Missing fallback | P1 | — | GPU failure with no CPU path | #598 |
| Cache inconsistency | P1 | 1.2% (6) | `pull` says cached, `list` shows empty | #602 |
| Misleading message | P3 | — | Valid model labeled "Garbage" | #599 |
| Phantom subcommand | P2 | — | `--help` lists subcommand that errors | #587 |
| Version sentinel bug | P3 | — | Version string shows `(unknown)` | #597 |
| Arch divergence | P0/P1 | 2.0% (10) | Passes x86_64, panics aarch64 | #557, #550, #556 |
| GPU backend-specific | P0/P1 | 8.2% (41) | Works CPU+CUDA, panics WGPU | #598, #471, #573 |
| Div-by-zero/underflow | P0 | 1.2% (6) | Unsigned subtraction underflow | #492, #497, #498 |
| Build/CI breakage | P1 | 1.4% (7) | Nightly fails, check-cfg broken | #589, #590, #593 |
| Contract drift | P1 | — | Benchmark contradicts YAML obligation | — |
| Missing contract | P2 | — | apr subcommand with no YAML | — |
| Missing Lean proof | P3 | — | Contract has no `pv lean-status` proof | — |
| Contract schema | P1 | 8.2% (41) | YAML fails `pv lint` | #588 |

---

## QA Protocols (from historical patterns)

The skill MUST execute these **protocol-level** checks beyond the per-command grid, because bug patterns in the issue history map to systemic failures not caught by naive invocation.

1. **Silent-Flag Protocol** — For every accepted flag, run the command with AND without the flag; if output is byte-identical, the flag is a no-op (P2). Catches 23+ historical issues.
2. **Exit-Code Contradiction Protocol** — grep output for `\b(error|failed|FAIL|✗)\b`; if matched and exit code is 0, flag as exit-code lie (P1). Catches #601 family.
3. **Flag-Echo Protocol** — When the user passes `--rank 16`, parse the command's own output; if it reports a different value ("Rank: 256"), the flag is silently overridden (P2). Catches #568.
4. **Cross-Subcommand Consistency Protocol** — Run `{inspect, check, oracle, tensors, rosetta inspect, serve plan}` on the same model; diff detected `{family, dtype names, param count, tokenizer}`. Any mismatch is P1. Catches #600, #605.
5. **Cache Registry Integrity Protocol** — `pull X` → `list` must contain X → `rm X` → `list` must not contain X. Catches #602.
6. **GPU/CPU Parity Protocol** — Run same prompt on `--device cpu` vs `--device gpu`; output similarity >= 0.95 cosine AND tok/s ratio within 20x. Catches #573.
7. **NaN/Inf Sentinel Protocol** — Grep inference output for `\b(NaN|nan|[+-]?[Ii]nf|[+-]?[Ii]nfinity)\b` in tensor/metric values. Any match is P0. Catches #598, #563.
8. **Version Sanity Protocol** — `apr --version` must not contain `unknown`, `<empty>`, or `0000000`. Catches #597.
9. **Phantom Subcommand Protocol** — Every subcommand listed in `apr --help` must execute without returning "unknown subcommand" or "not yet implemented". Catches #587.
10. **JSON Schema Stability Protocol** — Every `--json` invocation must: (a) produce valid JSON, (b) not contain f32 precision artifacts on fields typed as integer/ratio, (c) preserve all CLI-output fields. Catches #596, #508, #510.
11. **Default-Defamation Protocol** — Never emit "Garbage", "broken", "corrupt" labels when running with default flags on a known-good model. If defaults produce insufficient samples, warn instead of defame. Catches #599.
12. **Hardware Cascade Protocol** — When GPU init fails, CPU fallback must engage silently AND correctness must be preserved. No CPU-fallback -> NaN cascade (#568 -> OOM -> CPU fallback -> NaN). P0 if cascade produces corrupt output.

---

## Contract Coverage Invariant

Every apr CLI subcommand must map to at least one provable-contract YAML in `contracts/`, and every contract must carry a Lean 4 proof verified by `pv lean-status`. This is subsumed by **Invariant B** (grade-A contract per recipe) but stated explicitly for the `/qa` skill:

```
∀ subcommand s ∈ apr.subcommands \ {help}:
  ∃ contract C ∈ contracts/: s ∈ C.bindings

∀ contract C ∈ contracts/:
  pv lint C = PASS
  pv lean-status C ≥ L2
```

**Status (2026-04-22)**: 11 contracts / 57 subcommands. CLI recipe parity: 57/57 = 100%. Fleet: yoga deployed (ca687120), intel/jetson pending. Gaps are filed as **P2 (missing contract)** or **P3 (missing Lean proof)** issues.

The audit uses the `pv` CLI from `../provable-contracts`:

```bash
pv lint contracts/              # 8-gate quality validation
pv lean-status contracts/       # Lean 4 proof status per contract
pv proof-status contracts/      # L1–L5 hierarchical proof levels
pv coverage contracts/          # cross-contract obligation coverage
pv audit contracts/             # traceability chain audit (paper → contract → test)
```

---

## Invocation

```bash
# From the apr-cookbook project directory:
/qa                                      # Default model, all reachable targets
/qa /path/to/model.gguf                  # Specific model, all targets
/qa --targets=intel,jetson               # Subset of targets
/qa --targets=local                      # Local-only (skip SSH)
```

## Issue Filing Convention

- One issue per distinct bug (group related exit-code bugs)
- Title: `<subcommand>: <concise description>`
- Body: Description, Reproduction (exact commands), Expected, Version
- Severity label in body: P0/P1/P2/P3
