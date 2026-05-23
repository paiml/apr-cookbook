# Expand Cookbooks Specification

**Version**: 1.0.0
**Status**: PROPOSED
**MSRV**: 1.89 (inherits from apr-cookbook v6.0)
**Date**: 2026-05-05
**Repository**: [github.com/paiml/apr-cookbook](https://github.com/paiml/apr-cookbook)
**Sovereign Stack**: APR-MONO v0.31.2 + sister crates

---

## Executive Summary

apr-cookbook v6.0.0 (centralize-cookbooks) consolidated three predecessor cookbooks. Since then, **aprender 0.31.0..0.31.2 (Unreleased)** shipped substantial new capability — a Claude Messages API drop-in (`apr serve anthropic`), a 21-row Claude Code parity surface (`apr code`), GPU/CPU output-parity oracle bisection (`apr trace --save-tensor` + `apr diff --values` on APRT stage tensors), MCP M5 transports (SSE, WebSocket, notifications), end-to-end publish manifests, plus **6 new sister crates** published at 0.31.2 (`aprender-mcp`, `aprender-tsp`, `aprender-shell`, `aprender-monte-carlo`, `aprender-cgp`, `aprender-contracts-macros`).

The cookbook has **zero recipes** for any of those surfaces. A user landing on the cookbook today learns nothing about: how to bisect a silent-GPU-gibberish bug; how to embed an MCP server in a Rust app; how to use the new code-agent surface; how to publish a model end-to-end; or that the sister crates exist at all.

This spec proposes ~50 new recipes across 6 new categories and 6 existing categories, plus the per-subcrate ≥3-recipe requirement (18 recipes alone). Net: apr-cookbook grows from 28 categories → 34 categories, ~388 recipes → ~440 recipes.

---

## Component Documents

| Document | Purpose |
|----------|---------|
| [scope.md](expand-cookbooks/scope.md) | Charter expansion, naming decisions, non-goals |
| [gap-inventory.md](expand-cookbooks/gap-inventory.md) | Per-capability gap, ranked by user-facing impact |
| [recipe-catalog.md](expand-cookbooks/recipe-catalog.md) | All planned recipes with names, paths, contracts, citations |
| [subcrate-coverage.md](expand-cookbooks/subcrate-coverage.md) | ≥3-recipe coverage per sister crate (the hard requirement) |
| [tickets.md](expand-cookbooks/tickets.md) | PMAT ticket breakdown (PMAT-072..083) |

---

## Acceptance Criteria

The expand-cookbooks initiative is **done** when, and only when:

1. **Recipe count**: every Tier-1/2/3/4 capability listed in [gap-inventory.md](expand-cookbooks/gap-inventory.md) has at least one recipe; every sister crate listed in [subcrate-coverage.md](expand-cookbooks/subcrate-coverage.md) has **≥3** recipes.
2. **IIUR grade**: every new recipe satisfies `contracts/recipe-iiur-v1.yaml` (Rust binaries) or `contracts/recipe-iiur-config-v1.yaml` (declarative wrappers).
3. **CLI parity (Invariant A) extended**: every new `apr` subcommand (`apr code`, `apr trace --save-tensor`, `apr serve anthropic`, `apr serve plan` with HF, `apr publish` end-to-end, `apr diff --values` APRT) is covered by at least one recipe.
4. **Cargo.toml**: new sister-crate dev-dependencies declared (`aprender-mcp`, `aprender-tsp`, `aprender-shell`, `aprender-monte-carlo`, `aprender-cgp`, `aprender-contracts-macros`).
5. **Recipe table**: README regenerated to reflect new total.
6. **mdBook**: each new category gets an overview chapter; existing SUMMARY.md gets the new sections wired in.
7. **Recipe-catalog spec**: `docs/specifications/components/recipe-catalog.md` extended with the 6 new categories and recipe tables.

---

## Non-Goals

- **No re-implementation** of aprender-tsp/shell/monte-carlo/cgp logic. Recipes call the published APIs.
- **No coverage of unstable/experimental aprender APIs** (anything marked `#[doc(hidden)]` or behind `#[cfg(feature = "experimental")]`).
- **No GPU-required CI** — recipes that exercise GPU paths must `#[cfg_attr(not(gpu_available), ignore)]` their tests, falling back to CPU-only smoke at minimum.
- **No standalone binaries/services** — every recipe is a `cargo run --example <name>` artifact, like the rest of the cookbook.
- **No version pin to 0.31.2** — recipes use the same `^0.31` as existing cookbook deps; later releases pull through.

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| New sister crates may not be API-stable yet (v0.31.x is still pre-1.0) | Pin to exact `^0.31.2`; failure on minor bumps surfaces as a cookbook test fail, which is the canary we want |
| `aprender-tsp` / `-monte-carlo` / `-shell` may have GPL/strong copyleft transitive deps | Each subcrate gets a one-time `cargo deny check licenses` run before recipe authoring; results captured in subcrate-coverage.md notes |
| `apr code` recipes need Anthropic-style state on disk (`.apr/agents/`, `.apr/skills/`) | Recipes use `tempfile::tempdir()` for isolation per IIUR; nothing escapes the recipe's scope |
| GPU-path recipes (CPU/GPU parity, MoE rayon dispatch bench) may require CUDA on the runner | Mark `#[cfg_attr(not(any(target_arch = "x86_64", feature = "cuda")), ignore)]`; CI runs CPU-only path |
| `apr publish` recipes upload to crates.io/HF Hub | Use `--dry-run` mode where supported; otherwise `tempfile::tempdir()` as the upload destination |
| Inventory bloat — ~50 recipes is a big sprint | Decompose into 12 PMAT tickets (PMAT-072..083), one per category or capability cluster; each ticket is independently shippable |

---

## Cross-References

- Parent spec: [apr-cookbook.md](apr-cookbook.md) — IIUR, falsification discipline, Six Coverage Invariants
- Predecessor: [centralize-cookbooks.md](centralize-cookbooks.md) — v6.0.0 umbrella consolidation
- Memory: `memory/MEMORY.md` — current 28-category structure post-centralize-cookbooks

---

## Approval

This spec moves to `Status: ACTIVE` after:
1. Repository owner approval (Noah Gift)
2. PMAT-072 created and assigned

Until then, no new examples land and no Cargo.toml changes are made.
