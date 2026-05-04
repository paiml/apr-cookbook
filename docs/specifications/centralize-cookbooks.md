# Centralize Cookbooks Specification

**Version**: 1.0.0
**Status**: PROPOSED
**MSRV**: 1.89 (inherits from apr-cookbook v5.0)
**Date**: 2026-05-04
**Repository**: [github.com/paiml/apr-cookbook](https://github.com/paiml/apr-cookbook)
**Sovereign Stack**: APR-MONO v0.31.2

---

## Executive Summary

This is a **one-time migration spec**. It folds three sibling cookbook/library repositories into `apr-cookbook`, expanding its scope from "the .apr format cookbook" to **the umbrella sovereign-stack cookbook** — a single technical manual covering model bundling, deployment configs, data loading, and visualization, all in idiomatic Rust with IIUR contracts.

After execution, three repositories are archived (read-only, with REDIRECT.md pointing into apr-cookbook subdirectories). No source artifact is lost; every recipe, every book chapter, and every machine config has a destination.

The repository name `apr-cookbook` is **preserved**: renaming would invalidate published crates.io references, the live Pages URL, and CI badges across consuming projects. The README and book hero text re-frame scope; the slug stays.

**Sources** (counts as of 2026-05-04):
- `sovereign-ai-cookbook` — 14 deployment YAMLs + 10 stacks + jetson machine configs
- `alimentar` — 18 Rust examples + 103 mdBook chapters (data loading)
- `presentar` — 28 YAML/.prs declarative configs + 121 mdBook chapters (visualization)

**Net additions**: 60 recipe artifacts + 224 book chapters across **4 new top-level categories** (`deployment-stacks/`, `data-loading/`, `visualization/`, `machines/`), bringing the cookbook from 24 → 28 categories.

---

## Component Documents

| Document | Purpose |
|----------|---------|
| [scope.md](centralize-cookbooks/scope.md) | Charter expansion, naming decision, non-goals |
| [source-inventory.md](centralize-cookbooks/source-inventory.md) | Per-source file inventory with paths and counts |
| [migration-mapping.md](centralize-cookbooks/migration-mapping.md) | Source artifact → destination path table |
| [iiur-conformance.md](centralize-cookbooks/iiur-conformance.md) | How declarative artifacts (YAML/.prs) earn IIUR grades |
| [book-consolidation.md](centralize-cookbooks/book-consolidation.md) | mdBook merge plan and SUMMARY.md restructure |
| [tickets.md](centralize-cookbooks/tickets.md) | PMAT ticket breakdown (PMAT-065 → PMAT-070) |
| [archive-checklist.md](centralize-cookbooks/archive-checklist.md) | Gating criteria + execution steps for repo archival |

---

## Acceptance Criteria

The migration is **done** when, and only when, all of the following are true:

1. **Inventory parity**: `scripts/centralize-verify.sh` reports zero source artifacts without a destination (hash-mapped, not name-mapped — renames are tracked in [migration-mapping.md](centralize-cookbooks/migration-mapping.md)).
2. **IIUR grade**: Every migrated recipe satisfies either `contracts/recipe-iiur-v1.yaml` (Rust) or `contracts/recipe-iiur-config-v1.yaml` (declarative, see [iiur-conformance.md](centralize-cookbooks/iiur-conformance.md)).
3. **Six Coverage Invariants** (see APR Cookbook v5.0): A–F still pass after extension to the four new categories. Invariant A (CLI parity) is unaffected — sovereign/alimentar/presentar artifacts do not represent `apr` subcommands; they constitute new categories with their own coverage rules added in [tickets.md](centralize-cookbooks/tickets.md).
4. **Book builds**: `mdbook build book/` succeeds with the consolidated SUMMARY.md; no broken cross-references.
5. **Falsification preserved**: `cargo test --test falsification` still passes; no F-claim is silently dropped or weakened during migration.
6. **Archive-readiness commit**: Each source repo has a final commit `archive: redirect to apr-cookbook` containing only a top-level `REDIRECT.md`. The original `main` HEAD before that commit is tagged `pre-archive-2026-05`.

Only after (1)–(6) green do the `gh api -X PATCH … archived=true` calls in [archive-checklist.md](centralize-cookbooks/archive-checklist.md) run.

---

## Non-Goals

- **No renaming** of the apr-cookbook repository, the published crate, the Pages URL, or the GitHub org.
- **No schema migration** of `.apr`, sovereign recipe YAML, or presentar `.prs`. Formats are preserved verbatim; only locations change.
- **No re-implementation** of alimentar/presentar library code. Those crates remain published on crates.io independently; the cookbook only absorbs their **examples** and **books**, not their `src/`.
- **No GitHub org consolidation**. paiml/* org membership is unchanged. Archiving is per-repo.
- **No CI cross-wiring**. Each migrated example builds inside apr-cookbook's existing workspace; no remote workflow triggers.

---

## Risk Register

| Risk | Mitigation |
|------|------------|
| Inventory drift between spec and reality during multi-week migration | `scripts/centralize-verify.sh` runs in CI on every PR touching `examples/{deployment-stacks,data-loading,visualization,machines}/`; drift fails the gate |
| Declarative-config recipes can't enforce idempotence (deployment YAMLs touch real machines) | [iiur-conformance.md](centralize-cookbooks/iiur-conformance.md) defines `recipe-iiur-config-v1.yaml` with parse-only obligations; full idempotence is opt-in via `cfg(integration)` |
| presentar `.prs` files require `presentar` crate to validate; circular dep risk | Add `presentar` as `[dev-dependencies]` only; no runtime dep |
| sovereign YAMLs reference `forjar` recipes that live in another repo | Migrate forjar recipe references as cited links + a smoke test that loads the YAML; no execution |
| Two repos archived prematurely, blocking a missed source artifact | Archive checklist is gated on `centralize-verify.sh` passing AND a 7-day quiet period after the migration PR merges |
| Loss of git history for migrated examples | `git log --follow` works after `git mv`; preserve via `git filter-repo --paths` import where history is non-trivial (alimentar examples have multi-commit histories worth keeping) |

---

## Cross-References

- Parent spec: [apr-cookbook.md](apr-cookbook.md) — defines IIUR, falsification discipline, Six Coverage Invariants
- Memory: `memory/MEMORY.md` — current 24-category structure and PMAT workflow

---

## Approval

This spec moves to `Status: ACTIVE` after:
1. Repository owner approval (Noah Gift)
2. PMAT-065 created and assigned

Until then, no source artifacts are moved and no archive commands are run.
