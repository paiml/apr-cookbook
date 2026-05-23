# Centralize Cookbooks Specification

**Version**: 1.1.0
**Status**: AMENDED (post-migration consolidation 2026-05-06)
**MSRV**: 1.89 (inherits from apr-cookbook v5.0)
**Date**: 2026-05-04 (v1.0.0); amended 2026-05-06 (v1.1.0)
**Repository**: [github.com/paiml/apr-cookbook](https://github.com/paiml/apr-cookbook)
**Sovereign Stack**: APR-MONO v0.31.2

---

## Amendment 2026-05-06 — Post-Migration Library Consolidation (v1.1.0)

After PMAT-066 archived `paiml/alimentar` and migrated its 18 examples into `apr-cookbook/examples/data-loading/`, **the alimentar library source itself was folded into the APR-MONO monorepo** at `aprender/crates/aprender-data/` (package `aprender-data`, lib name `alimentar`, v0.31.2 published to crates.io). Both the standalone repo and the migrated cookbook now point upstream to a single source of truth.

**Implications:**

1. **paiml/alimentar is ARCHIVED.** The git URL still resolves (read-only) and `REDIRECT.md` points here, but no further commits land there. The `archive-log.txt` entry from 2026-05-05 stands.

2. **Canonical source lives in aprender.** New aprender-data features, bug fixes, and version bumps happen in `aprender/crates/aprender-data/` only. The `repository` field in that crate's Cargo.toml still reads `github.com/paiml/alimentar` (preserved for crates.io continuity); update is not required since the URL redirects.

3. **apr-cookbook MUST show recipes.** `examples/data-loading/` is the canonical recipe gallery for `aprender-data` (lib `alimentar`). It is the ONLY user-facing surface that demonstrates the library. As aprender-data ships new APIs (new dataset formats, new transforms, new REPL commands), `examples/data-loading/` MUST be extended to cover them — same IIUR template, same Verdict-enum + arXiv-citation discipline as PMAT-110+ apr CLI recipe expansions.

4. **Dependency wiring is correct as of v0.31.2.** `Cargo.toml` line 123:
   ```toml
   alimentar = { version = "0.31.2", package = "aprender-data", features = ["doctest"] }
   ```
   No change needed when aprender-data ships a patch; bump the version pin and re-run the data-loading example suite.

5. **F-invariant for data-loading.** The original 18 migrated recipes satisfy the per-API floor at the time of migration. New aprender-data subcommands (`alimentar repl`, `alimentar registry`, `alimentar ingest`, etc.) added after 2026-05-06 each need ≥3 recipes in `examples/data-loading/` to maintain F-invariant parity with apr CLI surfaces — track via PMAT-200-series tickets when expansion ramps up.

The rest of this spec (and all component documents under `centralize-cookbooks/`) remains accurate for the one-time migration mechanics. Where it says "alimentar crate stays published on crates.io," read that as "aprender-data is published on crates.io with `alimentar` as its lib name and the `paiml/alimentar` URL preserved for SemVer continuity."

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

## Implementation Status (as of 2026-05-05)

Spec moved to `Status: ACTIVE` on 2026-05-04 after owner approval. Tickets PMAT-065..069 implemented and pushed as 4 PRs against `paiml/apr-cookbook`. PMAT-070 deferred per spec gates.

### Per-ticket status

| Ticket | Scope | PR | Branch | Local impl | CI required checks | Unified gates (advisory) |
|--------|-------|----|----|------------|---------|---------|
| PMAT-065 | sovereign migration (14 wrappers, 10 stacks, jetson, contract, 30 book chapters) | [#249](https://github.com/paiml/apr-cookbook/pull/249) | `feat/centralize-pmat-065-sovereign` | ✅ 42 wrapper tests green | ✅ all 5 required pass | lint-gate ✅ pass; cpu-gates ❌ runner OOM (transient) |
| PMAT-066 | alimentar migration (18 examples + 71 chapters + IIUR retrofit) | [#250](https://github.com/paiml/apr-cookbook/pull/250) | `feat/centralize-pmat-066-alimentar` | ✅ 18 example tests green | ✅ all 5 required pass | lint-gate ✅ pass; cpu-gates ❌ container collision (transient) |
| PMAT-067 | presentar migration (28 declarative configs + 1 validator + 52 chapters) | [#251](https://github.com/paiml/apr-cookbook/pull/251) | `feat/centralize-pmat-067-presentar` | ✅ 4 validator tests green | ✅ all 5 required pass | lint-gate ✅ pass; cpu-gates ⏳ pending |
| PMAT-068+069 | invariants + spec v6.0.0 + README | [#252](https://github.com/paiml/apr-cookbook/pull/252) | `feat/centralize-pmat-068-069-spec-update` | ✅ docs-only | ✅ all 5 required pass | lint-gate ⏳ pending; cpu-gates ⏳ pending |
| PMAT-070 | archive 3 source repos | — | — | ⏸️ DEFERRED — gated on PMAT-065..069 merged + 7-day quiet + verifier | — | — |

### Required-check status (branch-protection gating)

All 4 PRs are **mergeable on required checks**: `CI Status`, `Quality Gate (stable)`, `Lean Gate`, `Build book`, `Verify recipe table is deterministic` are green on every PR. The `unified` workflow is advisory-only (`continue-on-error: true`) per `.github/workflows/unified-gate-advisory.yml` and does not block merge.

### Infrastructure dependency surfaced and resolved (PMAT-161)

During CI verification of the 4 PRs, three structural gaps in the self-hosted runner stack were discovered:

1. **`forjar` not on PATH on `mac-server`** — `_preflight` in `paiml/infra` exited 127. Resolved by `ssh noah@192.168.50.100 cargo install --locked forjar` (forjar 1.4.1 installed at `~/.cargo/bin/forjar`; runner picks it up via existing `.path`).
2. **`pmat` not on PATH on `mac-server`** — lint-gate's L4 step exited 127. Resolved by `cargo install --locked pmat` (pmat 3.16.0; lint-gate now passes on all PRs that have re-run).
3. **`machines/clean-room/gates/apr-cookbook.sh` missing in `paiml/infra`** — forjar's `gate-script` file resource exited 1. Structural omission — apr-cookbook had never had a gate script even though the workflow expected one.

Resolved by [paiml/infra#82](https://github.com/paiml/infra/pull/82), merged at commit `089592c`. The PR carries:
- `_preflight` self-heal: `cargo install --locked forjar` if missing AND cargo present (so future runner gaps recover automatically)
- New `gates/apr-cookbook.sh` (the structural omission)
- `stack-tools` manifest cleanup: drop `batuta` (lib-only, no binary on crates.io); document `copia --features cli`

### Residual cpu-gates flakes

After the PMAT-161 infra fixes, cpu-gates re-runs on PRs #249/#250 hit *runtime resource contention* on the runner host:

- PR #249: `system-deps failed: exit code 137` (SIGKILL = OOM during apt-install in clean-room)
- PR #250: `system-deps failed: container is not running` (Docker container died — likely OOM by neighbor)

Cause: 4 PRs hitting mac-server concurrently; 4 simultaneous clean-room containers each running `apt-get install` exceeded host memory. NOT a code or infra bug.

Mitigation: re-run cpu-gates sequentially (one at a time) once the burst settles. `unified / cpu-gates` is advisory and does not block merge — the user may admin-merge when Quality Gate is green per `MEMORY.md` "pmat work flow" cheatsheet.

### Outstanding actions

1. **Merge order**: 249 → 250 → 251 → 252 (each later PR documents work added by earlier ones; PMAT-068+069 references categories created by 065/066/067).
2. **Re-run PMAT-070 gate criteria** once 249–252 merge: 7-day quiet period, `scripts/centralize-verify.sh --strict` passes, source repos still have crates.io publishing intact.
3. **Tag `v6.0.0`** on apr-cookbook main after PMAT-068+069 merges (per ticket spec).
4. **PMAT-070** (archive): tag `pre-archive-2026-05` on each source repo, open REDIRECT.md PR, set archive bit, log to `archive-log.txt`. Order: sovereign → presentar → alimentar with 24h pauses.

---

## Approval

This spec moved to `Status: ACTIVE` on 2026-05-04 after:
1. ✅ Repository owner approval (Noah Gift)
2. ✅ PMAT-065 created and assigned (and 5 sibling tickets PMAT-066..070)

Source artifacts are migrated; archive commands gated per [archive-checklist.md](centralize-cookbooks/archive-checklist.md).
