# PMAT Ticket Breakdown

Centralization is decomposed into **6 PMAT tickets**, executed in dependency order. Each ticket is independently shippable, testable, and reversible.

Ticket numbers below are placeholders (`PMAT-065..105`); actual numbers are assigned by `pmat work add` at execution time.

---

## PMAT-065 — Migrate sovereign-ai-cookbook

**Priority**: P1
**Estimate**: 2–3 days
**Depends on**: spec approval (this document accepted)

### Scope

1. Create `examples/deployment-stacks/{recipes,stacks}/` directory tree.
2. Copy all 14 sovereign recipe YAMLs verbatim per [migration-mapping.md](migration-mapping.md).
3. Copy all 10 sovereign stack directories (renaming the second `09-` to `10-`).
4. Generate the 14 Rust loader wrappers per [iiur-conformance.md](iiur-conformance.md) Strategy A. Use `scripts/gen-deployment-wrappers.sh` (new, part of this ticket).
5. Copy `machines/jetson/` to `examples/machines/jetson/`.
6. Author the **Deployment Stacks** book section per [book-consolidation.md](book-consolidation.md). One chapter per recipe, one per stack, one for jetson, one overview.
7. Add the new contract `contracts/recipe-iiur-config-v1.yaml` per [iiur-conformance.md](iiur-conformance.md) §"Class 3".
8. Add `serde_yaml` to workspace dev-dependencies (already present? verify; add if missing).
9. Wire the new wrappers into `Cargo.toml` `[[example]]` entries.

### Definition of Done

- All 14 wrappers compile and `cargo test --example <wrapper>` passes for each.
- All 10 stacks present at destination paths.
- `contracts/recipe-iiur-config-v1.yaml` exists; `pv lint contracts/` passes.
- `mdbook build book/` succeeds with new Deployment Stacks chapters.
- `scripts/centralize-verify.sh --source sovereign` returns clean.

### Override exposure

The ticket should NOT need overrides for `lint`, `supply-chain`, or `file-size`. If `manifest` blocks complete due to mass file additions, override with `--override-claims manifest --ticket PMAT-065`.

---

## PMAT-066 — Migrate alimentar examples + book

**Priority**: P1
**Estimate**: 3–4 days (citation lookup is the slow part)
**Depends on**: PMAT-065 (uses the same `examples/` workspace conventions)

### Scope

1. Create `examples/data-loading/`.
2. Copy all 18 alimentar examples verbatim.
3. Run `scripts/iiur-retrofit-alimentar.sh` (new, part of this ticket) to apply the per-file IIUR header retrofit per [iiur-conformance.md](iiur-conformance.md) §"Class 1".
4. Manual citation lookup: for each example, find an arXiv/DOI for the relevant concept (drift detection, federated learning, hub publishing, etc.). Insert into the doc header. Where no clean citation exists, use `Citation: N/A — see PMAT-066` and create a follow-up ticket.
5. Copy `alimentar/book/src/**` (excluding `development/` and `ecosystem/`) to `book/src/data-loading/`.
6. Run `scripts/rewrite-book-links.sh --section data-loading`.
7. Wire the 18 examples into `Cargo.toml` `[[example]]` entries.
8. Add `alimentar`, `arrow`, `parquet` to workspace dev-dependencies.

### Definition of Done

- All 18 examples build, test, and pass clippy with no `#![allow(unwrap_used,...)]` escape hatches.
- Each example's doc header has Contract + Citation lines (Citation may be N/A with a follow-up ticket).
- `mdbook build book/` succeeds with the new Data Loading section.
- `scripts/centralize-verify.sh --source alimentar` returns clean.
- `cargo test --test contracts` accepts all 18 retrofitted examples.

### Override exposure

`file-size` may trigger if any retrofitted example exceeds 500 lines after header additions; if so, split into helper module (don't override).

---

## PMAT-067 — Migrate presentar examples + book

**Priority**: P1
**Estimate**: 2–3 days
**Depends on**: PMAT-065 (uses `recipe-iiur-config-v1.yaml` contract)

### Scope

1. Create `examples/visualization/{ald,apr,charts,dashboards,edge_cases,prs}/`.
2. Copy all 28 declarative configs verbatim (preserve subdir layout).
3. Author the single validator wrapper `examples/visualization/load_visualization.rs` per [iiur-conformance.md](iiur-conformance.md) §"Class 2 Strategy B".
4. Copy `presentar/book/src/**` (excluding `development/` and `ecosystem/`) to `book/src/visualization/`.
5. Run `scripts/rewrite-book-links.sh --section visualization`.
6. Wire `load_visualization` into `Cargo.toml` `[[example]]`.
7. Add `presentar` (latest crates.io) to workspace dev-dependencies.

### Definition of Done

- `load_visualization` example builds, runs, and passes its 2 tests.
- All 28 configs present at destination paths.
- `mdbook build book/` succeeds with the new Visualization section.
- `scripts/centralize-verify.sh --source presentar` returns clean.

### Override exposure

If `presentar` crates.io version doesn't yet support all 28 fixture schemas (some may be from main branch), gate the failing fixtures behind `#[ignore]` and create a follow-up ticket. Do NOT override IIUR contract checks.

---

## PMAT-068 — Update Six Coverage Invariants

**Priority**: P2
**Estimate**: 1 day
**Depends on**: PMAT-065, PMAT-066, PMAT-067 (all sources migrated before computing extended invariants)

### Scope

1. Update `scripts/coverage-invariants.sh` per [iiur-conformance.md](iiur-conformance.md) §"Coverage Invariants — Update".
2. Update `memory/MEMORY.md` "Six Coverage Invariants" section to reflect the new denominators and carve-outs.
3. Update `docs/specifications/components/recipe-catalog.md` to add the 4 new categories (deployment-stacks, data-loading, visualization, machines) with their recipe tables.
4. Re-run all six invariants; CI gate must stay green.

### Definition of Done

- `scripts/coverage-invariants.sh` exits 0.
- Recipe catalog reflects all migrated artifacts.
- MEMORY.md "Project Stats" line updated to reflect new total category and example counts.

### Override exposure

None expected. If invariant E (docs contract coverage) drops below threshold because of new chapters lacking contract refs, fix the chapters — don't override.

---

## PMAT-069 — Update master spec to v6.0.0

**Priority**: P2
**Estimate**: 0.5 day
**Depends on**: PMAT-065, PMAT-066, PMAT-067, PMAT-068

### Scope

1. Bump `docs/specifications/apr-cookbook.md` Version field from 5.0.0 → 6.0.0.
2. Rewrite Executive Summary per [scope.md](scope.md) §"After (apr-cookbook v6.0, post-centralization)".
3. Update Technology Stack diagram to add the 4 new categories.
4. Update README.md hero text and category list.
5. Update `book/src/SUMMARY.md` to the consolidated layout per [book-consolidation.md](book-consolidation.md) §"SUMMARY.md ordering policy".
6. Tag release: `git tag v6.0.0` after merge.

### Definition of Done

- README + spec + book reflect umbrella scope.
- v6.0.0 git tag pushed.
- crates.io readme regen succeeds (if applicable).

---

## PMAT-070 — Archive source repositories

**Priority**: P3
**Estimate**: 0.5 day
**Depends on**: PMAT-069 (all migration work merged) + 7-day quiet period
**See also**: [archive-checklist.md](archive-checklist.md)

### Scope

1. Run `scripts/centralize-verify.sh --strict` (full mode, no source skips). Must exit 0.
2. For each of `sovereign-ai-cookbook`, `alimentar`, `presentar`:
   a. Open a final PR adding `REDIRECT.md` at repo root with content per archive-checklist template.
   b. Merge PR.
   c. Tag `pre-archive-2026-05` at the commit BEFORE the redirect (preserves last live HEAD).
   d. Run `gh api -X PATCH repos/paiml/<repo> -f archived=true`.
3. Update apr-cookbook README "Related Repositories" section: strikethrough + redirect note for each archived repo.

### Definition of Done

- All 3 source repos show "Archived" badge on github.com.
- Each contains REDIRECT.md as its only top-level diff vs. last live state.
- `pre-archive-2026-05` tag exists in each.
- apr-cookbook README reflects archived status.

### Override exposure

`github-sync` may block ticket completion because the apr-cookbook tree is clean but source repos have just been archived — that's expected; override with `--override-claims github-sync --ticket PMAT-070` only after verifying archive bits set.

---

## Dependency Graph

```
PMAT-065 (sovereign) ─┐
                      ├─→ PMAT-068 (invariants) ─→ PMAT-069 (spec bump) ─→ PMAT-070 (archive)
PMAT-066 (alimentar) ─┤
                      │
PMAT-067 (presentar) ─┘
```

PMAT-065/101/102 can be parallelized across 3 days if work is distributed; otherwise serialize.

---

## Backout Plan

Each ticket is reversible by `git revert <merge-commit>` until PMAT-070 lands. After PMAT-070 (archive bit set on source repos), un-archive is one API call (`-f archived=false`) but the redirect commit remains in their history — handle by force-resetting back to `pre-archive-2026-05` if a true rollback is needed (destructive; requires explicit user approval).
