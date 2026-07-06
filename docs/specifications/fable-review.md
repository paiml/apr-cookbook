# apr-cookbook — EV-Ordered, Falsifiable Engineering Roadmap (Fable 5 Review)

**Date:** 2026-07-05
**Analysis pinned to:** apr-cookbook `origin/main` = `03a84e4c26a7631581f54928075df12720c3887c` (last main commit 2026-05-23); aprender `origin/main` = `542c213ef62012ffcfb06f00287770f64ac209c6` (fetched 2026-07-05 23:30, HEAD committed 2026-07-05 12:19 UTC)
**Method:** 8 parallel audit agents (CI gates, test-contract reality, docs-claims, Makefile gates, book drift, upstream lag, id/publish status, per-category surface map), every load-bearing claim adversarially re-verified by independent skeptic agents (32 agents total, 0 errors). Two auditor claims were REFUTED and corrected during verification (merge-gate ruleset; release-pipeline failure mechanism) — the corrected versions appear below.

**Operating assumption (§0):** live access WAS available — the repo checked out locally (all claims pinned to `origin/main`), a fetched `../aprender` checkout, the GitHub API via `gh`, and crates.io. Everything below is artifact-backed; the only session-supplied evidence is `~/Desktop/top50.md`, marked UNVERIFIED-provenance throughout.

---

## 7a. Verification ledger

| claim | snapshot value | HEAD value | source | verdict |
|---|---|---|---|---|
| Standalone vs consolidated | Project guide: consolidated into APR-MONO | Standalone external consumer: aprender workspace `members` (aprender `Cargo.toml` at `542c213ef`) has no cookbook crate; its line 504 comment names apr-cookbook as downstream; cookbook consumes crates.io pins with `[patch.crates-io]` commented out (`Cargo.toml:144-148`) | aprender `Cargo.toml` members block + `:504`; cookbook `Cargo.toml:72,144` | **VERIFIED** (guide is STALE) |
| aprender pin vs crates.io vs HEAD | pin 0.31.2 | Pin 0.31.2 across all **9** aprender-family deps (`Cargo.toml:72-74,121,124,131-136`) vs crates.io **0.51.0** (2026-06-21) vs aprender HEAD **0.59.0** (`Cargo.toml:118` @542c213ef). Lag = **20 minors published / 28 vs HEAD**. Installed `apr` CLI = 0.49.1 | crates.io API; `git -C ../aprender show origin/main:Cargo.toml`; `apr --version` | **VERIFIED** |
| Pin-lag watchdog exists | — | **None.** `dependabot.yml` covers github-actions only; `Makefile:444-453` update targets manual, uncalled by any workflow; 20-minor lag accumulated over 6-week dormancy (last main commit 2026-05-23) | `.github/dependabot.yml`; `Makefile:444-453`; `git log origin/main --since=2026-06-21` (empty) | **VERIFIED** (absent) |
| Recipe count | 420 (v6.1.0 README) | Registry = **1825** `[[example]]` blocks; all 1825 paths exist on disk. README holds **four contradictory totals**: 420@L24, 341@L60, 496@L139 (its own table above sums to 501), 1825@L151 (auto-generated, matches registry) | `Cargo.toml`; `README.md:24,60,139,151`; `scripts/generate-recipe-table.sh` | **STALE** (registry-truth = 1825) |
| Category count | 34 | **37** dirs in `examples/`, **36** with registered examples (`examples/machines/` has 0 `.rs` — a Jetson canary harness) | `ls -d examples/*/`; `find examples/machines` | **STALE** |
| Crate 0.1.0 vs product v6.4.0 | disagree | Confirmed contradiction: v6.4.0 exists **only** in `README.md:12` prose; git tags stop at v6.1.0; CHANGELOG's latest release is `[0.1.0] 2024-12-02` and its Unreleased section still claims "121 recipes / aprender 0.25" | `Cargo.toml:3`; `git ls-remote --tags`; `CHANGELOG.md:11-12,145` | **VERIFIED** (contradiction real) |
| "4 tests per recipe" | claimed each recipe | True **only for finetune**: 155/155 finetune files carry the contract (1 name variant: `chatml_falsifier_holds`); **0/1670** other examples have it. 17,345 ad-hoc `#[test]`s exist across 1829/1983 files | `pmat query --literal 'fn recipe_runs' --files-with-matches`; `examples/finetune/t3_instruction_chat_template.rs:64` | **STALE** as global claim |
| 4-test contract CI-enforced? | "100% pass as standalone binaries" | **NO.** Required job runs `cargo test --lib --tests` ("NOT examples", `ci.yml:66-71`); weekly cron only `cargo build --examples --release` (`examples-build.yml:49`) — `#[cfg(test)]` bodies are **never even compiled** by CI. Zero `cargo test --example`/`--examples`/`cargo run --example` in any of the 10 workflows. Contract YAMLs name test commands (`finetune-gen.sh:200-210`) CI never runs | `ci.yml:64-71`; `examples-build.yml:49`; workflow grep | **VERIFIED** (negative) |
| "100% pass rate" | claimed v6.3.0 | Sampled live: `cargo test --example t4_online_dpo` and `t4_mpo` = 8/8 pass (~1.3s/example warm). Full 1825-sweep not run anywhere | live run output; `README.md:16` | **UNVERIFIED at scale** |
| `falsifier_breaks` breaks | "negative control" (`finetune/README.md:88`) | ≥2 of 155 are **self-documented vacuous tautologies** (`t4_online_dpo.rs:35-38` "0 steps → vacuously true", `t4_orpo_mistral.rs:49-52`) — mutation-verified: `let hash = 0;` at `src/finetune/online_alt.rs:24` cannot turn them red. Distribution: 35 negation, 75 inverted-threshold, 26 degenerate-eq, 19 plain-positive | mutation reasoning in scratch worktree; classification of all 155 bodies | **STALE** (discipline claim) |
| `deterministic` scope | — | All 155 = same function called twice **in-process**, one hardcoded seed, `assert_eq` on return — no subprocess, no byte-artifact, no second seed, no platform/WASM matrix (repo ships wasm recipes) | `t4_online_dpo.rs:40-45` et al. | **VERIFIED** (single-point) |
| Required checks = `ci / gate` + `workspace-test` | stack convention | Classic protection requires only **"CI Status"**; org ruleset 13878864 "Green Main — unified gate enforcement" (active, updated 2026-07-05 17:35 +02:00) additionally requires contexts `gate`/`kani`/`lake-build`/`workspace-test` — **no workflow in this repo ever reports those names** (jobs report "Quality Gate (stable)", "Lean Gate", "Kani Gate"). Effect: every PR is `mergeStateStatus=BLOCKED` even when fully green (verified on PR #431) → **every merge requires OrganizationAdmin bypass, so all gates are bypassed at merge time** | `gh api .../branches/main/protection`; `gh api .../rules/branches/main`; PR #431 GraphQL | **STALE** convention / **VERIFIED** breakage |
| 95% coverage CI-enforced | CLAUDE.md target | **No.** `coverage.yml` = weekly cron, lib+tests only (examples excluded), **no threshold flag**, codecov `fail_ci_if_error:false`. Threshold configs conflict three ways: `.pmat-gates.toml` 80/60/20 vs `.pmat/tdg-rules.toml` 95/80/10 vs CLAUDE.md 95 | `coverage.yml:9-58`; `.pmat-gates.toml`; `.pmat/tdg-rules.toml` | **STALE** |
| Book renders live source? | drift risk flagged | **Hybrid + already broken**: 75 chapters transclude via `{{#include}}`; ~328 hand-written rust blocks across 170/411 files; `whisper-transcribe.md` documents a program that **never existed** in the repo; **5 includes broken since 0a16a90 (2026-04-06)** ship raw `{{#include}}` text to the public Pages site (mdBook 0.4.36 logs 5 ERRORs, exits 0); `book.yml` triggers on `book/**` only so example renames never rebuild; 4 categories = 405 recipes (finetune 155, tui 177, lint 57, mcp 16) have zero book presence | include-resolution audit; `book.yml:10-21,42`; drift diffs | **VERIFIED** (fired defect) |
| architecture-demos gate enforced | "manifest-driven CI gate" | Static file-existence + count-string check, **manifest→disk direction only**; recipe/registration checks apply only to `in-progress` families (currently 0 → vacuous); `manifest.schema.yaml` is a dead variable in both gen scripts; 0 network calls. **Drift fired**: aprender added 9 descriptors 2026-05-13..16 closing tickets aprender#1586-1594 — `gpt_bigcode` absent from manifest, 8 families (bloom, falcon, granite, internlm2_5, nemotron, olmo, stablelm, starcoder2) still falsely "blocked" | `architecture-demos-gen.sh:327-475`; `manifest.yaml` (last_updated 2026-05-07); aprender `contracts/model-families/` git log | **VERIFIED** (fired) |
| Publishes to crates.io / clean-room | v0.1.0, not in clean-room list | Real pipeline exists (`release.yml`: tag v* → paiml/infra clean-room-gate → verify → OIDC `cargo publish`), no `publish=false` — but it has **never succeeded**: both runs (v6.0.0/v6.1.0) died at **startup with 0 jobs** because `clean-room-gate.yml` does not exist at the pinned infra SHA `ba2f56a7` (only `clean-room.yml` does). Tag-v6.x-vs-0.1.0 is a second, **latent** blocker behind it. Crate absent from crates.io; zero GitHub releases | `release.yml:16-36,81-84,110-117`; `gh api .../runs/<id>/jobs` (total_count=0); infra contents at pinned SHA | **VERIFIED** |
| MSRV agreement | 1.89 | aprender HEAD `rust-version = "1.89"` — equal; no toolchain blocker for a bump | both `Cargo.toml`s | **VERIFIED** |
| Registry integrity | — | 0 registered-but-absent; 0 orphaned recipes; the 158-file gap (1983 vs 1825) is fully accounted: 101 `types.rs` + 39 `helpers.rs` + 13 `tests.rs` + 5 `proptests.rs`, all siblings of the 106 dir-based recipes | comm-based set diff; fn-main scan | **VERIFIED** (invariant true, unenforced) |
| Ticket-id convention | check `.pmat/` | Two PMAT series: `roadmap.yaml` ids APR-001..027 + PMAT-028..145 (+5 free-text ids) = the pmat-work ledger namespace, **stale**; commit refs run a separate higher series to **PMAT-690**; 7 roadmap items "inprogress" though merged; hook enforces `Refs (APR-\d+\|PMAT-\d+\|#\d+\|…)` | `docs/roadmaps/roadmap.yaml` @origin/main; `.githooks/commit-msg:21`; commit log | **VERIFIED** (drifted) |
| top50 GGUF validation | (user-supplied this session) | 43/50 top HF GGUF models ❌, 7 ⚠️ no-file; generated 2026-07-05 23:24 local; **no committed harness** in cookbook or aprender emits this table | `~/Desktop/top50.md`; provenance grep both repos | **UNVERIFIED** (provenance) |

**Stable doctrine vs drifting inventory:** the 4-test contract definition (`docs/specifications/fine-tuning-cookbook/recipe-template.md:95-98`), contract-YAML validation in-process (`tests/contracts.rs`, 190 contracts, merge-path), and the generated-table mechanism are **stable doctrine**. Recipe counts, the pin, the manifest's family statuses, README prose, and `roadmap.yaml` are **drifting inventory** — every number above dated accordingly.

---

## 7b. EV-ranked backlog

Id convention note (verified, not guessed): new ids continue the **commit-ref PMAT series** from above its observed max (PMAT-690), because `roadmap.yaml`'s own namespace (max PMAT-145) is a stale mirror — reconciling that is itself item PMAT-713.

```yaml
- id: PMAT-700
  surface: infra
  type: gate
  ev_rank: 1
  ev_rationale: Cost ~0, unblocks EVERYTHING — org ruleset 13878864 demands contexts (gate/kani/lake-build/workspace-test) no workflow reports, so every PR is BLOCKED-when-green and every merge is an admin bypass; until fixed, all other gates in this list are theater at the merge boundary (CF-4 at policy level).
  definition_of_done: ci.yml job/check names emit the four ruleset contexts (or the org ruleset is corrected); PR #431 (green, currently BLOCKED) becomes mergeable WITHOUT --admin. RED-mutation — a PR with a failing Quality Gate must remain BLOCKED for a non-admin.
  blocked_by: possibly org-admin access to ruleset 13878864 (paiml org; see 7d-6)
  artifact_on_completion: PR (ci.yml rename) + before/after `gh pr view 431 --json mergeStateStatus` receipt
  workflow_note: ticket → branch fix/green-main-contexts → PR → merge must be the LAST admin-bypass merge

- id: PMAT-701
  surface: conformance
  type: gate
  ev_rank: 2
  ev_rationale: The repo's central claim (620 finetune tests, "100% pass") has zero CI witness — test bodies are never even compiled by CI; a touched-files-scoped gate sidesteps the documented disk blow-up (ci.yml:67-69, aprender#1599) while making the contract falsifiable per-PR.
  definition_of_done: (a) per-PR job in the CI-Status needs-list runs `cargo test --example <e>` for every example whose file the PR touches; (b) weekly lane runs the full `cargo test --examples` sharded, filing an issue on failure. RED-mutation — change an assert in examples/finetune/t1_eval_accuracy.rs falsifier_holds to assert!(false) in a PR touching it → CI Status RED; revert → GREEN.
  blocked_by: PMAT-700 (else the gate is admin-bypassed)
  artifact_on_completion: falsifier (the gate) + one intentionally-red draft-PR run as receipt
  workflow_note: ticket → branch → PR → ci / gate (CI Status)

- id: PMAT-702
  surface: conformance
  type: contract
  ev_rank: 3
  ev_rationale: Gates-or-theater (§4.2) — ≥2 of 155 falsifier_breaks are mutation-proof tautologies, so the falsification discipline the cookbook SELLS is decorative exactly where it claims discriminating power.
  definition_of_done: t4_online_dpo.rs:35-38 and t4_orpo_mistral.rs:49-52 rewritten to assert a real violation under the perturbation they name; classify the other 17 plain-positive falsifiers and fix or justify each. RED-mutation — `let hash = 0;` at src/finetune/online_alt.rs:24 must turn t4_online_dpo falsifier_breaks_on_perturbed_input RED (today it stays green).
  blocked_by: none (lands before or with PMAT-701's weekly lane)
  artifact_on_completion: falsifier + mutation receipt in the PR description
  workflow_note: ticket → branch → PR → ci / gate; append case-file per §4.7 (escaped-defect, vacuous negative controls shipped in v6.3.0)

- id: PMAT-703
  surface: pin-freshness
  type: freshness
  ev_rank: 4
  ev_rationale: The defining threat has NO watch-signal — 20 minors of lag accumulated silently in 6 weeks of dormancy; a lag gate is cheap and converts drift from invisible to RED.
  definition_of_done: scheduled workflow compares each Cargo.toml aprender-family pin to crates.io latest (curl with UA per crates.io policy) and fails + opens/refreshes a bump PR when minor-lag > 2. RED-mutation — with crates.io at 0.51.0 and the pin at 0.31.2, the first dispatch run must fail (it will, today).
  blocked_by: none
  artifact_on_completion: falsifier (workflow) + its first genuinely-red run
  workflow_note: ticket → branch → PR → ci / gate

- id: PMAT-704
  surface: pin-freshness
  type: infra
  ev_rank: 5
  ev_rationale: 1825 recipes teaching a 20-minor-stale API is active mis-teaching; highest-cost item here, which is exactly why the conformance gate must exist first to price the breakage honestly.
  definition_of_done: all 9 pins read 0.51.x; `cargo test --lib --tests --all-features` green; PMAT-701's full example-test lane green (or every breakage five-whys'd to a fixed recipe or a filed aprender ticket — no --skip, §4.4); Cargo.lock carries ONE version each of aprender-core/-compute (0.29.x duplicates gone, shrinking the ci disk pressure that justified excluding examples).
  blocked_by: PMAT-701, PMAT-703
  artifact_on_completion: PR + breakage-triage table (recipe → root cause → fix/ticket)
  workflow_note: ticket → branch → PR → ci / gate; expect multi-PR; book/API prose updates ride PMAT-712

- id: PMAT-705
  surface: docs-contract
  type: gate
  ev_rank: 6
  ev_rationale: Four contradictory totals live in the one region the existing recipe-table gate structurally cannot see (all outside the markers); the truth-computing script already exists — extend it, don't write prose.
  definition_of_done: generate-recipe-table.sh stamps headline total, category count, and the per-category table inside new marker pairs; recipe-table check runs per-PR AND is required (via PMAT-700's contexts); `make docs-validate` (pmat validate-readme --fail-on-contradiction, Makefile:147-172 — currently local-only) wired into ci.yml. RED-mutation — edit any prose count (e.g. '341'→'340') → check exits 1.
  blocked_by: PMAT-700
  artifact_on_completion: falsifier + README with zero hand-written counts
  workflow_note: ticket → branch → PR → ci / gate

- id: PMAT-706
  surface: book
  type: gate
  ev_rank: 7
  ev_rationale: Escaped defect LIVE on the public teaching surface since 2026-04-23 (5 chapters render raw {{#include}} text) because mdBook 0.4.36 exits 0 on missing includes and book.yml never sees examples/** changes — fix + permanent falsifier per §4.7.
  definition_of_done: 5 includes repointed to <name>/main.rs; book.yml paths gain examples/** + Cargo.toml; build step fails on `grep -rq '{{#include' book/book/` or mdbook ERROR lines; post-mortem case-file appended. RED-mutation — rename an included example on a scratch branch → book job RED.
  blocked_by: none
  artifact_on_completion: falsifier + post-mortem + restored Pages pages
  workflow_note: ticket → branch → PR → ci / gate → Pages deploy on merge

- id: PMAT-707
  surface: spec-pin
  type: gate
  ev_rank: 8
  ev_rationale: The one manifest gate the README brags about checks one direction, skips certified families entirely (in-progress set empty → vacuous), and has already silently diverged — 9 upstream descriptors, 8 families falsely 'blocked', gpt_bigcode invisible.
  definition_of_done: --check asserts certified families' recipe .rs + [[example]] registration + contract exist AND rejects unmanifested inference_*_smoke.rs; manifest.schema.yaml actually applied (dead SCHEMA var, both gen scripts); new CI step diffs aprender contracts/model-families vs manifest names; manifest adds gpt_bigcode and re-statuses the 8 unblocked families. RED-mutations — (1) `git rm examples/inference/inference_bert_smoke.rs` → exit 1; (2) `status: certifed` typo → exit 1; (3) delete gpt_bigcode entry post-sync → exit 1.
  blocked_by: none (family recipe IMPLEMENTATION for the 9 is follow-on work, gated by this)
  artifact_on_completion: falsifier + reconciled manifest
  workflow_note: ticket → branch → PR → ci / gate

- id: PMAT-708
  surface: spec-pin
  type: contract
  ev_rank: 9
  ev_rationale: SPEC-HF-PUBLISH-001 is referenced only via a mutable main-branch URL — a silent-drift channel that is merely lucky (spec unchanged since its single commit 52c52c95b).
  definition_of_done: README link pins blob SHA + 'Version 1.0.0'; cron step fetches the spec header and fails if Document Version ≠ pinned. RED-mutation — set expected version to 9.9.9 → check RED.
  blocked_by: none
  artifact_on_completion: falsifier
  workflow_note: ticket → branch → PR → ci / gate

- id: PMAT-709
  surface: docs-contract
  type: gate
  ev_rank: 10
  ev_rationale: The registry↔disk invariant is TRUE today (0 ghosts, 0 orphans) and nothing keeps it true — a dropped-in recipe silently becomes unrunnable via cargo run --example.
  definition_of_done: CI step asserts (a) every [[example]] path exists, (b) every examples/**/*.rs is registered OR is a {types,helpers,tests,proptests}.rs sibling of a registered dir-recipe, (c) no empty category dir. RED-mutations — `touch examples/simd/orphan.rs` with fn main → RED; delete one registered file → RED.
  blocked_by: PMAT-700
  artifact_on_completion: falsifier
  workflow_note: ticket → branch → PR → ci / gate

- id: PMAT-710
  surface: infra
  type: contract
  ev_rank: 11
  ev_rationale: Clean-room named FIRST per §4.5 — the publish pipeline is startup-broken (clean-room-gate.yml absent at pinned infra SHA ba2f56a7) with a latent tag/version blocker behind it; today "unpublished" is an accident, not a decision.
  definition_of_done: EITHER `publish = false` in Cargo.toml + publish job removed (decision recorded) OR release.yml points at an existing paiml/infra reusable workflow @SHA AND Cargo.toml version aligned with the next tag AND CHANGELOG backfilled with dated 6.x entries; a v* tag then runs gate→verify→publish green end-to-end. RED-check retained — tag/version mismatch must still fail verify (release.yml:81-84 already does; keep it).
  blocked_by: paiml/infra fix (external; see 7d-5); PMAT-713 for CHANGELOG backfill
  artifact_on_completion: PR + a green (or intentionally-absent) Release run
  workflow_note: ticket → branch → PR → ci / gate → clean-room gate → publish

- id: PMAT-711
  surface: conformance
  type: contract
  ev_rank: 12
  ev_rationale: 'deterministic' currently asserts call-twice-in-process purity — near-tautological — while the README promises bit-identical output and the repo ships WASM recipes it never cross-checks.
  definition_of_done: deterministic tests compare serialized outputs across ≥2 seeds and, for artifact-writing recipes, across two subprocess invocations; OR finetune/README.md:89 re-worded to the honest scope. RED-mutation — inject HashMap-iteration-order dependence into one recipe's sim → its deterministic test RED.
  blocked_by: PMAT-701 (tests must be CI-executed for strengthening to matter)
  artifact_on_completion: falsifier
  workflow_note: ticket → branch → PR → ci / gate

- id: PMAT-712
  surface: book
  type: fixture
  ev_rank: 13
  ev_rationale: Hand-copied chapters are provably unreliable (whisper chapter documents a program that NEVER existed; distributed chapter uses removed APIs) and 405 recipes across 4 categories have zero book presence — transclude or explicitly scope, never copy.
  definition_of_done: the 15 hand-copied recipe chapters transclude their sources; 7 orphaned flat book pages deleted; stale facts fixed (Rust 1.75→1.89, nonexistent 'apr-cookbook = 0.1' crates.io install instructions, 5 hardcoded v0.31.2 strings centralized); SUMMARY either gains finetune/tui/lint/mcp sections or introduction.md documents the exclusion and the PMAT-706 gate whitelists them.
  blocked_by: PMAT-706; PMAT-704 (write against the bumped API, not 0.31.2)
  artifact_on_completion: PR (book renders 100% of recipe code from live source)
  workflow_note: ticket → branch → PR → ci / gate → Pages deploy

- id: PMAT-713
  surface: infra
  type: infra
  ev_rank: 14
  ev_rationale: Truth-reconciliation bundle — every tracker an auditor reads currently lies; roadmap.yaml (max PMAT-145 vs commit series 690, 7 merged-but-'inprogress' items, 5 free-text ids), three-way threshold conflict (80/60/20 vs 95/80/10 vs 95), CLAUDE.md (16-of-36 category table, false 'path deps' claim), CHANGELOG (2024-era), vestigial always-green targets (SUB_PROJECTS loops, centralize-verify.sh), machines/ dir ambiguity, 4 unattended dependabot PRs (#428-431, all ruleset-BLOCKED).
  definition_of_done: roadmap.yaml has zero inprogress items whose ticket appears in a merged commit subject, all ids match ^(APR|PMAT)-\d+$, and its numbering is reconciled with the live series; exactly ONE threshold config survives and matches CLAUDE.md; CLAUDE.md dependency paragraph matches Cargo.toml:72; vestigial targets removed; machines/ moved out of examples/ or given a registered example; dependabot PRs merged/closed.
  blocked_by: PMAT-700 (to merge the PRs without bypass)
  artifact_on_completion: PR
  workflow_note: ticket → branch → PR → ci / gate

- id: PMAT-714
  surface: conformance
  type: freshness
  ev_rank: 15
  ev_rationale: User-supplied top50.md (2026-07-05) shows 43/50 top HF GGUF models failing validation — recipes passing self-tests while the real-model surface fails is the exact CF-4 gap; but the harness itself is uncommitted and unverified, so provenance comes first.
  definition_of_done: the top-50 harness committed (script + model list + apr version stamp); scheduled lane pulls N real GGUF models through apr import/validate via the cookbook's conversion recipes and publishes a pass-table; failures filed as aprender tickets with model+tensor specifics. RED-mutation — point the harness at a known-good fixture model and corrupt one tensor header → lane RED.
  blocked_by: 7d-1 (provenance), PMAT-704 (meaningless against a 20-minor-stale pin), likely aprender-side fixes
  artifact_on_completion: fixture + benchmark table (the committed successor of top50.md)
  workflow_note: ticket → branch → PR → ci / gate; failures cross-filed to paiml/aprender
```

---

## 7c. Do-not-do list

1. **No new recipes, categories, or finetune tiers** (Tier-5 RL, more sister-crate categories, family recipes beyond PMAT-707's sync) until PMAT-700..705 land — §4.1: 1825 ungated recipes already rot against a 20-minor-stale pin; recipe N+1 has ~zero marginal EV.
2. **No naive `cargo test --examples` in the required per-PR path** — ci.yml:67-69 documents the disk blow-up (aprender#1599 duplicate-dep multiplier); scoped-touched + weekly-full is the shape.
3. **No more compile-only example lanes** — compiles ≠ runs ≠ correct (§4.2); another `cargo build --examples` variant adds green without evidence.
4. **No new recipes written against the 0.31.2 API** — each one is pre-paid migration debt for PMAT-704 and teaches deprecated code.
5. **No recipe-count competition with Ludwig/Unsloth/TRL/LLaMA-Factory/Axolotl** — README.md:16 names them only as curriculum mirrors (competitive framing is *not from repo doctrine*); at 1825 the count is already won and meaningless — executed conformance is the differentiator.
6. **No hand-edited counts anywhere in README** — only generate-recipe-table.sh may write numbers (its own header says so); PMAT-705 extends its territory instead.
7. **No new hand-copied book code blocks** — 2 of 3 sampled are wrong, one documents a never-existed program; transclusion or nothing.
8. **No mdBook 0.5.x bump bundled with anything** — v0.5.2 hard-fails this book for an unrelated reason (book.toml:17 `fa-github`); do it alone, after PMAT-706's gate exists.
9. **No re-enabling unified-gate-advisory auto-trigger** until paiml/infra runners get sccache+pmat — disabled for cause (f56842c, 2026-05-11).
10. **No backfilling v6.2–v6.4 git tags before PMAT-710** — every `v*` tag fires the startup-broken release pipeline and mints another failure artifact.
11. **No importing aprender's pillar/beat/threat-register frames or T-numbers** — verified absent from this repo's doctrine (§2); the frame above is derived from the cookbook's own correctness surface.

---

## 7d. UNVERIFIED / needs-live-access appendix

1. **top50.md provenance** — artifact needed: the exact command/harness (and `apr` version — locally 0.49.1) that generated `~/Desktop/top50.md` at 2026-07-05 23:24; no committed script in apr-cookbook or aprender emits that table. Blocks PMAT-714.
2. **"100% pass rate" at full scale** — artifact: one complete `cargo test --examples` sweep log (sampled 2/1825 = 8/8 pass; ~1.3 s/example warm ⇒ ≈40 min serial). PMAT-701's weekly lane produces this permanently.
3. **README:142 "330/341 pass under 10s" runtime baseline** — denominator provably stale; artifact: a re-run of that demo harness at 1825.
4. **Live Pages HTML for the 5 broken chapters** — breakage verified in a reproduced local 0.4.36 build + deploy history, not by fetching the live site; artifact: `curl https://paiml.github.io/apr-cookbook/recipes/c-training/autograd-custom-ops.html | grep '{{#include'`.
5. **paiml/infra clean-room workflow at infra HEAD** — `clean-room-gate.yml` is absent at the pinned SHA; whether it exists (renamed?) at HEAD of the private infra repo is unresolvable from here. Artifact: `gh api repos/paiml/infra/contents/.github/workflows` at HEAD. Blocks PMAT-710.
6. **Green-Main ruleset intent** — ruleset 13878864 was updated 2026-07-05 17:35 +02:00; whether `gate`/`kani`/`lake-build`/`workspace-test` are the org-wide job names repos must adopt, or the ruleset is misconfigured for this repo, needs the org admin. Determines PMAT-700's direction (rename jobs vs fix ruleset).
7. **Cross-platform / WASM determinism** — no artifact exists anywhere in the repo; PMAT-711's matrix lane would create the first one.
8. **aprender 0.51.0-vs-0.59.0 publish gap** — why 8 minors at HEAD are unpublished (release cadence? blocked?) affects whether PMAT-704 targets 0.51.0 or waits; artifact: aprender tags/CHANGELOG or its release workflow runs.

---

## Summary

The cookbook's inventory is real (1825 registered recipes, all on disk, zero ghosts) but its enforcement is almost entirely theater — no CI executes or even compiles a single recipe test, the only truthful README number is the one a script writes, the flagship falsification discipline contains mutation-proof falsifiers, the public book has shipped broken pages for three months, and the merge gate itself is unsatisfiable so every merge bypasses all checks. Meanwhile the pin lag (0.31.2 → 0.51.0 published → 0.59.0 HEAD) is the compounding threat with no watch-signal. The backlog therefore spends its first six items making existing claims falsifiable and enforced, and only then pays the migration (PMAT-704) — after which, and only after which, breadth work is EV-positive again.
