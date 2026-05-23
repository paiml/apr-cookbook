# Archive Checklist

This document is the runbook for **PMAT-070**. It is gated on PMAT-065 through PMAT-069 being merged AND a 7-day quiet period elapsing, so that downstream consumers have a chance to surface broken assumptions before the source repos go read-only.

**Do not execute any step in this document until the gate criteria below all return ✅.**

---

## Gate Criteria

Run, in order, and confirm each:

```bash
# 1. All migration tickets merged to apr-cookbook main
gh pr list --repo paiml/apr-cookbook --state merged --search "in:title PMAT-10"
# Expect: PMAT-065, PMAT-066, PMAT-067, PMAT-068, PMAT-069 all present and merged

# 2. Inventory verifier passes in strict mode
cd ~/src/apr-cookbook && bash scripts/centralize-verify.sh --strict
# Expect: exit 0, "all source artifacts accounted for"

# 3. CI green on apr-cookbook main for the merge commit and at least 5 subsequent commits
gh run list --repo paiml/apr-cookbook --branch main --limit 6 --json conclusion --jq '.[].conclusion'
# Expect: 6× "success"

# 4. 7-day quiet period observed since PMAT-069 merge
gh pr view <PMAT-069-PR-NUMBER> --repo paiml/apr-cookbook --json mergedAt --jq .mergedAt
# Expect: mergedAt + 7 days <= today

# 5. No open issues on source repos asking about pending work
for repo in sovereign-ai-cookbook alimentar presentar; do
  gh issue list --repo paiml/$repo --state open --label "blocking-archive"
done
# Expect: empty for each

# 6. Crates.io listings for alimentar and presentar still publish from their own repos
cargo search alimentar | head -1
cargo search presentar | head -1
# Expect: each crate visible (proves we did not accidentally take over publishing)
```

If any criterion fails, **stop**. Do not proceed to execution. Open a follow-up ticket.

---

## REDIRECT.md Template

Each archived repo gets exactly one new top-level file: `REDIRECT.md`. Template:

```markdown
# This repository has moved

The content of `<repo-name>` has been consolidated into the
**APR Cookbook** umbrella project as part of the sovereign-stack
documentation centralization (spec: docs/specifications/centralize-cookbooks).

| Where it used to live | Where it lives now |
|-----------------------|--------------------|
| `<source-path-1>` | https://github.com/paiml/apr-cookbook/tree/main/<dest-path-1> |
| `<source-path-2>` | https://github.com/paiml/apr-cookbook/tree/main/<dest-path-2> |
| ... | ... |

This repository is now archived (read-only). Open issues and pull requests
have been closed. For new contributions, please use:

- **Cookbook examples and book**: https://github.com/paiml/apr-cookbook
- **Crate source code (if applicable)**: still published on crates.io as `<crate-name>` from a separate maintenance branch — see https://crates.io/crates/<crate-name>

Last live commit before archive: `<commit-sha>`
Tag preserving pre-archive state: `pre-archive-2026-05`

For full migration rationale, see:
https://github.com/paiml/apr-cookbook/blob/main/docs/specifications/centralize-cookbooks.md
```

The path table is filled per-repo from [migration-mapping.md](migration-mapping.md). The runbook below produces it mechanically.

---

## Per-Repo Execution

Repeat the block below for each of `sovereign-ai-cookbook`, `alimentar`, `presentar`. Substitute `$REPO` accordingly.

### Step 1 — Tag pre-archive HEAD

```bash
cd ~/src/$REPO
git fetch origin main
git checkout main
git pull --ff-only
git tag -a pre-archive-2026-05 -m "Pre-archive snapshot (centralize-cookbooks PMAT-070)"
git push origin pre-archive-2026-05
```

This tag is the rollback point. It is created BEFORE the redirect commit so the last live state is recoverable.

### Step 2 — Generate REDIRECT.md

```bash
cd ~/src/apr-cookbook
bash scripts/gen-redirect-md.sh --repo $REPO > /tmp/$REPO-REDIRECT.md
```

The script reads [migration-mapping.md](migration-mapping.md), filters to the entries belonging to `$REPO`, and emits a populated REDIRECT.md.

Manually review `/tmp/$REPO-REDIRECT.md` before committing. Confirm:
- Path table covers every migrated entry
- crate-name reference is correct (or absent for sovereign-ai-cookbook which has no crate)
- Last live commit sha matches `git rev-parse pre-archive-2026-05`

### Step 3 — Open redirect PR

```bash
cd ~/src/$REPO
git checkout -b chore/archive-redirect
cp /tmp/$REPO-REDIRECT.md REDIRECT.md
git add REDIRECT.md
git commit -m "archive: redirect to apr-cookbook (centralize-cookbooks PMAT-070)"
git push -u origin chore/archive-redirect
gh pr create \
  --title "archive: redirect to apr-cookbook" \
  --body "Final commit before archive. Spec: https://github.com/paiml/apr-cookbook/blob/main/docs/specifications/centralize-cookbooks.md"
```

Wait for CI to pass on the PR (it should — the only change is one new markdown file). Merge:

```bash
gh pr merge --squash --delete-branch
```

### Step 4 — Set archive bit

```bash
gh api -X PATCH repos/paiml/$REPO -f archived=true
gh repo view paiml/$REPO --json isArchived --jq .isArchived
# Expect: true
```

### Step 5 — Confirm and log

```bash
echo "$(date -Iseconds) archived $REPO at HEAD=$(git -C ~/src/$REPO rev-parse main)" \
  >> ~/src/apr-cookbook/docs/specifications/centralize-cookbooks/archive-log.txt
```

`archive-log.txt` is committed to apr-cookbook as durable evidence of the archive event.

---

## Order of operations

The three repos are archived in this order, with a 24-hour pause between each, so that a problem surfaced after the first archive can be acted on before the next:

1. **sovereign-ai-cookbook** first — lowest external consumer count
2. **24-hour pause** — monitor issues, monitor crates.io if a deploy breaks
3. **presentar** second — moderate external consumer count
4. **24-hour pause**
5. **alimentar** last — highest external consumer count (data-loading is widely depended on)

---

## Post-archive cleanup (apr-cookbook side)

After all three repos are archived:

1. Update apr-cookbook `README.md` "Related Repositories" section. For each archived repo:
   ```markdown
   - ~~[sovereign-ai-cookbook](https://github.com/paiml/sovereign-ai-cookbook)~~ — archived; consolidated into [examples/deployment-stacks](examples/deployment-stacks)
   ```
2. Add a "History" subsection citing the centralize-cookbooks spec.
3. Update memory `MEMORY.md` to record the archive event with date.
4. Push to apr-cookbook main (via PR — main is protected).

---

## Reversal Procedure (emergency only)

If a critical regression is discovered post-archive and rollback is required:

```bash
# 1. Un-archive
gh api -X PATCH repos/paiml/$REPO -f archived=false

# 2. Reset main to pre-archive state (DESTRUCTIVE, requires explicit user approval)
cd ~/src/$REPO
git fetch origin --tags
git checkout main
git reset --hard pre-archive-2026-05
git push --force-with-lease origin main   # only with user approval; main may be protected

# 3. Re-open relevant PRs/issues that were auto-closed by the archive
# (GitHub does not auto-reopen; manual triage required)
```

This procedure is **not** part of normal operation. It is documented here only because the spec promises reversibility — and reversibility that isn't documented is unreliable. Invoking it requires explicit owner approval per CLAUDE.md branch-protection rules.

---

## Done

When all three repos report `isArchived: true`, REDIRECT.md is the only top-level diff vs. their pre-archive HEAD, the apr-cookbook README has been updated, and `archive-log.txt` is committed:

- Mark PMAT-070 complete via `pmat work complete PMAT-070`
- Tag the apr-cookbook commit that includes the README update + archive-log as `centralization-complete-2026-05`
- Notify in the team channel referenced from MEMORY.md

The umbrella cookbook is the canonical source from this point forward.
