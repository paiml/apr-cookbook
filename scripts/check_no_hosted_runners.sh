#!/usr/bin/env bash
# check_no_hosted_runners.sh -- no workflow job may run on a GitHub-hosted runner (#441).
#
# Fleet rule (operator, 2026-09-10): no hosted GitHub runners. A hosted runner also holds no
# models and no GPU, so it can never run a model-taking recipe.
#
# The jobs not yet migrated are listed in .github/hosted-runner-baseline.txt as
# "<workflow file> <job id>". That list is a RATCHET:
#   - a hosted job NOT in the baseline -> FAIL (a new one was added)
#   - a baseline row whose job is no longer hosted -> FAIL (stale: delete the row, so the
#     list only ever shrinks and cannot hide a future regression under an old name)
# The goal is an empty baseline.
#
# "Hosted" means a runs-on label naming a GitHub-hosted image: ubuntu-*, windows-*, macos-*.
#
# BLIND SPOT, by construction: a job that is a `uses:` call to another repo's reusable
# workflow has no local runs-on, so its runner is decided in that repo. The FOUR this repo
# calls, checked by hand on 2026-09-23/24:
#   ci.yml         paiml/.github sovereign-ci.yml@69482db7   runs_on input default self-hosted clean-room
#   pr-gate.yml    paiml/.github pr-gate.yml@main            [self-hosted, Linux, X64, clean-room, intel]
#                  (branch-pinned: it can move under this note -- pin it by sha)
#   unified-gate-advisory.yml  paiml/.github unified-gate.yml@b6c24635   self-hosted clean-room / gpu
#   release.yml    paiml/infra clean-room-gate.yml@ba2f56a7   not readable from this repo's token
# Re-check each when its pin moves.
#
# Usage: scripts/check_no_hosted_runners.sh [--self-test]
# Exit: 0 clean | 1 violation | 2 usage/ENV
set -euo pipefail

BASELINE=${HOSTED_BASELINE:-.github/hosted-runner-baseline.txt}
WF_DIR=${WF_DIR:-.github/workflows}

hosted_jobs() { # prints "<file> <job>" for every job whose runs-on names a hosted image
    python3 - "$WF_DIR" << 'PY'
import pathlib, re, sys, yaml
HOSTED = re.compile(r"^(ubuntu|windows|macos)-")
EXPR = re.compile(r"\$\{\{\s*matrix\.([A-Za-z0-9_-]+)\s*\}\}")
for f in sorted(pathlib.Path(sys.argv[1]).glob("*.y*ml")):
    d = yaml.safe_load(f.read_text()) or {}
    for jid, j in (d.get("jobs") or {}).items():
        ro = j.get("runs-on")
        shape_ok = True
        if isinstance(ro, dict):
            # Mapping form {group: ..., labels: ...} is how GitHub-HOSTED larger runners are
            # selected; read both keys, and refuse any other key as unverifiable.
            shape_ok = set(ro) <= {"group", "labels"}
            lab = ro.get("labels", [])
            labels = (lab if isinstance(lab, list) else [lab]) + ([ro["group"]] if isinstance(ro.get("group"), str) else [])
        else:
            labels = ro if isinstance(ro, list) else [ro] if isinstance(ro, str) else []
        matrix = ((j.get("strategy") or {}).get("matrix") or {})
        expanded, unverifiable = [], False
        for l in labels:
            if not isinstance(l, str) or "${{" not in l:
                expanded.append(l)
                continue
            # An expression hides the label. `${{ matrix.<k> }}` is resolved against the job's
            # own matrix values; anything else cannot be checked, so it is refused rather than
            # passed (a hosted image behind an expression would otherwise read as clean).
            m = EXPR.fullmatch(l.strip())
            vals = matrix.get(m.group(1)) if m else None
            if isinstance(vals, list) and all(isinstance(v, (str, list)) for v in vals):
                for v in vals:
                    expanded.extend(v if isinstance(v, list) else [v])
            else:
                unverifiable = True
        if unverifiable or not shape_ok or any(isinstance(l, str) and HOSTED.match(l) for l in expanded):
            print("%s %s" % (f.name, jid))
PY
}

check() {
    local actual base new stale nwf
    command -v python3 > /dev/null || { echo "ENV: python3 missing" >&2; return 2; }
    # Zero workflow files is broken wiring (a wrong WF_DIR, an empty checkout), never "0 hosted
    # jobs": refuse instead of printing PASS (quorum finding, #454).
    nwf=$(find "$WF_DIR" -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) 2> /dev/null | wc -l)
    if [ "$nwf" -eq 0 ]; then
        echo "REFUSED: no workflow files under $WF_DIR -- zero files is broken wiring, not a clean pool"
        return 2
    fi
    actual=$(hosted_jobs | sort)
    base=$( { grep -vE '^\s*(#|$)' "$BASELINE" 2> /dev/null || true; } | sort)
    new=$(comm -23 <(printf '%s\n' "$actual" | sed '/^$/d') <(printf '%s\n' "$base" | sed '/^$/d'))
    stale=$(comm -13 <(printf '%s\n' "$actual" | sed '/^$/d') <(printf '%s\n' "$base" | sed '/^$/d'))
    if [ -n "$new" ]; then
        printf 'FAIL  hosted-runner job(s) not in %s (fleet rule: no hosted runners):\n' "$BASELINE"
        printf '        %s\n' "$new"
    fi
    if [ -n "$stale" ]; then
        printf 'FAIL  stale baseline row(s), those jobs are no longer hosted -- delete them so the list only shrinks:\n'
        printf '        %s\n' "$stale"
    fi
    [ -z "$new" ] && [ -z "$stale" ] || return 1
    printf 'PASS  %s hosted job(s) remain, all in the shrinking baseline\n' "$(printf '%s\n' "$actual" | sed '/^$/d' | wc -l | tr -d ' ')"
}

self_test() {
    local t bad=0 rc
    t=$(mktemp -d)
    case "$t" in /tmp/tmp.*) ;; *) echo "REFUSED: mktemp gave $t" >&2; return 2 ;; esac
    mkdir -p "$t/wf"
    printf 'jobs:\n  a:\n    runs-on: [self-hosted, Linux, clean-room]\n  b:\n    runs-on: ubuntu-latest\n' > "$t/wf/x.yml"
    printf 'x.yml b\n' > "$t/base"
    row() { if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1 (rc $3, wanted $2)"; bad=$((bad + 1)); fi; }
    rc=0; WF_DIR="$t/wf" BASELINE="$t/base" check > /dev/null || rc=$?
    row "a hosted job listed in the baseline -> PASS" 0 "$rc"
    printf '  c:\n    runs-on: ubuntu-24.04\n' >> "$t/wf/x.yml"
    rc=0; WF_DIR="$t/wf" BASELINE="$t/base" check > /dev/null || rc=$?
    row "MUST-RED: a NEW hosted job (ubuntu-24.04) not in the baseline -> FAIL" 1 "$rc"
    printf 'jobs:\n  a:\n    runs-on: [self-hosted, Linux, clean-room]\n' > "$t/wf/x.yml"
    rc=0; WF_DIR="$t/wf" BASELINE="$t/base" check > /dev/null || rc=$?
    row "MUST-RED: a baseline row for a job no longer hosted -> FAIL (ratchet)" 1 "$rc"
    : > "$t/base"
    rc=0; WF_DIR="$t/wf" BASELINE="$t/base" check > /dev/null || rc=$?
    row "no hosted jobs and an empty baseline -> PASS (the goal)" 0 "$rc"
    printf 'jobs:\n  m:\n    runs-on: macos-14\n' > "$t/wf/y.yml"
    rc=0; WF_DIR="$t/wf" BASELINE="$t/base" check > /dev/null || rc=$?
    row "MUST-RED: macos-14 counts as hosted too" 1 "$rc"
    rm -f "$t/wf/y.yml"; : > "$t/base"
    printf 'jobs:\n  mx:\n    strategy:\n      matrix:\n        os: [ubuntu-latest, self-hosted]\n    runs-on: ${{ matrix.os }}\n' > "$t/wf/x.yml"
    rc=0; WF_DIR="$t/wf" BASELINE="$t/base" check > /dev/null || rc=$?
    row "MUST-RED: a hosted image behind \${{ matrix.os }} is resolved and caught" 1 "$rc"
    printf 'jobs:\n  ex:\n    runs-on: ${{ inputs.runner }}\n' > "$t/wf/x.yml"
    rc=0; WF_DIR="$t/wf" BASELINE="$t/base" check > /dev/null || rc=$?
    row "MUST-RED: an unresolvable runs-on expression is refused, not passed" 1 "$rc"
    printf 'jobs:\n  ok:\n    strategy:\n      matrix:\n        r: [[self-hosted, Linux, clean-room]]\n    runs-on: ${{ matrix.r }}\n' > "$t/wf/x.yml"
    rc=0; WF_DIR="$t/wf" BASELINE="$t/base" check > /dev/null || rc=$?
    row "a matrix of self-hosted label lists resolves and passes" 0 "$rc"
    mkdir -p "$t/empty"
    rc=0; WF_DIR="$t/empty" BASELINE="$t/base" check > /dev/null || rc=$?
    row "MUST-RED: zero workflow files is refused (rc 2), never PASS" 2 "$rc"
    printf 'jobs:\n  big:\n    runs-on:\n      group: larger-runners\n      labels: ubuntu-latest\n' > "$t/wf/x.yml"
    rc=0; WF_DIR="$t/wf" BASELINE="$t/base" check > /dev/null || rc=$?
    row "MUST-RED: mapping-form runs-on {group, labels: ubuntu-latest} (hosted larger runners)" 1 "$rc"
    printf 'jobs:\n  odd:\n    runs-on:\n      image: whatever\n' > "$t/wf/x.yml"
    rc=0; WF_DIR="$t/wf" BASELINE="$t/base" check > /dev/null || rc=$?
    row "MUST-RED: a mapping runs-on with an unknown key is refused as unverifiable" 1 "$rc"
    if [ -n "$t" ] && [ "$t" != "/" ] && [ -d "$t" ]; then
        case "$t" in /tmp/tmp.*) rm -rf -- "$t" ;; *) ;; esac
    fi
    echo "check_no_hosted_runners self-test: $([ "$bad" = 0 ] && echo PASS || echo FAIL) ($bad failed)"
    [ "$bad" = 0 ]
}

case "${1:-}" in
    "") check ;;
    --self-test) self_test ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
