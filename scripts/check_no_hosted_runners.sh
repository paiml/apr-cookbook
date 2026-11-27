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
# PARSER (#470): yq v4 (mikefarah, go-yaml v3, pinned v4.52.2) turns each workflow into JSON and
# jq applies the rules. Bash only; no interpreter. A file yq refuses, or one with more than one
# YAML document, is a FAIL: the gate never passes what it cannot read.
# Known differences, all in the refusing direction: duplicate keys are refused (yq), a quoted
# "yes"/"10" in a matrix is UNVERIFIABLE (yq drops the quoting), and a `<<` merge key is
# resolved by go-yaml.
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

# One JSON document per workflow file in, "<file> <job>" lines out, for every job that is hosted,
# unverifiable or not shape-checkable. --arg file names the workflow file.
here=$(dirname "${BASH_SOURCE[0]}")
readonly JQ_HOSTED_RULES="$here/check_no_hosted_runners.jq"

hosted_jobs() { # prints "<file> <job>" for every job whose runs-on names a hosted image
    local f fail=0
    while IFS= read -r f; do
        if ! yq -o json '.' "$f" | jq -r -s --arg file "${f##*/}" -f "$JQ_HOSTED_RULES"; then
            echo "FAIL  ${f##*/}: not readable (yq refused it, more than one YAML document, or a job outside the shape the gate reads)" >&2; fail=1; fi
    done <<< "$WF_FILES"
    if [ "$fail" != 0 ]; then
        echo "FAIL  a workflow could not be parsed (see above): the gate never passes what it cannot parse" >&2
    fi
    return "$fail"
}

# in_list <line> <list>: pure-bash membership over newline-separated lists (no subprocess).
in_list() {
    local nl=$'\n'
    local found=1
    [[ "$nl$2$nl" == *"$nl$1$nl"* ]] && found=0
    return "$found"
}

check() {
    local actual base new="" stale="" line nwf WF_FILES
    if ! command -v yq > /dev/null 2>&1; then
        echo "ENV: yq missing" >&2
        return 2
    fi
    ver=$(yq --version 2> /dev/null)
    if [[ "$ver" != *"version v4."* ]]; then
        echo "ENV: yq v4 (mikefarah) required" >&2
        return 2
    fi
    if ! command -v jq > /dev/null 2>&1; then
        echo "ENV: jq missing" >&2
        return 2
    fi
    # Zero workflow files is broken wiring (a wrong WF_DIR, an empty checkout), never "0 hosted
    # jobs": refuse instead of printing PASS (quorum finding, #454).
    if [ ! -d "$WF_DIR" ]; then
        echo "ENV: workflow dir $WF_DIR does not exist" >&2
        return 2
    fi
    # Symlinks count as workflow files (main's glob read them): a link to a hosted job must FAIL, not vanish.
    WF_FILES=$(find "$WF_DIR" -maxdepth 1 \( -type f -o -type l \) \( -name '*.yml' -o -name '*.yaml' \) | sort)
    nwf=$(printf '%s\n' "$WF_FILES" | sed '/^$/d' | wc -l)
    if [ "$nwf" -eq 0 ]; then
        echo "REFUSED: no workflow files under $WF_DIR -- zero files is broken wiring, not a clean pool"
        return 2
    fi
    actual=$(hosted_jobs) || return 1
    if [ -e "$BASELINE" ] && [ ! -r "$BASELINE" ]; then
        echo "ENV: baseline $BASELINE exists but is unreadable" >&2
        return 2
    fi
    base=$(grep -v -E "^ *#" "$BASELINE" 2> /dev/null | grep -v -E "^ *$" | sort) || base=""
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        in_list "$line" "$base" || new="$new        $line"$'\n'
    done <<< "$actual"
    while IFS= read -r line; do
        [ -n "$line" ] || continue
        in_list "$line" "$actual" || stale="$stale        $line"$'\n'
    done <<< "$base"
    if [ -n "$new" ]; then
        printf 'FAIL  hosted-runner job(s) not in %s (fleet rule: no hosted runners):\n' "$BASELINE"
        printf '%s' "$new"
    fi
    if [ -n "$stale" ]; then
        printf 'FAIL  stale baseline row(s), those jobs are no longer hosted -- delete them so the list only shrinks:\n'
        printf '%s' "$stale"
    fi
    if [ -n "$new" ] || [ -n "$stale" ]; then
        return 1
    fi
    local n=0
    while IFS= read -r line; do [ -n "$line" ] && n=$((n + 1)); done <<< "$actual"
    printf 'PASS  %s hosted job(s) remain, all in the shrinking baseline\n' "$n"
}

# gate_rc <wf dir> <baseline>: runs the gate on that tree and prints its exit code.
gate_rc() { # sets RC (the gate exit code for WF_DIR=$1, BASELINE=$2); no subshell
    RC=0; WF_DIR="$1" BASELINE="$2" check > /dev/null 2>&1 || RC=$?
}

self_test() {
    local bad=0 wf base
    # The managed temp root: TMPDIR (the runner's), without a trailing slash; /tmp when unset.
    ST_ROOT="${TMPDIR:-/tmp}"; ST_ROOT="${ST_ROOT%/}"; [ -n "$ST_ROOT" ] || ST_ROOT=/tmp
    IFS= read -r ST_DIR < <(mktemp -d)
    if [[ "$ST_DIR" != "$ST_ROOT"/tmp.* ]]; then
        echo "REFUSED: mktemp gave $ST_DIR" >&2
        return 2
    fi
    trap 'case "${ST_DIR:?}" in "$ST_ROOT"/tmp.*) rm -rf -- "${ST_DIR:?}" ;; esac' EXIT
    wf="$ST_DIR/wf"; base="$ST_DIR/base"
    mkdir -p "$wf"
    printf 'jobs:\n  a:\n    runs-on: [self-hosted, Linux, clean-room]\n  b:\n    runs-on: ubuntu-latest\n' > "$wf/x.yml"
    printf 'x.yml b\n' > "$base"
    row() { gate_rc "$3" "$4"; if [ "$2" = "$RC" ]; then echo "ok    $1"; else echo "FAIL  $1 (rc $RC, wanted $2)"; bad=$((bad + 1)); fi; }
    row "a hosted job listed in the baseline -> PASS" 0 "$wf" "$base"
    printf '  c:\n    runs-on: ubuntu-24.04\n' >> "$wf/x.yml"
    row "MUST-RED: a NEW hosted job (ubuntu-24.04) not in the baseline -> FAIL" 1 "$wf" "$base"
    printf 'jobs:\n  a:\n    runs-on: [self-hosted, Linux, clean-room]\n' > "$wf/x.yml"
    row "MUST-RED: a baseline row for a job no longer hosted -> FAIL (ratchet)" 1 "$wf" "$base"
    : > "$base"
    row "no hosted jobs and an empty baseline -> PASS (the goal)" 0 "$wf" "$base"
    printf 'jobs:\n  m:\n    runs-on: macos-14\n' > "$wf/y.yml"
    row "MUST-RED: macos-14 counts as hosted too" 1 "$wf" "$base"
    : > "$wf/y.yml"
    printf 'jobs:\n  mx:\n    strategy:\n      matrix:\n        os: [ubuntu-latest, self-hosted]\n    runs-on: ${{ matrix.os }}\n' > "$wf/x.yml"
    row "MUST-RED: a hosted image behind \${{ matrix.os }} is resolved and caught" 1 "$wf" "$base"
    printf 'jobs:\n  ex:\n    runs-on: ${{ inputs.runner }}\n' > "$wf/x.yml"
    row "MUST-RED: an unresolvable runs-on expression is refused, not passed" 1 "$wf" "$base"
    printf 'jobs:\n  ok:\n    strategy:\n      matrix:\n        r: [[self-hosted, Linux, clean-room]]\n    runs-on: ${{ matrix.r }}\n' > "$wf/x.yml"
    row "a matrix of self-hosted label lists resolves and passes" 0 "$wf" "$base"
    mkdir -p "$ST_DIR/empty"
    row "MUST-RED: zero workflow files is refused (rc 2), never PASS" 2 "$ST_DIR/empty" "$base"
    row "MUST-RED: a missing workflow dir is refused (rc 2)" 2 "$ST_DIR/no-such-dir" "$base"
    mkdir -p "$ST_DIR/lw"
    printf 'jobs:\n  s:\n    runs-on: ubuntu-latest\n' > "$ST_DIR/lw/hosted.txt"
    (cd "$ST_DIR/lw" && ln -sfn hosted.txt x.yml)
    row "MUST-RED: a symlinked workflow file to a hosted job is read, not skipped" 1 "$ST_DIR/lw" "$base"
    printf 'jobs:\n  big:\n    runs-on:\n      group: larger-runners\n      labels: ubuntu-latest\n' > "$wf/x.yml"
    row "MUST-RED: mapping-form runs-on {group, labels: ubuntu-latest} (hosted larger runners)" 1 "$wf" "$base"
    printf 'jobs:\n  odd:\n    runs-on:\n      image: whatever\n' > "$wf/x.yml"
    row "MUST-RED: a mapping runs-on with an unknown key is refused as unverifiable" 1 "$wf" "$base"
    if [ "$bad" = 0 ]; then res=PASS; else res=FAIL; fi
    echo "check_no_hosted_runners self-test: $res ($bad failed)"
    [ "$bad" = 0 ]
}

case "${1:-}" in
    "") check ;;
    --self-test) self_test ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
