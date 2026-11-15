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
    local f
    # Plain awk reading the block/flow YAML subset the workflows use (no Python, no yq).
    # Anything it cannot read (aliases, anchors, block scalars, flow mappings) is reported
    # as unverifiable, so an unreadable runs-on fails closed instead of passing.
    while IFS= read -r -d "" f; do
        awk -v sq="'" -v fname="${f##*/}" '
        function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
        function quoted(s,   a, z) { a = substr(s, 1, 1); z = substr(s, length(s)); return (length(s) >= 2 && (a == "\"" || a == sq) && z == a) }
        # a double-quoted scalar with a backslash may hold an escape (\x75 = u): unreadable, flag the job
        function unq(s) { s = trim(s); if (substr(s, 1, 1) == "\"" && index(s, "\\")) esc = 1; if (length(s) >= 2 && quoted(s)) s = substr(s, 2, length(s) - 2); return s }
        # fail closed: report file:line and why, once per file
        function unv(why) { if (!ub) print "UNVERIFIABLE " fname ":" FNR ": " why; ub = 1 }
        # parse "key: value" (plain, "dq" or sq key, space before colon ok) into K, V; 0 if not that shape
        function parsekey(t,   a, m, k) {
            a = substr(t, 1, 1)
            if (a == "\"") m = match(t, /^"([^"\\]|\\.)*"[ \t]*:([ \t]|$)/)
            else if (a == sq) m = match(t, /^\047([^\047]|\047\047)*\047[ \t]*:([ \t]|$)/)
            else if (a ~ /[?\[{*&!|>%@`#,-]/) return 0
            else m = match(t, /^[^:]*:([ \t]|$)/)
            K = ""; V = ""; if (!m) return 0
            k = substr(t, 1, RLENGTH); sub(/:[ \t]?$/, "", k); K = unq(k); V = trim(substr(t, RLENGTH + 1))
            return 1
        }
        function nonstr(s) { s = trim(s); return (!quoted(s) && s ~ /^(-?[0-9][0-9._]*|true|false|True|False|TRUE|FALSE|null|Null|NULL|~)$/) }
        function strip_comment(s,   i, c, q, out) {
            q = ""; out = ""
            for (i = 1; i <= length(s); i++) {
                c = substr(s, i, 1)
                if (q == "") {
                    if (c == "\"" || c == sq) q = c
                    else if (c == "#" && (i == 1 || substr(s, i - 1, 1) ~ /[ \t]/)) break
                } else if (c == q) q = ""
                out = out c
            }
            return out
        }
        # split a flow-list body (text between the outer brackets) on top-level commas into R[1..n]
        function flow_split(s, R,   i, c, q, d, n, cur) {
            n = 0; cur = ""; q = ""; d = 0
            for (i = 1; i <= length(s); i++) {
                c = substr(s, i, 1)
                if (q != "") { if (c == q) q = ""; cur = cur c; continue }
                if (c == "\"" || c == sq) { q = c; cur = cur c; continue }
                if (c == "[" || c == "{") d++
                if (c == "]" || c == "}") d--
                if (c == "," && d == 0) { R[++n] = trim(cur); cur = ""; continue }
                cur = cur c
            }
            if (trim(cur) != "") R[++n] = trim(cur)
            return n
        }
        function flow_body(v) { v = trim(v); if (v ~ /^\[.*\]$/) return substr(v, 2, length(v) - 2); return "\001" }
        function unreadable(v) { return (v ~ /^[*&|>!{]/ || (v ~ /^\[/ && v !~ /\]$/)) }
        # add the labels of one runs-on value (scalar or flow list) to the label list L[1..nl]
        function add_labels(v,   b, k, n, R) {
            v = trim(v)
            if (v == "" || v == "~" || v == "null") return
            if (unreadable(v)) { bad = 1; return }
            if (v ~ /^\[/) {
                n = flow_split(flow_body(v), R)
                for (k = 1; k <= n; k++) if (!nonstr(R[k]) && R[k] !~ /^\[/) L[++nl] = unq(R[k])
            } else if (!nonstr(v)) L[++nl] = unq(v)
        }
        function flush_job(   i, k, n, R, R2, j, l, key, m, ok, nmx, vals, x, host) {
            if (jid == "") return
            nl = 0; bad = 0
            if (robuf_n > 0 || rov != "") {
                if (rov != "") add_labels(rov)
                else if (robuf[1] ~ /^-/) {
                    for (i = 1; i <= robuf_n; i++) if (robuf[i] ~ /^-[ \t]/ || robuf[i] == "-") add_labels(substr(robuf[i], 2))
                } else {
                    for (i = 1; i <= robuf_n; i++) {
                        l = robuf[i]
                        if (rind[i] != rmin) continue
                        key = unq(substr(l, 1, index(l, ":") - 1)); x = trim(substr(l, index(l, ":") + 1))
                        if (key != "group" && key != "labels") { bad = 1; continue }
                        if (key == "group") { if (x != "" && x !~ /^[\[{]/ && !nonstr(x)) L[++nl] = unq(x); continue }
                        if (x != "") add_labels(x)
                        else for (j = i + 1; j <= robuf_n && rind[j] > rmin; j++)
                            if (robuf[j] ~ /^-[ \t]/) add_labels(substr(robuf[j], 2))
                    }
                }
            }
            host = 0
            for (i = 1; i <= nl; i++) {
                l = L[i]
                if (index(l, "${{") == 0) { if (l ~ /^(ubuntu|windows|macos)-/) host = 1; continue }
                x = trim(l)
                if (x ~ /^\$\{\{[ \t]*matrix\.[A-Za-z0-9_-]+[ \t]*\}\}$/) {
                    key = x; sub(/^\$\{\{[ \t]*matrix\./, "", key); sub(/[ \t]*\}\}$/, "", key)
                    if (!(key in mcnt) || mbad[key]) { bad = 1; continue }
                    for (j = 1; j <= mcnt[key]; j++) if (mval[key, j] ~ /^(ubuntu|windows|macos)-/) host = 1
                } else bad = 1
            }
            if (bad || host || esc) print fname " " jid
        }
        # register one matrix element (scalar or flow list) of key
        function mat_add(key, e,   n, R, k) {
            e = trim(e)
            if (e ~ /^\[/ && e ~ /\]$/) {
                n = flow_split(flow_body(e), R)
                for (k = 1; k <= n; k++) mval[key, ++mcnt[key]] = unq(R[k])
            } else if (e == "" || e ~ /^[*&|>!{\[]/ || nonstr(e)) mbad[key] = 1
            else mval[key, ++mcnt[key]] = unq(e)
        }
        function mat_set(key, v,   n, R, k) {
            v = trim(v); mcnt[key] = 0; mbad[key] = 0
            if (v == "") { mpend = key; return }
            mpend = ""
            if (v ~ /^\[.*\]$/) { n = flow_split(flow_body(v), R); for (k = 1; k <= n; k++) mat_add(key, R[k]) }
            else mbad[key] = 1
        }
        FNR == 1 { flush_job(); jid = ""; injobs = 0; ji = 0; ub = 0; esc = 0 }
        {
            line = strip_comment($0); sub(/[ \t\r]+$/, "", line)
            if (trim(line) == "") next
            if (line ~ /^ *\t/) { unv("tab in indentation (not valid YAML)"); next }
            match(line, /^ */); ind = RLENGTH; t = substr(line, ind + 1)
            if (ind == 0) {
                flush_job(); jid = ""; injobs = (t ~ /^jobs:[ \t]*$/); ji = 0
                if (t ~ /^(\?|\{|\[|---[ \t]+[^ \t#])/) unv("complex key, flow or inline document at top level")
                else if (!injobs && parsekey(t) && K == "jobs") unv("jobs key not in plain block form (quoted, spaced or flow value)")
                next
            }
            if (!injobs) next
            if (ji == 0) ji = ind
            if (ind == ji) {
                if (!parsekey(t)) { unv("job entry is not a plain or quoted \"id:\" key"); next }
                if (V != "") unv("job value is not a block mapping (flow, alias or anchor)")
                flush_job()
                jid = K; esc = 0
                ki = 0; mode = ""; rov = ""; robuf_n = 0; rmin = 0
                for (k in mcnt) { delete mcnt[k]; delete mbad[k] }
                delete mval; si = 0; mi = 0; mpend = ""; inmat = 0
                next
            }
            if (jid == "") next
            if (ki == 0 && ind > ji) ki = ind
            if (ind == ki && t !~ /^-([ \t]|$)/) {
                mode = ""; inmat = 0
                if (!parsekey(t)) unv("job key is not a plain or quoted key (complex key, anchor or flow)")
                else if (K == "<<") unv("merge key in a job")
                if (K == "runs-on") { mode = "ro"; rov = V; robuf_n = 0; rmin = 0 }
                else if (K == "strategy") { mode = "st"; si = 0 }
                next
            }
            if (mode == "ro" && rov == "") {
                robuf[++robuf_n] = t; rind[robuf_n] = ind
                if (rmin == 0 || ind < rmin) rmin = ind
                next
            }
            if (mode == "st") {
                if (!inmat) {
                    if (si == 0) si = ind
                    if (ind == si && t ~ /^matrix:/) { inmat = 1; mi = 0; mpend = ""; x = trim(substr(t, 8)); if (x != "") inmat = 0 }
                    next
                }
                if (ind <= si) { inmat = 0; next }
                if (mi == 0) mi = ind
                if (t ~ /^-[ \t]/ && mpend != "") { mat_add(mpend, substr(t, 2)); next }
                if (ind == mi && t ~ /^[^-][^:]*:/) mat_set(unq(substr(t, 1, index(t, ":") - 1)), substr(t, index(t, ":") + 1))
            }
        }
        END { flush_job() }
        ' "$f"
    done < <(find "$WF_DIR" -maxdepth 1 -type f -name '*.y*ml' -print0)
}

check() {
    local actual base new stale nwf unv
    # Zero workflow files is broken wiring (a wrong WF_DIR, an empty checkout), never "0 hosted
    # jobs": refuse instead of printing PASS (quorum finding, #454).
    nwf=$(find "$WF_DIR" -maxdepth 1 -type f \( -name '*.yml' -o -name '*.yaml' \) 2> /dev/null | wc -l)
    if [ "$nwf" -eq 0 ]; then
        echo "REFUSED: no workflow files under $WF_DIR -- zero files is broken wiring, not a clean pool"
        return 2
    fi
    actual=$(hosted_jobs | sort)
    unv=$(printf '%s\n' "$actual" | grep '^UNVERIFIABLE ' || true)
    actual=$(printf '%s\n' "$actual" | { grep -v '^UNVERIFIABLE ' || true; })
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
    if [ -n "$unv" ]; then
        printf 'FAIL  workflow file(s) outside the supported YAML subset, cannot be verified (fail closed):\n'
        printf '        %s\n' "$unv"
    fi
    [ -z "$new" ] && [ -z "$stale" ] && [ -z "$unv" ] || return 1
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
    # adversarial forms (review of #484): each valid YAML spelling of a hosted job must be RED,
    # either caught (hosted) or refused as unverifiable; none may pass.
    adv() { # adv <label> <file name> <printf format>
        rm -rf "${t:?}/adv"; mkdir -p "$t/adv"; printf "$3" > "$t/adv/$2"
        rc=0; WF_DIR="$t/adv" BASELINE="$t/base" check > /dev/null || rc=$?
        row "MUST-RED: $1" 1 "$rc"
    }
    J='jobs:\n  a:\n    runs-on: '
    adv 'quoted key "runs-on":' w.yml "${J//runs-on: /\"runs-on\": }ubuntu-latest\n"
    adv "runs-on : (space before colon)" w.yml "${J//runs-on:/runs-on :} macos-14\n"
    adv 'explicit ? runs-on key' w.yml 'jobs:\n  a:\n    ? runs-on\n    : ubuntu-latest\n'
    adv 'quoted "jobs" key' w.yml '"jobs":\n  a:\n    runs-on: ubuntu-latest\n'
    adv 'jobs as a flow mapping' w.yml 'jobs: {a: {runs-on: ubuntu-latest}}\n'
    adv 'a job as a flow mapping' w.yml 'jobs:\n  a: {runs-on: ubuntu-latest}\n'
    adv 'escaped double-quoted label "\\x75buntu-latest"' w.yml "${J}\"\\\\x75buntu-latest\"\n"
    adv 'dotfile workflow .hidden.yml' .hidden.yml "${J}ubuntu-latest\n"
    adv 'tab-indented runs-on' w.yml 'jobs:\n  a:\n\truns-on: ubuntu-latest\n'
    adv 'merge key <<: *base in a job' w.yml 'jobs:\n  a:\n    <<: *base\n    runs-on: self-hosted\n'
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
