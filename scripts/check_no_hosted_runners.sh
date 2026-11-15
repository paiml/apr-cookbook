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
    # Allow-list parser (awk): only a small canonical YAML subset is positively recognised;
    # every line it cannot classify in the regions that matter (top level, job ids, job keys,
    # runs-on, strategy.matrix) is reported UNVERIFIABLE, so unreadable fails closed.
    local f
    while IFS= read -r -d "" f; do
        if [ ! -f "$f" ] || [ ! -r "$f" ]; then
            printf 'UNVERIFIABLE %s:0: not a readable regular file\n' "${f##*/}"; continue
        fi
        if ! iconv -f UTF-8 -t UTF-8 < "$f" > /dev/null 2>&1; then
            printf 'UNVERIFIABLE %s:0: not valid UTF-8\n' "${f##*/}"; continue
        fi
        awk -v sq="'" -v fname="${f##*/}" '
        function trim(s) { sub(/^[ \t]+/, "", s); sub(/[ \t]+$/, "", s); return s }
        # fail closed: report file:line and why, once per file
        function unv(why) { if (!ub) print "UNVERIFIABLE " fname ":" FNR ": " why; ub = 1 }
        # a scalar on one line: plain, "dq" (no escape) or sq (no doubled quote); sets SV
        function scalar_ok(v,   a) {
            v = trim(v); if (v == "") return 0
            a = substr(v, 1, 1)
            if (a == "\"") { if (v ~ /^"[^"\\]*"$/) { SV = substr(v, 2, length(v) - 2); return 1 } return 0 }
            if (a == sq) { if (v ~ /^\047[^\047]*\047$/) { SV = substr(v, 2, length(v) - 2); return 1 } return 0 }
            if (index("[]{}*&!|>%@`#,?:-", a)) return 0
            if (v ~ /: / || v ~ /:$/ || v ~ /["\047]/) return 0
            SV = v; return 1
        }
        # a key line "key:" (plain [A-Za-z0-9_-]+ or quoted likewise, space before colon ok): sets K, V
        function skey(t,   m, k) {
            K = ""; V = ""
            if (substr(t, 1, 1) == "\"") m = match(t, /^"[A-Za-z0-9_-]*"[ \t]*:([ \t]|$)/)
            else if (substr(t, 1, 1) == sq) m = match(t, /^\047[A-Za-z0-9_-]*\047[ \t]*:([ \t]|$)/)
            else m = match(t, /^[A-Za-z0-9_-]+[ \t]*:([ \t]|$)/)
            if (!m) return 0
            k = substr(t, 1, RLENGTH); sub(/[ \t]*:[ \t]?$/, "", k)
            if (k ~ /^["\047]/) k = substr(k, 2, length(k) - 2)
            K = k; V = trim(substr(t, RLENGTH + 1)); return 1
        }
        function nonstr(s) { return (s ~ /^(-?[0-9][0-9._]*|true|false|True|False|TRUE|FALSE|null|Null|NULL|~)$/) }
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
        # split a flow-list body on top-level commas into R[1..n]
        function flow_split(s, R,   i, c, q, n, cur, d) {
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
        # lines outside the classified regions still get a cheap well-formedness check: a quoted
        # scalar or flow collection that starts a value must close on its own line (block-scalar
        # content is opaque); a miss is UNVERIFIABLE, never "not hosted"
        function balanced(v,   i, c, q, d) {
            q = ""; d = 0
            for (i = 1; i <= length(v); i++) {
                c = substr(v, i, 1)
                if (q != "") { if (c == "\\" && q == "\"") i++; else if (c == q) q = ""; continue }
                if (c == "\"" || c == sq) q = c
                else if (c == "[" || c == "{") d++
                else if (c == "]" || c == "}") { if (--d < 0) return 0 }
            }
            return (q == "" && d == 0)
        }
        function chk_value(t, ind,   col, v, a) {
            col = ind
            while (t ~ /^-[ \t]+/) { match(t, /^-[ \t]+/); col += RLENGTH; t = substr(t, RLENGTH + 1) }
            if (match(t, /^[A-Za-z0-9_.\/-]+:[ \t]+/)) { v = trim(substr(t, RLENGTH + 1)) } else v = trim(t)
            if (v ~ /^[|>][-+0-9]*$/) { bsind = (col > ind ? col : ind); bsc = 0; return }
            a = substr(v, 1, 1)
            if (a == "\"" && v !~ /^"([^"\\]|\\.)*"$/) unv("double-quoted scalar does not close on its line")
            else if (a == sq && v !~ /^\047([^\047]|\047\047)*\047$/) unv("single-quoted scalar does not close on its line")
            else if ((a == "[" || a == "{") && !balanced(v)) unv("flow collection does not close on its line")
        }
        # indentation must nest, as block YAML does: a dedent lands on an open level, a deeper line
        # follows a key with an empty value (or continues a plain scalar that is not itself "k: v"),
        # a list item sits only in a list level or directly under its own key; else not valid YAML
        function chk_indent(t, ind,   col, t2, v, isk, dash) {
            col = ind; t2 = t; dash = (t2 ~ /^-([ \t]|$)/)
            while (t2 ~ /^-([ \t]+|$)/) { match(t2, /^-([ \t]+|$)/); col += RLENGTH; t2 = substr(t2, RLENGTH + 1) }
            isk = match(t2, /^[^ \t][^:]*:([ \t]+|$)/)
            v = isk ? trim(substr(t2, RLENGTH + 1)) : t2
            if (sp > 0 && pplain && ind > pind && !isk) return
            if (sp == 0) { stk[sp = 1] = ind; knd[1] = (dash ? "s" : "m"); lk[1] = "" }
            else {
                while (sp > 0 && stk[sp] > ind) sp--
                if (sp == 0 || stk[sp] != ind) {
                    if (!(popen && ind > pind)) unv("indentation does not nest")
                    stk[++sp] = ind; knd[sp] = (dash ? "s" : "m"); lk[sp] = ""
                }
            }
            if (dash) {
                if (knd[sp] == "m" && !(lk[sp] == "d" || (popen && pind == ind))) unv("list item among mapping keys")
                lk[sp] = "d"
            } else {
                if (!isk) unv("line is neither a key nor a list item")
                if (knd[sp] == "s") unv("mapping key at a list level")
                lk[sp] = (v == "" ? "o" : "k")
            }
            if (dash && isk && t2 != "") { stk[++sp] = col; knd[sp] = "m"; lk[sp] = (v == "" ? "o" : "k") }
            pind = (dash && isk ? col : ind); popen = (t2 == "" || (isk && v == ""))
            pplain = (v != "" && v !~ /^[\[{"\047|>]/)
        }
        # one runs-on label: nonstr scalars never name an image; anything not a closed scalar is refused
        function add_label(v) {
            v = trim(v)
            if (nonstr(v)) return 1
            if (!scalar_ok(v)) return 0
            L[++nl] = SV; return 1
        }
        # the value after "runs-on:" / "labels:" on the same line: scalar or one-line flow list of scalars
        function add_value(v,   n, k, R) {
            v = trim(v)
            if (v ~ /^\[.*\]$/) {
                n = flow_split(substr(v, 2, length(v) - 2), R)
                for (k = 1; k <= n; k++) if (!add_label(R[k])) return 0
                return 1
            }
            return add_label(v)
        }
        function bad_ro(why) { unv("job " jid ": runs-on " why); return 0 }
        # runs-on of the job: scalar, flow list, block list or {group, labels}; else refuse
        function read_runs_on(   i, k, cur, curv, itemind, x) {
            if (rov != "") { if (rdeep) return bad_ro("scalar continues on another line"); return add_value(rov) || bad_ro("value is not a plain/closed-quoted scalar or one-line list") }
            if (robuf_n == 0) return 1
            if (robuf[1] ~ /^-([ \t]|$)/) {
                for (i = 1; i <= robuf_n; i++) {
                    if (robuf[i] !~ /^-[ \t]+[^ \t]/ || rind[i] != rind[1]) return bad_ro("block list item is bare, nested or ragged")
                    if (!add_label(substr(robuf[i], 2))) return bad_ro("block list item is not a closed scalar")
                }
                return 1
            }
            cur = ""; curv = "x"; itemind = 0
            for (i = 1; i <= robuf_n; i++) {
                if (rind[i] == rmin) {
                    if (!skey(robuf[i]) || (K != "group" && K != "labels")) return bad_ro("mapping has a key other than group/labels")
                    cur = K; curv = V; itemind = 0
                    if (K == "group") { if (!add_label(V) || V == "") return bad_ro("group is not a scalar"); continue }
                    if (V != "" && !add_value(V)) return bad_ro("labels value is not a scalar or one-line list")
                } else {
                    if (cur != "labels" || curv != "" || robuf[i] !~ /^-[ \t]+[^ \t]/) return bad_ro("mapping content is not a labels block list")
                    if (itemind == 0) itemind = rind[i]
                    if (rind[i] != itemind || !add_label(substr(robuf[i], 2))) return bad_ro("labels item is not a closed scalar")
                }
            }
            return 1
        }
        function flush_job(   i, j, l, x, key, host) {
            if (jid == "") return
            if (nb == 0) unv("job " jid " has no mapping body")
            nl = 0
            if (read_runs_on()) {
                host = 0
                for (i = 1; i <= nl; i++) {
                    l = L[i]
                    if (index(l, "${{") == 0) { if (tolower(l) ~ /^(ubuntu|windows|macos)-/) host = 1; continue }
                    x = trim(l)
                    if (x ~ /^\$\{\{[ \t]*matrix\.[A-Za-z0-9_-]+[ \t]*\}\}$/) {
                        key = x; sub(/^\$\{\{[ \t]*matrix\./, "", key); sub(/[ \t]*\}\}$/, "", key)
                        if (!(key in mcnt) || mbad[key]) { unv("job " jid ": matrix." key " is absent or not a list of scalars"); continue }
                        for (j = 1; j <= mcnt[key]; j++) if (tolower(mval[key, j]) ~ /^(ubuntu|windows|macos)-/) host = 1
                    } else unv("job " jid ": runs-on expression other than matrix.<key>")
                }
                if (host) print fname " " jid
            }
            jid = ""
        }
        function mat_add(key, e,   n, R, k) {
            e = trim(e)
            if (e ~ /^\[.*\]$/) {
                n = flow_split(substr(e, 2, length(e) - 2), R)
                for (k = 1; k <= n; k++) { if (nonstr(R[k]) || !scalar_ok(R[k])) mbad[key] = 1; else mval[key, ++mcnt[key]] = SV }
            } else if (nonstr(e) || !scalar_ok(e)) mbad[key] = 1
            else mval[key, ++mcnt[key]] = SV
        }
        function mat_set(key, v,   n, R, k) {
            v = trim(v); mcnt[key] = 0; mbad[key] = 0
            if (v == "") { mpend = key; return }
            mpend = ""
            if (v ~ /^\[.*\]$/) { n = flow_split(substr(v, 2, length(v) - 2), R); for (k = 1; k <= n; k++) mat_add(key, R[k]) }
            else mbad[key] = 1
        }
        FNR == 1 { jid = ""; injobs = 0; ub = 0; first = 1; seenjobs = 0; bsind = -1; bsc = 0; sp = 0; pplain = 0; popen = 0; pind = 0 }
        {
            if (index($0, "\357\273\277")) unv("byte order mark")
            if (index($0, "\r")) unv("carriage return (CRLF)")
            if ($0 ~ /[\001-\010\013\014\016-\037\177]/) unv("control character")
            if (bsind >= 0 && $0 ~ /[^ \t]/) { match($0, /^ */); if (RLENGTH > bsind && (bsc == 0 || RLENGTH >= bsc)) { if (bsc == 0) bsc = RLENGTH; next } bsind = -1 }
            if (bsind >= 0) next
            line = strip_comment($0); sub(/[ \t]+$/, "", line); raw = $0; sub(/[ \t]+$/, "", raw)
            if (trim(line) == "") { if (trim(raw) != "") pplain = 0; next }
            if (line ~ /^ *\t/) { unv("tab in indentation (not valid YAML)"); next }
            match(line, /^ */); ind = RLENGTH; t = substr(line, ind + 1)
            chk_value(t, ind)
            chk_indent(t, ind)
            if (length(line) != length(raw)) pplain = 0
            if (first) { first = 0; if (ind != 0) unv("top-level mapping is not at column 0") }
            if (ind == 0) {
                flush_job(); injobs = 0; ji = 0
                if (!skey(t) || t ~ /^["\047]/) unv("top-level line is not a plain key (document marker, list, quoted or complex key)")
                else if (K == "jobs") {
                    if (seenjobs++) unv("second top-level jobs key")
                    if (V != "") unv("jobs value is not a block mapping (flow, alias or anchor)")
                    injobs = 1
                }
                next
            }
            if (!injobs) next
            if (ji == 0) ji = ind
            if (ind < ji) { unv("indentation under jobs is inconsistent"); next }
            if (ind == ji) {
                flush_job()
                if (!skey(t)) { unv("job entry is not a plain or quoted \"id:\" key"); next }
                if (V != "") unv("job value is not a block mapping (flow, alias or anchor)")
                jid = K; nb = 0; ki = 0; mode = ""; lastempty = 0; rov = ""; rdeep = 0; robuf_n = 0; rmin = 0
                for (k in mcnt) { delete mcnt[k]; delete mbad[k] }
                delete mval; si = 0; mi = 0; mpend = ""; inmat = 0; seenmat = 0
                next
            }
            nb++
            if (ki == 0) ki = ind
            if (ind < ki) { unv("job " jid ": key indentation is inconsistent"); next }
            if (ind == ki && t !~ /^-([ \t]|$)/) {
                mode = ""; inmat = 0
                if (!skey(t)) { unv("job " jid ": key is not plain or quoted (complex key, anchor, flow or merge)"); next }
                lastempty = (V == "")
                if (K == "runs-on") { mode = "ro"; rov = V; rdeep = 0; robuf_n = 0; rmin = 0 }
                else if (K == "strategy") { mode = "st"; si = 0; seenmat = 0; if (V != "") unv("job " jid ": strategy is not a block mapping") }
                next
            }
            if (ind == ki) {
                if (!lastempty || mode == "st") unv("job " jid ": list item at key indent after a non-block key")
            }
            if (mode == "ro") {
                if (rov != "") { rdeep = 1; next }
                robuf[++robuf_n] = t; rind[robuf_n] = ind
                if (rmin == 0 || ind < rmin) rmin = ind
                next
            }
            if (mode == "st") {
                if (ind == ki) next
                if (si == 0) si = ind
                if (ind < si) { unv("job " jid ": strategy indentation is inconsistent"); next }
                if (!inmat) {
                    if (ind == si) {
                        if (!skey(t)) { unv("job " jid ": strategy key is not plain"); next }
                        if (K == "matrix") {
                            if (seenmat++) unv("job " jid ": second matrix key")
                            if (V == "") { inmat = 1; mi = 0; mpend = "" }
                        }
                    }
                    next
                }
                if (ind <= si) {
                    inmat = 0
                    if (!skey(t)) { unv("job " jid ": strategy key is not plain"); next }
                    if (K == "matrix") unv("job " jid ": second matrix key")
                    next
                }
                if (mi == 0) mi = ind
                if (ind < mi) { unv("job " jid ": matrix indentation is inconsistent"); next }
                if (t ~ /^-([ \t]|$)/ && mpend != "") { mat_add(mpend, substr(t, 2)); next }
                if (ind == mi) {
                    if (!skey(t)) { unv("job " jid ": matrix key is not plain or quoted"); next }
                    mat_set(K, V)
                }
            }
        }
        END { flush_job() }
        ' "$f" || printf 'UNVERIFIABLE %s:0: awk failed\n' "${f##*/}"
    done < <(find -L "$WF_DIR" -maxdepth 1 -name '*.y*ml' -print0)
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
        rm -rf "${t:?}/adv"; mkdir -p "$t/adv"; printf -- "$3" > "$t/adv/$2"
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
    # round 2 (allow-list): the construct is refused or caught, never read as "not hosted"
    adv 'top-level mapping indented by 2' w.yml '  on: push\n  jobs:\n    a:\n      runs-on: ubuntu-latest\n'
    adv 'UTF-8 BOM before jobs:' w.yml '\357\273\277jobs:\n  a:\n    runs-on: ubuntu-latest\n'
    adv 'double-quoted label that continues on the next line' w.yml "${J}\"ubuntu-latest\n      \"\n"
    adv 'single-quoted label that continues on the next line' w.yml "${J}'ubuntu-\n      latest'\n"
    adv 'two documents, the second one clean' w.yml 'jobs:\n  ok:\n    runs-on: self-hosted\n---\nfoo: 1\n'
    adv 'second document indented, hosted job in it' w.yml 'jobs:\n  ok:\n    runs-on: self-hosted\n--- \n  jobs:\n    a:\n      runs-on: ubuntu-latest\n'
    adv 'job keys at inconsistent indent' w.yml 'jobs:\n  a:\n        name: x\n    runs-on: ubuntu-latest\n'
    adv 'top-level list "- jobs"' w.yml '- jobs\n'
    adv 'bare "-" block item under runs-on' w.yml 'jobs:\n  a:\n    runs-on:\n      -\n        ubuntu-latest\n'
    rm -rf "${t:?}/adv"; mkdir -p "$t/adv" "$t/tgt"
    printf 'jobs:\n  h:\n    runs-on: ubuntu-latest\n' > "$t/tgt/h.yml"; ln -s "$t/tgt/h.yml" "$t/adv/h.yml"
    printf 'jobs:\n  ok:\n    runs-on: self-hosted\n' > "$t/adv/ok.yml"
    rc=0; WF_DIR="$t/adv" BASELINE="$t/base" check > /dev/null || rc=$?
    row "MUST-RED: a hosted job in a symlinked workflow file" 1 "$rc"
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
