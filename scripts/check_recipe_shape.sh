#!/usr/bin/env bash
# check_recipe_shape.sh -- SHACL-validate the cookbook recipes (cookbook-recipe-v1, #440, aprender#3769).
#
#   1. evidence/recipes/recipes.jsonl must equal what the recipes derive (recipes_to_jsonl.sh --check),
#      so the shape never judges a stale copy.
#   2. `pv lint contracts --gate shapes --shape cookbook-recipe-v1` must pass with EXACTLY as many focus
#      nodes as there are recipes (a gate that saw fewer nodes judged less than it claims).
#
# --self-test has five groups of rows:
#   (a) focus_count reads by_shape only; a missing yq or jq is ENV (rc 2, message names the tool);
#   (b) throwaway fixture repos, each with ONE planted defect in the derived rows, which the REAL pv
#       shapes gate must refuse (rc 1, naming the property), and the unmodified real rows must pass;
#       plus one end-to-end row planting the defect in a recipe YAML, derived then gated;
#   (c) derivation parity with the deleted Python on odd YAML (empty, null, list, multi-document,
#       non-mapping models/expect, CRLF CURRENT, ASCII escapes, floats, falsy fields);
#   (d) an empty corpus stays rc 2 through check();
#   (e) the bashrs lint of recipes_to_jsonl.sh exits 0 (skipped when bashrs is absent), and this
#       header names the no-yq exit case.
# A shape that cannot go red validates nothing.
#
#
# Usage: scripts/check_recipe_shape.sh [--self-test]
# Exit: 0 conforms | 1 violation / drift / count mismatch | 2 ENV (no pv, no jq, no yq)
set -euo pipefail

SHAPE=cookbook-recipe-v1
PV=${PV:-pv}

need() { command -v "$1" > /dev/null 2>&1 || { echo "ENV: $1 is missing" >&2; exit 2; }; }

gate() { # gate <contracts dir> -> prints the pv JSON, returns pv's rc
    "$PV" lint "$1" --gate shapes --shape "$SHAPE" --format json 2> /dev/null
}

focus_count() { # focus_count <pv json> -> the number of cookbook-recipe-v1 focus nodes judged (by_shape only)
    local n
    n=$(jq -r --arg s "$SHAPE" '[.by_shape[]? | select(startswith($s + "=")) | split("=")[1] | tonumber] | .[0] // 0' 2> /dev/null || true)
    printf '%s' "${n:-0}"
}

check() {
    local out rc n want
    rc=0; bash scripts/recipes_to_jsonl.sh --check || rc=$?
    [ "$rc" = 0 ] || return "$rc"
    rc=0; out=$(gate contracts) || rc=$?
    n=$(printf '%s' "$out" | focus_count)
    want=$(grep -c . evidence/recipes/recipes.jsonl)
    if [ "$rc" != 0 ]; then
        echo "FAIL  $SHAPE: pv shapes gate rc $rc"
        printf '%s\n' "$out" | grep -E '"(message|shape|path|property)"' | head -n 10 | cut -c1-240 || true
        return 1
    fi
    if [ "$n" != "$want" ]; then
        echo "FAIL  $SHAPE judged $n focus node(s), but there are $want recipe row(s)"
        return 1
    fi
    echo "PASS  $SHAPE: $n recipe(s) SHACL-validated (pv shapes gate), derived evidence in sync"
}

self_test() {
    local t bad=0
    t=$(mktemp -d)
    case "$t" in /tmp/tmp.*) ;; *) echo "REFUSED: mktemp gave $t" >&2; return 2 ;; esac
    row() { if [ "$2" = "$3" ]; then echo "ok    $1"; else echo "FAIL  $1 (got $3, wanted $2)"; bad=$((bad + 1)); fi; }
    # RED 1: focus_count reads by_shape only; a message that repeats the token must not inflate it.
    row "focus_count reads by_shape only (a message cannot inflate it)" "2" "$(printf '%s' '{"results":[{"message":"cookbook-recipe-v1=99"}],"by_shape":["cookbook-recipe-v1=2"]}' | focus_count)"
    # RED 2: no yq on PATH is an ENV failure (rc 2 and the message "ENV: yq is missing"), not a generic rc 1.
    local nb="$t/noyq" yrc=0 yerr0; mkdir -p "$nb"; ln -sf "$(command -v jq)" "$nb/jq"; ln -sf "$(command -v "$PV")" "$nb/pv"
    yerr0=$(PV=pv PATH="$nb" "$BASH" scripts/check_recipe_shape.sh 2>&1 > /dev/null) || yrc=$?
    row "missing yq is ENV: rc 2 and names yq" "2 ENV: yq is missing" "$yrc $yerr0"
    # fixture <name> <jq filter editing the FIRST real row> [<property>]: a tiny repo holding the
    # contract and the real rows with ONE defect planted in the first row. Echoes "<rc> <focus> <named>".
    fixture() {
        local d="$t/$1" rc out
        mkdir -p "$d/contracts" "$d/evidence/recipes"
        cp contracts/$SHAPE.yaml "$d/contracts/"
        { head -n 1 evidence/recipes/recipes.jsonl | jq -cS "$2"; tail -n +2 evidence/recipes/recipes.jsonl; } > "$d/evidence/recipes/recipes.jsonl"
        rc=0; out=$(gate "$d/contracts") || rc=$?
        # A must-RED row passes only if the violation names ITS property: rc 1 for some other reason
        # (an orthogonal failure) would pass either way and prove nothing about this property.
        local named=-
        if [ -n "${3:-}" ]; then
            if printf '%s' "$out" | grep -q "recipe/$3"; then named=named; else named=NOT-named; fi
        fi
        printf '%s %s %s' "$rc" "$(printf '%s' "$out" | focus_count)" "$named"
    }
    local n; n=$(grep -c . evidence/recipes/recipes.jsonl)
    row "the real rows conform: rc 0 and every recipe is a focus node" "0 $n -" "$(fixture ok '.')"
    row "MUST-RED: argv[0] is not apr" "1 named" "$(fixture argv0 '.argv0_is_apr = false' argv0_is_apr | cut -d' ' -f1,3)"
    row "MUST-RED: a {model:slot} that names no pinned model" "1 named" "$(fixture slots '.placeholders_resolve = false' placeholders_resolve | cut -d' ' -f1,3)"
    row "MUST-RED: a model pinned by a non-sha256" "1 named" "$(fixture sha '.model_sha256 = ["latest"]' model_sha256 | cut -d' ' -f1,3)"
    row "MUST-RED: a model ref that is not an hf:// file path" "1 named" "$(fixture ref '.model_ref = ["Qwen3.5-4B"]' model_ref | cut -d' ' -f1,3)"
    row "MUST-RED: expect judges only the exit code" "1 named" "$(fixture outchk '.has_output_check = false' has_output_check | cut -d' ' -f1,3)"
    row "MUST-RED: no PASS receipt from the CURRENT release" "1 named" "$(fixture rcpt '.receipt_current_pass = false' receipt_current_pass | cut -d' ' -f1,3)"
    row "MUST-RED: the receipt field missing altogether (minCount)" "1 named" "$(fixture rcptmiss 'del(.receipt_current_pass)' receipt_current_pass | cut -d' ' -f1,3)"
    row "MUST-RED: an undeclared host" "1 named" "$(fixture host '.host = ["laptop"]' host | cut -d' ' -f1,3)"
    row "MUST-RED: an undeclared field (the shape is closed)" "1 named" "$(fixture closed '.sneaky = "x"' sneaky | cut -d' ' -f1,3)"
    row "MUST-RED: a schema other than cookbook-recipe/v1" "1 named" "$(fixture schema '.schema = "cookbook-recipe/v0"' schema | cut -d' ' -f1,3)"
    # END TO END through the DERIVATION: the defect is planted in a recipe YAML, not in the rows, so
    # a derivation that computed argv0_is_apr=true regardless would turn this row green.
    local e="$t/e2e" rc2=0 yaml
    mkdir -p "$e/contracts" "$e/evidence/recipes"
    cp contracts/$SHAPE.yaml "$e/contracts/"
    cp -r recipes "$e/recipes"
    cp -r receipts "$e/receipts" 2> /dev/null || true
    yaml="$e/recipes/apr/capability/capability-registry-json.yaml"
    grep -q '\["apr", "capability"' "$yaml" || { echo "ENV: e2e anchor missing in $yaml" >&2; return 2; }
    sed -i '0,/\["apr", "capability"/s//["cargo", "capability"/' "$yaml"
    bash scripts/recipes_to_jsonl.sh --recipes "$e/recipes/apr" --receipts "$e/receipts" --out "$e/evidence/recipes/recipes.jsonl" > /dev/null
    local eout; eout=$(gate "$e/contracts") || rc2=$?
    # Editing the recipe also invalidates its receipt, so rc 1 alone could come from
    # receipt_current_pass; the row requires argv0_is_apr ITSELF to be named.
    printf '%s' "$eout" | grep -q "recipe/argv0_is_apr" && rc2="$rc2 named" || rc2="$rc2 NOT-named"
    row "MUST-RED end to end: a recipe YAML whose argv[0] is 'cargo', derived then gated, names argv0_is_apr" "1 named" "$rc2"
    # #494 lane B findings: the DERIVATION must match the deleted Python derivation on odd inputs.
    # rj <name> <yaml text> [<CURRENT text>]: derive one recipe into $t/<name>; echoes "<rc>|<row>".
    rj() {
        local d="$t/rj-$1" rc=0 sha
        mkdir -p "$d/recipes" "$d/receipts"
        printf '%b' "$2" > "$d/recipes/r.yaml"
        if [ -n "${3:-}" ]; then
            sha=$(sha256sum "$d/recipes/r.yaml" | cut -d' ' -f1)
            printf '%b' "$3" > "$d/receipts/CURRENT"
            mkdir -p "$d/receipts/0.2.0-abc/h1"
            printf '{"verdict":"PASS","apr_version":"0.2.0-abc","recipe_sha256":"%s"}' "$sha" > "$d/receipts/0.2.0-abc/h1/a.json"
        fi
        bash scripts/recipes_to_jsonl.sh --recipes "$d/recipes" --receipts "$d/receipts" --out "$d/out.jsonl" > /dev/null 2>&1 || rc=$?
        printf '%s|%s' "$rc" "$(cat "$d/out.jsonl" 2> /dev/null || true)"
    }
    local base='id: a\nargv: [apr, run]\nexpect: {exit: 0}\n'
    row "derive: empty .yaml is rc 1 (Python AttributeError)" "1" "$(rj empty '' | cut -d'|' -f1)"
    row "derive: comment-only .yaml is rc 1" "1" "$(rj comment '# nothing\n' | cut -d'|' -f1)"
    row "derive: null .yaml is rc 1" "1" "$(rj null 'null\n' | cut -d'|' -f1)"
    row "derive: list .yaml is rc 1" "1" "$(rj list '- a\n- b\n' | cut -d'|' -f1)"
    row "derive: multi-document .yaml is rc 1 (Python ComposerError)" "1" "$(rj multi "${base}---\nid: b\nargv: [apr, run]\n" | head -n 1 | cut -d'|' -f1)"
    row "derive: CURRENT with CRLF still counts (Python .strip())" "true" "$(rj crlf "${base}hosts: [h1]\n" '0.2.0-abc\r\n' | grep -o '"receipt_current_pass": true' | cut -d' ' -f2 || true)"
    row "derive: CURRENT with leading blank lines still counts" "true" "$(rj blank "${base}hosts: [h1]\n" '\n\n0.2.0-abc\n' | grep -o '"receipt_current_pass": true' | cut -d' ' -f2 || true)"
    row "derive: non-ASCII is escaped \\uXXXX (ensure_ascii)" "ok" "$(rj uni "${base}min_apr: \"0.2.0\xe2\x80\x94beta\"\n" | grep -qF '0.2.0\u2014beta' && echo ok || echo no)"
    row "derive: whole-number float keeps its point (1.0)" "ok" "$(rj flt "${base}min_apr: 1.0\n" | grep -qF '"min_apr": 1.0' && echo ok || echo no)"
    row "derive: falsy models: \"\" and expect: \"\" succeed (Python or {})" "0" "$(rj falsy 'id: a\nargv: [apr, run]\nmodels: ""\nexpect: ""\n' | cut -d'|' -f1)"
    # #494 quorum findings (4ed9940): the same derivation parity, plus rc/ENV plumbing.
    row "derive: multi-doc, leading empty document ('---' twice) is rc 1 (Python ComposerError)" "1" "$(rj mdA '---\n---\nid: a\nargv: [apr, run]\n' | cut -d'|' -f1)"
    row "derive: multi-doc, leading comment-only document is rc 1" "1" "$(rj mdB '---\n# c\n---\nid: a\nargv: [apr, run]\n' | cut -d'|' -f1)"
    row "derive: multi-doc, CRLF twin of the empty-leading-document case is rc 1" "1" "$(rj mdC '---\r\n---\r\nid: a\r\nargv: [apr, run]\r\n' | cut -d'|' -f1)"
    row "derive: multi-doc, CRLF twin of the comment-leading case is rc 1" "1" "$(rj mdD '---\r\n# c\r\n---\r\nid: a\r\nargv: [apr, run]\r\n' | cut -d'|' -f1)"
    # RED (cop fail-open): a tab after '---' is still a document marker; the guard counted only '--- ' and '---' at EOL.
    row "derive: multi-doc, empty leading document with a TAB-marked '---' is rc 1" "1" "$(rj mdF '---\t\n---\nid: a\nargv: [apr, run]\n' | cut -d'|' -f1)"
    row "derive: multi-doc, TAB-marked second '---' is rc 1" "1" "$(rj mdG "${base}---\tid: b\nargv: [apr, run]\n" | cut -d'|' -f1)"
    row "derive: single document with ONE leading '---' still derives (rc 0)" "0" "$(rj mdE '---\nid: a\nargv: [apr, run]\n' | cut -d'|' -f1)"
    row "derive: models: [{}] (non-mapping) is rc 1 (Python AttributeError)" "1" "$(rj nmod 'id: a\nargv: [apr, run]\nmodels: [{}]\n' | cut -d'|' -f1)"
    row "derive: expect: [1] (non-mapping) is rc 1" "1" "$(rj nexp 'id: a\nargv: [apr, run]\nexpect: [1]\n' | cut -d'|' -f1)"
    # C: an empty corpus must stay rc 2 THROUGH check() (a fixture repo with scripts but no recipes).
    local ec="$t/empty" crc=0
    mkdir -p "$ec/recipes/apr" "$ec/evidence/recipes" "$ec/contracts"
    cp -r scripts "$ec/scripts"
    (cd "$ec" && "$BASH" scripts/check_recipe_shape.sh > /dev/null 2>&1) || crc=$?
    row "check: empty corpus stays rc 2 through check()" "2" "$crc"
    # D: the missing-tool rows assert the rc AND the message, linking the REAL "$PV" (not a hardcoded pv).
    local nj="$t/nojq" jrc=0 jerr
    mkdir -p "$nj"; ln -sf "$(command -v yq)" "$nj/yq"; ln -sf "$(command -v "$PV")" "$nj/pv"
    jerr=$(PV=pv PATH="$nj" "$BASH" scripts/check_recipe_shape.sh 2>&1 > /dev/null) || jrc=$?
    row "missing jq is ENV: rc 2 and names jq" "2 ENV: jq is missing" "$jrc $jerr"
    local bl=skipped
    if command -v bashrs > /dev/null 2>&1; then bl=0; bashrs lint scripts/recipes_to_jsonl.sh > /dev/null 2>&1 || bl=$?; fi
    row "bashrs lint exits 0 on recipes_to_jsonl.sh (or bashrs absent)" "ok" "$([ "$bl" = 0 ] || [ "$bl" = skipped ] && echo ok || echo "rc $bl")"
    row "header exit-code line names the no-yq case" "ok" "$(grep -E '^# Exit:' scripts/check_recipe_shape.sh | grep -q 'no yq' && echo ok || echo no)"
    if [ -n "$t" ] && [ "$t" != "/" ] && [ -d "$t" ]; then
        case "$t" in /tmp/tmp.*) rm -rf -- "$t" ;; *) ;; esac
    fi
    echo "check_recipe_shape self-test: $([ "$bad" = 0 ] && echo PASS || echo FAIL) ($bad failed)"
    [ "$bad" = 0 ]
}

need jq
need yq
need "$PV"
case "${1:-}" in
    "") check ;;
    --self-test) self_test ;;
    *) echo "usage: $0 [--self-test]" >&2; exit 2 ;;
esac
