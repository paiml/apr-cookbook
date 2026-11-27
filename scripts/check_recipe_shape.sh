#!/usr/bin/env bash
# check_recipe_shape.sh -- SHACL-validate the cookbook recipes (cookbook-recipe-v1, #440, aprender#3769).
#
#   1. evidence/recipes/recipes.jsonl must equal what the recipes derive (recipes_to_jsonl.sh --check),
#      so the shape never judges a stale copy.
#   2. `pv lint contracts --gate shapes --shape cookbook-recipe-v1` must pass with EXACTLY as many focus
#      nodes as there are recipes (a gate that saw fewer nodes judged less than it claims).
#
# --self-test builds throwaway fixture repos, each with ONE planted defect in the derived rows, and
# requires the REAL pv shapes gate to refuse every one (rc 1, naming the shape), and the unmodified real
# rows to pass. A shape that cannot go red validates nothing.
#
# Usage: scripts/check_recipe_shape.sh [--self-test]
# Exit: 0 conforms | 1 violation / drift / count mismatch | 2 ENV (no pv, no jq)
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
    bash scripts/recipes_to_jsonl.sh --check || return 1
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
    # RED 2: no yq on PATH is an ENV failure (rc 2), not a generic rc 1 from the derivation.
    local nb="$t/noyq" yrc=0; mkdir -p "$nb"; ln -sf "$(command -v jq)" "$nb/jq"; ln -sf "$(command -v pv)" "$nb/pv"
    PATH="$nb" "$BASH" scripts/check_recipe_shape.sh > /dev/null 2>&1 || yrc=$?
    row "missing yq is ENV: rc 2" "2" "$yrc"
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
