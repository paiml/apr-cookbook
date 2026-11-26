#!/usr/bin/env bash
# recipes_to_jsonl.sh -- derive evidence/recipes/recipes.jsonl from recipes/apr/**.yaml for the
# cookbook-recipe-v1 shape (#440, bash port #474). Output is byte-identical to the former derivation.
#
# pv's shapes gate validates RDF built by its generic `json` extractor, which turns each JSONL line
# into one node of scalar properties. It cannot see list positions ("argv[0] is apr") or join two
# fields ("every {model:<slot>} names a pinned model"). So this script computes those facts, and the
# receipt coverage, as explicit fields. contracts/cookbook-recipe-v1.yaml then REQUIRES them.
# The JSONL is DERIVED and never hand-edited: --check regenerates it and fails on any difference.
#
# A row per recipe:
#   id, schema, verb, recipe_sha256
#   argv0_is_apr          argv[0] == "apr"
#   placeholders_resolve  every {model:<slot>} in argv names a slot in models, and every slot is used
#   model_sha256 / model_ref (repeated, one per model slot)
#   expect_exit           the expected exit code
#   has_output_check      expect also judges output (stdout_contains / stdout_json / stdout_json_is_object)
#   min_apr, host (repeated), backend (repeated)
#   receipt_current_pass  for EVERY declared host: receipts/<CURRENT>/<host>/<id>.json says PASS,
#                         names the CURRENT binary, and carries this recipe's sha256 (#442's rule)
#
# Zero recipes is REFUSED (rc 2): an empty corpus would let the shape gate decline quietly.
# Usage: recipes_to_jsonl.sh [--recipes DIR] [--receipts DIR] [--out FILE] [--check]
# Needs: yq (v4, YAML to JSON), jq, sha256sum, sed, sort, find.
set -euo pipefail

RECIPES=recipes/apr
RECEIPTS=receipts
OUT=evidence/recipes/recipes.jsonl
CHECK=0

usage() {
    printf 'usage: recipes_to_jsonl.sh [--recipes DIR] [--receipts DIR] [--out FILE] [--check]\n'
}

bad_usage() {
    usage >&2
    exit 2
}

while [ $# -gt 0 ]; do
    case "$1" in
        --recipes)  [ $# -ge 2 ] || bad_usage; RECIPES=$2; shift 2 ;;
        --receipts) [ $# -ge 2 ] || bad_usage; RECEIPTS=$2; shift 2 ;;
        --out)      [ $# -ge 2 ] || bad_usage; OUT=$2; shift 2 ;;
        --check)    CHECK=1; shift ;;
        -h|--help)  usage && exit 0 ;;
        *)          bad_usage ;;
    esac
done

# list_recipes: every *.yaml file under $RECIPES, ordered by path components (the order the
# path-sorted derivation used). A \x01 separator sorts below every filename byte, so "run" < "run-x".
list_recipes() {
    [ -d "$RECIPES" ] || return 0
    find "$RECIPES" -type f -name '*.yaml' -print \
        | sed 's|/|\x01|g' | LC_ALL=C sort | sed 's|\x01|/|g'
}

# receipt_current_pass ID SHA HOST...: rc 0 only when every declared host's receipt for the CURRENT
# binary says PASS, names that binary (its version tail after the first "-") and carries SHA.
receipt_current_pass() {
    local rid=$1 sha=$2 cur csha h
    shift 2
    [ -r "$RECEIPTS/CURRENT" ] || return 1
    read -r cur < "$RECEIPTS/CURRENT" || true
    case "$cur" in
        *-*) csha=$(cut -d- -f2- <<< "$cur") ;;
        *)   return 1 ;;
    esac
    [ -n "$csha" ] || return 1
    [ $# -gt 0 ] || return 1
    for h in "$@"; do
        jq -e --arg sha "$sha" --arg csha "$csha" \
            '.verdict == "PASS" and ((.apr_version // "") | contains($csha)) and .recipe_sha256 == $sha' \
            "$RECEIPTS/$cur/$h/$rid.json" >/dev/null 2>&1 || return 1
    done
    return 0
}

# JSON filter: one recipe (YAML already converted to JSON on stdin) to one row, rendered with the
# same spacing and sorted keys as the former derivation (", " and ": "; null fields dropped).
read -r -d '' ROW_FILTER <<'JQ' || true
def orn: if . == null or . == false or . == "" or . == 0 or ((type | IN("array", "object")) and length == 0) then null else . end;
def py:
    if type == "object" then
        "{" + ([keys[] as $k | ($k | tojson) + ": " + (.[$k] | py)] | join(", ")) + "}"
    elif type == "array" then
        "[" + ([.[] | py] | join(", ")) + "]"
    else tojson end;
(.argv // []) as $argv
| (.models // {}) as $models
| (.expect // {}) as $expect
| ([$argv[] | strings | capture("^\\{model:(?<s>[A-Za-z0-9_-]+)\\}$")? | .s]) as $used
| {
    id: .id,
    schema: .schema,
    verb: (if ($argv | length) > 1 then $argv[1] else null end),
    recipe_sha256: $sha,
    argv0_is_apr: (($argv | length) > 0 and $argv[0] == "apr"),
    placeholders_resolve: (($used | unique) == ($models | keys | unique)),
    model_sha256: ([$models[] | (((. | orn) // {}) | .sha256)] | if length == 0 then null else . end),
    model_ref: ([$models[] | (((. | orn) // {}) | .ref)] | if length == 0 then null else . end),
    expect_exit: $expect.exit,
    has_output_check: (($expect | has("stdout_contains")) or ($expect | has("stdout_json")) or ($expect | has("stdout_json_is_object"))),
    min_apr: .min_apr,
    host: (.hosts | orn),
    backend: (.backends | orn),
    receipt_current_pass: $rcp
  }
| with_entries(select(.value != null))
| py
JQ

# row FILE: print the JSONL row for one recipe file.
row() {
    local f=$1 sha yj rid rcp
    local -a hosts=()
    sha=$(sha256sum "$f" | cut -d' ' -f1)
    yj=$(yq -o=json -I=0 '.' "$f")
    rid=$(jq -r '.id // "None"' <<< "$yj")
    mapfile -t hosts < <(jq -r '(.hosts // [])[] | tostring' <<< "$yj")
    if [ "${#hosts[@]}" -gt 0 ] && receipt_current_pass "$rid" "$sha" "${hosts[@]}"; then
        rcp=true
    else
        rcp=false
    fi
    jq -r --arg sha "$sha" --argjson rcp "$rcp" "$ROW_FILTER" <<< "$yj"
}

# build: one row per recipe, in list_recipes order.
build() {
    local f
    while IFS= read -r f; do
        [ -n "$f" ] || continue
        row "$f"
    done < <(list_recipes)
}

TMP=$(mktemp "${TMPDIR:-/tmp}/recipes-to-jsonl.XXXXXX")
trap 'rm -f "$TMP"' EXIT
build > "$TMP"

if [ ! -s "$TMP" ]; then
    printf 'REFUSED: no recipes under %s -- an empty corpus would let the shape gate decline quietly\n' "$RECIPES"
    exit 2
fi
N=$(wc -l < "$TMP" | tr -d ' ')

if [ "$CHECK" = 1 ]; then
    if [ -f "$OUT" ] && cmp -s "$TMP" "$OUT"; then
        printf 'PASS  %s matches the %s recipe(s)\n' "$OUT" "$N"
        exit 0
    fi
    printf 'FAIL  %s is not what the recipes derive -- run scripts/recipes_to_jsonl.sh (never hand-edit it)\n' "$OUT"
    exit 1
fi

mkdir -p "$(dirname "$OUT")"
cp "$TMP" "$OUT"
printf 'wrote %s: %s recipe row(s)\n' "$OUT" "$N"
