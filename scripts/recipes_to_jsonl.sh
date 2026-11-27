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
# Output is ASCII-only (\uXXXX escapes, like json.dumps); a recipe that is not ONE YAML mapping exits 1.
# Needs: yq (v4, YAML to JSON), jq, sha256sum, sed, sort, find.
set -euo pipefail

# a missing tool must be an environment failure (rc 2), not a recipe failure (rc 1)
need() {
    if ! command -v "$1" > /dev/null 2>&1; then
        printf 'ENV: %s is missing\n' "$1" >&2
        exit 2
    fi
}
need yq
need jq

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
    # strip like Python's read_text().strip(): leading and trailing whitespace (CR, blank lines)
    cur=$(jq -Rrs 'sub("^\\s+"; "") | sub("\\s+$"; "")' "$RECEIPTS/CURRENT") || return 1
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
def h4: . as $n | [12, 8, 4, 0] | map((($n / pow(2; .)) | floor) % 16 | "0123456789abcdef"[. : . + 1]) | join("");
def esc: tojson | explode | map(if . < 128 then ([.] | implode) elif . < 65536 then "\\u" + h4 else (. - 65536) as $c | "\\u" + (55296 + ($c / 1024 | floor) | h4) + "\\u" + (56320 + ($c % 1024) | h4) end) | join("");
def py:
    if type == "object" and has("__float") then "\(.__float).0"
    elif type == "object" then
        "{" + ([keys[] as $k | ($k | esc) + ": " + (.[$k] | py)] | join(", ")) + "}"
    elif type == "array" then
        "[" + ([.[] | py] | join(", ")) + "]"
    elif type == "string" then esc
    else tojson end;
((.argv | orn) // []) as $argv
| ((.models | orn) // {}) as $models
| ((.expect | orn) // {}) as $expect
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
    min_apr: (if $minfloat and (.min_apr | type) == "number" and .min_apr == (.min_apr | floor) and (.min_apr | fabs) < 1e16 then {__float: .min_apr} else .min_apr end),
    host: (.hosts | orn),
    backend: (.backends | orn),
    receipt_current_pass: $rcp
  }
| with_entries(select(.value != null))
| py
JQ

# a truthy models or expect must itself be a mapping (Python calls .values() / .get() on it)
read -r -d '' MAPPING_FILTER <<'JQ' || true
def truthy: . != null and . != false and . != "" and . != 0 and ((type | IN("array", "object") | not) or length > 0);
[(.models, .expect) | select(truthy) | select(type != "object")] | length == 0
JQ

# row FILE: print the JSONL row for one recipe file.
row() {
    local f=$1 sha yj rid rcp nmark mf=false
    local -a hosts=()
    sha=$(sha256sum "$f" | cut -d' ' -f1)
    # yq drops leading empty or comment-only documents, so more than one column-0 document marker
    # (CR stripped) is rejected before parsing, as Python's safe_load refuses a multi-document stream
    # every CR (CRLF or bare CR) is a line break to YAML, so turn each into LF before counting
    nmark=$(sed 's/\r/\n/g' "$f" | grep -c -E '^---([[:space:]]|$)' || true)
    if [ "$nmark" -gt 1 ]; then
        printf 'ERROR: %s is not a single YAML mapping (more than one document marker)\n' "$f" >&2
        exit 1
    fi
    yj=$(yq -o=json -I=0 '.' "$f")
    # one YAML document that is a mapping, like yaml.safe_load + r.get(...) (Python rc 1 otherwise)
    if ! jq -s -e 'length == 1 and (.[0] | type == "object")' <<< "$yj" > /dev/null 2>&1; then
        printf 'ERROR: %s is not a single YAML mapping (empty, null, multi-document or not an object)\n' "$f" >&2
        exit 1
    fi
    # a truthy models or expect must itself be a mapping (Python calls .values() / .get() on it)
    if ! jq -e "$MAPPING_FILTER" <<< "$yj" > /dev/null 2>&1; then
        printf 'ERROR: %s has a models or expect that is not a mapping\n' "$f" >&2
        exit 1
    fi
    # a whole-number float (min_apr: 1.0) must keep its point, as Python's json.dumps does
    [ "$(yq '.min_apr | tag' "$f")" = '!!float' ] && mf=true
    rid=$(jq -r '.id // "None"' <<< "$yj")
    mapfile -t hosts < <(jq -r '(.hosts | arrays)[]? | tostring' <<< "$yj")
    if [ "${#hosts[@]}" -gt 0 ] && receipt_current_pass "$rid" "$sha" "${hosts[@]}"; then
        rcp=true
    else
        rcp=false
    fi
    # a field the derivation cannot read (argv not a list, a models value not a mapping) is a malformed
    # recipe: rc 1 with a message, never jq's own rc 5 (the gate contract is rc 1 for malformed input)
    local out
    out=$(jq -r --arg sha "$sha" --argjson rcp "$rcp" --argjson minfloat "$mf" "$ROW_FILTER" <<< "$yj") || {
        printf 'ERROR: %s has a field the derivation cannot read: argv must be a list, models a mapping of mappings\n' "$f" >&2
        exit 1
    }
    printf '%s\n' "$out"
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
cat "$TMP" > "$OUT"
printf 'wrote %s: %s recipe row(s)\n' "$OUT" "$N"
