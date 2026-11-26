#!/usr/bin/env bash
# Interim argv check for recipes/apr/**.yaml against ONE apr binary's --help (#3769 draft).
#
# NOT the cookbook-recipe-v1 SHACL shape (aprender#3769 done_when 1, not built yet). It
# checks one thing: every `--flag` a recipe's argv uses is listed by `apr <verb> --help`
# on the given binary. A removed or misspelt flag is a failure. Nothing is run.
# Usage: check_recipe_argv_surface.sh <apr-binary> [recipes-dir]; exit 1 on any miss.
#
# Bash port of the former checker (apr-cookbook#473). Pure bash and coreutils:
# recipes are read with a small YAML scanner that covers the forms the corpus uses
# (a top-level `id:` scalar, a flow or block `argv:` sequence of plain, "double" or
# 'single' quoted scalars).
set -euo pipefail
export LC_ALL=C

if [ "$#" -lt 1 ]; then
  echo "usage: check_recipe_argv_surface.sh <apr-binary> [recipes-dir]" >&2
  exit 1
fi
apr=$1
if [ "$#" -ge 2 ]; then
  root=$2
else
  root=recipes/apr
fi
if [ -z "$root" ]; then
  root=.
fi
if ! command -v "$apr" >/dev/null 2>&1; then
  echo "check_recipe_argv_surface.sh: cannot run apr binary: $apr" >&2
  exit 1
fi

# ---- YAML scanner state -------------------------------------------------------
S=""      # text being scanned
POS=0     # cursor into S
SCALAR="" # last scalar parsed
ITEMS=()  # argv items parsed from one recipe
NONSTR=0  # 1 when an argv item is an unquoted YAML non-string (int, bool, null, ...)

# True when an unquoted YAML scalar would load as a non-string (PyYAML rules, the subset
# that matters here). The former checker crashed on such an argv item.
is_nonstr_scalar() {
  local v=$1
  case $v in
    '' | '~' | null | Null | NULL | true | True | TRUE | false | False | FALSE) return 0 ;;
    yes | Yes | YES | no | No | NO | on | On | ON | off | Off | OFF) return 0 ;;
  esac
  [[ $v =~ ^[-+]?[0-9][0-9_]*$ ]] && return 0
  [[ $v =~ ^0x[0-9a-fA-F_]+$ || $v =~ ^0o[0-7_]+$ || $v =~ ^0b[01_]+$ ]] && return 0
  [[ $v =~ ^[-+]?[0-9_]*\.[0-9_]*([eE][-+]?[0-9]+)?$ ]] && [[ $v =~ [0-9] ]] && return 0
  [[ $v =~ ^[-+]?[0-9][0-9_]*[eE][-+]?[0-9]+$ ]] && return 0
  [[ $v =~ ^[-+]?\.(inf|Inf|INF)$ || $v == .nan || $v == .NaN || $v == .NAN ]] && return 0
  [[ $v =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}$ ]] && return 0
  return 1
}

# Skip spaces, tabs, line breaks and `#` comments (only ever called between tokens).
skip_ws() {
  local c
  while :; do
    c=${S:POS:1}
    case $c in
      ' ' | $'\t' | $'\n' | $'\r') POS=$((POS + 1)) ;;
      '#')
        while [ -n "${S:POS:1}" ] && [ "${S:POS:1}" != $'\n' ]; do
          POS=$((POS + 1))
        done
        ;;
      *) return 0 ;;
    esac
  done
}

# Parse one scalar at POS into SCALAR. $1 is `flow` or `block`.
parse_scalar() {
  local c out="" esc prev=" "
  c=${S:POS:1}
  if [ "$c" = '"' ]; then
    POS=$((POS + 1))
    while :; do
      c=${S:POS:1}
      if [ -z "$c" ]; then
        return 1
      fi
      POS=$((POS + 1))
      if [ "$c" = '"' ]; then
        break
      fi
      if [ "$c" = '\' ]; then
        esc=${S:POS:1}
        POS=$((POS + 1))
        case $esc in
          n) out+=$'\n' ;;
          t) out+=$'\t' ;;
          r) out+=$'\r' ;;
          0) ;;
          x) POS=$((POS + 2)) ;;
          *) out+=$esc ;;
        esac
      else
        out+=$c
      fi
    done
  elif [ "$c" = "'" ]; then
    POS=$((POS + 1))
    while :; do
      c=${S:POS:1}
      if [ -z "$c" ]; then
        return 1
      fi
      POS=$((POS + 1))
      if [ "$c" = "'" ]; then
        if [ "${S:POS:1}" = "'" ]; then
          out+="'"
          POS=$((POS + 1))
        else
          break
        fi
      else
        out+=$c
      fi
    done
  else
    while :; do
      c=${S:POS:1}
      if [ -z "$c" ] || [ "$c" = $'\n' ] || [ "$c" = $'\r' ]; then
        break
      fi
      if [ "$1" = flow ]; then
        case $c in
          ',' | ']' | '[' | '{' | '}') break ;;
        esac
      fi
      if [ "$c" = '#' ] && { [ "$prev" = ' ' ] || [ "$prev" = $'\t' ]; }; then
        break
      fi
      out+=$c
      prev=$c
      POS=$((POS + 1))
    done
    out="${out%"${out##*[![:space:]]}"}"
    if is_nonstr_scalar "$out"; then
      NONSTR=1
    fi
  fi
  SCALAR=$out
}

# Parse the argv sequence at POS into ITEMS. Flow `[a, b]` or block `- a` lines.
parse_argv() {
  local c
  ITEMS=()
  skip_ws
  c=${S:POS:1}
  if [ "$c" = '[' ]; then
    POS=$((POS + 1))
    while :; do
      skip_ws
      c=${S:POS:1}
      case $c in
        '') return 1 ;;
        ']')
          POS=$((POS + 1))
          return 0
          ;;
        ',') POS=$((POS + 1)) ;;
        '[' | '{') return 1 ;;
        *)
          parse_scalar flow || return 1
          ITEMS+=("$SCALAR")
          ;;
      esac
    done
  elif [ "$c" = '-' ] && [ "${S:POS+1:1}" = ' ' ]; then
    while :; do
      skip_ws
      c=${S:POS:1}
      if [ "$c" != '-' ] || [ "${S:POS+1:1}" != ' ' ]; then
        break
      fi
      POS=$((POS + 2))
      while [ "${S:POS:1}" = ' ' ] || [ "${S:POS:1}" = $'\t' ]; do
        POS=$((POS + 1))
      done
      parse_scalar block || return 1
      ITEMS+=("$SCALAR")
    done
    [ "${#ITEMS[@]}" -gt 0 ]
  else
    return 1
  fi
}

# Read the top-level `id:` scalar of FILE into RID.
read_id() {
  local f=$1 ln
  ln=$({ grep -En '^id:' -- "$f" || true; } | tail -n 1 | cut -d: -f1)
  if [ -z "$ln" ]; then
    return 1
  fi
  S=$(tail -n +"$ln" -- "$f")
  S=${S#id:}
  POS=0
  skip_ws
  parse_scalar block || return 1
  RID=$SCALAR
  [ -n "$RID" ]
}

# Read the top-level `argv:` sequence of FILE into ITEMS.
read_argv() {
  local f=$1 ln
  ln=$({ grep -En '^argv:' -- "$f" || true; } | tail -n 1 | cut -d: -f1)
  if [ -z "$ln" ]; then
    return 1
  fi
  S=$(tail -n +"$ln" -- "$f")
  S=${S#argv:}
  POS=0
  parse_argv
}

# Quote a flag as the former checker repr() did, for the missing-flag list.
quote_repr() {
  local s=$1 q="'"
  if [[ $s == *"'"* && $s != *'"'* ]]; then
    q='"'
  fi
  s=${s//\\/\\\\}
  s=${s//$'\n'/\\n}
  s=${s//$'\t'/\\t}
  s=${s//$'\r'/\\r}
  if [ "$q" = "'" ]; then
    s=${s//\'/\\\'}
  fi
  printf '%s%s%s' "$q" "$s" "$q"
}

# ---- main -----------------------------------------------------------------------
mapfile -t files < <(find "$root" -name '*.yaml' 2>/dev/null | tr '/' '\001' | sort | tr '\001' '/')

bad=0
n=0
declare -A listed_set=()
for f in "${files[@]}"; do
  if ! read_id "$f"; then
    echo "check_recipe_argv_surface.sh: $f: no top-level id" >&2
    exit 1
  fi
  NONSTR=0
  if ! read_argv "$f"; then
    echo "check_recipe_argv_surface.sh: $f: no argv sequence" >&2
    exit 1
  fi
  if [ "$NONSTR" -ne 0 ]; then
    echo "check_recipe_argv_surface.sh: $f: argv holds a non-string item (the former checker crashed on it)" >&2
    exit 1
  fi
  if [ "${ITEMS[0]}" != apr ]; then
    echo "check_recipe_argv_surface.sh: $f: argv[0] is not apr" >&2
    exit 1
  fi
  if [ "${#ITEMS[@]}" -lt 2 ]; then
    echo "check_recipe_argv_surface.sh: $f: argv has no verb" >&2
    exit 1
  fi
  verb=${ITEMS[1]}

  helptext=$("$apr" "$verb" --help 2>/dev/null || true)
  listed_set=()
  while IFS= read -r flag; do
    if [ -n "$flag" ]; then
      listed_set[$flag]=1
    fi
  done < <(printf '%s\n' "$helptext" | grep -oE -- '--[a-z][a-z0-9-]*' || true)

  missing=()
  for ((i = 2; i < ${#ITEMS[@]}; i++)); do
    item=${ITEMS[i]}
    if [[ $item == --* ]] && [ -z "${listed_set[$item]+x}" ]; then
      missing+=("$item")
    fi
  done

  n=$((n + 1))
  if [ "${#missing[@]}" -gt 0 ]; then
    bad=$((bad + 1))
    list=""
    for m in "${missing[@]}"; do
      if [ -n "$list" ]; then
        list+=", "
      fi
      list+=$(quote_repr "$m")
    done
    printf 'FAIL %s: [%s] not in `apr %s --help`\n' "$RID" "$list" "$verb"
  else
    printf 'ok   %s\n' "$RID"
  fi
done

vout=$("$apr" --version 2>/dev/null || true; printf x)
vout=${vout%x}
if [ -z "$vout" ]; then
  echo "check_recipe_argv_surface.sh: apr --version printed nothing" >&2
  exit 1
fi
ver=${vout%%$'\n'*}
printf '%d recipe(s), %d failed, against %s\n' "$n" "$bad" "$ver"

if [ "$bad" -gt 0 ] || [ "$n" -eq 0 ]; then
  exit 1
fi
exit 0
