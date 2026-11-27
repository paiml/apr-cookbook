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
WHY=""    # reason a scan failed closed
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
  [[ $v =~ ^[0-9]{4}-[0-9]{1,2}-[0-9]{1,2}($|[Tt\ \t]) ]] && return 0
  [[ $v =~ ^[-+]?[0-9][0-9_]*(:[0-5]?[0-9])+(\.[0-9_]*)?$ ]] && return 0
  [[ $v =~ ^[-+]0x ]] && return 0
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

# True for a character that may follow a block `-` indicator: space, tab or line end.
is_blank_or_eol() {
  case $1 in
    '' | ' ' | $'\t' | $'\n' | $'\r') return 0 ;;
  esac
  return 1
}

# Decode a `\x`, `\u` or `\U` escape of $1 hex digits at POS into `out` (the caller's
# local). ASCII is decoded; anything else (and NUL) becomes U+FFFD, which no flag matches.
decode_hex() {
  local n=$1 h cp
  h=${S:POS:n}
  if [ "${#h}" -ne "$n" ] || [[ ! $h =~ ^[0-9a-fA-F]+$ ]]; then
    return 1
  fi
  POS=$((POS + n))
  cp=$((16#$h))
  if [ "$cp" -gt 1114111 ]; then
    unsupported "escape beyond U+10FFFF"
    return 1
  fi
  if [ "$cp" -ge 1 ] && [ "$cp" -lt 128 ]; then
    printf -v h '%b' "$(printf '\\x%02x' "$cp")"
    out+=$h
  else
    out+=$'\357\277\275'
  fi
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
      if [ "$c" = $'\n' ]; then
        # A line break inside a quoted scalar folds in YAML; not supported: fail closed.
        return 1
      fi
      if [ "$c" = '\' ]; then
        esc=${S:POS:1}
        POS=$((POS + 1))
        case $esc in
          n) out+=$'\n' ;;
          t | $'\t') out+=$'\t' ;;
          r) out+=$'\r' ;;
          a) out+=$'\a' ;;
          b) out+=$'\b' ;;
          e) out+=$'\e' ;;
          f) out+=$'\f' ;;
          v) out+=$'\v' ;;
          ' ' | '"' | '/' | '\') out+=$esc ;;
          0 | N | _ | L | P) out+=$'\357\277\275' ;;
          x) decode_hex 2 || return 1 ;;
          u) decode_hex 4 || return 1 ;;
          U) decode_hex 8 || return 1 ;;
          *) return 1 ;;
        esac
      else
        out+=$c
      fi
    done
  elif [ "$c" = "'" ]; then
    POS=$((POS + 1))
    while :; do
      c=${S:POS:1}
      if [ -z "$c" ] || [ "$c" = $'\n' ]; then
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

# Fail closed: record the unsupported construct in WHY and return 1.
unsupported() {
  WHY="unsupported YAML construct: $1"
  return 1
}

# True when the previous character is a space or tab (so a `#` there starts a comment).
prev_is_blank() {
  [ "${S:POS-1:1}" = ' ' ] || [ "${S:POS-1:1}" = $'\t' ]
}

# Fail closed on an unquoted argv item that is not a plain string scalar: a tag, anchor,
# alias, directive, nested collection or mapping. $1 is the item's first character and
# $2 its parsed text.
check_plain_item() {
  case $1 in
    '!') unsupported "tag (!) on an argv item"; return 1 ;;
    '&') unsupported "anchor (&) on an argv item"; return 1 ;;
    '*') unsupported "alias (*) as an argv item"; return 1 ;;
    '[' | '{') unsupported "nested collection as an argv item"; return 1 ;;
    '?' | '%' | '@' | '`') unsupported "indicator $1 starting an argv item"; return 1 ;;
    ':') unsupported "mapping indicator (:) starting an argv item"; return 1 ;;
    '-') if is_blank_or_eol "${2:1:1}"; then unsupported "nested sequence as an argv item"; return 1; fi ;;
  esac
  if [[ $2 == *": "* || $2 == *":" || $2 == *":"$'\t'* ]]; then
    unsupported "mapping (key: value) as an argv item"
    return 1
  fi
}

# After the sequence: only blanks, comments and line breaks may follow before the end of
# the text or a new column-0 key; anything else is a continuation this scanner skips.
check_seq_end() {
  skip_ws
  if [ -n "${S:POS:1}" ] && [ "${S:POS-1:1}" != $'\n' ]; then
    unsupported "text after or inside the argv sequence"
  fi
}

# Column (0-based) of POS within its line; fails when POS is on the `argv:` line itself.
col_of() {
  local i=$POS
  COL=0
  while [ "$i" -gt 0 ]; do
    i=$((i - 1))
    if [ "${S:i:1}" = $'\n' ]; then
      return 0
    fi
    COL=$((COL + 1))
  done
  return 1
}

# Parse the argv sequence at POS into ITEMS. Flow `[a, b]` or block `- a` lines.
parse_argv() {
  local c start first expect=item dash_col=-1
  ITEMS=()
  skip_ws
  c=${S:POS:1}
  if [ "$c" = '[' ]; then
    POS=$((POS + 1))
    while :; do
      skip_ws
      c=${S:POS:1}
      case $c in
        '') unsupported "unterminated flow sequence" || return 1 ;;
        ']')
          POS=$((POS + 1))
          check_seq_end
          return
          ;;
        ',')
          [ "$expect" = sep ] || unsupported "empty entry in flow sequence" || return 1
          expect=item
          POS=$((POS + 1))
          ;;
        '[' | '{' | '}') unsupported "nested collection or stray $c in flow sequence" || return 1 ;;
        *)
          [ "$expect" = item ] || unsupported "flow scalar continued over lines or without a comma" || return 1
          first=$c
          start=$POS
          parse_scalar flow || return 1
          # A scalar that consumed nothing would loop forever: fail closed.
          [ "$POS" -gt "$start" ] || unsupported "unreadable flow item" || return 1
          case $first in '"' | "'") ;; *) check_plain_item "$first" "$SCALAR" || return 1 ;; esac
          ITEMS+=("$SCALAR")
          expect=sep
          ;;
      esac
          done
  elif [ "$c" = '-' ] && is_blank_or_eol "${S:POS+1:1}"; then
    while :; do
      skip_ws
      c=${S:POS:1}
      if [ "$c" != '-' ] || ! is_blank_or_eol "${S:POS+1:1}"; then
        break
      fi
      col_of || unsupported "block sequence on the argv: line" || return 1
      if [ "$dash_col" -lt 0 ]; then
        dash_col=$COL
      elif [ "$COL" -ne "$dash_col" ]; then
        unsupported "block item indented differently from the first (wrapped scalar or nesting)" || return 1
      fi
      POS=$((POS + 1))
      while [ "${S:POS:1}" = ' ' ] || [ "${S:POS:1}" = $'\t' ]; do
        POS=$((POS + 1))
      done
      c=${S:POS:1}
      # A null item (`-` then a line break), a block scalar (`|`, `>`) or a comment-only
      # item is a form this scanner does not read: fail closed rather than skip it.
      case $c in
        '' | $'\n' | $'\r' | '#' | '|' | '>') unsupported "null item, comment-only item or block scalar in argv" || return 1 ;;
      esac
      first=$c
      parse_scalar block || return 1
      case $first in '"' | "'") ;; *) check_plain_item "$first" "$SCALAR" || return 1 ;; esac
      ITEMS+=("$SCALAR")
      # Nothing but a comment may follow the item on its line (no wrapped scalars).
      while [ "${S:POS:1}" = ' ' ] || [ "${S:POS:1}" = $'\t' ]; do
        POS=$((POS + 1))
      done
      c=${S:POS:1}
      case $c in
        '' | $'\n' | $'\r') ;;
        '#') prev_is_blank || unsupported "# not preceded by a blank" || return 1 ;;
        *) unsupported "text after a quoted argv item" || return 1 ;;
      esac
    done
    check_seq_end || return 1
    [ "${#ITEMS[@]}" -gt 0 ]
  else
    unsupported "argv is neither a flow nor a block sequence"
  fi
}

# Print the line number of the one column-0 `$2:` key of FILE. Fails closed on none, on a
# second one (it could sit inside a multi-line quoted scalar) and on a multi-document file.
single_key_line() {
  local f=$1 key=$2 lines count
  if grep -Eq '^(---|\.\.\.)' -- "$f"; then
    WHY="unsupported YAML construct: document marker (multi-document file)"
    return 1
  fi
  lines=$({ grep -En "^$key:" -- "$f" || true; } | cut -d: -f1)
  count=$(printf '%s' "$lines" | grep -c . || true)
  if [ "$count" -ne 1 ]; then
    WHY="expected exactly one top-level $key: key, found $count"
    return 1
  fi
  printf '%s\n' "$lines"
}

# Read the top-level `id:` scalar of FILE into RID.
read_id() {
  local f=$1 ln
  ln=$(single_key_line "$f" id) || return 1
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
  ln=$(single_key_line "$f" argv) || return 1
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
    echo "check_recipe_argv_surface.sh: $f: ${WHY:-no top-level id}" >&2
    exit 1
  fi
  NONSTR=0
  WHY=""
  if ! read_argv "$f"; then
    echo "check_recipe_argv_surface.sh: $f: ${WHY:-no argv sequence}" >&2
    exit 1
  fi
  if [ "$NONSTR" -ne 0 ]; then
    echo "check_recipe_argv_surface.sh: $f: argv holds a non-string item (the former checker crashed on it)" >&2
    exit 1
  fi
  if [ "${#ITEMS[@]}" -lt 1 ] || [ "${ITEMS[0]}" != apr ]; then
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
