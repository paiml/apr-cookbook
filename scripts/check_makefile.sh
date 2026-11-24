#!/usr/bin/env bash
# check_makefile.sh -- the Makefile must not report work it did not do.
#
# Four defects were measured on main at d0de8d6 (2026-11-24); each has a check here:
#   1. `.PHONY: release install` with no rule: `make release` printed "Nothing to be done"
#      and exited 0.                                   -> PHONY-NO-RULE / RULE-NOT-PHONY
#   2. `for proj in $(SUB_PROJECTS)` with the variable never set: `make sub-test` looped over
#      nothing and printed "All sub-projects tests passed!", exit 0.   -> UNDEFINED-VAR
#   3. `MAKEFLAGS += -j$(nproc)` over ordered prerequisite lists: format and lint --fix ran
#      beside check and test.                          -> FORCED-JOBS / NOT-SERIAL
#   4. quality-gate counted tests with `[ "$TEST_COUNT" -lt 50 ]` where TEST_COUNT held one
#      number PER test binary, so `[` failed with "integer expression expected" and the
#      else-branch printed "Test count acceptable"; a low count only warned.  -> --test-count
#
# Usage:
#   scripts/check_makefile.sh [MAKEFILE]        check MAKEFILE (default: Makefile)
#   scripts/check_makefile.sh --test-count N    read `cargo test` output on stdin; exit 1
#                                               unless the passed tests sum to at least N
# Exit: 0 clean | 1 violation | 2 usage
set -euo pipefail

if [ "${1:-}" = "--test-count" ]; then
  min=${2:-}
  case "$min" in '' | *[!0-9]*) echo "usage: $0 --test-count N" >&2; exit 2 ;; esac
  # "test result: ok. 12 passed; 0 failed; ..." -- one line per test binary; sum them.
  read -r results passed < <(awk '
    /^test result:/ { n++; for (i = 2; i <= NF; i++) if ($i == "passed;") s += $(i - 1) }
    END { print n + 0, s + 0 }')
  echo "  Tests: $passed passed over $results test binaries (minimum $min)"
  if [ "$results" -eq 0 ]; then
    echo "  FAIL: no 'test result:' line on stdin -- nothing was counted"
    exit 1
  fi
  if [ "$passed" -lt "$min" ]; then
    echo "  FAIL: test count $passed is below the minimum $min"
    exit 1
  fi
  echo "  ✓ Test count acceptable"
  exit 0
fi

mk=${1:-Makefile}
[ -f "$mk" ] || { echo "no such Makefile: $mk" >&2; exit 2; }
dir=$(dirname "$mk")
# A caller's -j or MAKEFLAGS must not leak into what we measure.
run_make() { env -u MAKEFLAGS -u MFLAGS -u MAKELEVEL make -C "$dir" -f "$(basename "$mk")" "$@"; }

fail=0
phony=$(sed -n 's/^\.PHONY:[[:space:]]*//p' "$mk" | tr -s '[:space:]' '\n' | sed '/^$/d' | sort -u)
rules=$(sed -n 's/^\([A-Za-z0-9_][A-Za-z0-9_.-]*\):\([^=].*\)\{0,1\}$/\1/p' "$mk" | sort -u)

for t in $(comm -23 <(echo "$phony") <(echo "$rules")); do
  echo "PHONY-NO-RULE   $t: listed in .PHONY but has no rule (make $t does nothing, exit 0)"
  fail=$((fail + 1))
done
for t in $(comm -13 <(echo "$phony") <(echo "$rules")); do
  echo "RULE-NOT-PHONY  $t: has a rule but is not in .PHONY"
  fail=$((fail + 1))
done

# Recipe variables expand only when the recipe is chosen, so dry-run every rule.
checked=0
for t in $rules; do
  checked=$((checked + 1))
  out=$(run_make --warn-undefined-variables -n "$t" 2>&1 || true)
  while IFS= read -r line; do
    [ -n "$line" ] || continue
    echo "UNDEFINED-VAR   $t: $line"
    fail=$((fail + 1))
  done < <(printf '%s\n' "$out" | grep -E "warning: undefined variable" | sort -u || true)
done

flags=$(run_make -p -n help 2>/dev/null | sed -n 's/^MAKEFLAGS = //p' | head -n 1)
case " $flags " in
  *" -j"*)
    echo "FORCED-JOBS     the Makefile forces parallel jobs (MAKEFLAGS = $flags)"
    fail=$((fail + 1))
    ;;
esac
if ! grep -qE '^\.NOTPARALLEL:[[:space:]]*$' "$mk"; then
  echo "NOT-SERIAL      no bare .NOTPARALLEL: -- 'make -j validate' would run the ordered steps at once"
  fail=$((fail + 1))
fi

nphony=$(echo "$phony" | sed '/^$/d' | wc -l)
echo "checked $mk: $checked rule(s), $nphony .PHONY name(s), $fail violation(s)"
if [ "$checked" -eq 0 ]; then
  echo "FAIL: 0 rules found -- refusing to report green over nothing"
  exit 1
fi
[ "$fail" -eq 0 ]
