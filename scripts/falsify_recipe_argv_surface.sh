#!/usr/bin/env bash
# Falsifier for check_recipe_argv_surface.sh (Refs apr-cookbook#495, #473).
# Each tests/fixtures/argv-surface/refuse/*.yaml must exit 1; each accept/*.yaml must exit 0.
# Exit 2 (could not measure) or any other code is a failure of the falsifier itself.
# Usage: falsify_recipe_argv_surface.sh [apr-binary]   (default: the fake apr fixture)
set -euo pipefail
here=$(cd "$(dirname "$0")/.." && pwd)
fx=$here/tests/fixtures/argv-surface
apr=${1:-$fx/fake-apr.sh}
tmp=$(mktemp -d)
trap 'rm -rf "${tmp:?}"' EXIT
fail=0
run_one() { # $1 file $2 want
  rm -rf "${tmp:?}/d" && mkdir "$tmp/d" && cp "$1" "$tmp/d/r.yaml"
  local rc=0
  bash "$here/scripts/check_recipe_argv_surface.sh" "$apr" "$tmp/d" >/dev/null 2>&1 || rc=$?
  if [ "$rc" -ne "$2" ]; then
    echo "FALSIFIER FAIL: $(basename "$1") exit $rc, want $2" >&2
    fail=$((fail + 1))
  fi
}
for f in "$fx"/refuse/*.yaml; do run_one "$f" 1; done
for f in "$fx"/accept/*.yaml; do run_one "$f" 0; done
if [ "$fail" -ne 0 ]; then echo "falsifier: $fail failure(s)" >&2; exit 1; fi
echo "falsifier: all refuse fixtures exit 1, all accept fixtures exit 0"
