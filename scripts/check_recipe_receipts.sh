#!/usr/bin/env bash
# check_recipe_receipts.sh -- the required receipt check (#442), Rust (#472).
# Runs src/bin/check_recipe_receipts.rs; no Python on this path.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.." || exit 2
exec cargo run -q --bin check_recipe_receipts -- "$@"
