#!/usr/bin/env bash
# check_recipe_receipts.sh -- the required receipt check (#442), Rust (#472).
# Runs src/bin/check_recipe_receipts.rs; no Python on this path. The caller's working
# directory is kept (the default --recipes/--receipts are relative to it, as in the
# Python original), so the crate is found with --manifest-path, not by cd.
set -uo pipefail
repo="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)" || exit 2
exec cargo run -q --manifest-path "$repo/Cargo.toml" --bin check_recipe_receipts -- "$@"
