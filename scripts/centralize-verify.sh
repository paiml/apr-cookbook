#!/usr/bin/env bash
# centralize-verify.sh — Verify PMAT-065/066/067 source artifacts have a destination.
#
# Per docs/specifications/centralize-cookbooks/migration-mapping.md, every source
# YAML/example/chapter/config from sovereign-ai-cookbook, alimentar, and presentar
# must exist at its declared destination in apr-cookbook.
#
# Usage:
#   ./scripts/centralize-verify.sh           # standard mode (warn on missing source repos)
#   ./scripts/centralize-verify.sh --strict  # fail if any source repo unavailable
#
# Exit codes:
#   0 — all source artifacts have destinations (or sources unavailable in non-strict mode)
#   1 — at least one source artifact missing from destination
#   2 — invocation error
#
# Added by PMAT-070 (centralize-cookbooks archive runbook precondition).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
STRICT=0
[ "${1:-}" = "--strict" ] && STRICT=1

SOVEREIGN_SRC="${SOVEREIGN_AI_COOKBOOK_SRC:-$HOME/src/sovereign-ai-cookbook}"
ALIMENTAR_SRC="${ALIMENTAR_SRC:-$HOME/src/alimentar}"
PRESENTAR_SRC="${PRESENTAR_SRC:-$HOME/src/presentar}"

failures=0
checked=0
warned=0

check_dest() {
    # $1=source-path-for-error-msg  $2=destination-path-relative-to-REPO_ROOT
    local src="$1" dest="$2"
    checked=$((checked + 1))
    if [ ! -e "$REPO_ROOT/$dest" ]; then
        printf 'MISSING: %s -> %s (destination does not exist)\n' "$src" "$dest" >&2
        failures=$((failures + 1))
    fi
}

check_source_repo() {
    local repo_path="$1" repo_name="$2"
    if [ ! -d "$repo_path" ]; then
        if [ "$STRICT" = "1" ]; then
            printf 'STRICT: source repo %s not available at %s\n' "$repo_name" "$repo_path" >&2
            failures=$((failures + 1))
            return 1
        else
            printf 'WARN: source repo %s not available at %s (skipping)\n' "$repo_name" "$repo_path" >&2
            warned=$((warned + 1))
            return 1
        fi
    fi
    return 0
}

# ── sovereign-ai-cookbook (PMAT-065) ─────────────────────────
if check_source_repo "$SOVEREIGN_SRC" sovereign-ai-cookbook; then
    # 14 recipes -> examples/deployment-stacks/recipes/ + Rust wrappers
    for yaml in "$SOVEREIGN_SRC"/recipes/*.yaml; do
        base="$(basename "$yaml" .yaml)"
        check_dest "sovereign/recipes/$base.yaml" "examples/deployment-stacks/recipes/$base.yaml"
        wrapper_name="${base//-/_}"
        check_dest "sovereign/recipes/$base.yaml [wrapper]" "examples/deployment-stacks/$wrapper_name.rs"
    done
    # 10 stacks -> examples/deployment-stacks/stacks/ (with 09-qwen-coder -> 10-qwen-coder)
    for stack_dir in "$SOVEREIGN_SRC"/stacks/*/; do
        stack_name="$(basename "$stack_dir")"
        # Apply rename per migration-mapping.md
        if [ "$stack_name" = "09-qwen-coder" ]; then
            stack_name="10-qwen-coder"
        fi
        check_dest "sovereign/stacks/$(basename "$stack_dir")/" "examples/deployment-stacks/stacks/$stack_name"
    done
    # Jetson machine config
    if [ -d "$SOVEREIGN_SRC/machines/jetson" ]; then
        check_dest "sovereign/machines/jetson/" "examples/machines/jetson"
    fi
fi

# ── alimentar (PMAT-066) ─────────────────────────────────────
if check_source_repo "$ALIMENTAR_SRC" alimentar; then
    # 18 examples -> examples/data-loading/ (verbatim filenames; IIUR retrofit applied)
    for rs in "$ALIMENTAR_SRC"/examples/*.rs; do
        base="$(basename "$rs")"
        check_dest "alimentar/examples/$base" "examples/data-loading/$base"
    done
    # Book chapters (excluding development/ and ecosystem/ per spec)
    for sub in 100-examples appendix architecture backends cli dataloader dataset datasets hf-hub transforms; do
        if [ -d "$ALIMENTAR_SRC/book/src/$sub" ]; then
            check_dest "alimentar/book/src/$sub/" "book/src/data-loading/$sub"
        fi
    done
fi

# ── presentar (PMAT-067) ─────────────────────────────────────
if check_source_repo "$PRESENTAR_SRC" presentar; then
    # 28 declarative configs across 6 subdirs
    for sub in ald apr charts dashboards edge_cases prs; do
        if [ -d "$PRESENTAR_SRC/examples/$sub" ]; then
            for cfg in "$PRESENTAR_SRC"/examples/"$sub"/*; do
                base="$(basename "$cfg")"
                check_dest "presentar/examples/$sub/$base" "examples/visualization/$sub/$base"
            done
        fi
    done
    # The single Rust validator wrapper
    check_dest "[wrapper] visualization validator" "examples/visualization/load_visualization.rs"
    # Book chapters (excluding development/ and ecosystem/)
    for sub in advanced appendix architecture examples getting-started layout quality; do
        if [ -d "$PRESENTAR_SRC/book/src/$sub" ]; then
            check_dest "presentar/book/src/$sub/" "book/src/visualization/$sub"
        fi
    done
fi

# ── Summary ──────────────────────────────────────────────────
printf '\n=== centralize-verify summary ===\n'
printf '  checked:  %d artifacts\n' "$checked"
printf '  warnings: %d (source repo unavailable)\n' "$warned"
printf '  failures: %d (destination missing)\n' "$failures"

if [ "$failures" -eq 0 ]; then
    printf '\n  every source artifact has a destination\n'
    exit 0
else
    printf '\n  %d artifacts missing from destination\n' "$failures" >&2
    exit 1
fi
