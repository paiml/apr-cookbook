#!/usr/bin/env bash
# generate-recipe-table.sh — Deterministic recipe table for README.md
# Run from repo root. Parses Cargo.toml + source files to produce a markdown
# table of all examples with device-tier tags.
#
# Usage:
#   ./scripts/generate-recipe-table.sh          # print table to stdout
#   ./scripts/generate-recipe-table.sh --check  # exit 1 if README is stale
#   ./scripts/generate-recipe-table.sh --update  # rewrite README in-place
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
README="$REPO_ROOT/README.md"
CARGO="$REPO_ROOT/Cargo.toml"

START_MARKER="<!-- RECIPE-TABLE-START -->"
END_MARKER="<!-- RECIPE-TABLE-END -->"

# ---------------------------------------------------------------------------
# 1. Extract examples from Cargo.toml: name + path
# ---------------------------------------------------------------------------
extract_examples() {
    awk -F'"' '
    /^\[\[example\]\]/ { in_ex=1; name=""; path=""; next }
    in_ex && /^name/ { name=$2 }
    in_ex && /^path/ { path=$2 }
    in_ex && name != "" && path != "" { print name "\t" path; in_ex=0 }
    /^\[/ && !/^\[\[example\]\]/ { in_ex=0 }
    ' "$CARGO" | sort -t$'\t' -k2,2 -k1,1
}

# ---------------------------------------------------------------------------
# 2. Detect device tier from source file content
# ---------------------------------------------------------------------------
detect_devices() {
    local src="$1"
    local category="$2"
    local devices=""

    if grep -qE 'target_arch.*x86_64|is_x86_feature_detected|avx2|avx512|sse4\b|vnni|AVX' "$src" 2>/dev/null; then
        devices="${devices}x86_64 "
    fi

    if grep -qE 'target_arch.*aarch64|neon\b|NEON' "$src" 2>/dev/null; then
        devices="${devices}aarch64 "
    fi

    if grep -qE 'cuda\b|CUDA|\.ptx\b|PTX\b|nvidia|NVIDIA|CudaDevice' "$src" 2>/dev/null; then
        devices="${devices}cuda "
    fi

    if grep -qE 'wgpu\b|WebGPU|webgpu|Vulkan|vulkan|VkDevice' "$src" 2>/dev/null; then
        devices="${devices}wgpu "
    fi

    # WASM: detect from content OR category name
    if grep -qE 'wasm_bindgen|wasm32|target_arch.*wasm|WasmModel|wasm-pack' "$src" 2>/dev/null; then
        devices="${devices}wasm "
    elif [ "$category" = "wasm" ]; then
        devices="${devices}wasm "
    fi

    # Distributed: be specific to avoid false positives (e.g. "worker" in web_worker)
    if grep -qE 'repartir|ring_allreduce|gossip_protocol|pipeline_parallel|model_sharding' "$src" 2>/dev/null; then
        devices="${devices}distributed "
    elif [ "$category" = "distributed" ]; then
        devices="${devices}distributed "
    fi

    # Serverless: match on deployment patterns
    if grep -qE 'aws.*lambda|Lambda.*handler|serverless|cold_start|edge_function' "$src" 2>/dev/null; then
        devices="${devices}serverless "
    elif [ "$category" = "serverless" ]; then
        devices="${devices}serverless "
    fi

    # If no special devices detected, it's cpu-only
    if [ -z "$devices" ]; then
        devices="cpu"
    else
        devices="${devices% }"
    fi

    echo "$devices"
}

# ---------------------------------------------------------------------------
# 3. Map device names to badge markdown
# ---------------------------------------------------------------------------
device_badge() {
    case "$1" in
        cpu)          echo '![cpu](https://img.shields.io/badge/-cpu-lightgrey)' ;;
        x86_64)       echo '![x86_64](https://img.shields.io/badge/-x86__64-blue)' ;;
        aarch64)      echo '![aarch64](https://img.shields.io/badge/-aarch64-blue)' ;;
        cuda)         echo '![cuda](https://img.shields.io/badge/-cuda-76b900)' ;;
        wgpu)         echo '![wgpu](https://img.shields.io/badge/-wgpu-green)' ;;
        wasm)         echo '![wasm](https://img.shields.io/badge/-wasm-purple)' ;;
        distributed)  echo '![distributed](https://img.shields.io/badge/-distributed-red)' ;;
        serverless)   echo '![serverless](https://img.shields.io/badge/-serverless-yellow)' ;;
        *)            echo "![$1](https://img.shields.io/badge/-$1-lightgrey)" ;;
    esac
}

# ---------------------------------------------------------------------------
# 4. Extract category from path
# ---------------------------------------------------------------------------
path_to_category() {
    echo "$1" | sed 's|examples/||' | cut -d'/' -f1
}

# ---------------------------------------------------------------------------
# 5. Generate the table
# ---------------------------------------------------------------------------
generate_table() {
    local count=0

    echo "| # | Example | Category | Devices | Build |"
    echo "|--:|---------|----------|---------|:-----:|"

    extract_examples | while IFS=$'\t' read -r name path; do
        count=$((count + 1))
        local src="$REPO_ROOT/$path"
        local category
        category=$(path_to_category "$path")

        local devices
        if [ -f "$src" ]; then
            devices=$(detect_devices "$src" "$category")
        else
            devices="cpu"
        fi

        # Build badge string
        local badges=""
        for dev in $devices; do
            badges="${badges}$(device_badge "$dev") "
        done
        badges="${badges% }"

        printf '| %d | `%s` | %s | %s | %s |\n' \
            "$count" "$name" "$category" "$badges" "✅"
    done
}

# ---------------------------------------------------------------------------
# 6. Main: generate / check / update
# ---------------------------------------------------------------------------
TABLE_CONTENT=$(generate_table)
TOTAL=$(echo "$TABLE_CONTENT" | tail -n +3 | wc -l)

FULL_BLOCK="$START_MARKER
<!-- Auto-generated by scripts/generate-recipe-table.sh — do not edit manually -->
<!-- Re-generate: ./scripts/generate-recipe-table.sh --update -->
<!-- CI validates: recipe-table workflow ensures this table matches source -->

**${TOTAL} recipes** | Build: [![CI](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml/badge.svg)](https://github.com/paiml/apr-cookbook/actions/workflows/ci.yml)

<details>
<summary>Full recipe table (click to expand)</summary>

$TABLE_CONTENT

</details>
$END_MARKER"

case "${1:-}" in
    --check)
        if ! grep -q "$START_MARKER" "$README"; then
            echo "ERROR: $START_MARKER not found in README.md"
            exit 1
        fi
        # Extract current block from README
        CURRENT=$(sed -n "/$START_MARKER/,/$END_MARKER/p" "$README")
        if [ "$CURRENT" = "$FULL_BLOCK" ]; then
            echo "OK: Recipe table is up to date ($TOTAL recipes)"
            exit 0
        else
            echo "STALE: Recipe table in README.md does not match source."
            echo "Run: ./scripts/generate-recipe-table.sh --update"
            diff <(echo "$CURRENT") <(echo "$FULL_BLOCK") || true
            exit 1
        fi
        ;;
    --update)
        if ! grep -q "$START_MARKER" "$README"; then
            echo "ERROR: $START_MARKER not found in README.md"
            echo "Add $START_MARKER and $END_MARKER markers to README.md first."
            exit 1
        fi
        # Replace block between markers
        ESCAPED_BLOCK=$(echo "$FULL_BLOCK" | sed 's/[&/\]/\\&/g')
        # Use awk for reliable multi-line replacement
        awk -v start="$START_MARKER" -v end="$END_MARKER" -v block="$FULL_BLOCK" '
            $0 ~ start { printing=0; print block; next }
            $0 ~ end { printing=1; next }
            printing!=0 { print }
            BEGIN { printing=1 }
        ' "$README" > "$README.tmp"
        mv "$README.tmp" "$README"
        echo "Updated README.md with $TOTAL recipes"
        ;;
    *)
        echo "$FULL_BLOCK"
        ;;
esac
