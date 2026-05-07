#!/usr/bin/env bash
# architecture-demos-gen.sh — reconcile manifest with on-disk recipes/contracts.
# Modes: --check (CI gate, read-only), --update (write stubs), --diff (preview).
# Targets: recipes, contracts, cargo, coverage-matrix, summary, all (default).
#
# See docs/specifications/architecture-demos/generator.md.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/docs/specifications/architecture-demos/manifest.yaml"
SCHEMA="$ROOT/docs/specifications/architecture-demos/manifest.schema.yaml"
COVERAGE_MATRIX="$ROOT/docs/specifications/architecture-demos/coverage-matrix.md"
CONTRACTS_DIR="$ROOT/contracts"
RECIPES_DIR="$ROOT/examples/inference"
CARGO_TOML="$ROOT/Cargo.toml"
CARGO_SECTION_HEADER="# --- architecture-demos (PMAT-300+) ---"

MODE="${1:-}"
TARGET="all"
shift || true
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target) TARGET="$2"; shift 2 ;;
        *) shift ;;
    esac
done

if [[ -z "$MODE" ]]; then
    echo "usage: $0 {--check|--update|--diff} [--target {recipes,contracts,cargo,coverage-matrix,summary,all}]" >&2
    exit 1
fi

# Output buffers — for --diff and --check we collect findings, for --update we apply them.
DRIFT=()

if ! command -v yq >/dev/null 2>&1; then
    echo "ERROR: yq is required (https://github.com/mikefarah/yq)" >&2
    exit 3
fi

# ---------- Manifest readers ----------
families_with_status() {
    local status="$1"
    yq -r ".families[] | select(.status == \"$status\") | .name" "$MANIFEST"
}

field_for() {
    local name="$1"; local field="$2"
    yq -r ".families[] | select(.name == \"$name\") | .$field // \"\"" "$MANIFEST"
}

# ---------- Recipe stub generator ----------
emit_recipe_stub() {
    local family="$1"
    local recipe_path="$2"
    local citation
    citation="$(field_for "$family" citation)"
    local target="$ROOT/$recipe_path"
    if [[ -e "$target" ]]; then
        return 0
    fi
    mkdir -p "$(dirname "$target")"
    cat > "$target" <<EOF
//! # ${family^} Smoke Inference
//!
//! Load a synthetic ${family} micro-checkpoint via \`aprender::rosetta\`,
//! run a deterministic forward pass, emit a \`Verdict::Ok\` value with
//! the resulting logits checksum.
//!
//! Demonstrates the **${family^^}.smoke** recipe per
//! \`docs/specifications/architecture-demos.md\`.
//!
//! IIUR Contract: contracts/recipe-iiur-v1.yaml
//! Provable-contract: contracts/inference-${family//_/-}-smoke-v1.yaml (grade A; lean_status: wip)
//! Citation: ${citation}
//!
//! Run with: cargo run --example inference_${family}_smoke
//!
//! Added by PMAT-300+ (architecture-demos: ${family} coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SmokeVerdict {
    Ok {
        family: String,
        format: String,
        logits_checksum: u32,
        layer_count: u32,
    },
    LoaderUnavailable {
        reason: String,
    },
    InvalidFixture,
}

pub fn smoke(fixture_path: &str, format: &str) -> SmokeVerdict {
    if !std::path::Path::new(fixture_path).exists() {
        return SmokeVerdict::InvalidFixture;
    }
    // todo!() — replace with actual aprender::rosetta::load_family call when fixture lands.
    SmokeVerdict::LoaderUnavailable {
        reason: format!("loader call for ${family} not yet wired (format={format})"),
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("inference_${family}_smoke")?;
    let fixture = "tests/fixtures/architectures/${family}/model.safetensors";
    println!("safetensors: {:?}", smoke(fixture, "safetensors"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn smoke_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_fixture_returns_invalid() {
        assert_eq!(smoke("/no/such/path", "safetensors"), SmokeVerdict::InvalidFixture);
    }

    #[test]
    fn loader_unavailable_when_path_exists_but_loader_unwired() {
        // Until the real loader is wired, an existing-but-not-loadable file
        // surfaces LoaderUnavailable. After the loader lands, this test flips
        // to assert SmokeVerdict::Ok { family: "$family", .. }.
        let v = smoke("/dev/null", "safetensors");
        assert!(matches!(v, SmokeVerdict::InvalidFixture | SmokeVerdict::LoaderUnavailable { .. }));
    }
}
EOF
}

# ---------- Provable-contract stub generator ----------
emit_contract_stub() {
    local family="$1"
    local contract_path="$2"
    local target="$ROOT/$contract_path"
    if [[ -e "$target" ]]; then
        return 0
    fi
    local family_pascal
    family_pascal="$(echo "$family" | sed -E 's/(^|_)([a-z])/\U\2/g')"
    mkdir -p "$(dirname "$target")"
    local today
    today="$(date +%Y-%m-%d)"
    cat > "$target" <<EOF
metadata:
  version: "1.0.0"
  created: "$today"
  author: "PAIML Engineering"
  description: "${family_pascal} smoke inference invariants — load, forward, determinism"
  references:
    - "$(field_for "$family" citation)"
    - "docs/specifications/architecture-demos.md"
  depends_on:
    - "recipe-iiur-v1"
  tags:
    - architecture-demos
    - inference
    - smoke
    - $family

kernel_structure:
  phases:
    - name: setup
      description: "Construct RecipeContext; verify fixture path exists"
      invariant: "context.temp_dir exists; fixture path is readable"
    - name: load
      description: "Dispatch aprender::rosetta loader for the $family family"
      invariant: "load_family('$family', fixture, fmt) returns Ok(Model)"
    - name: forward
      description: "Run a seeded forward pass through the loaded model"
      invariant: "model.forward_smoke(&[42; 4]) returns logits of expected shape"
    - name: verify
      description: "Compare verdicts across two consecutive smoke() calls"
      invariant: "smoke(f, fmt) == smoke(f, fmt) byte-for-byte"
    - name: teardown
      description: "Drop RecipeContext; verify temp_dir cleanup"
      invariant: "!context.temp_dir.exists() after Drop"

equations:
  loader_dispatch:
    formula: "load_family(family, fixture, fmt) ↦ Ok(Model)"
    domain: "family ∈ supported_families; fixture: Path; fmt ∈ {safetensors, apr, gguf}"
    codomain: "Result<Model, LoaderError>"
    invariants:
      - "Dispatch is deterministic in (family, fixture, fmt)"
      - "Returned Model.family() == family"
    preconditions:
      - "Path::new(fixture).exists()"
      - "fmt ∈ {safetensors, apr, gguf}"
    postconditions:
      - "Model.family() == '$family'"
      - "Model.layer_count() == 2"
    lean_theorem: "Theorems.${family_pascal}.LoaderDispatch"
    tolerance: 0
    lean:
      theorem: "Theorems.${family_pascal}.LoaderDispatch"
      status: wip
      module: "lean/Theorems/${family_pascal}.lean"

  tensor_validation:
    formula: "validate_tensor_names(model, expected) ↦ Ok(())"
    domain: "model: Model; expected: Vec<String>"
    codomain: "Result<(), ContractError>"
    invariants:
      - "All expected tensor names are present in the model"
      - "No unexpected tensor names appear"
    preconditions:
      - "model.layer_count() > 0"
      - "expected.len() > 0"
    postconditions:
      - "expected ⊆ model.tensor_names()"
    lean_theorem: "Theorems.${family_pascal}.TensorValidation"
    tolerance: 0
    lean:
      theorem: "Theorems.${family_pascal}.TensorValidation"
      status: wip
      module: "lean/Theorems/${family_pascal}.lean"

  forward_determinism:
    formula: "smoke(f, fmt) == smoke(f, fmt)"
    domain: "f: Path; fmt: String"
    codomain: "SmokeVerdict"
    invariants:
      - "Two calls with identical inputs produce equal verdicts"
    preconditions:
      - "Path::new(f).exists()"
      - "rng_seed_for_smoke == 42"
    postconditions:
      - "logits_checksum_run_1 == logits_checksum_run_2"
    lean_theorem: "Theorems.${family_pascal}.ForwardDeterminism"
    tolerance: 0
    lean:
      theorem: "Theorems.${family_pascal}.ForwardDeterminism"
      status: wip
      module: "lean/Theorems/${family_pascal}.lean"

falsification_tests:
  - id: FALSIFY-${family^^}-001
    rule: "Loader dispatches and returns correct family identifier"
    prediction: "load_family('$family', fixture, 'safetensors') ↦ Ok with .family() == '$family'"
    test: "Run smoke('tests/fixtures/architectures/$family/model.safetensors', 'safetensors') and assert SmokeVerdict::Ok with matching family"
    if_fails: "Upstream rosetta loader misroutes or returns a different family identifier"
  - id: FALSIFY-${family^^}-002
    rule: "Forward pass is deterministic across two consecutive calls"
    prediction: "Identical inputs produce identical SmokeVerdict outputs"
    test: "Call smoke twice on the same fixture and assert byte-equal verdicts"
    if_fails: "Non-determinism leaked from RNG, threading, or uninitialized memory"
  - id: FALSIFY-${family^^}-003
    rule: "Missing fixture path returns InvalidFixture, not panic"
    prediction: "smoke('/no/such/path', 'safetensors') ↦ SmokeVerdict::InvalidFixture"
    test: "Pass a non-existent path and assert InvalidFixture variant"
    if_fails: "Loader panics on missing input instead of returning a clean verdict"

kani_harnesses:
  - id: KANI-${family^^}-001
    obligation: "Deterministic verdict on equal inputs"
    bound: 4
    strategy: bounded_int
    harness: "kani_harnesses::${family}_smoke_deterministic"
  - id: KANI-${family^^}-002
    obligation: "InvalidFixture is total (no panic on bad path)"
    bound: 4
    strategy: bounded_int
    harness: "kani_harnesses::${family}_smoke_invalid_fixture_total"

proof_obligations:
  - type: invariant
    property: "Loader dispatch returns Ok for ${family}"
    formal: "load_family('$family', fixture, fmt) ↦ Ok(Model) where Model.family() == '$family'"
    tolerance: 0.0
    applies_to: loader_dispatch
    lean:
      theorem: "Theorems.${family_pascal}.LoaderDispatch"
      module: "ProvableContracts.ArchitectureDemos.${family_pascal}"
  - type: invariant
    property: "Tensor names match expected layout"
    formal: "validate_tensor_names(model, expected) ↦ Ok(())"
    tolerance: 0.0
    applies_to: tensor_validation
    lean:
      theorem: "Theorems.${family_pascal}.TensorValidation"
      module: "ProvableContracts.ArchitectureDemos.${family_pascal}"
  - type: invariant
    property: "Forward pass is deterministic"
    formal: "smoke(f, fmt) == smoke(f, fmt)"
    tolerance: 0.0
    applies_to: forward_determinism
    lean:
      theorem: "Theorems.${family_pascal}.ForwardDeterminism"
      module: "ProvableContracts.ArchitectureDemos.${family_pascal}"

qa_gate:
  id: F-${family^^}-001
  name: "${family_pascal} Smoke Inference Contract"
  description: "Smoke load + deterministic forward pass for the ${family} architecture family"
  checks:
    - "loader_dispatch_returns_ok"
    - "tensor_validation_passes"
    - "forward_determinism"
  pass_criteria: "All three falsification tests pass; SmokeVerdict is Ok across all declared formats"
EOF
}

# ---------- Cargo.toml block writer ----------
ensure_cargo_block() {
    local action="$1"  # check | update
    local in_progress=()
    while IFS= read -r f; do
        in_progress+=("$f")
    done < <(families_with_status in-progress)

    # Speech families are tracked but their recipes live in examples/speech/, not examples/inference/.
    # Filter to families whose recipe_path is in examples/inference/.
    local need_blocks=()
    for f in "${in_progress[@]}"; do
        local rp; rp="$(field_for "$f" recipe_path)"
        if [[ "$rp" == examples/inference/* ]]; then
            need_blocks+=("$f")
        fi
    done

    if [[ "$action" == "update" ]]; then
        # Strip any existing section + everything after the header marker.
        if grep -q "^${CARGO_SECTION_HEADER}$" "$CARGO_TOML"; then
            local line; line=$(grep -n "^${CARGO_SECTION_HEADER}$" "$CARGO_TOML" | cut -d: -f1)
            head -n $((line - 1)) "$CARGO_TOML" > "$CARGO_TOML.tmp"
            # Drop trailing blank lines.
            sed -i -e :a -e '/^[[:space:]]*$/{$d;N;ba' -e '}' "$CARGO_TOML.tmp"
            mv "$CARGO_TOML.tmp" "$CARGO_TOML"
        fi
        {
            echo ""
            echo "$CARGO_SECTION_HEADER"
            echo ""
            for f in "${need_blocks[@]}"; do
                local rp; rp="$(field_for "$f" recipe_path)"
                echo "[[example]]"
                echo "name = \"inference_${f}_smoke\""
                echo "path = \"$rp\""
                echo ""
            done
        } >> "$CARGO_TOML"
    else
        # check mode: confirm each block is present.
        for f in "${need_blocks[@]}"; do
            if ! grep -q "name = \"inference_${f}_smoke\"" "$CARGO_TOML"; then
                DRIFT+=("Cargo.toml missing [[example]] block for inference_${f}_smoke")
            fi
        done
    fi
}

# ---------- Coverage matrix regenerator ----------
regen_coverage_matrix() {
    local action="$1"
    local certified_count in_progress_count blocked_count total
    certified_count=$(yq -r '.families[] | select(.status == "certified") | .name' "$MANIFEST" | wc -l)
    in_progress_count=$(yq -r '.families[] | select(.status == "in-progress") | .name' "$MANIFEST" | wc -l)
    blocked_count=$(yq -r '.families[] | select(.status == "blocked") | .name' "$MANIFEST" | wc -l)
    total=$((certified_count + in_progress_count + blocked_count))

    # The current coverage-matrix.md is hand-maintained as a starter; CI only
    # checks the summary line. A future expansion of this function can fully
    # regenerate the body from the manifest.
    local expected_summary="**Totals:** ${certified_count} certified · ${in_progress_count} in-progress · ${blocked_count} blocked · **${total} total**"
    if grep -qF "$expected_summary" "$COVERAGE_MATRIX"; then
        return 0
    fi
    if [[ "$action" == "update" ]]; then
        sed -i -E "s/\*\*Totals:\*\*.*total\*\*/${expected_summary//\//\\/}/" "$COVERAGE_MATRIX"
    else
        DRIFT+=("coverage-matrix.md summary line mismatch (expected: $expected_summary)")
    fi
}

# ---------- Summary block regenerator (in manifest.yaml) ----------
regen_summary() {
    local action="$1"
    local certified in_progress blocked total
    certified=$(yq -r '.families[] | select(.status == "certified") | .name' "$MANIFEST" | wc -l)
    in_progress=$(yq -r '.families[] | select(.status == "in-progress") | .name' "$MANIFEST" | wc -l)
    blocked=$(yq -r '.families[] | select(.status == "blocked") | .name' "$MANIFEST" | wc -l)
    total=$((certified + in_progress + blocked))

    if [[ "$action" == "update" ]]; then
        yq -i ".summary.certified = $certified | .summary.in_progress = $in_progress | .summary.blocked = $blocked | .summary.total = $total" "$MANIFEST"
    else
        local mc mp mb mt
        mc=$(yq -r '.summary.certified' "$MANIFEST")
        mp=$(yq -r '.summary.in_progress' "$MANIFEST")
        mb=$(yq -r '.summary.blocked' "$MANIFEST")
        mt=$(yq -r '.summary.total' "$MANIFEST")
        if [[ "$mc" != "$certified" || "$mp" != "$in_progress" || "$mb" != "$blocked" || "$mt" != "$total" ]]; then
            DRIFT+=("manifest.yaml summary block stale (expected $certified/$in_progress/$blocked/$total, got $mc/$mp/$mb/$mt)")
        fi
    fi
}

# ---------- Driver ----------
do_recipes() {
    local action="$1"
    while IFS= read -r f; do
        local rp; rp="$(field_for "$f" recipe_path)"
        if [[ -z "$rp" || "$rp" != examples/inference/* ]]; then
            continue
        fi
        if [[ ! -e "$ROOT/$rp" ]]; then
            if [[ "$action" == "update" ]]; then
                emit_recipe_stub "$f" "$rp"
                echo "  + $rp"
            else
                DRIFT+=("missing recipe: $rp (family $f)")
            fi
        fi
    done < <(families_with_status in-progress)
}

do_contracts() {
    local action="$1"
    for status in in-progress certified; do
        while IFS= read -r f; do
            local cp; cp="$(field_for "$f" provable_contract)"
            if [[ -z "$cp" || "$cp" == "null" ]]; then continue; fi
            if [[ ! -e "$ROOT/$cp" ]]; then
                if [[ "$action" == "update" ]]; then
                    emit_contract_stub "$f" "$cp"
                    echo "  + $cp"
                else
                    DRIFT+=("missing contract: $cp (family $f)")
                fi
            fi
        done < <(families_with_status "$status")
    done
}

run_target() {
    local action="$1"
    case "$TARGET" in
        recipes)         do_recipes "$action" ;;
        contracts)       do_contracts "$action" ;;
        cargo)           ensure_cargo_block "$action" ;;
        coverage-matrix) regen_coverage_matrix "$action" ;;
        summary)         regen_summary "$action" ;;
        all)
            do_recipes "$action"
            do_contracts "$action"
            ensure_cargo_block "$action"
            regen_coverage_matrix "$action"
            regen_summary "$action"
            ;;
        *) echo "ERROR: unknown --target $TARGET" >&2; exit 1 ;;
    esac
}

case "$MODE" in
    --check)
        run_target check
        if [[ ${#DRIFT[@]} -gt 0 ]]; then
            echo "DRIFT detected:" >&2
            printf '  - %s\n' "${DRIFT[@]}" >&2
            exit 1
        fi
        echo "architecture-demos: in sync"
        ;;
    --update)
        run_target update
        echo "architecture-demos: updated"
        ;;
    --diff)
        run_target check
        if [[ ${#DRIFT[@]} -gt 0 ]]; then
            printf '%s\n' "${DRIFT[@]}"
        fi
        ;;
    *) echo "usage: $0 {--check|--update|--diff} [--target ...]" >&2; exit 1 ;;
esac
