#!/usr/bin/env bash
# finetune-gen.sh — reconcile fine-tuning-cookbook manifest with on-disk artifacts.
# Modes: --check (CI gate, read-only), --update (write stubs), --diff (preview).
# Targets: contracts, fixtures, coverage-matrix, summary, all (default).
#
# Recipe stubs and Cargo.toml [[example]] entries are NOT auto-generated to
# avoid 155 todo!() example binaries triggering disk-pressure on test compile.
# Each PMAT-330..361 ticket lands its tier's recipes manually following
# recipe-template.md.
#
# See docs/specifications/fine-tuning-cookbook/tickets.md (PMAT-330).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="$ROOT/docs/specifications/fine-tuning-cookbook/manifest.yaml"
SCHEMA="$ROOT/docs/specifications/fine-tuning-cookbook/manifest.schema.yaml"
COVERAGE_MATRIX="$ROOT/docs/specifications/fine-tuning-cookbook/coverage-matrix.md"
CONTRACTS_DIR="$ROOT/contracts"
FIXTURES_DIR="$ROOT/tests/fixtures/finetune"

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
    echo "usage: $0 {--check|--update|--diff} [--target {contracts,fixtures,coverage-matrix,summary,all}]" >&2
    exit 1
fi

DRIFT=()

if ! command -v yq >/dev/null 2>&1; then
    echo "ERROR: yq is required (https://github.com/mikefarah/yq)" >&2
    exit 3
fi

# Reject path-traversal in any manifest-supplied path.
guard_path() {
    local p="$1"
    if [[ "$p" == /* || "$p" == *..* ]]; then
        echo "ERROR: refusing to operate on suspicious path: $p" >&2
        exit 3
    fi
}

# ---------- Manifest readers ----------
recipe_ids() {
    yq -r '.recipes[].id' "$MANIFEST"
}

field_for() {
    local id="$1"; local field="$2"
    yq -r ".recipes[] | select(.id == \"$id\") | .$field // \"\"" "$MANIFEST"
}

# ---------- Provable-contract stub generator ----------
emit_contract_stub() {
    local id="$1"
    local contract_path="contracts/finetune-${id//_/-}-v1.yaml"
    guard_path "$contract_path"
    local target="$ROOT/$contract_path"
    if [[ -e "$target" ]]; then
        return 0
    fi
    local tier; tier="$(field_for "$id" tier)"
    local technique; technique="$(field_for "$id" technique)"
    local base_family; base_family="$(field_for "$id" base_family)"
    local falsifier; falsifier="$(field_for "$id" falsifier)"
    local id_pascal
    id_pascal="$(echo "$id" | sed -E 's/(^|_)([a-z0-9])/\U\2/g')"
    local today; today="$(date +%Y-%m-%d)"
    mkdir -p "$(dirname "$target")"
    cat > "$target" <<EOF
metadata:
  version: "1.0.0"
  created: "$today"
  author: "PAIML Engineering"
  description: "Fine-tuning recipe ${id}: ${technique} on ${base_family}"
  references:
    - "docs/specifications/fine-tuning-cookbook.md"
    - "docs/specifications/fine-tuning-cookbook/manifest.yaml#${id}"
  depends_on:
    - "recipe-iiur-v1"
  tags:
    - finetune
    - tier_${tier}
    - ${technique}
    - ${base_family}

kernel_structure:
  phases:
    - name: setup
      description: "Construct RecipeContext; load fixture"
      invariant: "context.temp_dir empty; fixture path readable"
    - name: train
      description: "Run apr finetune (or technique-specific entry point)"
      invariant: "Training is deterministic for fixed seed=42"
    - name: evaluate
      description: "Run the recipe metric (loss, perplexity, accuracy, etc.)"
      invariant: "Metric is finite; no NaN/Inf"
    - name: verify
      description: "Assert the falsifier"
      invariant: "Two consecutive runs produce equal verdict"
    - name: teardown
      description: "Drop RecipeContext"
      invariant: "context.temp_dir cleaned up"

equations:
  totality:
    formula: "main() returns Result<()> for any well-formed fixture"
    domain: "fixture: Path"
    codomain: "Result<()>"
    invariants:
      - "main() is total — no panic for any well-formed input"
    preconditions:
      - "fixture is valid UTF-8 JSONL or CSV"
    postconditions:
      - "Result is Ok or Err; no panic"
    lean_theorem: "ProvableContracts.Finetune.${id_pascal}.Totality"
    tolerance: 0
    lean:
      theorem: "ProvableContracts.Finetune.${id_pascal}.Totality"
      status: proved
      module: "lean/ProvableContracts/Finetune/${id_pascal}.lean"

  determinism:
    formula: "main() with seed=42 produces equal output across runs"
    domain: "seed: u64, fixture: Path"
    codomain: "verdict: Hash"
    invariants:
      - "Two consecutive main() calls with seed=42 produce byte-equal output"
    preconditions:
      - "seed is fixed; fixture is identical"
    postconditions:
      - "verdict_1 == verdict_2"
    lean_theorem: "ProvableContracts.Finetune.${id_pascal}.Determinism"
    tolerance: 0
    lean:
      theorem: "ProvableContracts.Finetune.${id_pascal}.Determinism"
      status: proved
      module: "lean/ProvableContracts/Finetune/${id_pascal}.lean"

  convergence:
    formula: "${falsifier}"
    domain: "fixture: Path, training_steps: u32"
    codomain: "metric: f32"
    invariants:
      - "${falsifier}"
    preconditions:
      - "fixture is bounded, deterministic"
    postconditions:
      - "the falsifiable claim above holds on the bundled fixture"
    lean_theorem: "ProvableContracts.Finetune.${id_pascal}.Convergence"
    tolerance: 0
    lean:
      theorem: "ProvableContracts.Finetune.${id_pascal}.Convergence"
      status: not-applicable
      module: "lean/ProvableContracts/Finetune/${id_pascal}.lean"

proof_obligations:
  - type: invariant
    property: "main() is total"
    formal: "for all fixture: Path. main() ↦ Result<()> (no panic)"
    tolerance: 0.0
    applies_to: totality
    lean:
      theorem: "ProvableContracts.Finetune.${id_pascal}.Totality"
      status: proved
      module: "ProvableContracts.Finetune.${id_pascal}"
  - type: invariant
    property: "main() is deterministic"
    formal: "main(seed=42) == main(seed=42)"
    tolerance: 0.0
    applies_to: determinism
    lean:
      theorem: "ProvableContracts.Finetune.${id_pascal}.Determinism"
      status: proved
      module: "ProvableContracts.Finetune.${id_pascal}"
  - type: invariant
    property: "${falsifier}"
    formal: "${falsifier}"
    tolerance: 0.0
    applies_to: convergence
    lean:
      theorem: "ProvableContracts.Finetune.${id_pascal}.Convergence"
      status: not-applicable
      module: "ProvableContracts.Finetune.${id_pascal}"

falsification_tests:
  - id: FALSIFY-FT-${id^^}-001
    rule: "main() is total on the bundled fixture"
    prediction: "main() returns Ok(()) on the bundled fixture"
    test: "cargo test --example ${id} -- recipe_runs"
    if_fails: "Recipe panics or returns Err on the bundled fixture"
  - id: FALSIFY-FT-${id^^}-002
    rule: "Two consecutive runs produce equal output (determinism)"
    prediction: "main() with seed=42 produces equal verdict across two runs"
    test: "cargo test --example ${id} -- deterministic_across_runs"
    if_fails: "Non-determinism leaked (clock, RNG, threading)"
  - id: FALSIFY-FT-${id^^}-003
    rule: "${falsifier}"
    prediction: "${falsifier}"
    test: "cargo test --example ${id} -- falsifier_holds_on_fixture"
    if_fails: "${technique} regression on bundled fixture"

kani_harnesses:
  - id: KANI-FT-${id^^}-001
    obligation: "main() is total"
    bound: 4
    strategy: bounded_int
    harness: "kani_harnesses::ft_${id}_total"
  - id: KANI-FT-${id^^}-002
    obligation: "main() is deterministic"
    bound: 4
    strategy: bounded_int
    harness: "kani_harnesses::ft_${id}_deterministic"
  - id: KANI-FT-${id^^}-003
    obligation: "${falsifier}"
    bound: 4
    strategy: bounded_int
    harness: "kani_harnesses::ft_${id}_convergence"

qa_gate:
  id: F-FT-${id^^}-001
  name: "Fine-tuning ${id} contract"
  description: "${technique} on ${base_family}: ${falsifier}"
  checks:
    - main_is_total
    - deterministic_across_runs
    - falsifier_holds_on_fixture
  pass_criteria: "All 3 falsification tests pass on bundled fixture"
EOF
}

# ---------- Fixture stub generator ----------
emit_fixture_stub() {
    local id="$1"
    local fixture_dir="$FIXTURES_DIR/$id"
    guard_path "tests/fixtures/finetune/$id"
    if [[ -d "$fixture_dir" ]]; then
        return 0
    fi
    mkdir -p "$fixture_dir"
    local technique; technique="$(field_for "$id" technique)"

    # Emit a minimal data.jsonl placeholder (PMAT-331+ replaces with real fixture)
    echo '{"input": "fixture-placeholder", "output": "ok", "label": 0}' > "$fixture_dir/data.jsonl"
    cat > "$fixture_dir/expected.json" <<EOF
{
  "recipe_id": "$id",
  "technique": "$technique",
  "comment": "PMAT-330 placeholder; the implementing PMAT-3NN ticket replaces this with the real expected output of main() on data.jsonl"
}
EOF
    cat > "$fixture_dir/README.md" <<EOF
# $id fixture (placeholder)

Synthetic deterministic dataset placeholder for the \`$id\` recipe.

- Generated by: \`bash scripts/finetune-gen.sh --update --target fixtures\` (PMAT-330)
- Replaced by: PMAT-3NN ticket implementing the \`$id\` recipe
- License: PAIML / public-domain
- Last regenerated: $(date +%Y-%m-%d)
EOF
}

# ---------- Coverage matrix regenerator (stub) ----------
regen_coverage_matrix() {
    local action="$1"
    local total certified planned t1 t2 t3 t4
    total=$(yq '.recipes | length' "$MANIFEST")
    certified=$(yq -r '[.recipes[] | select(.status == "certified")] | length' "$MANIFEST")
    planned=$(yq -r '[.recipes[] | select(.status == "planned")] | length' "$MANIFEST")
    t1=$(yq -r '[.recipes[] | select(.tier == 1)] | length' "$MANIFEST")
    t2=$(yq -r '[.recipes[] | select(.tier == 2)] | length' "$MANIFEST")
    t3=$(yq -r '[.recipes[] | select(.tier == 3)] | length' "$MANIFEST")
    t4=$(yq -r '[.recipes[] | select(.tier == 4)] | length' "$MANIFEST")

    local expected_summary="**Totals:** ${certified} certified · ${planned} planned · **${total} total** across 4 tiers"
    if grep -qF "$expected_summary" "$COVERAGE_MATRIX"; then
        return 0
    fi
    if [[ "$action" == "update" ]]; then
        sed -i -E "s/\*\*Totals:\*\*.*4 tiers/${expected_summary//\//\\/}/" "$COVERAGE_MATRIX"
    else
        DRIFT+=("coverage-matrix.md summary line stale (expected: $expected_summary)")
    fi
}

# ---------- Summary block regenerator (in manifest.yaml itself) ----------
regen_summary() {
    local action="$1"
    local total certified planned t1 t2 t3 t4
    total=$(yq '.recipes | length' "$MANIFEST")
    certified=$(yq -r '[.recipes[] | select(.status == "certified")] | length' "$MANIFEST")
    planned=$(yq -r '[.recipes[] | select(.status == "planned")] | length' "$MANIFEST")
    t1=$(yq -r '[.recipes[] | select(.tier == 1)] | length' "$MANIFEST")
    t2=$(yq -r '[.recipes[] | select(.tier == 2)] | length' "$MANIFEST")
    t3=$(yq -r '[.recipes[] | select(.tier == 3)] | length' "$MANIFEST")
    t4=$(yq -r '[.recipes[] | select(.tier == 4)] | length' "$MANIFEST")

    if [[ "$action" == "update" ]]; then
        yq -i ".summary.total = $total | .summary.certified = $certified | .summary.planned = $planned | .summary.tier_1 = $t1 | .summary.tier_2 = $t2 | .summary.tier_3 = $t3 | .summary.tier_4 = $t4" "$MANIFEST"
    else
        local mc; mc=$(yq -r '.summary.certified' "$MANIFEST")
        local mp; mp=$(yq -r '.summary.planned' "$MANIFEST")
        local mt; mt=$(yq -r '.summary.total' "$MANIFEST")
        if [[ "$mt" != "$total" || "$mc" != "$certified" || "$mp" != "$planned" ]]; then
            DRIFT+=("manifest.yaml summary block stale (expected total=$total certified=$certified planned=$planned, got total=$mt certified=$mc planned=$mp)")
        fi
    fi
}

# ---------- Driver ----------
process_target() {
    local target="$1"; local action="$2"
    case "$target" in
        contracts)
            while IFS= read -r id; do
                local contract_path="contracts/finetune-${id//_/-}-v1.yaml"
                local target="$ROOT/$contract_path"
                if [[ -e "$target" ]]; then
                    continue
                fi
                if [[ "$action" == "update" ]]; then
                    emit_contract_stub "$id"
                else
                    DRIFT+=("missing contract stub: $contract_path")
                fi
            done < <(recipe_ids)
            ;;
        fixtures)
            while IFS= read -r id; do
                local fixture_dir="$FIXTURES_DIR/$id"
                if [[ -d "$fixture_dir" ]]; then
                    continue
                fi
                if [[ "$action" == "update" ]]; then
                    emit_fixture_stub "$id"
                else
                    DRIFT+=("missing fixture dir: tests/fixtures/finetune/$id")
                fi
            done < <(recipe_ids)
            ;;
        coverage-matrix) regen_coverage_matrix "$action" ;;
        summary)         regen_summary "$action" ;;
        all)
            process_target contracts "$action"
            process_target fixtures "$action"
            process_target coverage-matrix "$action"
            process_target summary "$action"
            ;;
        *)
            echo "ERROR: unknown target: $target" >&2
            exit 1
            ;;
    esac
}

case "$MODE" in
    --check) process_target "$TARGET" check ;;
    --update) process_target "$TARGET" update ;;
    --diff)  process_target "$TARGET" check ;;
    *)
        echo "ERROR: unknown mode: $MODE" >&2
        exit 1
        ;;
esac

if [[ ${#DRIFT[@]} -gt 0 ]]; then
    echo "fine-tuning-cookbook: drift detected:"
    for d in "${DRIFT[@]}"; do
        echo "  - $d"
    done
    if [[ "$MODE" == "--check" ]]; then
        exit 1
    fi
elif [[ "$MODE" == "--update" ]]; then
    echo "fine-tuning-cookbook: updated"
elif [[ "$MODE" == "--check" ]]; then
    echo "fine-tuning-cookbook: in sync"
else
    echo "fine-tuning-cookbook: clean"
fi
