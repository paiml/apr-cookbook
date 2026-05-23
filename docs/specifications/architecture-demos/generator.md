# Generator Script

`scripts/architecture-demos-gen.sh` reconciles the manifest against on-disk recipes and Cargo.toml entries. It has three modes — `--check`, `--update`, and `--diff` — and one optional `--target <area>` flag.

## Modes

### `--check` (default in CI)

Read-only verification. Exits non-zero if any of:
- A `status: certified` or `in-progress` family lacks `recipe_path` on disk.
- A `status: certified` or `in-progress` family lacks `provable_contract` YAML on disk.
- A `status: certified` family's contract fails `pv lint` or scores below grade A.
- A recipe at `examples/inference/inference_<family>_smoke.rs` has no manifest entry.
- A contract at `contracts/inference-<family>-smoke-v1.yaml` has no manifest entry.
- A `[[example]]` block in `Cargo.toml` references a recipe path that doesn't exist.
- The manifest's `lean_status:` field disagrees with what `pv lint` reports for the contract (e.g., manifest claims `proved` but `lean.status: wip` in the YAML).
- `coverage-matrix.md` is stale relative to `manifest.yaml` (compared by `last_updated` field + content hash).

This is the gate wired into the unified CI workflow. PRs that touch `manifest.yaml` or `examples/inference/inference_*_smoke.rs` must pass `--check`.

```bash
bash scripts/architecture-demos-gen.sh --check
```

### `--update` (developer-local)

Writes:
1. **Recipe stubs** for any `status: in-progress` family lacking a recipe — emits `examples/inference/inference_<family>_smoke.rs` from [recipe-template.md](recipe-template.md), filling in family name, citation, fixture path, and both contract paths (IIUR + per-family provable). Stubs include a `todo!()` in the verdict body so they fail tests until the developer fills in the upstream API call.
2. **Provable-contract YAML stubs** at `contracts/inference-<family>-smoke-v1.yaml` from the skeleton in [recipe-template.md § Per-family contract skeleton](recipe-template.md). All `lean.status:` start at `wip`, all `tolerance:` start at 0, `lean.module:` references a not-yet-existing `lean/Theorems/<Family>.lean`. Developer fills in concrete pre/postconditions when authoring the recipe body.
3. **Cargo.toml entries** under the `# --- architecture-demos (PMAT-<NNN>) ---` section, alphabetized by family name.
4. **`coverage-matrix.md`** regenerated from manifest.
5. **`manifest.yaml summary` block** recomputed from family counts.

Running `--update` after `--check` failed should produce a clean tree.

```bash
bash scripts/architecture-demos-gen.sh --update
git status   # review generated files before committing
```

### `--diff`

Prints a unified diff of what `--update` would change without writing. Useful for PR review.

```bash
bash scripts/architecture-demos-gen.sh --diff
```

## Optional `--target` flag

Restricts the generator to a single area:

| `--target` | Effect |
|------------|--------|
| `recipes` | Only emit/check recipe stubs |
| `contracts` | Only emit/check provable-contract YAML stubs at `contracts/inference-<family>-smoke-v1.yaml` |
| `cargo` | Only emit/check `[[example]]` blocks in Cargo.toml |
| `coverage-matrix` | Only regenerate `coverage-matrix.md` |
| `summary` | Only recompute the `summary:` block in manifest.yaml |

Default (no `--target`) operates on all five.

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Manifest, recipes, Cargo.toml, and coverage-matrix.md are all in sync |
| 1 | Drift detected (use `--diff` to inspect, `--update` to fix) |
| 2 | Manifest schema validation failed (run `pv lint` against manifest.schema.yaml) |
| 3 | Internal generator error |

## Implementation Notes

- The script is bash + `yq` for YAML parsing — keep it dependency-light so it runs on the same self-hosted runners that build the cookbook.
- Recipe stubs use placeholder `todo!()` calls inside `smoke()`; the developer replaces these with the actual `aprender::rosetta` API. Stub tests reference `tests/fixtures/architectures/<family>/model.safetensors` which is generated separately by `architecture-demos-gen-fixture.py`.
- Cargo.toml editing is idempotent — the section header is the anchor; regeneration replaces the whole section, preserving order from the manifest.
- `coverage-matrix.md` is regenerated wholesale; manual edits are silently overwritten. The "Build Steps" section at the bottom is part of the template.

## Adding to CI

Wire into `.github/workflows/architecture-demos.yml`:

```yaml
- name: Architecture-demos coverage
  run: bash scripts/architecture-demos-gen.sh --check
```

The job is required (not advisory) — drift between manifest and disk is a merge blocker, the same way the existing recipe-table determinism check is.

## Local Workflow

```bash
# 1. Edit manifest to flip a family from blocked → in-progress
$EDITOR docs/specifications/architecture-demos/manifest.yaml

# 2. Generate stubs (recipe + provable-contract YAML + Cargo.toml entry)
bash scripts/architecture-demos-gen.sh --update

# 3. Implement the recipe (replace todo!() with real loader call)
$EDITOR examples/inference/inference_<family>_smoke.rs

# 4. Author the per-family provable-contract (fill in pre/postconditions,
#    lean_theorem refs; keep lean.status: wip until proofs land)
$EDITOR contracts/inference-<family>-smoke-v1.yaml

# 5. Generate the fixture
python3 scripts/architecture-demos-gen-fixture.py --family <family>

# 6. Verify
cargo test --example inference_<family>_smoke
pv lint contracts/inference-<family>-smoke-v1.yaml
pv score contracts/inference-<family>-smoke-v1.yaml --summary
bash scripts/architecture-demos-gen.sh --check

# 7. Flip the manifest entry to status: certified, regenerate matrix
$EDITOR docs/specifications/architecture-demos/manifest.yaml
bash scripts/architecture-demos-gen.sh --update
```
