# PMAT-436 — aprender 0.69.1 draft, with the recipe stack folded in (PR #438)

## What this PR is

- The 0.69.1 README banner and the first `apr`-binary recipes (#436).
- Four stacked PRs, squash-folded into it on the cop's ruling (2026-09-24). Each is closed as "folded into #438", and its branch is kept:
  - #452: recipe runner plus receipts (#439)
  - #456: the `cookbook-recipe-v1` shape (#440)
  - #458: inspect / tensors / validate recipes (#439)
  - #455: the receipts-required check (#442)

## Scope split: what this PR does NOT do

#440 and #442 are delivered here only as **scripts run by hand**. Their `done_when` asks for **CI** jobs. The CI wiring (`.github/workflows/ci.yml`: a recipes job reported through `gate`) is PR #459. It is a workflow change, which the cop arms separately. That is why this PR says `Refs #439/#440/#442` and closes none of them. `recipes/README.md` says the same: each check is "run by hand in this change; CI runs it through `gate` once PR #459 lands".

## Measured on this tree (head of this PR, against the published apr 0.69.1 release asset)

The asset is `apr-v0.69.1-x86_64-unknown-linux-gnu-cuda`, sha256 `c4bbd7ae…`, and prints `apr 0.69.1 (v0.69.1+no-git)`.

| Command | Result |
|---|---|
| `python3 scripts/check_recipe_argv_surface.py <asset>` | rc 0, 8 recipes, 0 failed |
| the same, on a scratch copy with `--json` misspelt `--jsonn` | rc 1, `FAIL inspect-json-qwen35-4b: ['--jsonn'] not in apr inspect --help` |
| `bash scripts/check_recipe_shape.sh` | rc 0: 8 recipes SHACL-validated by `pv`, derived evidence in sync |
| `bash scripts/check_recipe_shape.sh --self-test` | rc 0 (0 failed) |
| `python3 scripts/check_recipe_receipts.py` | rc 0: 16 recipe×host cells against `0.69.1-v0.69.1+no-git`, 0 problems |
| `python3 scripts/check_recipe_receipts.py --self-test` | rc 0, 8/8 rows ok |
| `python3 scripts/run_recipes.py --self-test` | rc 0 (0 failed) |
| `python3 scripts/recipes_to_jsonl.py --check` | rc 0. The file was regenerated after the fold and is byte-identical to #458's |

## Not run locally

`cargo test` / `clippy` for the `tests/contracts.rs` registration (from #456). gx10 had 93G free, and a private target dir would take it below the 90G floor. CI's `ci / test` and `ci / lint` run them on this head.
