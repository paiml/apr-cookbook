#!/usr/bin/env python3
"""Generate per-recipe, per-stack, and machine chapter stubs.

Chapter stubs land under book/src/deployment-stacks/. Re-run is idempotent:
existing stubs are overwritten. Manual edits to chapter prose WILL be
clobbered. To customize, edit this script.

Added by PMAT-065 (centralize-cookbooks migration).
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES = REPO_ROOT / "examples" / "deployment-stacks" / "recipes"
STACKS = REPO_ROOT / "examples" / "deployment-stacks" / "stacks"
BOOK = REPO_ROOT / "book" / "src" / "deployment-stacks"

GH = "https://github.com/paiml/apr-cookbook"

RECIPE_DESC_RE = re.compile(r"^\s+description:\s*(.+?)\s*$", re.MULTILINE)
STACK_DESC_RE = re.compile(r"^description:\s*(.+?)\s*$", re.MULTILINE)


def extract_recipe_description(yaml_path: Path) -> str:
    text = yaml_path.read_text()
    m = RECIPE_DESC_RE.search(text)
    if m:
        return m.group(1).strip().strip('"')
    return "(no description in YAML -- see file header)"


def extract_stack_description(stack_dir: Path) -> str:
    forjar = stack_dir / "forjar.yaml"
    if forjar.exists():
        text = forjar.read_text()
        m = STACK_DESC_RE.search(text)
        if m:
            return m.group(1).strip()
    readme = stack_dir / "README.md"
    if readme.exists():
        lines = readme.read_text().splitlines()
        for ln in lines[:5]:
            ln = ln.strip().lstrip("#").strip()
            if ln:
                return ln
    return "(see README in stack directory)"


def write_recipe_chapter(yaml_path: Path) -> None:
    base = yaml_path.stem
    wrapper_name = base.replace("-", "_")
    desc = extract_recipe_description(yaml_path)
    out = BOOK / "recipes" / f"{base}.md"
    out.write_text(
        f"""# {base}

> {desc}

## Files

- **YAML recipe**: [`examples/deployment-stacks/recipes/{base}.yaml`]({GH}/blob/main/examples/deployment-stacks/recipes/{base}.yaml)
- **Rust wrapper**: [`examples/deployment-stacks/{wrapper_name}.rs`]({GH}/blob/main/examples/deployment-stacks/{wrapper_name}.rs)

## Run the wrapper

```bash
cargo run --example {wrapper_name}
cargo test --example {wrapper_name}
```

The wrapper loads the YAML, validates required fields (`recipe.name`, `version`, `description`, `inputs`), and exits without provisioning real infrastructure.

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/recipes/{base}.yaml \\
  --inputs <input_name>=<value>
```

See the YAML for the full input schema.

## Contract

This recipe is graded against [`contracts/recipe-iiur-config-v1.yaml`]({GH}/blob/main/contracts/recipe-iiur-config-v1.yaml).

## Provenance

Migrated from `sovereign-ai-cookbook/recipes/{base}.yaml` by PMAT-065 (centralize-cookbooks).
"""
    )


def write_stack_chapter(stack_dir: Path) -> None:
    stack_name = stack_dir.name
    desc = extract_stack_description(stack_dir)
    recipes_subdir = stack_dir / "recipes"
    if recipes_subdir.is_dir():
        recipe_lines = "\n".join(
            f"- `{r.name}`" for r in sorted(recipes_subdir.iterdir())
        )
    else:
        recipe_lines = "(stack uses recipes from the global recipes/ directory; see forjar.yaml)"

    out = BOOK / "stacks" / f"{stack_name}.md"
    out.write_text(
        f"""# Stack: {stack_name}

{desc}

## Files

- **Composition**: [`examples/deployment-stacks/stacks/{stack_name}/forjar.yaml`]({GH}/blob/main/examples/deployment-stacks/stacks/{stack_name}/forjar.yaml)
- **README**: [`examples/deployment-stacks/stacks/{stack_name}/README.md`]({GH}/blob/main/examples/deployment-stacks/stacks/{stack_name}/README.md)

## Recipes referenced

{recipe_lines}

## Real deployment via forjar

```bash
forjar apply examples/deployment-stacks/stacks/{stack_name}/forjar.yaml
```

## Provenance

Migrated from `sovereign-ai-cookbook/stacks/{stack_name}/` by PMAT-065 (centralize-cookbooks).
"""
    )


def write_machine_chapter() -> None:
    out = BOOK / "machines" / "jetson.md"
    out.write_text(
        f"""# Jetson Edge Machine

NVIDIA Jetson provisioning for edge inference.

## Files

- **Canary deployment**: [`examples/machines/jetson/canary/`]({GH}/tree/main/examples/machines/jetson/canary)
- **Makefile**: [`examples/machines/jetson/Makefile`]({GH}/blob/main/examples/machines/jetson/Makefile)

## Usage

```bash
cd examples/machines/jetson
make help
```

## Companion recipes

- `jetson-edge-base.yaml` -- base image provisioning
- Stacks `09-edge-inference` -- full edge inference deployment

## Provenance

Migrated from `sovereign-ai-cookbook/machines/jetson/` by PMAT-065 (centralize-cookbooks).
"""
    )


def write_forjar_integration() -> None:
    out = BOOK / "forjar-integration.md"
    out.write_text(
        """# forjar Integration

[forjar](https://github.com/paiml/forjar) is the Rust-native infrastructure-as-code engine that consumes the YAML recipes in this category. The cookbook ships only the **declarative configs and Rust loader wrappers**; forjar itself is a separate binary.

## Execution model

```text
+----------------------+         +--------+         +-----------------+
| recipe.yaml          | ------> | forjar | ------> | target machine  |
| (declarative config) |         | apply  |         | (provisioning)  |
+----------------------+         +--------+         +-----------------+
         |                                                    ^
         | included via include_str!                          |
         v                                                    | verifies
+----------------------+         +--------+                  | wrapper
| Rust wrapper         | ------> | cargo  |                  | schema
| (validates schema)   |         | test   |                  | matches
+----------------------+         +--------+                  |
```

The cookbook does not run `forjar apply` -- that requires real infrastructure and root privileges. The cookbook **does** run the wrappers in CI, which guarantees that any sovereign-side schema break breaks a cookbook test.

## Why both wrapper + YAML?

| Artifact | Source of truth for | Tested by |
|----------|---------------------|-----------|
| YAML recipe | Deployment shape, inputs, resources | forjar's own test suite (in the forjar repo) |
| Rust wrapper | Schema invariants required by the cookbook | `cargo test` in apr-cookbook CI |

When sovereign upstream changes a recipe schema (renames a field, drops `description`, etc.), the cookbook wrapper test fails -- that's the canary. The fix is either to update the wrapper expectation or to push the schema change through the upstream review.

## Cited references

- Morris, K. (2020). Infrastructure as Code (2nd ed). O'Reilly. ISBN: 978-1098114671
- forjar repository: [github.com/paiml/forjar](https://github.com/paiml/forjar)

## Provenance

Authored during PMAT-065 (centralize-cookbooks migration). No source content; written from scratch.
"""
    )


def write_indexes(yamls: list[Path], stack_dirs: list[Path]) -> None:
    (BOOK / "stacks" / "index.md").write_text(
        "# Stacks\n\n"
        "Multi-recipe compositions for full-stack sovereign AI deployments. Each stack\n"
        "wires several recipes together onto one or more machines.\n\n"
        "## Available stacks\n\n"
        + "\n".join(f"- [{d.name}]({d.name}.md)" for d in stack_dirs)
        + "\n"
    )
    (BOOK / "recipes" / "index.md").write_text(
        "# Recipes\n\n"
        "Per-service deployment recipes consumed by forjar. Each recipe ships with a\n"
        "matching Rust loader wrapper that validates the YAML schema in cookbook CI.\n\n"
        "## Available recipes\n\n"
        + "\n".join(f"- [{y.stem}]({y.stem}.md)" for y in yamls)
        + "\n"
    )
    (BOOK / "machines" / "index.md").write_text(
        "# Machines\n\n"
        "Per-platform machine provisioning configs.\n\n"
        "## Available machines\n\n"
        "- [Jetson](jetson.md) -- NVIDIA Jetson edge inference platform\n"
    )


def main() -> int:
    if not RECIPES.is_dir():
        print(f"error: {RECIPES} does not exist", file=sys.stderr)
        return 1
    if not STACKS.is_dir():
        print(f"error: {STACKS} does not exist", file=sys.stderr)
        return 1

    for sub in ("recipes", "stacks", "machines"):
        (BOOK / sub).mkdir(parents=True, exist_ok=True)

    yamls = sorted(RECIPES.glob("*.yaml"))
    stack_dirs = sorted([d for d in STACKS.iterdir() if d.is_dir()])

    for y in yamls:
        write_recipe_chapter(y)
    for d in stack_dirs:
        write_stack_chapter(d)
    write_machine_chapter()
    write_forjar_integration()
    write_indexes(yamls, stack_dirs)

    print(f"generated deployment-stacks book chapters under {BOOK}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
