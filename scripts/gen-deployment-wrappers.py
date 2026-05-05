#!/usr/bin/env python3
"""Generate one Rust loader wrapper per sovereign deployment recipe.

Re-run is idempotent: existing wrappers are overwritten verbatim from the
template, so manual edits to wrappers WILL be clobbered.

Added by PMAT-065 (centralize-cookbooks migration).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RECIPES_DIR = REPO_ROOT / "examples" / "deployment-stacks" / "recipes"
WRAPPERS_DIR = REPO_ROOT / "examples" / "deployment-stacks"

WRAPPER_TEMPLATE = '''\
//! # {title} -- Sovereign deployment recipe loader
//!
//! Loads the deployment recipe `{yaml_rel_path}`, validates its schema, and
//! exits without executing the deployment. Real provisioning is performed by
//! `forjar` against a target machine.
//!
//! Contract: contracts/recipe-iiur-config-v1.yaml
//! Citation: Morris, K. (2020). Infrastructure as Code (2nd ed). O'Reilly. ISBN: 978-1098114671
//!
//! Run with: cargo run --example {wrapper_name}
//!
//! Migrated from sovereign-ai-cookbook by PMAT-065.

use apr_cookbook::deployment_stack::validate_recipe;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const RECIPE_YAML: &str = include_str!("{yaml_rel_path}");

fn main() -> Result<()> {{
    let _ctx = RecipeContext::new("{wrapper_name}")?;
    let parsed: serde_yaml::Value = serde_yaml::from_str(RECIPE_YAML)?;
    let recipe = validate_recipe(&parsed)?;
    println!(
        "recipe={{}} version={{}} inputs={{}}",
        recipe.name, recipe.version, recipe.input_count
    );
    Ok(())
}}

#[cfg(test)]
mod tests {{
    use super::*;

    #[test]
    fn recipe_yaml_parses() {{
        let _: serde_yaml::Value =
            serde_yaml::from_str(RECIPE_YAML).expect("recipe YAML should parse");
    }}

    #[test]
    fn recipe_satisfies_schema() {{
        let parsed: serde_yaml::Value = serde_yaml::from_str(RECIPE_YAML).unwrap();
        let recipe = validate_recipe(&parsed).expect("recipe should validate");
        assert!(!recipe.name.is_empty());
        assert!(!recipe.version.is_empty());
        assert!(!recipe.description.is_empty());
        assert!(recipe.input_count > 0);
    }}

    #[test]
    fn wrapper_runs() {{
        main().expect("wrapper should run successfully");
    }}
}}
'''


def main() -> int:
    if not RECIPES_DIR.is_dir():
        print(f"error: {RECIPES_DIR} does not exist", file=sys.stderr)
        return 1

    yamls = sorted(RECIPES_DIR.glob("*.yaml"))
    for yaml_path in yamls:
        base = yaml_path.stem
        wrapper_name = base.replace("-", "_")
        wrapper_path = WRAPPERS_DIR / f"{wrapper_name}.rs"
        yaml_rel_path = f"recipes/{base}.yaml"
        title = " ".join(word.capitalize() for word in base.split("-"))

        wrapper_path.write_text(
            WRAPPER_TEMPLATE.format(
                title=title,
                yaml_rel_path=yaml_rel_path,
                wrapper_name=wrapper_name,
            )
        )
        print(f"generated: {wrapper_path}")

    print(f"\ngenerated {len(yamls)} deployment-stack wrappers")
    return 0


if __name__ == "__main__":
    sys.exit(main())
