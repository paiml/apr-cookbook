//! # Renacer Observability -- Sovereign deployment recipe loader
//!
//! Loads the deployment recipe `recipes/renacer-observability.yaml`, validates its schema, and
//! exits without executing the deployment. Real provisioning is performed by
//! `forjar` against a target machine.
//!
//! Contract: contracts/recipe-iiur-config-v1.yaml
//! Citation: Morris, K. (2020). Infrastructure as Code (2nd ed). O'Reilly. ISBN: 978-1098114671
//!
//! Run with: cargo run --example renacer_observability
//!
//! Migrated from sovereign-ai-cookbook by PMAT-065.

use apr_cookbook::deployment_stack::validate_recipe;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const RECIPE_YAML: &str = include_str!("recipes/renacer-observability.yaml");

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("renacer_observability")?;
    let parsed: serde_yaml::Value = serde_yaml::from_str(RECIPE_YAML)?;
    let recipe = validate_recipe(&parsed)?;
    println!(
        "recipe={} version={} inputs={}",
        recipe.name, recipe.version, recipe.input_count
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_yaml_parses() {
        let _: serde_yaml::Value =
            serde_yaml::from_str(RECIPE_YAML).expect("recipe YAML should parse");
    }

    #[test]
    fn recipe_satisfies_schema() {
        let parsed: serde_yaml::Value = serde_yaml::from_str(RECIPE_YAML).unwrap();
        let recipe = validate_recipe(&parsed).expect("recipe should validate");
        assert!(!recipe.name.is_empty());
        assert!(!recipe.version.is_empty());
        assert!(!recipe.description.is_empty());
        assert!(recipe.input_count > 0);
    }

    #[test]
    fn wrapper_runs() {
        main().expect("wrapper should run successfully");
    }
}
