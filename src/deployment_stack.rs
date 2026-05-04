//! Deployment-stack recipe validation helpers.
//!
//! Sovereign recipes are declarative YAML deployment configs (consumed by
//! `forjar` to provision real machines). The cookbook ships a Rust wrapper per
//! recipe that loads the embedded YAML, validates required fields, and exits
//! without executing the deployment. This module factors the parse/validate
//! logic so each wrapper stays under 50 lines.
//!
//! Contract: contracts/recipe-iiur-config-v1.yaml
//!
//! Added by PMAT-065 (centralize-cookbooks migration).

use crate::error::{CookbookError, Result};

/// Parsed shape of a sovereign deployment recipe.
pub struct ParsedRecipe<'a> {
    pub name: &'a str,
    pub version: &'a str,
    pub description: &'a str,
    pub input_count: usize,
}

/// Parse a sovereign deployment recipe YAML and assert required fields exist.
///
/// Required keys (per `sovereign-ai-cookbook` recipe schema):
/// - `recipe.name` (string)
/// - `recipe.version` (string)
/// - `recipe.description` (string)
/// - `recipe.inputs` (mapping)
///
/// Returns a borrowed view over the parsed value, plus the count of declared
/// inputs. The caller owns the parsed `serde_yaml::Value` and may inspect
/// recipe-specific fields beyond what this helper validates.
pub fn validate_recipe(value: &serde_yaml::Value) -> Result<ParsedRecipe<'_>> {
    let recipe = value
        .get("recipe")
        .ok_or_else(|| CookbookError::Validation("missing top-level `recipe` key".to_string()))?;

    let name = recipe.get("name").and_then(|v| v.as_str()).ok_or_else(|| {
        CookbookError::Validation("recipe.name missing or not a string".to_string())
    })?;

    let version = recipe
        .get("version")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            CookbookError::Validation("recipe.version missing or not a string".to_string())
        })?;

    let description = recipe
        .get("description")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            CookbookError::Validation("recipe.description missing or not a string".to_string())
        })?;

    let inputs = recipe
        .get("inputs")
        .and_then(|v| v.as_mapping())
        .ok_or_else(|| {
            CookbookError::Validation("recipe.inputs missing or not a mapping".to_string())
        })?;

    Ok(ParsedRecipe {
        name,
        version,
        description,
        input_count: inputs.len(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID: &str = r#"
recipe:
  name: test
  version: "1.0"
  description: "test recipe"
  inputs:
    foo: { type: string }
    bar: { type: int }
"#;

    #[test]
    fn validates_well_formed_recipe() {
        let v: serde_yaml::Value = serde_yaml::from_str(VALID).unwrap();
        let parsed = validate_recipe(&v).expect("should validate");
        assert_eq!(parsed.name, "test");
        assert_eq!(parsed.version, "1.0");
        assert_eq!(parsed.input_count, 2);
    }

    #[test]
    fn rejects_missing_recipe_key() {
        let v: serde_yaml::Value = serde_yaml::from_str("foo: bar").unwrap();
        assert!(validate_recipe(&v).is_err());
    }

    #[test]
    fn rejects_missing_inputs() {
        let v: serde_yaml::Value =
            serde_yaml::from_str("recipe:\n  name: x\n  version: \"1\"\n  description: y").unwrap();
        assert!(validate_recipe(&v).is_err());
    }
}
