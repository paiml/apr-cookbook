//! # Analysis — JSON Schema Draft 7 Meta-Validation
//!
//! aprender's MCP server runs JSON Schema Draft 7 meta-validation on every
//! tool input schema in CI (FALSIFY-MCP-002 strict). This recipe
//! demonstrates the meta-validation pattern: declare a tool inputSchema,
//! validate it against the Draft-7 meta-schema (the schema-of-schemas)
//! that defines what a valid Draft-7 schema must look like.
//!
//! Demonstrates the **AN+.4** recipe per
//! `docs/specifications/expand-cookbooks/recipe-catalog.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: JSON Schema Draft 7. https://json-schema.org/draft-07/json-schema-release-notes
//!
//! Run with: cargo run --example analysis_json_schema_draft7_meta_validation
//!
//! Added by PMAT-086 (expand-cookbooks: Tier 4 authoring patterns).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use serde_json::{json, Value};

/// Minimal Draft-7 well-formedness checker: a valid Draft-7 schema MUST
/// have `"type": "object"` (or another valid type) and (if object-type)
/// MAY have a `properties` map; `required` if present MUST be a string array.
/// Real meta-validation goes much further; this is the load-bearing subset.
fn validate_draft7_schema(schema: &Value) -> Result<()> {
    let schema_type = schema["type"].as_str().ok_or_else(|| {
        apr_cookbook::CookbookError::Validation(
            "FALSIFY-MCP-002: schema must declare `type` (string)".into(),
        )
    })?;
    if !matches!(
        schema_type,
        "object" | "string" | "number" | "integer" | "boolean" | "array" | "null"
    ) {
        return Err(apr_cookbook::CookbookError::Validation(format!(
            "FALSIFY-MCP-002: type `{schema_type}` not in Draft-7 type set"
        )));
    }
    if schema_type == "object" {
        if let Some(req) = schema.get("required") {
            let arr = req.as_array().ok_or_else(|| {
                apr_cookbook::CookbookError::Validation(
                    "FALSIFY-MCP-002: `required` must be an array".into(),
                )
            })?;
            for (i, item) in arr.iter().enumerate() {
                if !item.is_string() {
                    return Err(apr_cookbook::CookbookError::Validation(format!(
                        "FALSIFY-MCP-002: `required[{i}]` must be a string"
                    )));
                }
            }
        }
        if let Some(props) = schema.get("properties") {
            if !props.is_object() {
                return Err(apr_cookbook::CookbookError::Validation(
                    "FALSIFY-MCP-002: `properties` must be an object".into(),
                ));
            }
        }
    }
    Ok(())
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("analysis_json_schema_draft7_meta_validation")?;

    let valid_tool_schema = json!({
        "type": "object",
        "properties": {
            "model_path": {"type": "string"},
            "max_tokens": {"type": "integer"}
        },
        "required": ["model_path"]
    });
    validate_draft7_schema(&valid_tool_schema)?;
    println!("valid Draft-7 schema passes meta-validation");

    let invalid_no_type = json!({"properties": {}});
    let err = validate_draft7_schema(&invalid_no_type);
    println!("invalid schema (no type) rejected: {}", err.unwrap_err());

    let invalid_required = json!({"type": "object", "required": [42]});
    let err = validate_draft7_schema(&invalid_required);
    println!(
        "invalid schema (numeric required) rejected: {}",
        err.unwrap_err()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_validation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn missing_type_rejected() {
        let s = json!({"properties": {}});
        assert!(validate_draft7_schema(&s).is_err());
    }

    #[test]
    fn invalid_type_rejected() {
        let s = json!({"type": "fluffy"});
        assert!(validate_draft7_schema(&s).is_err());
    }

    #[test]
    fn non_string_required_entry_rejected() {
        let s = json!({"type": "object", "required": [42]});
        assert!(validate_draft7_schema(&s).is_err());
    }

    #[test]
    fn valid_object_schema_with_properties_passes() {
        let s = json!({
            "type": "object",
            "properties": {"x": {"type": "string"}},
            "required": ["x"]
        });
        assert!(validate_draft7_schema(&s).is_ok());
    }
}
