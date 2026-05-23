//! # Advanced Response Schema Match
//!
//! When LLM returns JSON-shaped output, validate against expected
//! schema fields before returning to client. Schema is a list of
//! required field names. Returns missing fields if invalid.
//!
//! Demonstrates the **ADV.31** recipe for PMAT-156 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: JSON Schema validation (RFC 8259 + draft-07).
//!
//! Run with: cargo run --example adv_response_schema_match
//!
//! Added by PMAT-156 (catalog 1027→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SchemaVerdict {
    Ok,
    Missing { fields: Vec<String> },
    EmptySchema,
    EmptyResponse,
}

pub fn validate(response_keys: &[&str], required_fields: &[&str]) -> SchemaVerdict {
    if required_fields.is_empty() {
        return SchemaVerdict::EmptySchema;
    }
    if response_keys.is_empty() {
        return SchemaVerdict::EmptyResponse;
    }
    let response_set: std::collections::BTreeSet<&str> = response_keys.iter().copied().collect();
    let missing: Vec<String> = required_fields
        .iter()
        .filter(|f| !response_set.contains(*f))
        .map(|f| (*f).to_string())
        .collect();
    if missing.is_empty() {
        SchemaVerdict::Ok
    } else {
        SchemaVerdict::Missing { fields: missing }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("adv_response_schema_match")?;

    println!(
        "ok: {:?}",
        validate(&["name", "age", "email"], &["name", "age"])
    );
    println!(
        "missing: {:?}",
        validate(&["name"], &["name", "age", "email"])
    );
    println!("empty schema: {:?}", validate(&["a"], &[]));
    println!("empty response: {:?}", validate(&[], &["name"]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn all_fields_present_ok() {
        assert_eq!(validate(&["a", "b", "c"], &["a", "b"]), SchemaVerdict::Ok);
    }

    #[test]
    fn missing_field_listed() {
        let v = validate(&["a"], &["a", "b", "c"]);
        if let SchemaVerdict::Missing { fields } = v {
            assert_eq!(fields, vec!["b".to_string(), "c".to_string()]);
        }
    }

    #[test]
    fn empty_schema_rejected() {
        assert_eq!(validate(&["a"], &[]), SchemaVerdict::EmptySchema);
    }

    #[test]
    fn empty_response_rejected() {
        assert_eq!(validate(&[], &["a"]), SchemaVerdict::EmptyResponse);
    }

    #[test]
    fn extra_fields_ok() {
        // Extra keys in response are fine.
        assert_eq!(
            validate(&["a", "b", "c", "d"], &["a", "b"]),
            SchemaVerdict::Ok
        );
    }

    #[test]
    fn case_sensitive() {
        let v = validate(&["Name"], &["name"]);
        assert!(matches!(v, SchemaVerdict::Missing { .. }));
    }

    #[test]
    fn order_does_not_matter() {
        assert_eq!(validate(&["b", "a"], &["a", "b"]), SchemaVerdict::Ok);
    }

    #[test]
    fn duplicate_in_response_ok() {
        // Duplicates in response don't break.
        assert_eq!(validate(&["a", "a", "b"], &["a", "b"]), SchemaVerdict::Ok);
    }

    #[test]
    fn all_missing_listed() {
        let v = validate(&["x", "y"], &["a", "b"]);
        if let SchemaVerdict::Missing { fields } = v {
            assert_eq!(fields.len(), 2);
        }
    }

    #[test]
    fn deterministic() {
        let a = validate(&["a"], &["a", "b"]);
        let b = validate(&["a"], &["a", "b"]);
        assert_eq!(a, b);
    }
}
