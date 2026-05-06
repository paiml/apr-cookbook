//! # Contracts-Macros Field Default Inference
//!
//! When a contract YAML omits an optional field, fall back to the
//! declared default. Returns a list of fields filled with defaults
//! (and any required fields still missing).
//!
//! Demonstrates the **CMM.49** recipe for PMAT-174 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: serde `#[serde(default)]` semantics.
//!
//! Run with: cargo run --example contracts_macros_field_default_inference
//!
//! Added by PMAT-174 (catalog 1189→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeMap;

#[derive(Debug, PartialEq)]
pub enum DefaultVerdict {
    Ok {
        filled: Vec<(String, String)>,
        defaulted: Vec<String>,
    },
    MissingRequired {
        fields: Vec<String>,
    },
    EmptySchema,
}

pub fn fill(schema: &[(&str, Option<&str>, bool)], provided: &[(&str, &str)]) -> DefaultVerdict {
    if schema.is_empty() {
        return DefaultVerdict::EmptySchema;
    }
    let provided_map: BTreeMap<&str, &str> = provided.iter().copied().collect();
    let mut filled: Vec<(String, String)> = Vec::new();
    let mut defaulted: Vec<String> = Vec::new();
    let mut missing: Vec<String> = Vec::new();
    for (name, default, required) in schema {
        if let Some(v) = provided_map.get(name) {
            filled.push(((*name).to_string(), (*v).to_string()));
            continue;
        }
        match (default, required) {
            (Some(d), _) => {
                filled.push(((*name).to_string(), (*d).to_string()));
                defaulted.push((*name).to_string());
            }
            (None, true) => missing.push((*name).to_string()),
            (None, false) => {
                // Optional with no default → omit silently.
            }
        }
    }
    if !missing.is_empty() {
        return DefaultVerdict::MissingRequired { fields: missing };
    }
    DefaultVerdict::Ok { filled, defaulted }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_field_default_inference")?;

    let schema = [
        ("name", None::<&str>, true),
        ("version", Some("0.1.0"), false),
        ("license", Some("MIT"), false),
    ];
    let provided = [("name", "demo")];
    println!("ok: {:?}", fill(&schema, &provided));

    let missing = [];
    println!("missing required: {:?}", fill(&schema, &missing));
    println!("empty schema: {:?}", fill(&[], &provided));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema_a() -> Vec<(&'static str, Option<&'static str>, bool)> {
        vec![("name", None, true), ("version", Some("0.1.0"), false)]
    }

    #[test]
    fn filler_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn provided_wins_over_default() {
        let v = fill(&schema_a(), &[("name", "demo"), ("version", "9.9.9")]);
        if let DefaultVerdict::Ok { filled, defaulted } = v {
            assert!(filled.iter().any(|(k, v)| k == "version" && v == "9.9.9"));
            assert!(defaulted.is_empty());
        }
    }

    #[test]
    fn default_used_when_missing() {
        let v = fill(&schema_a(), &[("name", "demo")]);
        if let DefaultVerdict::Ok { defaulted, .. } = v {
            assert_eq!(defaulted, vec!["version".to_string()]);
        }
    }

    #[test]
    fn missing_required_listed() {
        let v = fill(&schema_a(), &[]);
        if let DefaultVerdict::MissingRequired { fields } = v {
            assert_eq!(fields, vec!["name".to_string()]);
        }
    }

    #[test]
    fn empty_schema_rejected() {
        assert_eq!(fill(&[], &[("a", "1")]), DefaultVerdict::EmptySchema);
    }

    #[test]
    fn optional_no_default_omitted() {
        let schema = vec![("name", None, true), ("optional", None, false)];
        let v = fill(&schema, &[("name", "x")]);
        if let DefaultVerdict::Ok { filled, .. } = v {
            assert_eq!(filled.len(), 1);
        }
    }

    #[test]
    fn all_defaulted() {
        let schema = vec![("a", Some("1"), false), ("b", Some("2"), false)];
        let v = fill(&schema, &[]);
        if let DefaultVerdict::Ok { defaulted, .. } = v {
            assert_eq!(defaulted.len(), 2);
        }
    }

    #[test]
    fn no_defaults_used() {
        let v = fill(&schema_a(), &[("name", "x"), ("version", "1.0.0")]);
        if let DefaultVerdict::Ok { defaulted, .. } = v {
            assert!(defaulted.is_empty());
        }
    }

    #[test]
    fn multiple_required_missing() {
        let schema = vec![("a", None, true), ("b", None, true)];
        let v = fill(&schema, &[]);
        if let DefaultVerdict::MissingRequired { fields } = v {
            assert_eq!(fields.len(), 2);
        }
    }

    #[test]
    fn extra_provided_ignored() {
        let schema = vec![("a", None, true)];
        let v = fill(&schema, &[("a", "1"), ("b", "2")]);
        if let DefaultVerdict::Ok { filled, .. } = v {
            assert_eq!(filled.len(), 1);
        }
    }

    #[test]
    fn deterministic() {
        let s = schema_a();
        let p = [("name", "x")];
        let a = fill(&s, &p);
        let b = fill(&s, &p);
        assert_eq!(a, b);
    }
}
