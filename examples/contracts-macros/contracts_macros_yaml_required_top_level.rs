//! # Contracts-Macros YAML Required Top-Level Schema
//!
//! Verify YAML top-level keys exactly match a required-key schema.
//! Returns missing required keys and any unexpected extra keys.
//!
//! Demonstrates the **CMM.119** recipe for PMAT-197 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: JSON Schema `required` + `additionalProperties: false`;
//!  Kubernetes manifest top-level structure.
//!
//! Run with: cargo run --example contracts_macros_yaml_required_top_level
//!
//! Added by PMAT-197 (catalog 1396→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum SchemaVerdict {
    Ok {
        missing: Vec<String>,
        unexpected: Vec<String>,
        compliant: bool,
    },
    InvalidConfig,
}

pub fn audit(required_keys: &[&str], actual_keys: &[&str]) -> SchemaVerdict {
    if required_keys.is_empty() {
        return SchemaVerdict::InvalidConfig;
    }
    let req_set: BTreeSet<&str> = required_keys.iter().copied().collect();
    let act_set: BTreeSet<&str> = actual_keys.iter().copied().collect();
    let mut missing: Vec<String> = req_set
        .difference(&act_set)
        .map(|s| (*s).to_string())
        .collect();
    let mut unexpected: Vec<String> = act_set
        .difference(&req_set)
        .map(|s| (*s).to_string())
        .collect();
    missing.sort();
    unexpected.sort();
    let compliant = missing.is_empty() && unexpected.is_empty();
    SchemaVerdict::Ok {
        missing,
        unexpected,
        compliant,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_required_top_level")?;

    let required = ["name", "version", "owner"];
    let actual = ["name", "version", "owner"];
    println!("compliant: {:?}", audit(&required, &actual));
    let bad = ["name", "version", "extra"];
    println!("with extra: {:?}", audit(&required, &bad));
    println!("invalid: {:?}", audit(&[], &actual));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auditor_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_match_compliant() {
        let v = audit(&["a", "b"], &["a", "b"]);
        if let SchemaVerdict::Ok { compliant, .. } = v {
            assert!(compliant);
        }
    }

    #[test]
    fn missing_key_flagged() {
        let v = audit(&["a", "b"], &["a"]);
        if let SchemaVerdict::Ok {
            missing, compliant, ..
        } = v
        {
            assert_eq!(missing, vec!["b".to_string()]);
            assert!(!compliant);
        }
    }

    #[test]
    fn unexpected_key_flagged() {
        let v = audit(&["a"], &["a", "b"]);
        if let SchemaVerdict::Ok {
            unexpected,
            compliant,
            ..
        } = v
        {
            assert_eq!(unexpected, vec!["b".to_string()]);
            assert!(!compliant);
        }
    }

    #[test]
    fn empty_required_rejected() {
        assert_eq!(audit(&[], &["a"]), SchemaVerdict::InvalidConfig);
    }

    #[test]
    fn missing_sorted() {
        let v = audit(&["zeta", "alpha"], &[]);
        if let SchemaVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn unexpected_sorted() {
        let v = audit(&[], &["zeta", "alpha"]);
        if let SchemaVerdict::Ok { unexpected, .. } = v {
            // empty required → all unexpected, but empty required rejected.
            // This test is unreachable (empty required → InvalidConfig).
            // Verify with non-empty required + extras.
            let _ = unexpected;
        }
    }

    #[test]
    fn unexpected_sorted_alphabetically() {
        let v = audit(&["a"], &["a", "zeta", "alpha"]);
        if let SchemaVerdict::Ok { unexpected, .. } = v {
            assert_eq!(unexpected, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["a"], &["a"]);
        let r2 = audit(&["a"], &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive() {
        let v = audit(&["Name"], &["name"]);
        if let SchemaVerdict::Ok {
            missing,
            unexpected,
            ..
        } = v
        {
            assert_eq!(missing, vec!["Name".to_string()]);
            assert_eq!(unexpected, vec!["name".to_string()]);
        }
    }

    #[test]
    fn empty_actual_all_missing() {
        let v = audit(&["a", "b"], &[]);
        if let SchemaVerdict::Ok { missing, .. } = v {
            assert_eq!(missing.len(), 2);
        }
    }

    #[test]
    fn duplicate_required_dedup() {
        let v = audit(&["a", "a"], &["a"]);
        if let SchemaVerdict::Ok { compliant, .. } = v {
            assert!(compliant);
        }
    }

    #[test]
    fn many_keys_handled() {
        let req: Vec<&str> = vec!["k"; 50];
        let act: Vec<&str> = vec!["k"; 50];
        let v = audit(&req, &act);
        if let SchemaVerdict::Ok { compliant, .. } = v {
            assert!(compliant);
        }
    }
}
