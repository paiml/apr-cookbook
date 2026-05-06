//! # Contracts-Macros YAML Default Field Audit
//!
//! Audit YAML keys: keys flagged as "should-have-default" but
//! missing the `default:` sub-field are reported. Returns missing
//! list and how many of the required keys had defaults.
//!
//! Demonstrates the **CMM.77** recipe for PMAT-183 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: JSON Schema `default` keyword (draft 2020-12 §10.2);
//!  Kubernetes manifest defaulting conventions.
//!
//! Run with: cargo run --example contracts_macros_yaml_default_field_audit
//!
//! Added by PMAT-183 (catalog 1270→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum DefaultVerdict {
    Ok {
        missing_defaults: Vec<String>,
        with_defaults_count: u32,
    },
    InvalidConfig,
}

pub fn audit(required_keys: &[&str], keys_with_defaults: &[&str]) -> DefaultVerdict {
    if required_keys.is_empty() {
        return DefaultVerdict::InvalidConfig;
    }
    let with: BTreeSet<&str> = keys_with_defaults.iter().copied().collect();
    let mut missing: Vec<String> = required_keys
        .iter()
        .filter(|k| !with.contains(*k))
        .map(|k| (*k).to_string())
        .collect();
    missing.sort();
    missing.dedup();
    let with_defaults_count = required_keys.len() as u32 - missing.len() as u32;
    DefaultVerdict::Ok {
        missing_defaults: missing,
        with_defaults_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_default_field_audit")?;

    let required = ["timeout_sec", "retries", "tolerance"];
    let with = ["timeout_sec", "retries"];
    println!("audit: {:?}", audit(&required, &with));
    println!("invalid: {:?}", audit(&[], &[]));
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
    fn all_present_no_missing() {
        let v = audit(&["a", "b"], &["a", "b"]);
        if let DefaultVerdict::Ok {
            missing_defaults, ..
        } = v
        {
            assert!(missing_defaults.is_empty());
        }
    }

    #[test]
    fn missing_key_reported() {
        let v = audit(&["a", "b"], &["a"]);
        if let DefaultVerdict::Ok {
            missing_defaults, ..
        } = v
        {
            assert_eq!(missing_defaults, vec!["b".to_string()]);
        }
    }

    #[test]
    fn empty_required_rejected() {
        assert_eq!(audit(&[], &[]), DefaultVerdict::InvalidConfig);
    }

    #[test]
    fn with_count_correct() {
        let v = audit(&["a", "b", "c"], &["a", "c"]);
        if let DefaultVerdict::Ok {
            with_defaults_count,
            ..
        } = v
        {
            assert_eq!(with_defaults_count, 2);
        }
    }

    #[test]
    fn missing_sorted() {
        let v = audit(&["zeta", "alpha", "mu"], &[]);
        if let DefaultVerdict::Ok {
            missing_defaults, ..
        } = v
        {
            assert_eq!(
                missing_defaults,
                vec!["alpha".to_string(), "mu".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn all_missing() {
        let v = audit(&["a", "b"], &[]);
        if let DefaultVerdict::Ok {
            missing_defaults,
            with_defaults_count,
        } = v
        {
            assert_eq!(missing_defaults.len(), 2);
            assert_eq!(with_defaults_count, 0);
        }
    }

    #[test]
    fn extra_with_keys_ignored() {
        // Keys with defaults that aren't required don't matter.
        let v = audit(&["a"], &["a", "b", "c"]);
        if let DefaultVerdict::Ok {
            missing_defaults, ..
        } = v
        {
            assert!(missing_defaults.is_empty());
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["a", "b"], &["a"]);
        let r2 = audit(&["a", "b"], &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn duplicate_required_collapses() {
        let v = audit(&["a", "a", "b"], &["a"]);
        if let DefaultVerdict::Ok {
            missing_defaults, ..
        } = v
        {
            assert_eq!(missing_defaults, vec!["b".to_string()]);
        }
    }

    #[test]
    fn case_sensitive() {
        let v = audit(&["Timeout"], &["timeout"]);
        if let DefaultVerdict::Ok {
            missing_defaults, ..
        } = v
        {
            assert_eq!(missing_defaults, vec!["Timeout".to_string()]);
        }
    }

    #[test]
    fn all_present_count_matches_required_len() {
        let v = audit(&["a", "b", "c"], &["a", "b", "c"]);
        if let DefaultVerdict::Ok {
            with_defaults_count,
            ..
        } = v
        {
            assert_eq!(with_defaults_count, 3);
        }
    }
}
