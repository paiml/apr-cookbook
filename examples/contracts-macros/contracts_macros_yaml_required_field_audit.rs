//! # Contracts-Macros YAML Required Field Audit
//!
//! Find missing required keys in a YAML manifest. Returns missing
//! list and presence count.
//!
//! Demonstrates the **CMM.83** recipe for PMAT-185 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: JSON Schema `required` keyword (draft 2020-12 §6.5.3);
//!  Kubernetes manifest required fields.
//!
//! Run with: cargo run --example contracts_macros_yaml_required_field_audit
//!
//! Added by PMAT-185 (catalog 1288→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum RequiredVerdict {
    Ok {
        missing: Vec<String>,
        present_count: u32,
    },
    InvalidConfig,
}

pub fn audit(required: &[&str], present: &[&str]) -> RequiredVerdict {
    if required.is_empty() {
        return RequiredVerdict::InvalidConfig;
    }
    let present_set: BTreeSet<&str> = present.iter().copied().collect();
    let mut missing: Vec<String> = required
        .iter()
        .filter(|k| !present_set.contains(*k))
        .map(|k| (*k).to_string())
        .collect();
    missing.sort();
    missing.dedup();
    let required_set: BTreeSet<&str> = required.iter().copied().collect();
    let present_count = required_set
        .iter()
        .filter(|k| present_set.contains(*k))
        .count() as u32;
    RequiredVerdict::Ok {
        missing,
        present_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_required_field_audit")?;

    let required = ["name", "version", "owner", "tolerance"];
    let present = ["name", "version", "owner"];
    println!("audit: {:?}", audit(&required, &present));
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
        if let RequiredVerdict::Ok { missing, .. } = v {
            assert!(missing.is_empty());
        }
    }

    #[test]
    fn missing_keys_flagged() {
        let v = audit(&["a", "b", "c"], &["a"]);
        if let RequiredVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["b".to_string(), "c".to_string()]);
        }
    }

    #[test]
    fn empty_required_rejected() {
        assert_eq!(audit(&[], &["a"]), RequiredVerdict::InvalidConfig);
    }

    #[test]
    fn all_missing() {
        let v = audit(&["a", "b"], &[]);
        if let RequiredVerdict::Ok { missing, .. } = v {
            assert_eq!(missing.len(), 2);
        }
    }

    #[test]
    fn extra_present_keys_ignored() {
        let v = audit(&["a"], &["a", "b", "c"]);
        if let RequiredVerdict::Ok { missing, .. } = v {
            assert!(missing.is_empty());
        }
    }

    #[test]
    fn missing_sorted() {
        let v = audit(&["zeta", "alpha", "mu"], &[]);
        if let RequiredVerdict::Ok { missing, .. } = v {
            assert_eq!(
                missing,
                vec!["alpha".to_string(), "mu".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn present_count_correct() {
        let v = audit(&["a", "b", "c"], &["a", "c"]);
        if let RequiredVerdict::Ok { present_count, .. } = v {
            assert_eq!(present_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&["a", "b"], &["a"]);
        let r2 = audit(&["a", "b"], &["a"]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn case_sensitive() {
        let v = audit(&["Name"], &["name"]);
        if let RequiredVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["Name".to_string()]);
        }
    }

    #[test]
    fn duplicate_required_dedup() {
        let v = audit(&["a", "a", "b"], &["a"]);
        if let RequiredVerdict::Ok { missing, .. } = v {
            assert_eq!(missing, vec!["b".to_string()]);
        }
    }

    #[test]
    fn duplicate_present_no_double_count() {
        let v = audit(&["a"], &["a", "a", "a"]);
        if let RequiredVerdict::Ok { present_count, .. } = v {
            assert_eq!(present_count, 1);
        }
    }
}
