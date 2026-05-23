//! # Contracts-Macros Spec Drift Audit
//!
//! Detect recipes whose declared `spec_version` doesn't match the
//! current spec version. Returns sorted drift IDs and the latest
//! version observed.
//!
//! Demonstrates the **CMM.141** recipe for PMAT-204 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: contract-versioning patterns in OpenAPI 3.x; protobuf
//!  reserved-field drift detection.
//!
//! Run with: cargo run --example contracts_macros_spec_drift_audit
//!
//! Added by PMAT-204 (catalog 1459→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DriftVerdict {
    Ok {
        drifted_ids: Vec<String>,
        max_version_seen: u32,
    },
    InvalidConfig,
}

pub fn audit(recipes: &[(&str, u32)], current_version: u32) -> DriftVerdict {
    if recipes.is_empty() || current_version == 0 {
        return DriftVerdict::InvalidConfig;
    }
    let mut drifted: Vec<String> = recipes
        .iter()
        .filter(|(_, v)| *v != current_version)
        .map(|(id, _)| (*id).to_string())
        .collect();
    drifted.sort();
    let max_seen = recipes.iter().map(|(_, v)| *v).max().unwrap_or(0);
    DriftVerdict::Ok {
        drifted_ids: drifted,
        max_version_seen: max_seen,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_spec_drift_audit")?;

    let recipes = [("r1", 6), ("r2", 5), ("r3", 6)];
    println!("audit: {:?}", audit(&recipes, 6));
    println!("invalid: {:?}", audit(&[], 6));
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
    fn matching_version_no_drift() {
        let v = audit(&[("r", 6)], 6);
        if let DriftVerdict::Ok { drifted_ids, .. } = v {
            assert!(drifted_ids.is_empty());
        }
    }

    #[test]
    fn outdated_version_drifted() {
        let v = audit(&[("r", 5)], 6);
        if let DriftVerdict::Ok { drifted_ids, .. } = v {
            assert_eq!(drifted_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn ahead_version_also_drifted() {
        let v = audit(&[("r", 7)], 6);
        if let DriftVerdict::Ok { drifted_ids, .. } = v {
            assert_eq!(drifted_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn max_version_correct() {
        let v = audit(&[("a", 5), ("b", 7), ("c", 6)], 6);
        if let DriftVerdict::Ok {
            max_version_seen, ..
        } = v
        {
            assert_eq!(max_version_seen, 7);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 6), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn zero_current_rejected() {
        assert_eq!(audit(&[("a", 1)], 0), DriftVerdict::InvalidConfig);
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("r", 5)], 6);
        let r2 = audit(&[("r", 5)], 6);
        assert_eq!(r1, r2);
    }

    #[test]
    fn drifted_sorted() {
        let v = audit(&[("zeta", 5), ("alpha", 5)], 6);
        if let DriftVerdict::Ok { drifted_ids, .. } = v {
            assert_eq!(drifted_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let recipes: Vec<(&str, u32)> = (0..30).map(|_| ("r", 5)).collect();
        let v = audit(&recipes, 6);
        if let DriftVerdict::Ok { drifted_ids, .. } = v {
            assert_eq!(drifted_ids.len(), 30);
        }
    }

    #[test]
    fn no_drift_returns_empty() {
        let v = audit(&[("a", 6), ("b", 6)], 6);
        if let DriftVerdict::Ok { drifted_ids, .. } = v {
            assert!(drifted_ids.is_empty());
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = audit(&[("café", 5)], 6);
        if let DriftVerdict::Ok { drifted_ids, .. } = v {
            assert_eq!(drifted_ids, vec!["café".to_string()]);
        }
    }
}
