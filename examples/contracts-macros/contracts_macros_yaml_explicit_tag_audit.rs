//! # Contracts-Macros YAML Explicit Tag Audit
//!
//! Verify YAML scalars use explicit tags `!!type` rather than relying
//! on implicit type resolution. Returns sorted offending key paths.
//!
//! Demonstrates the **CMM.193** recipe for PMAT-222 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: YAML 1.2 §2.4 schema/tag resolution; libyaml strict-mode
//!  explicit-tag policy.
//!
//! Run with: cargo run --example contracts_macros_yaml_explicit_tag_audit
//!
//! Added by PMAT-222 (catalog 1621→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum TagAuditVerdict {
    Ok {
        offending_keys: Vec<String>,
        tagged_count: u32,
    },
    InvalidConfig,
}

/// Items: (key, has_explicit_tag).
pub fn audit(items: &[(&str, bool)]) -> TagAuditVerdict {
    if items.is_empty() {
        return TagAuditVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = items
        .iter()
        .filter(|(_, tagged)| !tagged)
        .map(|(k, _)| (*k).to_string())
        .collect();
    offenders.sort();
    let tagged_count = items.iter().filter(|(_, t)| *t).count() as u32;
    TagAuditVerdict::Ok {
        offending_keys: offenders,
        tagged_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_yaml_explicit_tag_audit")?;

    let items = [("a", true), ("b", false), ("c", true)];
    println!("audit: {:?}", audit(&items));
    println!("invalid: {:?}", audit(&[]));
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
    fn all_tagged_no_offender() {
        let v = audit(&[("a", true), ("b", true)]);
        if let TagAuditVerdict::Ok { offending_keys, .. } = v {
            assert!(offending_keys.is_empty());
        }
    }

    #[test]
    fn untagged_flagged() {
        let v = audit(&[("a", false)]);
        if let TagAuditVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["a".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[]), TagAuditVerdict::InvalidConfig);
    }

    #[test]
    fn tagged_count_correct() {
        let v = audit(&[("a", true), ("b", false), ("c", true)]);
        if let TagAuditVerdict::Ok { tagged_count, .. } = v {
            assert_eq!(tagged_count, 2);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("a", true)]);
        let r2 = audit(&[("a", true)]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn offenders_sorted() {
        let v = audit(&[("zeta", false), ("alpha", false)]);
        if let TagAuditVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(
                offending_keys,
                vec!["alpha".to_string(), "zeta".to_string()]
            );
        }
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, bool)> = (0..30).map(|_| ("k", false)).collect();
        let v = audit(&items);
        if let TagAuditVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys.len(), 30);
        }
    }

    #[test]
    fn unicode_key_supported() {
        let v = audit(&[("café", false)]);
        if let TagAuditVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys, vec!["café".to_string()]);
        }
    }

    #[test]
    fn no_offenders_returns_empty() {
        let v = audit(&[("a", true), ("b", true)]);
        if let TagAuditVerdict::Ok { offending_keys, .. } = v {
            assert!(offending_keys.is_empty());
        }
    }

    #[test]
    fn single_tagged_count_one() {
        let v = audit(&[("a", true)]);
        if let TagAuditVerdict::Ok { tagged_count, .. } = v {
            assert_eq!(tagged_count, 1);
        }
    }

    #[test]
    fn all_untagged_max_offenders() {
        let items = vec![("a", false), ("b", false), ("c", false)];
        let v = audit(&items);
        if let TagAuditVerdict::Ok { offending_keys, .. } = v {
            assert_eq!(offending_keys.len(), 3);
        }
    }
}
