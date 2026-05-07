//! # Contracts-Macros Recipe Freeze Check
//!
//! Validate that frozen recipes have not been edited since the freeze
//! timestamp. Returns sorted offending IDs (modified after freeze)
//! and the count still in compliance.
//!
//! Demonstrates the **CMM.160** recipe for PMAT-211 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: git tag-as-immutable conventions; PEP 541 frozen-package
//!  ownership rules.
//!
//! Run with: cargo run --example contracts_macros_recipe_freeze_check
//!
//! Added by PMAT-211 (catalog 1522→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum FreezeVerdict {
    Ok {
        offending_ids: Vec<String>,
        compliant_count: u32,
    },
    InvalidConfig,
}

/// Items: (id, freeze_ts, last_modified_ts).
pub fn check(items: &[(&str, u64, u64)]) -> FreezeVerdict {
    if items.is_empty() {
        return FreezeVerdict::InvalidConfig;
    }
    let mut offenders: Vec<String> = items
        .iter()
        .filter(|(_, freeze, last)| *last > *freeze)
        .map(|(id, _, _)| (*id).to_string())
        .collect();
    offenders.sort();
    let compliant = items.len() as u32 - offenders.len() as u32;
    FreezeVerdict::Ok {
        offending_ids: offenders,
        compliant_count: compliant,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_freeze_check")?;

    let items = [("r1", 1000u64, 999u64), ("r2", 1000, 1500)];
    println!("check: {:?}", check(&items));
    println!("invalid: {:?}", check(&[]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn checker_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn unmodified_after_freeze_compliant() {
        let v = check(&[("r", 1000, 999)]);
        if let FreezeVerdict::Ok { offending_ids, .. } = v {
            assert!(offending_ids.is_empty());
        }
    }

    #[test]
    fn modified_after_freeze_offender() {
        let v = check(&[("r", 1000, 1500)]);
        if let FreezeVerdict::Ok { offending_ids, .. } = v {
            assert_eq!(offending_ids, vec!["r".to_string()]);
        }
    }

    #[test]
    fn modified_at_freeze_compliant() {
        let v = check(&[("r", 1000, 1000)]);
        if let FreezeVerdict::Ok { offending_ids, .. } = v {
            assert!(offending_ids.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(check(&[]), FreezeVerdict::InvalidConfig);
    }

    #[test]
    fn compliant_count_correct() {
        let v = check(&[("a", 1000, 999), ("b", 1000, 1500), ("c", 1000, 800)]);
        if let FreezeVerdict::Ok {
            compliant_count, ..
        } = v
        {
            assert_eq!(compliant_count, 2);
        }
    }

    #[test]
    fn offenders_sorted() {
        let v = check(&[("zeta", 1000, 1500), ("alpha", 1000, 1500)]);
        if let FreezeVerdict::Ok { offending_ids, .. } = v {
            assert_eq!(offending_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = check(&[("r", 1000, 1500)]);
        let r2 = check(&[("r", 1000, 1500)]);
        assert_eq!(r1, r2);
    }

    #[test]
    fn many_items_handled() {
        let items: Vec<(&str, u64, u64)> = (0..30).map(|_| ("r", 1000u64, 1500u64)).collect();
        let v = check(&items);
        if let FreezeVerdict::Ok { offending_ids, .. } = v {
            assert_eq!(offending_ids.len(), 30);
        }
    }

    #[test]
    fn no_offenders_returns_empty() {
        let v = check(&[("a", 1000, 500), ("b", 1000, 800)]);
        if let FreezeVerdict::Ok { offending_ids, .. } = v {
            assert!(offending_ids.is_empty());
        }
    }

    #[test]
    fn unicode_id_supported() {
        let v = check(&[("café", 1000, 1500)]);
        if let FreezeVerdict::Ok { offending_ids, .. } = v {
            assert_eq!(offending_ids, vec!["café".to_string()]);
        }
    }

    #[test]
    fn boundary_one_after_freeze_offender() {
        let v = check(&[("r", 1000, 1001)]);
        if let FreezeVerdict::Ok { offending_ids, .. } = v {
            assert_eq!(offending_ids, vec!["r".to_string()]);
        }
    }
}
