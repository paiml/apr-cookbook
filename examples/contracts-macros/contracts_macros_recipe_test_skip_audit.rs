//! # Contracts-Macros Recipe Test Skip Audit
//!
//! Audit per-recipe skipped-test counts to flag any recipe exceeding
//! the allowed skip ratio. Returns sorted offending IDs and total
//! skipped tests.
//!
//! Demonstrates the **CMM.156** recipe for PMAT-209 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: pytest `xfail`/`skip` warning thresholds; rust `#[ignore]`
//!  test-suite hygiene practices.
//!
//! Run with: cargo run --example contracts_macros_recipe_test_skip_audit
//!
//! Added by PMAT-209 (catalog 1504→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum SkipAuditVerdict {
    Ok {
        offending_ids: Vec<String>,
        total_skipped: u32,
    },
    InvalidConfig,
}

/// Items: (id, total_tests, skipped_tests). Flag if skipped/total > threshold_pct/100.
pub fn audit(items: &[(&str, u32, u32)], threshold_pct: u32) -> SkipAuditVerdict {
    if items.is_empty() || threshold_pct > 100 {
        return SkipAuditVerdict::InvalidConfig;
    }
    for (_, total, skipped) in items {
        if *total == 0 || skipped > total {
            return SkipAuditVerdict::InvalidConfig;
        }
    }
    let mut offenders: Vec<String> = items
        .iter()
        .filter(|(_, total, skipped)| skipped * 100 > total * threshold_pct)
        .map(|(id, _, _)| (*id).to_string())
        .collect();
    offenders.sort();
    let total_skipped: u32 = items.iter().map(|(_, _, s)| *s).sum();
    SkipAuditVerdict::Ok {
        offending_ids: offenders,
        total_skipped,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_test_skip_audit")?;

    let items = [("r1", 10, 1), ("r2", 10, 5)];
    println!("threshold-20: {:?}", audit(&items, 20));
    println!("invalid: {:?}", audit(&[], 20));
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
    fn under_threshold_no_offender() {
        let v = audit(&[("a", 10, 1)], 20);
        if let SkipAuditVerdict::Ok { offending_ids, .. } = v {
            assert!(offending_ids.is_empty());
        }
    }

    #[test]
    fn over_threshold_flagged() {
        let v = audit(&[("a", 10, 5)], 20);
        if let SkipAuditVerdict::Ok { offending_ids, .. } = v {
            assert_eq!(offending_ids, vec!["a".to_string()]);
        }
    }

    #[test]
    fn at_threshold_no_offender() {
        // 2/10 = 20% exactly = at threshold
        let v = audit(&[("a", 10, 2)], 20);
        if let SkipAuditVerdict::Ok { offending_ids, .. } = v {
            assert!(offending_ids.is_empty());
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 20), SkipAuditVerdict::InvalidConfig);
    }

    #[test]
    fn threshold_over_100_rejected() {
        assert_eq!(audit(&[("a", 10, 1)], 101), SkipAuditVerdict::InvalidConfig);
    }

    #[test]
    fn zero_total_rejected() {
        assert_eq!(audit(&[("a", 0, 0)], 20), SkipAuditVerdict::InvalidConfig);
    }

    #[test]
    fn skipped_over_total_rejected() {
        assert_eq!(audit(&[("a", 10, 11)], 20), SkipAuditVerdict::InvalidConfig);
    }

    #[test]
    fn total_skipped_correct() {
        let v = audit(&[("a", 10, 1), ("b", 10, 3)], 50);
        if let SkipAuditVerdict::Ok { total_skipped, .. } = v {
            assert_eq!(total_skipped, 4);
        }
    }

    #[test]
    fn offenders_sorted() {
        let v = audit(&[("zeta", 10, 5), ("alpha", 10, 5)], 20);
        if let SkipAuditVerdict::Ok { offending_ids, .. } = v {
            assert_eq!(offending_ids, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let r1 = audit(&[("a", 10, 1)], 20);
        let r2 = audit(&[("a", 10, 1)], 20);
        assert_eq!(r1, r2);
    }

    #[test]
    fn no_skips_zero_total() {
        let v = audit(&[("a", 10, 0)], 20);
        if let SkipAuditVerdict::Ok { total_skipped, .. } = v {
            assert_eq!(total_skipped, 0);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let items: Vec<(&str, u32, u32)> = (0..30).map(|_| ("r", 10, 1)).collect();
        let v = audit(&items, 20);
        if let SkipAuditVerdict::Ok { total_skipped, .. } = v {
            assert_eq!(total_skipped, 30);
        }
    }
}
