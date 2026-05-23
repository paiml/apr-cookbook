//! # Contracts-Macros Recipe Diff Minimum
//!
//! Flag recipe change-sets below a minimum diff threshold (insufficient
//! to merit code review). Returns offending recipes and their diff
//! sizes (insertions + deletions).
//!
//! Demonstrates the **CMM.87** recipe for PMAT-186 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: trivial-PR-rejection conventions in code review research
//!  (Bird & Bacchelli, "Expectations Outcomes and Challenges of Modern
//!  Code Review", ICSE 2013).
//!
//! Run with: cargo run --example contracts_macros_recipe_diff_minimum
//!
//! Added by PMAT-186 (catalog 1297→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum DiffVerdict {
    Ok {
        too_small: Vec<String>,
        ok_count: u32,
    },
    InvalidConfig,
}

pub fn audit(diffs: &[(&str, u32, u32)], min_total: u32) -> DiffVerdict {
    if diffs.is_empty() || min_total == 0 {
        return DiffVerdict::InvalidConfig;
    }
    let mut too_small: Vec<String> = Vec::new();
    let mut ok_count = 0u32;
    for (name, ins, del) in diffs {
        if ins + del < min_total {
            too_small.push((*name).to_string());
        } else {
            ok_count += 1;
        }
    }
    too_small.sort();
    DiffVerdict::Ok {
        too_small,
        ok_count,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_diff_minimum")?;

    let diffs = [("r1", 50, 10), ("r2", 1, 0), ("r3", 0, 2)];
    println!("audit: {:?}", audit(&diffs, 10));
    println!("invalid: {:?}", audit(&[], 10));
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
    fn substantial_diff_passes() {
        let diffs = [("r1", 100, 50)];
        let v = audit(&diffs, 10);
        if let DiffVerdict::Ok {
            too_small,
            ok_count,
        } = v
        {
            assert!(too_small.is_empty());
            assert_eq!(ok_count, 1);
        }
    }

    #[test]
    fn trivial_diff_flagged() {
        let diffs = [("r1", 1, 0)];
        let v = audit(&diffs, 10);
        if let DiffVerdict::Ok { too_small, .. } = v {
            assert_eq!(too_small, vec!["r1".to_string()]);
        }
    }

    #[test]
    fn insertions_plus_deletions_counted() {
        let diffs = [("r1", 5, 5)];
        let v = audit(&diffs, 10);
        if let DiffVerdict::Ok {
            too_small,
            ok_count,
        } = v
        {
            // 5+5=10 == min_total → not too small.
            assert!(too_small.is_empty());
            assert_eq!(ok_count, 1);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 10), DiffVerdict::InvalidConfig);
    }

    #[test]
    fn zero_min_rejected() {
        let diffs = [("r1", 5, 5)];
        assert_eq!(audit(&diffs, 0), DiffVerdict::InvalidConfig);
    }

    #[test]
    fn too_small_sorted() {
        let diffs = [("zeta", 1, 0), ("alpha", 1, 0)];
        let v = audit(&diffs, 10);
        if let DiffVerdict::Ok { too_small, .. } = v {
            assert_eq!(too_small, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let diffs = [("r1", 5, 5)];
        let r1 = audit(&diffs, 10);
        let r2 = audit(&diffs, 10);
        assert_eq!(r1, r2);
    }

    #[test]
    fn ok_count_correct() {
        let diffs = [("r1", 50, 50), ("r2", 1, 0), ("r3", 100, 50)];
        let v = audit(&diffs, 10);
        if let DiffVerdict::Ok { ok_count, .. } = v {
            assert_eq!(ok_count, 2);
        }
    }

    #[test]
    fn boundary_at_min_passes() {
        let diffs = [("r1", 10, 0)];
        let v = audit(&diffs, 10);
        if let DiffVerdict::Ok { too_small, .. } = v {
            assert!(too_small.is_empty());
        }
    }

    #[test]
    fn one_below_min_flagged() {
        let diffs = [("r1", 9, 0)];
        let v = audit(&diffs, 10);
        if let DiffVerdict::Ok { too_small, .. } = v {
            assert_eq!(too_small, vec!["r1".to_string()]);
        }
    }

    #[test]
    fn many_recipes_handled() {
        let diffs: Vec<(&str, u32, u32)> = (0..15).map(|_| ("r", 50, 50)).collect();
        let v = audit(&diffs, 10);
        if let DiffVerdict::Ok { ok_count, .. } = v {
            assert_eq!(ok_count, 15);
        }
    }
}
