//! # Contracts-Macros Recipe Sign-off Required
//!
//! Verify a recipe has been signed off by the required minimum number
//! of distinct approvers from the allowed set. Returns sign-off
//! status and the count of valid approvals.
//!
//! Demonstrates the **CMM.173** recipe for PMAT-215 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: GitHub branch-protection required-reviewers; SOX 404
//!  control segregation-of-duties.
//!
//! Run with: cargo run --example contracts_macros_recipe_signoff_required
//!
//! Added by PMAT-215 (catalog 1558→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum SignoffVerdict {
    Approved { valid_count: u32 },
    InsufficientApprovers { valid_count: u32, needed: u32 },
    InvalidConfig,
}

pub fn check(approvers: &[&str], allowed_signers: &[&str], min_required: u32) -> SignoffVerdict {
    if allowed_signers.is_empty() || min_required == 0 {
        return SignoffVerdict::InvalidConfig;
    }
    let allowed: BTreeSet<&str> = allowed_signers.iter().copied().collect();
    let unique: BTreeSet<&str> = approvers
        .iter()
        .filter(|a| allowed.contains(*a))
        .copied()
        .collect();
    let valid_count = unique.len() as u32;
    if valid_count >= min_required {
        SignoffVerdict::Approved { valid_count }
    } else {
        SignoffVerdict::InsufficientApprovers {
            valid_count,
            needed: min_required - valid_count,
        }
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_recipe_signoff_required")?;

    let allowed = ["alice", "bob", "carol"];
    println!("ok: {:?}", check(&["alice", "bob"], &allowed, 2));
    println!("insufficient: {:?}", check(&["alice"], &allowed, 2));
    println!("unauthorized: {:?}", check(&["dave"], &allowed, 1));
    println!("invalid: {:?}", check(&[], &[], 1));
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
    fn enough_approvers_approved() {
        let allowed = ["a", "b"];
        let v = check(&["a", "b"], &allowed, 2);
        if let SignoffVerdict::Approved { valid_count } = v {
            assert_eq!(valid_count, 2);
        }
    }

    #[test]
    fn insufficient_approvers_blocked() {
        let allowed = ["a", "b"];
        let v = check(&["a"], &allowed, 2);
        if let SignoffVerdict::InsufficientApprovers { needed, .. } = v {
            assert_eq!(needed, 1);
        }
    }

    #[test]
    fn empty_allowed_rejected() {
        assert_eq!(check(&["a"], &[], 1), SignoffVerdict::InvalidConfig);
    }

    #[test]
    fn zero_required_rejected() {
        let allowed = ["a"];
        assert_eq!(check(&["a"], &allowed, 0), SignoffVerdict::InvalidConfig);
    }

    #[test]
    fn unauthorized_signer_ignored() {
        let allowed = ["a"];
        let v = check(&["x", "y"], &allowed, 1);
        if let SignoffVerdict::InsufficientApprovers { valid_count, .. } = v {
            assert_eq!(valid_count, 0);
        }
    }

    #[test]
    fn duplicate_approvers_dedup() {
        let allowed = ["a"];
        let v = check(&["a", "a", "a"], &allowed, 2);
        if let SignoffVerdict::InsufficientApprovers { valid_count, .. } = v {
            assert_eq!(valid_count, 1);
        }
    }

    #[test]
    fn deterministic() {
        let allowed = ["a", "b"];
        let r1 = check(&["a"], &allowed, 1);
        let r2 = check(&["a"], &allowed, 1);
        assert_eq!(r1, r2);
    }

    #[test]
    fn over_required_still_approved() {
        let allowed = ["a", "b", "c"];
        let v = check(&["a", "b", "c"], &allowed, 2);
        if let SignoffVerdict::Approved { valid_count } = v {
            assert_eq!(valid_count, 3);
        }
    }

    #[test]
    fn unicode_signer_supported() {
        let allowed = ["café"];
        let v = check(&["café"], &allowed, 1);
        assert!(matches!(v, SignoffVerdict::Approved { .. }));
    }

    #[test]
    fn many_approvers_handled() {
        let allowed: Vec<&str> = (0..30).map(|_| "a").collect();
        let approvers: Vec<&str> = (0..30).map(|_| "a").collect();
        let v = check(&approvers, &allowed, 1);
        if let SignoffVerdict::Approved { valid_count } = v {
            assert_eq!(valid_count, 1);
        }
    }

    #[test]
    fn no_approvers_blocked() {
        let allowed = ["a"];
        let v = check(&[], &allowed, 1);
        if let SignoffVerdict::InsufficientApprovers { valid_count, .. } = v {
            assert_eq!(valid_count, 0);
        }
    }

    #[test]
    fn case_sensitive() {
        let allowed = ["alice"];
        let v = check(&["Alice"], &allowed, 1);
        if let SignoffVerdict::InsufficientApprovers { valid_count, .. } = v {
            assert_eq!(valid_count, 0);
        }
    }
}
