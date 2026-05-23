//! # Contracts-Macros Proof Age Audit
//!
//! Flag Lean theorems whose proofs haven't been rechecked in
//! `max_age_days`. Returns the list of stale theorems sorted by age.
//!
//! Demonstrates the **CMM.41** recipe for PMAT-171 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SBOM / SLSA freshness audits.
//!
//! Run with: cargo run --example contracts_macros_proof_age_audit
//!
//! Added by PMAT-171 (catalog 1162→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AgeAuditVerdict {
    AllFresh,
    HasStale { stale: Vec<(String, u64)> },
    EmptyContract,
    InvalidAge,
}

pub fn audit(theorems: &[(&str, u64)], now_secs: u64, max_age_secs: u64) -> AgeAuditVerdict {
    if theorems.is_empty() {
        return AgeAuditVerdict::EmptyContract;
    }
    if max_age_secs == 0 {
        return AgeAuditVerdict::InvalidAge;
    }
    let mut stale: Vec<(String, u64)> = Vec::new();
    for (name, last_checked) in theorems {
        let age = now_secs.saturating_sub(*last_checked);
        if age > max_age_secs {
            stale.push(((*name).to_string(), age));
        }
    }
    if stale.is_empty() {
        return AgeAuditVerdict::AllFresh;
    }
    stale.sort_by_key(|b| std::cmp::Reverse(b.1));
    AgeAuditVerdict::HasStale { stale }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_proof_age_audit")?;

    let theorems = [("fresh", 1_000_000), ("stale", 100_000), ("very_old", 500)];
    println!("audit: {:?}", audit(&theorems, 1_000_000, 86_400 * 7));

    let all_fresh = [("a", 999_900), ("b", 999_950)];
    println!("all fresh: {:?}", audit(&all_fresh, 1_000_000, 86_400));
    println!("empty: {:?}", audit(&[], 1_000_000, 86_400));
    println!("invalid: {:?}", audit(&theorems, 1_000_000, 0));
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
    fn all_fresh() {
        let theorems = [("a", 950), ("b", 980)];
        let v = audit(&theorems, 1000, 100);
        assert_eq!(v, AgeAuditVerdict::AllFresh);
    }

    #[test]
    fn one_stale_listed() {
        let theorems = [("a", 950), ("old", 100)];
        let v = audit(&theorems, 1000, 100);
        if let AgeAuditVerdict::HasStale { stale } = v {
            assert_eq!(stale.len(), 1);
        }
    }

    #[test]
    fn stale_sorted_by_age_desc() {
        let theorems = [("a", 100), ("b", 500), ("c", 0)];
        let v = audit(&theorems, 1000, 100);
        if let AgeAuditVerdict::HasStale { stale } = v {
            // c is oldest (1000 - 0 = 1000), then a (900), then b (500).
            assert_eq!(stale[0].0, "c");
        }
    }

    #[test]
    fn empty_contract() {
        assert_eq!(audit(&[], 1000, 100), AgeAuditVerdict::EmptyContract);
    }

    #[test]
    fn invalid_zero_max_age() {
        let theorems = [("a", 100)];
        assert_eq!(audit(&theorems, 1000, 0), AgeAuditVerdict::InvalidAge);
    }

    #[test]
    fn boundary_at_max_age_fresh() {
        let theorems = [("a", 900)];
        // age = 100 == max_age_secs → not stale.
        assert_eq!(audit(&theorems, 1000, 100), AgeAuditVerdict::AllFresh);
    }

    #[test]
    fn just_over_boundary_stale() {
        let theorems = [("a", 899)];
        let v = audit(&theorems, 1000, 100);
        assert!(matches!(v, AgeAuditVerdict::HasStale { .. }));
    }

    #[test]
    fn future_timestamp_no_age() {
        // last_checked > now: saturating_sub → age = 0 → fresh.
        let theorems = [("a", 2000)];
        assert_eq!(audit(&theorems, 1000, 100), AgeAuditVerdict::AllFresh);
    }

    #[test]
    fn many_stale() {
        let theorems = [("a", 0), ("b", 0), ("c", 0)];
        let v = audit(&theorems, 1000, 100);
        if let AgeAuditVerdict::HasStale { stale } = v {
            assert_eq!(stale.len(), 3);
        }
    }

    #[test]
    fn deterministic() {
        let theorems = [("a", 100)];
        let a = audit(&theorems, 1000, 100);
        let b = audit(&theorems, 1000, 100);
        assert_eq!(a, b);
    }
}
