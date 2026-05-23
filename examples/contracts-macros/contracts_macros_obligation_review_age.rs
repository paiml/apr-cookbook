//! # Contracts-Macros Obligation Review Age
//!
//! Flag obligations whose last review is older than `max_age_days`.
//! Returns stale list and median-age metric.
//!
//! Demonstrates the **CMM.88** recipe for PMAT-187 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SOX audit-trail TTL conventions; SRE "stale incident
//!  response runbook" patterns.
//!
//! Run with: cargo run --example contracts_macros_obligation_review_age
//!
//! Added by PMAT-187 (catalog 1306→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum AgeVerdict {
    Ok {
        stale: Vec<String>,
        median_age: u32,
        mean_age: f64,
    },
    InvalidConfig,
}

pub fn audit(obligations: &[(&str, u32)], current_day: u32, max_age_days: u32) -> AgeVerdict {
    if obligations.is_empty() || max_age_days == 0 {
        return AgeVerdict::InvalidConfig;
    }
    let mut stale: Vec<String> = Vec::new();
    let mut ages: Vec<u32> = Vec::with_capacity(obligations.len());
    let mut total_age: u64 = 0;
    for (id, last_day) in obligations {
        if current_day < *last_day {
            return AgeVerdict::InvalidConfig;
        }
        let age = current_day - *last_day;
        ages.push(age);
        total_age += u64::from(age);
        if age > max_age_days {
            stale.push((*id).to_string());
        }
    }
    stale.sort();
    ages.sort_unstable();
    let median_age = ages[ages.len() / 2];
    let mean_age = total_age as f64 / obligations.len() as f64;
    AgeVerdict::Ok {
        stale,
        median_age,
        mean_age,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_review_age")?;

    let obligations = [("o1", 90), ("o2", 50), ("o3", 200)];
    println!("audit: {:?}", audit(&obligations, 250, 90));
    println!("invalid: {:?}", audit(&[], 250, 90));
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
    fn fresh_review_not_stale() {
        let obligations = [("o1", 80)];
        let v = audit(&obligations, 100, 30);
        if let AgeVerdict::Ok { stale, .. } = v {
            assert!(stale.is_empty());
        }
    }

    #[test]
    fn old_review_flagged_stale() {
        let obligations = [("old", 0)];
        let v = audit(&obligations, 200, 30);
        if let AgeVerdict::Ok { stale, .. } = v {
            assert_eq!(stale, vec!["old".to_string()]);
        }
    }

    #[test]
    fn empty_input_rejected() {
        assert_eq!(audit(&[], 100, 30), AgeVerdict::InvalidConfig);
    }

    #[test]
    fn zero_max_age_rejected() {
        let obligations = [("o", 50)];
        assert_eq!(audit(&obligations, 100, 0), AgeVerdict::InvalidConfig);
    }

    #[test]
    fn future_review_rejected() {
        let obligations = [("o", 200)];
        assert_eq!(audit(&obligations, 100, 30), AgeVerdict::InvalidConfig);
    }

    #[test]
    fn stale_sorted() {
        let obligations = [("zeta", 0), ("alpha", 0)];
        let v = audit(&obligations, 200, 30);
        if let AgeVerdict::Ok { stale, .. } = v {
            assert_eq!(stale, vec!["alpha".to_string(), "zeta".to_string()]);
        }
    }

    #[test]
    fn boundary_at_max_not_stale() {
        let obligations = [("o", 70)];
        let v = audit(&obligations, 100, 30);
        if let AgeVerdict::Ok { stale, .. } = v {
            assert!(stale.is_empty());
        }
    }

    #[test]
    fn boundary_one_over_max_stale() {
        let obligations = [("o", 69)];
        let v = audit(&obligations, 100, 30);
        if let AgeVerdict::Ok { stale, .. } = v {
            assert_eq!(stale, vec!["o".to_string()]);
        }
    }

    #[test]
    fn deterministic() {
        let obligations = [("o", 50)];
        let r1 = audit(&obligations, 100, 30);
        let r2 = audit(&obligations, 100, 30);
        assert_eq!(r1, r2);
    }

    #[test]
    fn median_correct() {
        let obligations = [("a", 90), ("b", 80), ("c", 70)];
        let v = audit(&obligations, 100, 100);
        if let AgeVerdict::Ok { median_age, .. } = v {
            // ages = [10, 20, 30] sorted → median = 20.
            assert_eq!(median_age, 20);
        }
    }

    #[test]
    fn mean_correct() {
        let obligations = [("a", 90), ("b", 80)];
        let v = audit(&obligations, 100, 100);
        if let AgeVerdict::Ok { mean_age, .. } = v {
            assert!((mean_age - 15.0).abs() < 1e-9);
        }
    }
}
