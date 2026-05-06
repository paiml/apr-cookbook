//! # apr qa --jobs — Parallel Test Worker Budget
//!
//! `apr qa --jobs <N>` runs N test workers in parallel. Constraints:
//! N ≥ 1; N ≤ logical_cpus × 2 (oversubscription past 2× context-
//! switches dominate); auto-pick = logical_cpus. This recipe builds
//! the validator + auto-picker.
//!
//! Demonstrates the **QA.6** recipe for PMAT-121 (apr qa coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender QA-001 + cargo --jobs/-j conventions
//!
//! Run with: cargo run --example cli_qa_parallel_test_budget
//!
//! Added by PMAT-121 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum BudgetVerdict {
    Ok,
    InvalidZero,
    OversubscribedWarning { ratio: f64 },
    SeverelyOversubscribed { ratio: f64 },
}

const SEVERE_OVERSUB_RATIO: f64 = 4.0;

pub fn validate(jobs: u32, logical_cpus: u32) -> BudgetVerdict {
    if jobs == 0 {
        return BudgetVerdict::InvalidZero;
    }
    if logical_cpus == 0 {
        return BudgetVerdict::Ok;
    }
    let ratio = f64::from(jobs) / f64::from(logical_cpus);
    if ratio > SEVERE_OVERSUB_RATIO {
        return BudgetVerdict::SeverelyOversubscribed { ratio };
    }
    if ratio > 2.0 {
        return BudgetVerdict::OversubscribedWarning { ratio };
    }
    BudgetVerdict::Ok
}

pub fn auto_pick(logical_cpus: u32) -> u32 {
    logical_cpus.max(1)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_qa_parallel_test_budget")?;

    let cpus = 8u32;
    for j in [0u32, 1, 8, 16, 17, 64] {
        println!("jobs={j} cpus={cpus}  →  {:?}", validate(j, cpus));
    }
    println!("auto({cpus} cpus) = {}", auto_pick(cpus));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn zero_jobs_invalid() {
        assert_eq!(validate(0, 8), BudgetVerdict::InvalidZero);
    }

    #[test]
    fn equal_to_cpus_passes() {
        assert_eq!(validate(8, 8), BudgetVerdict::Ok);
    }

    #[test]
    fn double_cpus_passes() {
        // 2× oversubscription is the maximum healthy ratio.
        assert_eq!(validate(16, 8), BudgetVerdict::Ok);
    }

    #[test]
    fn over_2x_warns() {
        let v = validate(17, 8);
        assert!(matches!(v, BudgetVerdict::OversubscribedWarning { .. }));
    }

    #[test]
    fn over_4x_severely_oversubscribed() {
        let v = validate(64, 8);
        assert!(matches!(v, BudgetVerdict::SeverelyOversubscribed { .. }));
    }

    #[test]
    fn unknown_cpus_passes_anything() {
        // logical_cpus = 0 → assume operator knows best, no oversubscription check.
        assert_eq!(validate(100, 0), BudgetVerdict::Ok);
    }

    #[test]
    fn auto_pick_returns_cpus() {
        assert_eq!(auto_pick(8), 8);
        assert_eq!(auto_pick(64), 64);
    }

    #[test]
    fn auto_pick_zero_clamps_to_one() {
        // If we don't know CPU count, default to single-threaded.
        assert_eq!(auto_pick(0), 1);
    }

    #[test]
    fn boundary_at_2x_passes() {
        // ratio == 2.0 exactly should still pass (strict > 2.0 for warning).
        assert_eq!(validate(4, 2), BudgetVerdict::Ok);
    }
}
