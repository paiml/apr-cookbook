//! # Contracts-Macros Obligation Coverage %
//!
//! Compute the percentage of declared obligations exercised by the
//! current test suite. Returns the coverage value plus first
//! uncovered obligation (alphabetically) for fast feedback.
//!
//! Demonstrates the **CMM.39** recipe for PMAT-170 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: code-coverage proxy applied to obligation-level coverage.
//!
//! Run with: cargo run --example contracts_macros_obligation_coverage_pct
//!
//! Added by PMAT-170 (catalog 1153→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use std::collections::BTreeSet;

#[derive(Debug, PartialEq)]
pub enum CoverageVerdict {
    Ok {
        coverage_pct: f64,
        uncovered: Vec<String>,
        first_uncovered: Option<String>,
    },
    EmptyContract,
}

pub fn compute(declared: &[&str], exercised: &[&str]) -> CoverageVerdict {
    if declared.is_empty() {
        return CoverageVerdict::EmptyContract;
    }
    let declared_set: BTreeSet<&str> = declared.iter().copied().collect();
    let exercised_set: BTreeSet<&str> = exercised.iter().copied().collect();
    let covered = declared_set.intersection(&exercised_set).count();
    let coverage_pct = (covered as f64 / declared_set.len() as f64) * 100.0;
    let uncovered: Vec<String> = declared_set
        .difference(&exercised_set)
        .map(|s| (*s).to_string())
        .collect();
    let first_uncovered = uncovered.first().cloned();
    CoverageVerdict::Ok {
        coverage_pct,
        uncovered,
        first_uncovered,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_coverage_pct")?;

    println!("full: {:?}", compute(&["a", "b", "c"], &["a", "b", "c"]));
    println!("partial: {:?}", compute(&["a", "b", "c"], &["a", "b"]));
    println!("zero: {:?}", compute(&["a", "b"], &[]));
    println!("empty contract: {:?}", compute(&[], &["a"]));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn computer_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn full_coverage() {
        let v = compute(&["a", "b"], &["a", "b"]);
        if let CoverageVerdict::Ok {
            coverage_pct,
            uncovered,
            ..
        } = v
        {
            assert!((coverage_pct - 100.0).abs() < 1e-9);
            assert!(uncovered.is_empty());
        }
    }

    #[test]
    fn partial_coverage() {
        let v = compute(&["a", "b", "c", "d"], &["a", "b"]);
        if let CoverageVerdict::Ok {
            coverage_pct,
            uncovered,
            ..
        } = v
        {
            assert!((coverage_pct - 50.0).abs() < 1e-9);
            assert_eq!(uncovered.len(), 2);
        }
    }

    #[test]
    fn zero_coverage() {
        let v = compute(&["a", "b"], &[]);
        if let CoverageVerdict::Ok { coverage_pct, .. } = v {
            assert!((coverage_pct - 0.0).abs() < 1e-9);
        }
    }

    #[test]
    fn extra_exercised_ignored() {
        let v = compute(&["a"], &["a", "b", "c"]);
        if let CoverageVerdict::Ok { coverage_pct, .. } = v {
            assert!((coverage_pct - 100.0).abs() < 1e-9);
        }
    }

    #[test]
    fn empty_contract_rejected() {
        assert_eq!(compute(&[], &["a"]), CoverageVerdict::EmptyContract);
    }

    #[test]
    fn first_uncovered_alphabetical() {
        let v = compute(&["banana", "apple", "cherry"], &[]);
        if let CoverageVerdict::Ok {
            first_uncovered, ..
        } = v
        {
            assert_eq!(first_uncovered, Some("apple".to_string()));
        }
    }

    #[test]
    fn first_uncovered_none_when_full() {
        let v = compute(&["a"], &["a"]);
        if let CoverageVerdict::Ok {
            first_uncovered, ..
        } = v
        {
            assert_eq!(first_uncovered, None);
        }
    }

    #[test]
    fn duplicates_dedup() {
        let v = compute(&["a", "a", "b"], &["a"]);
        if let CoverageVerdict::Ok { coverage_pct, .. } = v {
            // 1 of 2 unique → 50%.
            assert!((coverage_pct - 50.0).abs() < 1e-9);
        }
    }

    #[test]
    fn deterministic() {
        let a = compute(&["a", "b"], &["a"]);
        let b = compute(&["a", "b"], &["a"]);
        assert_eq!(a, b);
    }

    #[test]
    fn coverage_in_unit_range() {
        for &n in &[1, 5, 10] {
            let declared: Vec<&str> = (0..n).map(|_| "x").collect();
            let v = compute(&declared, &[]);
            if let CoverageVerdict::Ok { coverage_pct, .. } = v {
                assert!((0.0..=100.0).contains(&coverage_pct));
            }
        }
    }
}
