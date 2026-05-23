//! # Contracts-Macros Obligation Composition
//!
//! Compose obligations via And / Or combinators. Returns a verdict
//! reflecting which sub-obligations passed and which failed.
//!
//! Demonstrates the **CMM.52** recipe for PMAT-175 (catalog crosses 1200).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: classical conjunction/disjunction (Hoare logic).
//!
//! Run with: cargo run --example contracts_macros_obligation_compose
//!
//! Added by PMAT-175 (catalog 1198→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Combinator {
    And,
    Or,
}

#[derive(Debug, PartialEq)]
pub enum CompositionVerdict {
    Pass,
    Fail { failures: Vec<String> },
    EmptyObligations,
}

pub fn evaluate(obligations: &[(&str, bool)], combinator: Combinator) -> CompositionVerdict {
    if obligations.is_empty() {
        return CompositionVerdict::EmptyObligations;
    }
    let failures: Vec<String> = obligations
        .iter()
        .filter(|(_, p)| !*p)
        .map(|(n, _)| (*n).to_string())
        .collect();
    let any_pass = obligations.iter().any(|(_, p)| *p);
    match combinator {
        Combinator::And if failures.is_empty() => CompositionVerdict::Pass,
        Combinator::And => CompositionVerdict::Fail { failures },
        Combinator::Or if any_pass => CompositionVerdict::Pass,
        Combinator::Or => CompositionVerdict::Fail { failures },
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_compose")?;

    let all_pass = [("a", true), ("b", true)];
    println!("and pass: {:?}", evaluate(&all_pass, Combinator::And));

    let mixed = [("a", true), ("b", false)];
    println!("and fail: {:?}", evaluate(&mixed, Combinator::And));
    println!("or pass: {:?}", evaluate(&mixed, Combinator::Or));

    let all_fail = [("a", false), ("b", false)];
    println!("or fail: {:?}", evaluate(&all_fail, Combinator::Or));
    println!("empty: {:?}", evaluate(&[], Combinator::And));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn evaluator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn and_all_pass() {
        let v = evaluate(&[("a", true), ("b", true)], Combinator::And);
        assert_eq!(v, CompositionVerdict::Pass);
    }

    #[test]
    fn and_any_fail_means_fail() {
        let v = evaluate(&[("a", true), ("b", false)], Combinator::And);
        assert!(matches!(v, CompositionVerdict::Fail { .. }));
    }

    #[test]
    fn or_any_pass_means_pass() {
        let v = evaluate(&[("a", false), ("b", true)], Combinator::Or);
        assert_eq!(v, CompositionVerdict::Pass);
    }

    #[test]
    fn or_all_fail_means_fail() {
        let v = evaluate(&[("a", false), ("b", false)], Combinator::Or);
        assert!(matches!(v, CompositionVerdict::Fail { .. }));
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(
            evaluate(&[], Combinator::And),
            CompositionVerdict::EmptyObligations
        );
    }

    #[test]
    fn failures_listed() {
        let v = evaluate(&[("a", true), ("bad", false)], Combinator::And);
        if let CompositionVerdict::Fail { failures } = v {
            assert_eq!(failures, vec!["bad".to_string()]);
        }
    }

    #[test]
    fn single_obligation_pass() {
        let v = evaluate(&[("only", true)], Combinator::And);
        assert_eq!(v, CompositionVerdict::Pass);
    }

    #[test]
    fn single_obligation_fail() {
        let v = evaluate(&[("only", false)], Combinator::Or);
        assert!(matches!(v, CompositionVerdict::Fail { .. }));
    }

    #[test]
    fn many_obligations() {
        let obligations: Vec<(&str, bool)> = (0..50).map(|_| ("x", true)).collect();
        assert_eq!(
            evaluate(&obligations, Combinator::And),
            CompositionVerdict::Pass
        );
    }

    #[test]
    fn deterministic() {
        let o = [("a", true)];
        let a = evaluate(&o, Combinator::And);
        let b = evaluate(&o, Combinator::And);
        assert_eq!(a, b);
    }
}
