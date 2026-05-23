//! # Contracts-Macros Obligation Arity Validator
//!
//! Verify each equation's actual argument count matches its declared
//! arity in the contract. Flags arity mismatches per equation.
//!
//! Demonstrates the **CMM.29** recipe for PMAT-167 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Lambda calculus arity analysis.
//!
//! Run with: cargo run --example contracts_macros_obligation_arity
//!
//! Added by PMAT-167 (catalog 1126→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ArityVerdict {
    Ok {
        equation_count: u32,
    },
    Mismatch {
        equation: String,
        declared: u32,
        actual: u32,
    },
    EmptyContract,
}

pub fn validate(equations: &[(&str, u32, u32)]) -> ArityVerdict {
    if equations.is_empty() {
        return ArityVerdict::EmptyContract;
    }
    for (name, declared, actual) in equations {
        if declared != actual {
            return ArityVerdict::Mismatch {
                equation: (*name).to_string(),
                declared: *declared,
                actual: *actual,
            };
        }
    }
    ArityVerdict::Ok {
        equation_count: equations.len() as u32,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_obligation_arity")?;

    let consistent = vec![("add", 2, 2), ("mul", 2, 2), ("clamp", 3, 3)];
    println!("ok: {:?}", validate(&consistent));

    let mismatch = vec![("add", 2, 3)];
    println!("mismatch: {:?}", validate(&mismatch));
    println!("empty: {:?}", validate(&[]));
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
    fn consistent_arity_ok() {
        let v = validate(&[("a", 2, 2), ("b", 1, 1)]);
        if let ArityVerdict::Ok { equation_count } = v {
            assert_eq!(equation_count, 2);
        }
    }

    #[test]
    fn mismatch_reported() {
        let v = validate(&[("a", 2, 3)]);
        if let ArityVerdict::Mismatch {
            equation,
            declared,
            actual,
        } = v
        {
            assert_eq!(equation, "a");
            assert_eq!(declared, 2);
            assert_eq!(actual, 3);
        }
    }

    #[test]
    fn empty_rejected() {
        assert_eq!(validate(&[]), ArityVerdict::EmptyContract);
    }

    #[test]
    fn first_mismatch_reported() {
        let v = validate(&[("a", 2, 2), ("bad", 2, 4), ("c", 1, 99)]);
        if let ArityVerdict::Mismatch { equation, .. } = v {
            assert_eq!(equation, "bad");
        }
    }

    #[test]
    fn zero_arity_works() {
        let v = validate(&[("noop", 0, 0)]);
        assert!(matches!(v, ArityVerdict::Ok { .. }));
    }

    #[test]
    fn high_arity_works() {
        let v = validate(&[("many", 100, 100)]);
        assert!(matches!(v, ArityVerdict::Ok { .. }));
    }

    #[test]
    fn declared_zero_actual_one_mismatch() {
        let v = validate(&[("a", 0, 1)]);
        assert!(matches!(v, ArityVerdict::Mismatch { .. }));
    }

    #[test]
    fn declared_one_actual_zero_mismatch() {
        let v = validate(&[("a", 1, 0)]);
        assert!(matches!(v, ArityVerdict::Mismatch { .. }));
    }

    #[test]
    fn many_consistent() {
        let v = validate(&[("a", 1, 1), ("b", 2, 2), ("c", 3, 3), ("d", 4, 4)]);
        if let ArityVerdict::Ok { equation_count } = v {
            assert_eq!(equation_count, 4);
        }
    }

    #[test]
    fn deterministic() {
        let eqs = [("a", 2, 2)];
        let a = validate(&eqs);
        let b = validate(&eqs);
        assert_eq!(a, b);
    }
}
