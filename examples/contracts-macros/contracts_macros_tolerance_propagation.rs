//! # Contracts-Macros Tolerance Propagation
//!
//! When two equations are composed (`f(g(x))`), tolerances accumulate.
//! Conservative bound: `eps_total = |f'| × eps_g + eps_f`. Returns
//! the propagated tolerance for the composed contract.
//!
//! Demonstrates the **CMM.08** recipe for PMAT-160 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Higham (2002) Accuracy and Stability of Numerical Algorithms.
//!
//! Run with: cargo run --example contracts_macros_tolerance_propagation
//!
//! Added by PMAT-160 (catalog 1063→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum ToleranceVerdict {
    Ok { propagated_tol: f64 },
    InvalidInput,
}

pub fn propagate(inner_tol: f64, outer_tol: f64, outer_derivative_max: f64) -> ToleranceVerdict {
    if !inner_tol.is_finite()
        || !outer_tol.is_finite()
        || !outer_derivative_max.is_finite()
        || inner_tol < 0.0
        || outer_tol < 0.0
        || outer_derivative_max < 0.0
    {
        return ToleranceVerdict::InvalidInput;
    }
    let propagated_tol = outer_derivative_max * inner_tol + outer_tol;
    ToleranceVerdict::Ok { propagated_tol }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_tolerance_propagation")?;

    println!("typical: {:?}", propagate(0.01, 0.001, 2.0));
    println!("zero inner: {:?}", propagate(0.0, 0.001, 2.0));
    println!("invalid: {:?}", propagate(-0.01, 0.001, 2.0));
    println!("nan: {:?}", propagate(f64::NAN, 0.001, 2.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn propagator_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn typical_composition() {
        let v = propagate(0.01, 0.001, 2.0);
        if let ToleranceVerdict::Ok { propagated_tol } = v {
            // 2 × 0.01 + 0.001 = 0.021.
            assert!((propagated_tol - 0.021).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_inner_only_outer_remains() {
        let v = propagate(0.0, 0.001, 2.0);
        if let ToleranceVerdict::Ok { propagated_tol } = v {
            assert!((propagated_tol - 0.001).abs() < 1e-9);
        }
    }

    #[test]
    fn zero_outer_scales_inner() {
        let v = propagate(0.01, 0.0, 5.0);
        if let ToleranceVerdict::Ok { propagated_tol } = v {
            assert!((propagated_tol - 0.05).abs() < 1e-9);
        }
    }

    #[test]
    fn negative_inner_invalid() {
        assert_eq!(propagate(-0.01, 0.001, 2.0), ToleranceVerdict::InvalidInput);
    }

    #[test]
    fn negative_outer_invalid() {
        assert_eq!(propagate(0.01, -0.001, 2.0), ToleranceVerdict::InvalidInput);
    }

    #[test]
    fn negative_derivative_invalid() {
        assert_eq!(propagate(0.01, 0.001, -2.0), ToleranceVerdict::InvalidInput);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(
            propagate(f64::NAN, 0.001, 2.0),
            ToleranceVerdict::InvalidInput
        );
    }

    #[test]
    fn very_steep_outer_amplifies() {
        let v = propagate(0.01, 0.0, 1e6);
        if let ToleranceVerdict::Ok { propagated_tol } = v {
            assert!(propagated_tol > 100.0);
        }
    }

    #[test]
    fn all_zero_yields_zero() {
        let v = propagate(0.0, 0.0, 0.0);
        if let ToleranceVerdict::Ok { propagated_tol } = v {
            assert!((propagated_tol - 0.0).abs() < 1e-12);
        }
    }

    #[test]
    fn deterministic() {
        let a = propagate(0.01, 0.001, 2.0);
        let b = propagate(0.01, 0.001, 2.0);
        assert_eq!(a, b);
    }
}
