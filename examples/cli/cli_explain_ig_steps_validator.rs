//! # apr explain --integrated-gradients --steps — Step Count Validator
//!
//! Integrated Gradients approximates the path integral with a Riemann
//! sum of N steps. Empirical recommendations: N ≥ 20 for stable
//! attributions, N = 50 default, N > 200 hits diminishing returns.
//! This recipe builds the validator + completeness-check tolerance.
//!
//! Demonstrates the **EXP.5** recipe for PMAT-114 (apr explain coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: aprender EXP-001 + Sundararajan et al. 2017 (IG)
//!
//! Run with: cargo run --example cli_explain_ig_steps_validator
//!
//! Added by PMAT-114 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum StepsVerdict {
    Ok,
    BelowFloor { recommended: u32 },
    DiminishingReturns { recommended: u32 },
    InvalidZero,
}

const FLOOR: u32 = 20;
const CEILING: u32 = 200;
const DEFAULT_STEPS: u32 = 50;

pub fn classify(steps: u32) -> StepsVerdict {
    if steps == 0 {
        return StepsVerdict::InvalidZero;
    }
    if steps < FLOOR {
        return StepsVerdict::BelowFloor {
            recommended: DEFAULT_STEPS,
        };
    }
    if steps > CEILING {
        return StepsVerdict::DiminishingReturns {
            recommended: CEILING,
        };
    }
    StepsVerdict::Ok
}

pub fn completeness_tolerance(steps: u32) -> f64 {
    // Riemann error scales as O(1/N²) for trapezoidal-style sums.
    if steps == 0 {
        return f64::INFINITY;
    }
    1.0 / (f64::from(steps).powi(2))
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cli_explain_ig_steps_validator")?;

    for n in [0u32, 5, 20, 50, 200, 500] {
        println!(
            "steps={n:>3}  →  {:?}  tol≈{:.6}",
            classify(n),
            completeness_tolerance(n)
        );
    }
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
    fn zero_invalid() {
        assert_eq!(classify(0), StepsVerdict::InvalidZero);
    }

    #[test]
    fn under_floor_rejected() {
        let v = classify(5);
        assert!(matches!(v, StepsVerdict::BelowFloor { recommended: 50 }));
    }

    #[test]
    fn at_floor_optimal() {
        assert_eq!(classify(FLOOR), StepsVerdict::Ok);
    }

    #[test]
    fn default_steps_optimal() {
        assert_eq!(classify(DEFAULT_STEPS), StepsVerdict::Ok);
    }

    #[test]
    fn at_ceiling_optimal() {
        assert_eq!(classify(CEILING), StepsVerdict::Ok);
    }

    #[test]
    fn over_ceiling_diminishing_returns() {
        let v = classify(500);
        assert!(matches!(
            v,
            StepsVerdict::DiminishingReturns { recommended: 200 }
        ));
    }

    #[test]
    fn tolerance_decreases_with_more_steps() {
        let t1 = completeness_tolerance(20);
        let t2 = completeness_tolerance(200);
        assert!(t2 < t1);
    }

    #[test]
    fn tolerance_inverse_square() {
        // tol(50) / tol(100) should be 4 (50² → 100² is 4x).
        let t50 = completeness_tolerance(50);
        let t100 = completeness_tolerance(100);
        assert!((t50 / t100 - 4.0).abs() < 1e-9);
    }

    #[test]
    fn tolerance_zero_steps_infinite() {
        assert!(completeness_tolerance(0).is_infinite());
    }
}
