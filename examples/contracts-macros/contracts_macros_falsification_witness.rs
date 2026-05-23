//! # Contracts-Macros Falsification Witness
//!
//! When a contract postcondition fails, search for a witness: an
//! input where output ≠ expected within tolerance. Returns the
//! first counterexample found over a uniform grid.
//!
//! Demonstrates the **CMM.19** recipe for PMAT-164 (catalog crosses 1100).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: SmallCheck / QuickCheck-style enumerative falsification.
//!
//! Run with: cargo run --example contracts_macros_falsification_witness
//!
//! Added by PMAT-164 (catalog 1099→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WitnessVerdict {
    NoCounterexample,
    Found {
        input: f64,
        actual: f64,
        expected: f64,
        delta: f64,
    },
    InvalidConfig,
}

/// Contract claim: f(x) ≈ g(x) within tolerance for x in [low, high].
pub fn search<F, G>(f: F, g: G, low: f64, high: f64, tolerance: f64, samples: u32) -> WitnessVerdict
where
    F: Fn(f64) -> f64,
    G: Fn(f64) -> f64,
{
    if !low.is_finite()
        || !high.is_finite()
        || high <= low
        || !tolerance.is_finite()
        || tolerance < 0.0
        || samples < 2
    {
        return WitnessVerdict::InvalidConfig;
    }
    let step = (high - low) / f64::from(samples - 1);
    for i in 0..samples {
        let x = low + step * f64::from(i);
        let actual = f(x);
        let expected = g(x);
        let delta = (actual - expected).abs();
        if !actual.is_finite() || delta > tolerance {
            return WitnessVerdict::Found {
                input: x,
                actual,
                expected,
                delta,
            };
        }
    }
    WitnessVerdict::NoCounterexample
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_falsification_witness")?;

    // f(x) = x^2, g(x) = x*x — equivalent.
    let v1 = search(|x| x.powi(2), |x| x * x, -10.0, 10.0, 1e-9, 100);
    println!("equivalent: {v1:?}");

    // f(x) = x, g(x) = x + 0.01 — not equivalent.
    let v2 = search(|x| x, |x| x + 0.01, -10.0, 10.0, 1e-9, 100);
    println!("not equivalent: {v2:?}");

    let v3 = search(|x| x, |x| x, -10.0, 10.0, -1.0, 100);
    println!("invalid: {v3:?}");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn searcher_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn equivalent_no_counterexample() {
        let v = search(|x| x * x, |x| x.powi(2), -10.0, 10.0, 1e-9, 100);
        assert_eq!(v, WitnessVerdict::NoCounterexample);
    }

    #[test]
    fn off_by_constant_finds_witness() {
        let v = search(|x| x, |x| x + 0.5, 0.0, 10.0, 0.01, 100);
        assert!(matches!(v, WitnessVerdict::Found { .. }));
    }

    #[test]
    fn delta_carries_actual_value() {
        let v = search(|x| x, |x| x + 1.0, 0.0, 10.0, 0.01, 100);
        if let WitnessVerdict::Found { delta, .. } = v {
            assert!((delta - 1.0).abs() < 1e-9);
        }
    }

    #[test]
    fn invalid_neg_tolerance() {
        let v = search(|x| x, |x| x, -1.0, 1.0, -0.5, 10);
        assert_eq!(v, WitnessVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_low_ge_high() {
        let v = search(|x| x, |x| x, 5.0, 5.0, 0.5, 10);
        assert_eq!(v, WitnessVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_too_few_samples() {
        let v = search(|x| x, |x| x, -1.0, 1.0, 0.5, 1);
        assert_eq!(v, WitnessVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        let v = search(|x| x, |x| x, f64::NAN, 1.0, 0.5, 10);
        assert_eq!(v, WitnessVerdict::InvalidConfig);
    }

    #[test]
    fn nan_function_output_witness() {
        // If f(x) returns NaN anywhere, that's a counterexample (delta is NaN).
        let v = search(|_| f64::NAN, |x| x, 0.0, 10.0, 0.5, 10);
        assert!(matches!(v, WitnessVerdict::Found { .. }));
    }

    #[test]
    fn within_tolerance_passes() {
        let v = search(|x| x + 0.001, |x| x, 0.0, 10.0, 0.01, 100);
        assert_eq!(v, WitnessVerdict::NoCounterexample);
    }

    #[test]
    fn deterministic() {
        let a = search(|x| x, |x| x + 0.5, 0.0, 1.0, 0.01, 50);
        let b = search(|x| x, |x| x + 0.5, 0.0, 1.0, 0.01, 50);
        assert_eq!(a, b);
    }
}
