//! # Contracts-Macros Invariant Witness Pair
//!
//! For a postcondition over a numeric range, find one passing input
//! and one failing input. Returns both witnesses or NoFalsifier if
//! everything passes (or NoVerifier if nothing passes).
//!
//! Demonstrates the **CMM.56** recipe for PMAT-176 (post-milestone).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: model-based testing — pair-wise witness extraction.
//!
//! Run with: cargo run --example contracts_macros_invariant_witness_pair
//!
//! Added by PMAT-176 (catalog 1207→).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum WitnessPairVerdict {
    Pair { passing: f64, failing: f64 },
    NoFalsifier,
    NoVerifier,
    InvalidConfig,
}

pub fn search<F>(predicate: F, low: f64, high: f64, samples: u32) -> WitnessPairVerdict
where
    F: Fn(f64) -> bool,
{
    if !low.is_finite() || !high.is_finite() || high <= low || samples < 2 {
        return WitnessPairVerdict::InvalidConfig;
    }
    let step = (high - low) / f64::from(samples - 1);
    let mut passing: Option<f64> = None;
    let mut failing: Option<f64> = None;
    for i in 0..samples {
        let x = low + step * f64::from(i);
        if predicate(x) {
            if passing.is_none() {
                passing = Some(x);
            }
        } else if failing.is_none() {
            failing = Some(x);
        }
        if passing.is_some() && failing.is_some() {
            break;
        }
    }
    match (passing, failing) {
        (Some(p), Some(f)) => WitnessPairVerdict::Pair {
            passing: p,
            failing: f,
        },
        (Some(_), None) => WitnessPairVerdict::NoFalsifier,
        (None, Some(_)) => WitnessPairVerdict::NoVerifier,
        (None, None) => WitnessPairVerdict::InvalidConfig,
    }
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("contracts_macros_invariant_witness_pair")?;

    println!("partial: {:?}", search(|x| x > 0.5, 0.0, 1.0, 100));
    println!("all pass: {:?}", search(f64::is_finite, 0.0, 1.0, 100));
    println!("none pass: {:?}", search(|x| x > 10.0, 0.0, 1.0, 100));
    println!("invalid: {:?}", search(|_| true, 1.0, 0.0, 100));
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
    fn pair_found_for_threshold() {
        let v = search(|x| x > 0.5, 0.0, 1.0, 100);
        assert!(matches!(v, WitnessPairVerdict::Pair { .. }));
    }

    #[test]
    fn passing_below_threshold() {
        let v = search(|x| x > 0.5, 0.0, 1.0, 100);
        if let WitnessPairVerdict::Pair { passing, failing } = v {
            assert!(passing > 0.5);
            assert!(failing <= 0.5);
        }
    }

    #[test]
    fn always_true_no_falsifier() {
        let v = search(|x| x.is_finite(), 0.0, 1.0, 100);
        assert_eq!(v, WitnessPairVerdict::NoFalsifier);
    }

    #[test]
    fn always_false_no_verifier() {
        let v = search(|_| false, 0.0, 1.0, 100);
        assert_eq!(v, WitnessPairVerdict::NoVerifier);
    }

    #[test]
    fn invalid_low_ge_high() {
        let v = search(|_| true, 1.0, 0.0, 100);
        assert_eq!(v, WitnessPairVerdict::InvalidConfig);
    }

    #[test]
    fn invalid_few_samples() {
        let v = search(|_| true, 0.0, 1.0, 1);
        assert_eq!(v, WitnessPairVerdict::InvalidConfig);
    }

    #[test]
    fn nan_invalid() {
        let v = search(|_| true, f64::NAN, 1.0, 100);
        assert_eq!(v, WitnessPairVerdict::InvalidConfig);
    }

    #[test]
    fn dual_threshold() {
        let v = search(|x| (0.3..=0.7).contains(&x), 0.0, 1.0, 100);
        assert!(matches!(v, WitnessPairVerdict::Pair { .. }));
    }

    #[test]
    fn negative_range_works() {
        let v = search(|x| x > 0.0, -10.0, 10.0, 100);
        assert!(matches!(v, WitnessPairVerdict::Pair { .. }));
    }

    #[test]
    fn deterministic() {
        let a = search(|x| x > 0.5, 0.0, 1.0, 100);
        let b = search(|x| x > 0.5, 0.0, 1.0, 100);
        assert_eq!(a, b);
    }
}
