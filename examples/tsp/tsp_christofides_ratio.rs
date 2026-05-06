//! # TSP Christofides 1.5-Approximation Ratio
//!
//! Christofides (1976) bounds metric-TSP at 1.5× optimal. This recipe
//! validates that an observed tour length respects the bound: tour ≤
//! 1.5 × known_optimal. Useful as a regression check for new TSP
//! solvers.
//!
//! Demonstrates the **TSP.4** recipe for PMAT-129 (tsp coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Christofides, N. (1976). Worst-case analysis of a new heuristic for the TSP.
//!
//! Run with: cargo run --example tsp_christofides_ratio
//!
//! Added by PMAT-129 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const CHRISTOFIDES_RATIO: f64 = 1.5;

#[derive(Debug, PartialEq)]
pub enum RatioVerdict {
    WithinBound { ratio: f64 },
    ExceedsChristofides { ratio: f64, max: f64 },
    Optimal,
    InvalidInputs,
}

pub fn classify(tour_length: f64, optimal_length: f64) -> RatioVerdict {
    if !tour_length.is_finite() || !optimal_length.is_finite() {
        return RatioVerdict::InvalidInputs;
    }
    if optimal_length <= 0.0 || tour_length < 0.0 {
        return RatioVerdict::InvalidInputs;
    }
    let ratio = tour_length / optimal_length;
    if (ratio - 1.0).abs() < 1e-9 {
        return RatioVerdict::Optimal;
    }
    if ratio <= CHRISTOFIDES_RATIO {
        RatioVerdict::WithinBound { ratio }
    } else {
        RatioVerdict::ExceedsChristofides {
            ratio,
            max: CHRISTOFIDES_RATIO,
        }
    }
}

pub fn excess_pct(tour_length: f64, optimal_length: f64) -> Option<f64> {
    if optimal_length <= 0.0 || !tour_length.is_finite() || !optimal_length.is_finite() {
        return None;
    }
    Some((tour_length - optimal_length) / optimal_length * 100.0)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("tsp_christofides_ratio")?;

    let optimal = 100.0;
    for tour in [100.0, 110.0, 145.0, 160.0, -5.0] {
        println!(
            "tour={tour} optimal={optimal}  →  {:?}",
            classify(tour, optimal)
        );
    }
    println!("excess(120, 100) = {:?}%", excess_pct(120.0, 100.0));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classifier_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn exact_optimal_classified() {
        assert_eq!(classify(100.0, 100.0), RatioVerdict::Optimal);
    }

    #[test]
    fn within_bound_passes() {
        let v = classify(140.0, 100.0);
        assert!(matches!(v, RatioVerdict::WithinBound { .. }));
    }

    #[test]
    fn at_bound_passes() {
        // 150 / 100 = 1.5 exactly.
        let v = classify(150.0, 100.0);
        assert!(matches!(v, RatioVerdict::WithinBound { ratio } if (ratio - 1.5).abs() < 1e-9));
    }

    #[test]
    fn just_over_bound_rejected() {
        let v = classify(150.0001, 100.0);
        assert!(matches!(v, RatioVerdict::ExceedsChristofides { .. }));
    }

    #[test]
    fn way_over_bound_rejected() {
        let v = classify(300.0, 100.0);
        assert!(matches!(v, RatioVerdict::ExceedsChristofides { .. }));
    }

    #[test]
    fn zero_optimal_invalid() {
        assert_eq!(classify(100.0, 0.0), RatioVerdict::InvalidInputs);
    }

    #[test]
    fn negative_inputs_invalid() {
        assert_eq!(classify(-1.0, 100.0), RatioVerdict::InvalidInputs);
        assert_eq!(classify(100.0, -1.0), RatioVerdict::InvalidInputs);
    }

    #[test]
    fn nan_invalid() {
        assert_eq!(classify(f64::NAN, 100.0), RatioVerdict::InvalidInputs);
    }

    #[test]
    fn excess_pct_positive_when_over() {
        let pct = excess_pct(120.0, 100.0).unwrap();
        assert!((pct - 20.0).abs() < 1e-9);
    }

    #[test]
    fn excess_pct_zero_at_optimal() {
        let pct = excess_pct(100.0, 100.0).unwrap();
        assert!(pct.abs() < 1e-9);
    }
}
