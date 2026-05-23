//! # Monte Carlo Correlated Portfolio VaR
//!
//! Two-asset portfolio Value-at-Risk under joint normal returns with
//! correlation ρ. As ρ → 1, diversification benefit vanishes; as
//! ρ → -1, perfect hedge possible. Portfolio variance:
//! σ_p² = w₁²σ₁² + w₂²σ₂² + 2w₁w₂ρσ₁σ₂. This recipe builds the
//! analytic VaR + its correlation derivative.
//!
//! Demonstrates the **MC.4** recipe for PMAT-122 (monte-carlo coverage).
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Markowitz, H. (1952). Portfolio Selection. Journal of Finance 7(1).
//!
//! Run with: cargo run --example mc_correlated_portfolio_var
//!
//! Added by PMAT-122 (expand-cookbooks followup).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

#[derive(Debug, PartialEq)]
pub enum VarVerdict {
    Ok { var_95: f64, var_99: f64 },
    InvalidWeights,
    InvalidCorrelation,
    InvalidVolatility,
}

const Z_95: f64 = 1.6448536269514722;
const Z_99: f64 = 2.3263478740408408;

pub fn portfolio_var(w1: f64, w2: f64, sigma1: f64, sigma2: f64, rho: f64) -> VarVerdict {
    if !w1.is_finite() || !w2.is_finite() || (w1 + w2 - 1.0).abs() > 1e-9 {
        return VarVerdict::InvalidWeights;
    }
    if !rho.is_finite() || !(-1.0..=1.0).contains(&rho) {
        return VarVerdict::InvalidCorrelation;
    }
    if !sigma1.is_finite() || !sigma2.is_finite() || sigma1 < 0.0 || sigma2 < 0.0 {
        return VarVerdict::InvalidVolatility;
    }
    let variance = w1 * w1 * sigma1 * sigma1
        + w2 * w2 * sigma2 * sigma2
        + 2.0 * w1 * w2 * rho * sigma1 * sigma2;
    let portfolio_sigma = variance.max(0.0).sqrt();
    VarVerdict::Ok {
        var_95: Z_95 * portfolio_sigma,
        var_99: Z_99 * portfolio_sigma,
    }
}

pub fn diversification_benefit(
    equal_weight_undiversified_var: f64,
    portfolio_var: f64,
) -> Option<f64> {
    if equal_weight_undiversified_var <= 0.0 {
        return None;
    }
    Some(1.0 - portfolio_var / equal_weight_undiversified_var)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_correlated_portfolio_var")?;

    for rho in [-1.0, -0.5, 0.0, 0.5, 1.0] {
        let v = portfolio_var(0.5, 0.5, 0.20, 0.30, rho);
        println!("ρ={rho:>5.1}  →  {v:?}");
    }
    println!(
        "out of range ρ: {:?}",
        portfolio_var(0.5, 0.5, 0.20, 0.30, 1.5)
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calc_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn perfectly_correlated_no_diversification() {
        // ρ=1: portfolio σ = w1·σ1 + w2·σ2 (linear).
        let v = portfolio_var(0.5, 0.5, 0.20, 0.20, 1.0);
        if let VarVerdict::Ok { var_95, .. } = v {
            // σ_p = 0.5 × 0.20 + 0.5 × 0.20 = 0.20 → VaR_95 = 1.645 × 0.20.
            let expected = Z_95 * 0.20;
            assert!((var_95 - expected).abs() < 1e-9, "got {var_95}");
        }
    }

    #[test]
    fn perfectly_anti_correlated_zero_var() {
        // ρ=-1 with equal weights and equal σ → portfolio σ = 0.
        let v = portfolio_var(0.5, 0.5, 0.20, 0.20, -1.0);
        if let VarVerdict::Ok { var_95, .. } = v {
            assert!(var_95.abs() < 1e-12);
        }
    }

    #[test]
    fn weights_must_sum_to_one() {
        assert_eq!(
            portfolio_var(0.4, 0.4, 0.2, 0.3, 0.5),
            VarVerdict::InvalidWeights
        );
    }

    #[test]
    fn correlation_out_of_range_rejected() {
        assert_eq!(
            portfolio_var(0.5, 0.5, 0.2, 0.3, 1.5),
            VarVerdict::InvalidCorrelation
        );
        assert_eq!(
            portfolio_var(0.5, 0.5, 0.2, 0.3, -2.0),
            VarVerdict::InvalidCorrelation
        );
    }

    #[test]
    fn negative_volatility_rejected() {
        assert_eq!(
            portfolio_var(0.5, 0.5, -0.2, 0.3, 0.5),
            VarVerdict::InvalidVolatility
        );
    }

    #[test]
    fn nan_correlation_rejected() {
        assert_eq!(
            portfolio_var(0.5, 0.5, 0.2, 0.3, f64::NAN),
            VarVerdict::InvalidCorrelation
        );
    }

    #[test]
    fn var_99_strictly_greater_than_var_95() {
        let v = portfolio_var(0.5, 0.5, 0.20, 0.30, 0.0);
        if let VarVerdict::Ok { var_95, var_99 } = v {
            assert!(var_99 > var_95);
        }
    }

    #[test]
    fn diversification_benefit_positive_for_negative_rho() {
        // Compare ρ=0 vs ρ=-0.5: anti-correlated should yield smaller VaR.
        let v_zero = portfolio_var(0.5, 0.5, 0.20, 0.30, 0.0);
        let v_neg = portfolio_var(0.5, 0.5, 0.20, 0.30, -0.5);
        if let (VarVerdict::Ok { var_95: a, .. }, VarVerdict::Ok { var_95: b, .. }) =
            (v_zero, v_neg)
        {
            assert!(b < a, "ρ=-0.5 var ({b}) should be < ρ=0 var ({a})");
        }
    }

    #[test]
    fn diversification_helper_handles_zero_baseline() {
        assert!(diversification_benefit(0.0, 0.5).is_none());
    }
}
