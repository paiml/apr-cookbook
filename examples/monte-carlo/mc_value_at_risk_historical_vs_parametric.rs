//! # Value at Risk: Historical vs Parametric
//!
//! Compute 1-day 95% VaR on a synthetic returns series two ways:
//! - **Historical VaR**: 5th percentile of the empirical return distribution.
//! - **Parametric VaR**: μ + z₀.₉₅ × σ assuming normality (z₀.₉₅ ≈ 1.645).
//!
//! For normally-distributed inputs the two methods should converge as the
//! sample size → ∞. The recipe asserts they agree within tolerance for a
//! 10,000-sample Normal(μ, σ) input.
//!
//! Demonstrates the **MC.3** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md` —
//! the "two ways to skin the same cat" comparison that makes risk
//! analysts comfortable with the framework's Monte Carlo path matching
//! their textbook Normal-assumption VaR.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Jorion, P. (2007). Value at Risk: The New Benchmark for Managing Financial Risk (3rd ed). McGraw-Hill. ISBN: 978-0071464956
//!
//! Run with: cargo run --example mc_value_at_risk_historical_vs_parametric
//!
//! Added by PMAT-082 (expand-cookbooks: aprender-monte-carlo coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender::monte_carlo::prelude::{MonteCarloRng, VaR};

const N_RETURNS: usize = 10_000;
const MU: f64 = 0.001; // 0.1% daily mean return
const SIGMA: f64 = 0.02; // 2% daily volatility
const CONFIDENCE: f64 = 0.95;
const SEED: u64 = 42;

fn synthesize_returns() -> Vec<f64> {
    let mut rng = MonteCarloRng::new(SEED);
    (0..N_RETURNS).map(|_| rng.normal(MU, SIGMA)).collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_value_at_risk_historical_vs_parametric")?;

    let returns = synthesize_returns();

    let var_historical = VaR::historical(&returns, CONFIDENCE);
    // Parametric VaR for a Normal(μ, σ) input: -(μ - z * σ) at the (1-α) tail.
    // z₀.₉₅ ≈ 1.645 — closed-form rather than calling into stats crate.
    let z = 1.645;
    let var_parametric = -(MU - z * SIGMA);

    println!(
        "VaR_{:.0}% on {} synthetic Normal(μ={}, σ={}) returns:",
        CONFIDENCE * 100.0,
        N_RETURNS,
        MU,
        SIGMA
    );
    println!("  historical:  {:.6}", var_historical);
    println!("  parametric:  {:.6}", var_parametric);
    println!(
        "  abs diff:    {:.6}",
        (var_historical - var_parametric).abs()
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn comparison_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn historical_and_parametric_agree_for_normal_input() {
        let returns = synthesize_returns();
        let var_historical = VaR::historical(&returns, CONFIDENCE);
        let var_parametric = -(MU - 1.645 * SIGMA);
        // For 10k Normal samples the two should agree within 0.005 (absolute).
        let diff = (var_historical - var_parametric).abs();
        assert!(
            diff < 0.005,
            "historical VaR {:.6} vs parametric {:.6} differ by {:.6} (tol 0.005)",
            var_historical,
            var_parametric,
            diff
        );
    }

    #[test]
    fn var_is_non_negative() {
        // Convention: VaR is reported as a positive number (potential loss).
        let returns = synthesize_returns();
        let var = VaR::historical(&returns, CONFIDENCE);
        assert!(var >= 0.0, "VaR should be non-negative, got {var}");
    }
}
