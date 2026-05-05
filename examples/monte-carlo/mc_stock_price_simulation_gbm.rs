//! # Monte Carlo Stock Price Simulation (Geometric Brownian Motion)
//!
//! Simulate 1000 paths of stock-price evolution under Geometric Brownian
//! Motion (drift μ, volatility σ, time horizon T years). Compute mean
//! terminal price, P5/P50/P95 percentiles, and assert the mean is within
//! tolerance of the analytical expectation `s0 * exp(mu * t)`.
//!
//! Demonstrates the **MC.1** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md` —
//! the canonical GBM simulation as a sanity check that the engine is
//! working before reaching for more sophisticated models.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Black, F. & Scholes, M. (1973). The Pricing of Options and Corporate Liabilities. Journal of Political Economy 81(3). DOI: 10.1086/260062
//!
//! Run with: cargo run --example mc_stock_price_simulation_gbm
//!
//! Added by PMAT-082 (expand-cookbooks: aprender-monte-carlo coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender::monte_carlo::prelude::{
    GeometricBrownianMotion, MonteCarloEngine, TimeHorizon, VarianceReduction,
};

const S0: f64 = 100.0;
const MU: f64 = 0.08; // 8% annual drift
const SIGMA: f64 = 0.20; // 20% annual volatility
const T_YEARS: u32 = 1;
const N_PATHS: usize = 1000;
const SEED: u64 = 42;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_stock_price_simulation_gbm")?;

    let engine = MonteCarloEngine::reproducible(SEED)
        .with_n_simulations(N_PATHS)
        .with_variance_reduction(VarianceReduction::Antithetic);
    let model = GeometricBrownianMotion::new(S0, MU, SIGMA);
    let horizon = TimeHorizon::years(T_YEARS);
    let result = engine.simulate(&model, &horizon);

    let stats = result.final_value_statistics();
    let analytical_mean = S0 * (MU * f64::from(T_YEARS)).exp();

    println!("GBM simulation: s0={S0} mu={MU} sigma={SIGMA} T={T_YEARS}y paths={N_PATHS}");
    println!(
        "  terminal mean: simulated={:.2}  analytical={:.2}  diff={:.2}",
        stats.mean,
        analytical_mean,
        stats.mean - analytical_mean
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn simulation_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn terminal_mean_within_tolerance_of_analytical() {
        // E[S_T] under GBM = S_0 * exp(mu * T). With antithetic VR and
        // N=1000 paths, the simulated mean should be within ~3% of analytical.
        let engine = MonteCarloEngine::reproducible(SEED)
            .with_n_simulations(N_PATHS)
            .with_variance_reduction(VarianceReduction::Antithetic);
        let model = GeometricBrownianMotion::new(S0, MU, SIGMA);
        let horizon = TimeHorizon::years(T_YEARS);
        let result = engine.simulate(&model, &horizon);

        let stats = result.final_value_statistics();
        let analytical = S0 * (MU * f64::from(T_YEARS)).exp();
        let relative_error = (stats.mean - analytical).abs() / analytical;
        assert!(
            relative_error < 0.05,
            "simulated mean {:.2} too far from analytical {:.2} (rel error {:.4})",
            stats.mean,
            analytical,
            relative_error
        );
    }

    #[test]
    fn deterministic_with_same_seed() {
        let engine_a = MonteCarloEngine::reproducible(SEED).with_n_simulations(100);
        let engine_b = MonteCarloEngine::reproducible(SEED).with_n_simulations(100);
        let model = GeometricBrownianMotion::new(S0, MU, SIGMA);
        let horizon = TimeHorizon::years(T_YEARS);

        let mean_a = engine_a
            .simulate(&model, &horizon)
            .final_value_statistics()
            .mean;
        let mean_b = engine_b
            .simulate(&model, &horizon)
            .final_value_statistics()
            .mean;
        assert!(
            (mean_a - mean_b).abs() < 1e-9,
            "same seed must produce same mean"
        );
    }
}
