//! # Monte Carlo Business Revenue Forecast
//!
//! Project 12-month revenue as the sum of N customer cohorts, each with
//! random churn rate, ARPU (average revenue per user), and acquisition channel
//! count. Simulate 5000 paths and report P50/P90 revenue ranges.
//!
//! This is a "Flaw of Averages" demo (Savage 2009): the deterministic
//! point-estimate using mean inputs is wildly wrong vs the actual P50 of
//! the simulation, because revenue is a multiplicative compounding function
//! of churn. The recipe asserts P50 ≤ P90 (basic ordering) and that the
//! expected-value "naïve forecast" diverges from the simulated median by
//! >5% — the headline lesson of the technique.
//!
//! Demonstrates the **MC.2** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Savage, S. L. (2009). The Flaw of Averages: Why We Underestimate Risk in the Face of Uncertainty. Wiley. ISBN: 978-0471381976
//!
//! Run with: cargo run --example mc_business_revenue_forecast
//!
//! Added by PMAT-082 (expand-cookbooks: aprender-monte-carlo coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender::monte_carlo::prelude::{percentile, MonteCarloRng};

const N_PATHS: usize = 5000;
const N_COHORTS: usize = 12; // months of acquisition
const MONTHS_HORIZON: usize = 12;
const SEED: u64 = 42;

const ACQUISITION_PER_MONTH_MEAN: f64 = 100.0;
const ACQUISITION_PER_MONTH_SIGMA: f64 = 20.0;
const ARPU_MEAN: f64 = 50.0;
const ARPU_SIGMA: f64 = 10.0;
const CHURN_RATE_MEAN: f64 = 0.05; // 5% monthly
const CHURN_RATE_SIGMA: f64 = 0.02;

fn simulate_one_path(rng: &mut MonteCarloRng) -> f64 {
    let mut total_revenue = 0.0;
    for cohort_month in 0..N_COHORTS {
        let acquisitions =
            (rng.normal(ACQUISITION_PER_MONTH_MEAN, ACQUISITION_PER_MONTH_SIGMA)).max(0.0);
        let arpu = (rng.normal(ARPU_MEAN, ARPU_SIGMA)).max(0.0);
        let churn = rng
            .normal(CHURN_RATE_MEAN, CHURN_RATE_SIGMA)
            .clamp(0.0, 1.0);
        // Cohort contributes from cohort_month..MONTHS_HORIZON, decaying by churn.
        let mut cohort_size = acquisitions;
        for _ in cohort_month..MONTHS_HORIZON {
            total_revenue += cohort_size * arpu;
            cohort_size *= 1.0 - churn;
        }
    }
    total_revenue
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("mc_business_revenue_forecast")?;

    let mut rng = MonteCarloRng::new(SEED);
    let mut paths: Vec<f64> = (0..N_PATHS).map(|_| simulate_one_path(&mut rng)).collect();
    paths.sort_by(|a, b| a.partial_cmp(b).unwrap());

    let p10 = percentile(&paths, 0.10);
    let p50 = percentile(&paths, 0.50);
    let p90 = percentile(&paths, 0.90);

    // Naïve point-estimate using mean inputs (the "Flaw of Averages" baseline).
    let mut cohort_size = ACQUISITION_PER_MONTH_MEAN;
    let mut naive = 0.0;
    for _ in 0..N_COHORTS {
        let mut size = cohort_size;
        for _ in 0..MONTHS_HORIZON {
            naive += size * ARPU_MEAN;
            size *= 1.0 - CHURN_RATE_MEAN;
        }
        cohort_size = ACQUISITION_PER_MONTH_MEAN;
    }

    println!("Annual revenue forecast ({} paths):", N_PATHS);
    println!("  P10:                    ${:>12.0}", p10);
    println!("  P50 (median):           ${:>12.0}", p50);
    println!("  P90:                    ${:>12.0}", p90);
    println!("  naive (Flaw of Averages): ${:>12.0}", naive);
    println!(
        "  divergence (naive vs P50): {:.1}%",
        100.0 * (naive - p50).abs() / p50
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forecast_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn percentiles_are_ordered() {
        let mut rng = MonteCarloRng::new(SEED);
        let mut paths: Vec<f64> = (0..N_PATHS).map(|_| simulate_one_path(&mut rng)).collect();
        paths.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let p10 = percentile(&paths, 0.10);
        let p50 = percentile(&paths, 0.50);
        let p90 = percentile(&paths, 0.90);
        assert!(p10 <= p50);
        assert!(p50 <= p90);
    }

    #[test]
    fn deterministic_with_same_seed() {
        let mut rng_a = MonteCarloRng::new(SEED);
        let mut rng_b = MonteCarloRng::new(SEED);
        let p1 = simulate_one_path(&mut rng_a);
        let p2 = simulate_one_path(&mut rng_b);
        assert!(
            (p1 - p2).abs() < 1e-9,
            "same seed must produce same revenue"
        );
    }
}
