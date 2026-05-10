//! # Tier 1.3 — Tabular regression — Time series (AR(1))
//!
//! OLS 1-step-ahead forecast: y_t = ρ · y_{t-1} + ε.
//! Falsifier: 1-step-ahead RMSE on AR(1) ≤ RMSE_BOUND on the small-N fixture,
//! and ρ̂ shows positive autocorrelation (ρ̂ > 0.1).
//!
//! Run with: cargo run --example t1_tabular_regression_timeseries

use apr_cookbook::finetune::tabular_regression as tab;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_tabular_regression_timeseries/data.jsonl";
const RMSE_BOUND: f64 = 0.1;
const RHO_FLOOR: f64 = 0.1;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_tabular_regression_timeseries")?;
    let rows = tab::load_rows(FIXTURE, 1)?;
    let (weights, mse) = tab::fit_ols(&rows);
    let rmse = mse.sqrt();
    println!(
        "✓ AR(1) OLS: ρ̂={:.4} (autocorrelation > {RHO_FLOOR}), RMSE={:.6} (≤ {RMSE_BOUND})",
        weights[0], rmse
    );
    assert!(
        weights[0] > RHO_FLOOR,
        "falsifier: ρ̂ {} should show positive autocorrelation > {RHO_FLOOR}",
        weights[0]
    );
    assert!(
        rmse < RMSE_BOUND,
        "falsifier: 1-step RMSE {rmse} should be ≤ {RMSE_BOUND}"
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recipe_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn falsifier_holds_on_fixture() {
        let rows = tab::load_rows(FIXTURE, 1).expect("load");
        let (w, mse) = tab::fit_ols(&rows);
        assert!(w[0] > RHO_FLOOR, "ρ̂={}", w[0]);
        assert!(mse.sqrt() < RMSE_BOUND, "RMSE={}", mse.sqrt());
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Shuffled targets break the temporal structure — ρ̂ should drop near zero.
        let rows = tab::load_rows(FIXTURE, 1).expect("load");
        let mut shuffled: Vec<tab::Row> = rows.clone();
        let len = shuffled.len();
        for i in 0..len {
            let j = (i * 17 + 5) % len;
            let tmp = shuffled[i].target;
            shuffled[i].target = shuffled[j].target;
            shuffled[j].target = tmp;
        }
        let (w_shuf, _) = tab::fit_ols(&shuffled);
        assert!(
            w_shuf[0].abs() < 0.5,
            "shuffled targets should disturb ρ̂ estimate: w={}",
            w_shuf[0]
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = tab::load_rows(FIXTURE, 1).expect("a");
        let b = tab::load_rows(FIXTURE, 1).expect("b");
        assert_eq!(tab::fit_ols(&a).0, tab::fit_ols(&b).0);
    }
}
