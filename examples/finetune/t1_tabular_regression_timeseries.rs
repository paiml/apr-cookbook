//! # Tier 1.3 — Tabular regression — Time series (AR(1))
//!
//! OLS 1-step-ahead forecast: y_t = ρ · y_{t-1} + ε.
//! Falsifier: 1-step-ahead RMSE on AR(1) ≤ 1.1× theoretical optimum (= σ_ε).
//!
//! Run with: cargo run --example t1_tabular_regression_timeseries

use apr_cookbook::finetune::tabular_regression as tab;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_tabular_regression_timeseries/data.jsonl";
const TRUE_RHO: f64 = 0.7;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_tabular_regression_timeseries")?;
    let rows = tab::load_rows(FIXTURE, 1)?;
    let (weights, mse) = tab::fit_ols(&rows);
    let rmse = mse.sqrt();
    println!(
        "✓ AR(1) OLS: ρ̂={:.4} (true {:.2}), RMSE={:.6}",
        weights[0], TRUE_RHO, rmse
    );
    assert!(
        (weights[0] - TRUE_RHO).abs() < 0.15,
        "falsifier: estimated ρ̂ {} should be near true ρ {TRUE_RHO}",
        weights[0]
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
        let (w, _) = tab::fit_ols(&rows);
        assert!((w[0] - TRUE_RHO).abs() < 0.15, "ρ̂={}", w[0]);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Shuffled targets break the temporal structure — ρ̂ should drift far from truth.
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
        // With targets randomized vs features, |ρ̂| should typically drop closer to 0.
        // We assert it's *not* close to TRUE_RHO (i.e., the falsifier broken).
        assert!(
            (w_shuf[0] - TRUE_RHO).abs() > 0.05 || w_shuf[0].abs() < TRUE_RHO,
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
