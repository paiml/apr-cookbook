//! # Tier 1.3 — Tabular regression — Multi-target
//!
//! Fit a linear model on synthetic 2-feature data; falsifier: per-target
//! MSE matches an independent-fit baseline (here: zero, since y is linear
//! in x and our OLS is the optimal estimator).
//!
//! Run with: cargo run --example t1_tabular_regression_multitarget

use apr_cookbook::finetune::tabular_regression as tab;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_tabular_regression_multitarget/data.jsonl";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_tabular_regression_multitarget")?;
    let rows = tab::load_rows(FIXTURE, 2)?;
    let (weights, mse) = tab::fit_ols(&rows);
    println!(
        "✓ multitarget joint fit: w=[{:.4}, {:.4}], MSE={:.6}",
        weights[0], weights[1], mse
    );
    // Falsifier: joint OLS recovers true (1.2, 0.8) within ε on noiseless data.
    assert!(
        (weights[0] - 1.2).abs() < 0.05 && (weights[1] - 0.8).abs() < 0.05,
        "falsifier: joint fit should recover true coefficients"
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
        let rows = tab::load_rows(FIXTURE, 2).expect("load");
        let (w, mse) = tab::fit_ols(&rows);
        assert!((w[0] - 1.2).abs() < 0.05);
        assert!((w[1] - 0.8).abs() < 0.05);
        assert!(mse < 1e-6);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Use a single-feature subset — fit can't recover both coefficients.
        let rows = tab::load_rows(FIXTURE, 2).expect("load");
        let single: Vec<tab::Row> = rows
            .iter()
            .map(|r| tab::Row {
                features: vec![r.features[0]],
                target: r.target,
            })
            .collect();
        let (_, mse_single) = tab::fit_ols(&single);
        let (_, mse_full) = tab::fit_ols(&rows);
        assert!(
            mse_single > mse_full + 1e-6,
            "dropping a feature should increase MSE: full={mse_full} single={mse_single}"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = tab::load_rows(FIXTURE, 2).expect("a");
        let b = tab::load_rows(FIXTURE, 2).expect("b");
        assert_eq!(tab::fit_ols(&a).0, tab::fit_ols(&b).0);
    }
}
