//! # Tier 1.3 — Tabular regression — Energy
//!
//! OLS on 3-feature periodic synthetic data (hour × day × trend).
//! Falsifier: MAE on energy-consumption synthetic ≤ 5% of target range.
//!
//! Run with: cargo run --example t1_tabular_regression_energy

use apr_cookbook::finetune::tabular_regression as tab;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_tabular_regression_energy/data.jsonl";

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_tabular_regression_energy")?;
    let rows = tab::load_rows(FIXTURE, 3)?;
    let (weights, _mse) = tab::fit_ols(&rows);

    let predictions: Vec<f64> = rows
        .iter()
        .map(|r| {
            r.features
                .iter()
                .zip(weights.iter())
                .map(|(x, w)| x * w)
                .sum()
        })
        .collect();
    let targets: Vec<f64> = rows.iter().map(|r| r.target).collect();
    let target_range = targets.iter().cloned().fold(f64::NEG_INFINITY, f64::max)
        - targets.iter().cloned().fold(f64::INFINITY, f64::min);
    let mae = tab::mae(&predictions, &targets);
    let mae_pct = mae / target_range;

    println!(
        "✓ energy OLS: 3-feature, MAE={:.6} ({:.2}% of range {:.4})",
        mae,
        mae_pct * 100.0,
        target_range
    );
    assert!(
        mae_pct <= 0.05,
        "falsifier: MAE {mae} should be ≤ 5% of target range {target_range}"
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
        let rows = tab::load_rows(FIXTURE, 3).expect("load");
        let (w, _) = tab::fit_ols(&rows);
        assert_eq!(w.len(), 3);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Use random weights instead of OLS — MAE should be much larger.
        let rows = tab::load_rows(FIXTURE, 3).expect("load");
        let bad_weights = vec![10.0, -10.0, 10.0];
        let preds: Vec<f64> = rows
            .iter()
            .map(|r| {
                r.features
                    .iter()
                    .zip(bad_weights.iter())
                    .map(|(x, w)| x * w)
                    .sum()
            })
            .collect();
        let targets: Vec<f64> = rows.iter().map(|r| r.target).collect();
        let mae = tab::mae(&preds, &targets);
        assert!(mae > 1.0, "bad weights should yield MAE >> 0, got {mae}");
    }

    #[test]
    fn deterministic_across_runs() {
        let a = tab::load_rows(FIXTURE, 3).expect("a");
        let b = tab::load_rows(FIXTURE, 3).expect("b");
        assert_eq!(tab::fit_ols(&a).0, tab::fit_ols(&b).0);
    }
}
