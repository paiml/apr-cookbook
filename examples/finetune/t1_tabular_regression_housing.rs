//! # Tier 1.3 — Tabular regression — Housing (linear)
//!
//! OLS on synthetic 2-feature linear data y = 2*x1 + 3*x2 + ε.
//! Falsifier: MSE converges to noise floor σ² (≤ 0.04 here).
//!
//! Run with: cargo run --example t1_tabular_regression_housing

use apr_cookbook::finetune::tabular_regression as tab;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const FIXTURE: &str = "tests/fixtures/finetune/t1_tabular_regression_housing/data.jsonl";
const NOISE_FLOOR: f64 = 0.04; // σ² upper bound for the synthetic noise

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t1_tabular_regression_housing")?;
    let rows = tab::load_rows(FIXTURE, 2)?;
    let (weights, mse) = tab::fit_ols(&rows);
    println!(
        "✓ housing OLS: w=[{:.4}, {:.4}], MSE={:.6} (noise floor ≤ {})",
        weights[0], weights[1], mse, NOISE_FLOOR
    );
    assert!(
        mse <= NOISE_FLOOR,
        "falsifier: MSE {mse} should be at-or-below noise floor {NOISE_FLOOR}"
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
        assert_eq!(w.len(), 2);
        assert!(mse <= NOISE_FLOOR);
        // OLS recovers true coefficients within ε
        assert!((w[0] - 2.0).abs() < 0.1);
        assert!((w[1] - 3.0).abs() < 0.1);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Perturb targets randomly — MSE should exceed the noise floor.
        let mut rows = tab::load_rows(FIXTURE, 2).expect("load");
        for (i, r) in rows.iter_mut().enumerate() {
            r.target += (i as f64 % 5.0) * 0.3;
        }
        let (_, mse) = tab::fit_ols(&rows);
        assert!(
            mse > NOISE_FLOOR,
            "perturbed targets should violate noise-floor: mse={mse}"
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let r1 = tab::load_rows(FIXTURE, 2).expect("a");
        let r2 = tab::load_rows(FIXTURE, 2).expect("b");
        let (w1, m1) = tab::fit_ols(&r1);
        let (w2, m2) = tab::fit_ols(&r2);
        assert_eq!(w1, w2);
        assert_eq!(m1, m2);
    }
}
