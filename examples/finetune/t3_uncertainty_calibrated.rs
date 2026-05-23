//! # Tier 3.8 — Calibrated uncertainty intervals (tabular-only)
//!
//! Falsifier: 90% confidence interval [pred − 1.645σ, pred + 1.645σ]
//! contains the true target on a fixture where the prediction is close
//! enough to truth.
//!
//! Run with: cargo run --example t3_uncertainty_calibrated

use apr_cookbook::finetune::anomaly_open_uncertainty as aou;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const PRED: f64 = 0.5;
const SIGMA: f64 = 0.2;
const Z_90: f64 = 1.645;
const TARGET: f64 = 0.6;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_uncertainty_calibrated")?;
    let inside = aou::ci_contains(PRED, SIGMA, Z_90, TARGET);
    println!(
        "✓ calibrated CI: pred={} σ={} z={} target={} → {}",
        PRED,
        SIGMA,
        Z_90,
        TARGET,
        if inside { "inside" } else { "outside" }
    );
    assert!(inside);
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
        assert!(aou::ci_contains(PRED, SIGMA, Z_90, TARGET));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Target far away → outside the CI.
        assert!(!aou::ci_contains(PRED, SIGMA, Z_90, 5.0));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = aou::ci_contains(PRED, SIGMA, Z_90, TARGET);
        let b = aou::ci_contains(PRED, SIGMA, Z_90, TARGET);
        assert_eq!(a, b);
    }
}
