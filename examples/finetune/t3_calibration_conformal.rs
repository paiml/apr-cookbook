//! # Tier 3.3 — Conformal prediction calibration (tabular-only)
//!
//! Falsifier: conformal prediction at α=0.1 yields empirical coverage in
//! [0.85, 1.0] on a 100-sample calibration fixture (split-conformal upper
//! bound is at most 1.0 with finite samples).
//!
//! Run with: cargo run --example t3_calibration_conformal

use apr_cookbook::finetune::calibration as cal;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const ALPHA: f64 = 0.1;
const N: usize = 100;

fn fixture() -> (Vec<f64>, Vec<f64>) {
    let scores: Vec<f64> = (0..N).map(|i| i as f64 * 0.01).collect();
    // Varying residuals: target = score + (i-th cyclic residual in [0.01, 0.10])
    let targets: Vec<f64> = scores
        .iter()
        .enumerate()
        .map(|(i, s)| s + 0.01 + (i as f64 % 10.0) * 0.01)
        .collect();
    (scores, targets)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_calibration_conformal")?;
    let (scores, targets) = fixture();
    let cov = cal::conformal_coverage(&scores, &targets, ALPHA);
    println!("✓ conformal α={}: empirical coverage = {:.4}", ALPHA, cov);
    assert!(
        (0.85..=1.0).contains(&cov),
        "coverage must be in [0.85, 1.0], got {cov}"
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
        let (s, t) = fixture();
        let c = cal::conformal_coverage(&s, &t, ALPHA);
        assert!((0.85..=1.0).contains(&c));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // α=0.5 yields ~50% coverage, breaking the [0.85, 1.0] bound.
        let (s, t) = fixture();
        let c = cal::conformal_coverage(&s, &t, 0.5);
        assert!(c < 0.85);
    }

    #[test]
    fn deterministic_across_runs() {
        let (s, t) = fixture();
        let c1 = cal::conformal_coverage(&s, &t, ALPHA);
        let c2 = cal::conformal_coverage(&s, &t, ALPHA);
        assert_eq!(c1, c2);
    }
}
