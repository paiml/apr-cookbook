//! # CGP — Regression Detector: Baseline vs Current
//!
//! Use `aprender_cgp::analysis::regression::RegressionDetector` (bootstrap
//! confidence intervals per Hoefler & Belli) to compare two synthetic
//! benchmark sample series:
//! - **No-change**: same distribution → verdict `NoChange`.
//! - **Regression**: 10% slower mean with significant effect size →
//!   verdict `Regression`.
//!
//! Use as a CI gate template: capture timing samples, compare to a stored
//! baseline JSON, fail the build on `Regression` verdict.
//!
//! Demonstrates the **CGP.1** recipe per
//! `docs/specifications/expand-cookbooks/subcrate-coverage.md`.
//!
//! Contract: contracts/recipe-iiur-v1.yaml
//! Citation: Hoefler, T. & Belli, R. (2015). Scientific Benchmarking of Parallel Computing Systems. SC '15. DOI: 10.1145/2807591.2807644
//!
//! Run with: cargo run --example cgp_regression_detector_baseline_vs_current
//!
//! Added by PMAT-083 (expand-cookbooks: aprender-cgp coverage).

use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;
use aprender::monte_carlo::prelude::MonteCarloRng;
use cgp::analysis::regression::RegressionDetector;

const SEED: u64 = 42;
const N: usize = 100;

fn synth(rng: &mut MonteCarloRng, mean: f64, sigma: f64) -> Vec<f64> {
    (0..N).map(|_| rng.normal(mean, sigma)).collect()
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("cgp_regression_detector_baseline_vs_current")?;

    let mut rng = MonteCarloRng::new(SEED);
    let baseline = synth(&mut rng, 100.0, 5.0);
    let same = synth(&mut rng, 100.0, 5.0);
    let slower = synth(&mut rng, 110.0, 5.0); // ~10% slower

    let detector = RegressionDetector::new();
    let no_change = detector.compare(&baseline, &same);
    let regression = detector.compare(&baseline, &slower);

    println!(
        "no-change: verdict={:?} change_pct={:.2}% effect_size={:.2}",
        no_change.verdict, no_change.change_pct, no_change.effect_size_cohens_d
    );
    println!(
        "regression: verdict={:?} change_pct={:.2}% effect_size={:.2}",
        regression.verdict, regression.change_pct, regression.effect_size_cohens_d
    );

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use cgp::analysis::regression::Verdict;

    #[test]
    fn detector_runs() {
        main().expect("recipe execution failed");
    }

    #[test]
    fn ten_percent_slowdown_detected_as_regression() {
        let mut rng = MonteCarloRng::new(SEED);
        let baseline = synth(&mut rng, 100.0, 5.0);
        let slower = synth(&mut rng, 110.0, 5.0);
        let detector = RegressionDetector::new();
        let result = detector.compare(&baseline, &slower);
        assert!(
            matches!(result.verdict, Verdict::Regression),
            "expected Regression for 10% slowdown, got {:?} (change_pct={:.2}%)",
            result.verdict,
            result.change_pct
        );
    }

    #[test]
    fn empty_inputs_return_no_change() {
        let detector = RegressionDetector::new();
        let result = detector.compare(&[], &[]);
        assert!(matches!(result.verdict, Verdict::NoChange));
    }
}
