//! # Tier 3.3 — Isotonic regression calibration (tabular-only)
//!
//! Falsifier: isotonic regression calibration is monotonic in input score.
//!
//! Run with: cargo run --example t3_calibration_isotonic

use apr_cookbook::finetune::calibration as cal;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> (Vec<f64>, Vec<f64>) {
    let scores = vec![0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.95];
    let targets = vec![0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.7, 0.9, 0.85];
    (scores, targets)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_calibration_isotonic")?;
    let (scores, targets) = fixture();
    let calib = cal::isotonic_pav(&scores, &targets);
    println!(
        "✓ isotonic: input non-monotone targets → calibrated {:?}",
        calib
    );
    let mut indexed: Vec<(usize, f64)> = scores.iter().enumerate().map(|(i, s)| (i, *s)).collect();
    indexed.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    let calib_sorted: Vec<f64> = indexed.iter().map(|(i, _)| calib[*i]).collect();
    for w in calib_sorted.windows(2) {
        assert!(
            w[0] <= w[1] + 1e-12,
            "isotonic output must be non-decreasing: {} > {}",
            w[0],
            w[1]
        );
    }
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
        let c = cal::isotonic_pav(&s, &t);
        let mut idx: Vec<(usize, f64)> = s.iter().enumerate().map(|(i, x)| (i, *x)).collect();
        idx.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let cs: Vec<f64> = idx.iter().map(|(i, _)| c[*i]).collect();
        for w in cs.windows(2) {
            assert!(w[0] <= w[1] + 1e-12);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Verify that non-isotonic raw target is non-monotone (a separate fact
        // from the isotonic output, which IS monotone).
        let (_, targets) = fixture();
        let mut violations = 0;
        for w in targets.windows(2) {
            if w[0] > w[1] {
                violations += 1;
            }
        }
        assert!(violations > 0, "raw fixture targets must have ≥1 violation");
    }

    #[test]
    fn deterministic_across_runs() {
        let (s, t) = fixture();
        let c1 = cal::isotonic_pav(&s, &t);
        let c2 = cal::isotonic_pav(&s, &t);
        assert_eq!(c1, c2);
    }
}
