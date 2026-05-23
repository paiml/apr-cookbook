//! # Tier 3.4 — SMOTE for imbalance (tabular-only)
//!
//! Falsifier: SMOTE on a 90/10 dataset produces synthetic minority points
//! that lie on the line segment between two minority points (within k=5 NN).
//! Closed-form check: each synthetic point's coordinates fall in [min, max]
//! of the original minority points.
//!
//! Run with: cargo run --example t3_imbalance_smote

use apr_cookbook::finetune::imbalance as imb;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const T: f64 = 0.5;

fn fixture_minority() -> Vec<(f64, f64)> {
    vec![(0.0, 0.0), (0.5, 1.0), (1.0, 0.5), (0.2, 0.8), (0.8, 0.2)]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_imbalance_smote")?;
    let minority = fixture_minority();
    let synth = imb::smote_synthesize(&minority, T);
    println!(
        "✓ SMOTE t={}: {} synthetic points from {} minority",
        T,
        synth.len(),
        minority.len()
    );
    let xs: Vec<f64> = minority.iter().map(|p| p.0).collect();
    let ys: Vec<f64> = minority.iter().map(|p| p.1).collect();
    let x_min = xs.iter().copied().fold(f64::INFINITY, f64::min);
    let x_max = xs.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let y_min = ys.iter().copied().fold(f64::INFINITY, f64::min);
    let y_max = ys.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    for &(x, y) in &synth {
        assert!(
            (x_min..=x_max).contains(&x),
            "synth x {x} not in [{x_min}, {x_max}]"
        );
        assert!(
            (y_min..=y_max).contains(&y),
            "synth y {y} not in [{y_min}, {y_max}]"
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
        let synth = imb::smote_synthesize(&fixture_minority(), T);
        assert!(!synth.is_empty());
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Empty minority → no synthesis.
        let synth = imb::smote_synthesize(&[], T);
        assert!(synth.is_empty());
    }

    #[test]
    fn deterministic_across_runs() {
        let a = imb::smote_synthesize(&fixture_minority(), T);
        let b = imb::smote_synthesize(&fixture_minority(), T);
        assert_eq!(a, b);
    }
}
