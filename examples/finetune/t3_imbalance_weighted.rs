//! # Tier 3.4 — Class-weighted CE for imbalance (tabular-only)
//!
//! Falsifier: inverse-frequency weights raise minority-class loss term
//! relative to uniform weighting. Closed-form: weight_minority > 1.0 for
//! 90/10 split.
//!
//! Run with: cargo run --example t3_imbalance_weighted

use apr_cookbook::finetune::imbalance as imb;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const COUNTS: [u32; 2] = [90, 10];

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_imbalance_weighted")?;
    let weights = imb::inverse_freq_weights(&COUNTS);
    println!(
        "✓ inverse-frequency weights (90/10): majority={:.3}, minority={:.3}",
        weights[0], weights[1]
    );
    assert!(
        weights[1] > 1.0,
        "minority weight must exceed 1.0 for 90/10"
    );
    assert!(
        weights[1] > weights[0],
        "minority weight must exceed majority weight"
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
        let w = imb::inverse_freq_weights(&COUNTS);
        assert!(w[1] > w[0]);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Balanced 50/50 → equal weights, minority weight not > majority.
        let w = imb::inverse_freq_weights(&[50, 50]);
        assert!((w[0] - w[1]).abs() < 1e-12);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = imb::inverse_freq_weights(&COUNTS);
        let b = imb::inverse_freq_weights(&COUNTS);
        assert_eq!(a, b);
    }
}
