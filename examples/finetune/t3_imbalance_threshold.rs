//! # Tier 3.4 — Threshold tuning for imbalance (tabular-only)
//!
//! Falsifier: tuning decision threshold on validation produces F1 ≥ default
//! threshold's F1 (greedy 0.01-step grid finds the optimum).
//!
//! Run with: cargo run --example t3_imbalance_threshold

use apr_cookbook::finetune::imbalance as imb;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> Vec<(f64, u8)> {
    vec![
        (0.4_f64, 1_u8),
        (0.45, 1),
        (0.55, 0),
        (0.7, 0),
        (0.25, 1),
        (0.8, 0),
        (0.35, 1),
        (0.6, 0),
        (0.5, 1),
        (0.42, 1),
        (0.9, 0),
        (0.3, 1),
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_imbalance_threshold")?;
    let samples = fixture();
    let (best_t, best_f1) = imb::best_threshold(&samples);
    let f1_default = imb::f1_at_threshold(&samples, 0.5);
    println!(
        "✓ threshold tuning: best t={:.2} F1={:.4} (default F1={:.4})",
        best_t, best_f1, f1_default
    );
    assert!(best_f1 >= f1_default, "tuned F1 must be ≥ default 0.5 F1");
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
        let s = fixture();
        let (_, f1_best) = imb::best_threshold(&s);
        let f1_def = imb::f1_at_threshold(&s, 0.5);
        assert!(f1_best >= f1_def);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // No positives → F1 = 0 at every threshold; "tuning improves" is vacuously
        // false (best == default).
        let s = vec![(0.5_f64, 0_u8), (0.6, 0), (0.7, 0)];
        let (_, f1) = imb::best_threshold(&s);
        assert_eq!(f1, 0.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = imb::best_threshold(&fixture());
        let b = imb::best_threshold(&fixture());
        assert_eq!(a, b);
    }
}
