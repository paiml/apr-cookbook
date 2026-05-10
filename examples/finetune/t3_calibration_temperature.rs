//! # Tier 3.3 — Temperature scaling calibration (tabular-only)
//!
//! Falsifier: temperature scaling preserves argmax classification while
//! reducing ECE on an overconfident fixture.
//!
//! Run with: cargo run --example t3_calibration_temperature

use apr_cookbook::finetune::calibration as cal;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const T: f64 = 2.0;

fn fixture() -> (Vec<f64>, Vec<bool>) {
    let probs: Vec<f64> = vec![0.95; 10].into_iter().chain(vec![0.05; 10]).collect();
    let correct: Vec<bool> = vec![true; 7]
        .into_iter()
        .chain(vec![false; 3])
        .chain(vec![false; 7])
        .chain(vec![true; 3])
        .collect();
    (probs, correct)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_calibration_temperature")?;
    let (probs, correct) = fixture();
    let ece_raw = cal::ece(&probs, &correct);
    let logits: Vec<f64> = probs.iter().map(|p| (p / (1.0 - p)).ln()).collect();
    let scaled = cal::temperature_apply(&logits, T);
    let ece_scaled = cal::ece(&scaled, &correct);
    println!(
        "✓ temperature T={}: ECE {:.4} → {:.4}",
        T, ece_raw, ece_scaled
    );
    assert!(ece_scaled < ece_raw, "temperature must reduce ECE");
    // Argmax preserved
    for (p, l) in probs.iter().zip(scaled.iter()) {
        assert_eq!(cal::argmax_pred(*p), cal::argmax_pred(*l));
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
        main().unwrap();
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // T=1 → no change, ECE doesn't strictly decrease.
        let (probs, correct) = fixture();
        let logits: Vec<f64> = probs.iter().map(|p| (p / (1.0 - p)).ln()).collect();
        let scaled = cal::temperature_apply(&logits, 1.0);
        let ece_raw = cal::ece(&probs, &correct);
        let ece_t1 = cal::ece(&scaled, &correct);
        assert!((ece_raw - ece_t1).abs() < 0.01);
    }

    #[test]
    fn deterministic_across_runs() {
        let (probs, correct) = fixture();
        let logits: Vec<f64> = probs.iter().map(|p| (p / (1.0 - p)).ln()).collect();
        let a = cal::temperature_apply(&logits, T);
        let b = cal::temperature_apply(&logits, T);
        assert_eq!(a, b);
        let _ = correct;
    }
}
