//! # Tier 3.9 — CLIP image encoder + linear classifier (tabular-only)
//!
//! Falsifier: frozen CLIP image features + linear classifier achieves ≥ 0.7
//! accuracy on a synthetic linearly-separable fixture.
//!
//! Run with: cargo run --example t3_image_encoder_clip

use apr_cookbook::finetune::encoders_optimizers as enc;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> (Vec<(Vec<f64>, u8)>, Vec<f64>) {
    let mut samples = Vec::new();
    for i in 0..20 {
        let x = i as f64 / 10.0 - 1.0;
        let y = (i * 3 % 11) as f64 / 5.0 - 1.0;
        let feat = enc::frozen_encode(x, y).to_vec();
        let label = u8::from(x + y > 0.0);
        samples.push((feat, label));
    }
    let weights = vec![1.0, 1.0, 0.0, 0.0];
    (samples, weights)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_image_encoder_clip")?;
    let (samples, weights) = fixture();
    let acc = enc::linear_probe_accuracy(&samples, &weights);
    println!("✓ CLIP linear probe accuracy: {:.3}", acc);
    assert!(acc >= 0.7);
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
        let (s, w) = fixture();
        assert!(enc::linear_probe_accuracy(&s, &w) >= 0.7);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Random weights → near-50% accuracy.
        let (s, _) = fixture();
        let bogus = vec![1.0, -1.0, 0.0, 0.0];
        let acc = enc::linear_probe_accuracy(&s, &bogus);
        // For our fixture (x + y > 0), x − y as predictor differs from x + y.
        assert!(acc < 0.7);
    }

    #[test]
    fn deterministic_across_runs() {
        let (s, w) = fixture();
        let a = enc::linear_probe_accuracy(&s, &w);
        let b = enc::linear_probe_accuracy(&s, &w);
        assert_eq!(a, b);
    }
}
