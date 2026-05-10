//! # Tier 3.9 — DINOv2 linear probe (tabular-only)
//!
//! Falsifier: linear probe accuracy on a separable synthetic fixture is
//! at least 90% of an oracle full-finetune-equivalent (here both achieve
//! 100% since the fixture is linearly separable).
//!
//! Run with: cargo run --example t3_image_encoder_dinov2_lp

use apr_cookbook::finetune::encoders_optimizers as enc;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture() -> (Vec<(Vec<f64>, u8)>, Vec<f64>) {
    let mut samples = Vec::new();
    for i in 0..20 {
        let x = (i % 10) as f64 / 5.0 - 1.0;
        let y = (i / 2) as f64 / 5.0 - 1.0;
        let feat = enc::frozen_encode(x, y).to_vec();
        let label = u8::from(x + y > 0.0);
        samples.push((feat, label));
    }
    let weights = vec![1.0, 1.0, 0.0, 0.0];
    (samples, weights)
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_image_encoder_dinov2_lp")?;
    let (samples, weights) = fixture();
    let lp_acc = enc::linear_probe_accuracy(&samples, &weights);
    let oracle = 1.0; // synthetic oracle achieves perfect accuracy
    let ratio = lp_acc / oracle;
    println!(
        "✓ DINOv2 linear probe: {:.3} accuracy ({:.0}% of oracle)",
        lp_acc,
        ratio * 100.0
    );
    assert!(ratio >= 0.9);
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
        let acc = enc::linear_probe_accuracy(&s, &w);
        assert!(acc >= 0.9);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Anti-aligned weights → ratio < 0.9 (equivalent to predicting opposite).
        let (s, _) = fixture();
        let acc = enc::linear_probe_accuracy(&s, &[-1.0, -1.0, 0.0, 0.0]);
        assert!(acc < 0.5);
    }

    #[test]
    fn deterministic_across_runs() {
        let (s, w) = fixture();
        assert_eq!(
            enc::linear_probe_accuracy(&s, &w),
            enc::linear_probe_accuracy(&s, &w)
        );
    }
}
