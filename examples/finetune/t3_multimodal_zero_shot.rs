//! # Tier 3.5 — Zero-shot prompt classifier (llama family)
//!
//! Falsifier: zero-shot prompt-based classifier picks the class with highest
//! per-class log-likelihood ranking on a deterministic fixture.
//!
//! Run with: cargo run --example t3_multimodal_zero_shot

use apr_cookbook::finetune::multimodal as mm;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const LOG_PROBS: [f64; 4] = [-2.5, -1.0, -3.0, -0.5];
const EXPECTED_CLASS: usize = 3;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t3_multimodal_zero_shot")?;
    let pred = mm::zero_shot_predict(&LOG_PROBS);
    println!(
        "✓ zero-shot: log_probs={:?} → predicted class {}",
        LOG_PROBS, pred
    );
    assert_eq!(pred, EXPECTED_CLASS);
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
        assert_eq!(mm::zero_shot_predict(&LOG_PROBS), EXPECTED_CLASS);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Different log-probs → different argmax.
        let perturbed = [-0.1, -1.0, -2.0, -3.0];
        assert_eq!(mm::zero_shot_predict(&perturbed), 0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = mm::zero_shot_predict(&LOG_PROBS);
        let b = mm::zero_shot_predict(&LOG_PROBS);
        assert_eq!(a, b);
    }
}
