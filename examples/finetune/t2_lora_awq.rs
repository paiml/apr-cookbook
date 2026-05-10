//! # Tier 2.7 — LoRA on AWQ 4-bit base (mistral family)
//!
//! Falsifier: AWQ 4-bit activation-aware quantization preserves salient
//! channels (top-k by activation magnitude). Recipe checks that the salient
//! channel — selected as the one with highest |activation| — is recovered
//! with bounded magnitude.
//!
//! Run with: cargo run --example t2_lora_awq

use apr_cookbook::finetune::quantized_base as q;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn fixture_weights() -> Vec<f64> {
    vec![0.1, 0.2, 0.3, 0.4, 5.0]
}
fn fixture_activations() -> Vec<f64> {
    vec![0.1, 0.2, 0.3, 0.4, 10.0]
}
const TOP_K: usize = 1;
const SALIENT_LOWER: f64 = 4.0;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_lora_awq")?;
    let max_salient = q::awq_max_salient_error(&fixture_weights(), &fixture_activations(), TOP_K);
    println!(
        "✓ AWQ top-{} salient channel preserved at magnitude {:.3}",
        TOP_K, max_salient
    );
    assert!(
        max_salient > SALIENT_LOWER,
        "AWQ must preserve salient channel weight"
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
        let m = q::awq_max_salient_error(&fixture_weights(), &fixture_activations(), TOP_K);
        assert!(m > SALIENT_LOWER);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // All activations zero → no "salient" signal beyond magnitude 0.4 max.
        let acts = vec![0.0; 5];
        let m = q::awq_max_salient_error(&fixture_weights(), &acts, TOP_K);
        assert!(m <= SALIENT_LOWER);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = q::awq_max_salient_error(&fixture_weights(), &fixture_activations(), TOP_K);
        let b = q::awq_max_salient_error(&fixture_weights(), &fixture_activations(), TOP_K);
        assert_eq!(a, b);
    }
}
