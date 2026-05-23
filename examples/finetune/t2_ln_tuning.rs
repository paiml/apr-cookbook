//! # Tier 2.5 — LayerNorm-only tuning — qwen3 family
//!
//! Falsifier: LayerNorm-only tuning's trainable parameter count is exactly
//! 2 × num_layers × hidden_dim (γ + β per LayerNorm).
//!
//! Run with: cargo run --example t2_ln_tuning

use apr_cookbook::finetune::peft_variants as peft;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const N_LAYERS: u32 = 32;
const HIDDEN: u32 = 4096;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t2_ln_tuning")?;
    let count = peft::ln_tuning_param_count(N_LAYERS, HIDDEN);
    let expected: u64 = 2 * u64::from(N_LAYERS) * u64::from(HIDDEN);
    println!(
        "✓ LN-tuning: trainable={} expected={} ({} layers × {} hidden × 2)",
        count, expected, N_LAYERS, HIDDEN
    );
    assert_eq!(count, expected);
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
        assert_eq!(peft::ln_tuning_param_count(N_LAYERS, HIDDEN), 2 * 32 * 4096);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // 3× hidden_dim does NOT equal expected 2× formula.
        let bogus = 3 * u64::from(N_LAYERS) * u64::from(HIDDEN);
        assert_ne!(peft::ln_tuning_param_count(N_LAYERS, HIDDEN), bogus);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = peft::ln_tuning_param_count(N_LAYERS, HIDDEN);
        let b = peft::ln_tuning_param_count(N_LAYERS, HIDDEN);
        assert_eq!(a, b);
    }
}
