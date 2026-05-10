//! # Tier 4.5 — PPO adaptive KL coefficient (qwen3 family)
//!
//! Falsifier: PPO adaptive KL coefficient adjusts within target range —
//! coef doubles when KL > 2× target, halves when KL < 0.5× target.
//!
//! Run with: cargo run --example t4_rlhf_ppo_qwen

use apr_cookbook::finetune::rl_alignment as rl;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const TARGET_KL: f64 = 0.1;
const INITIAL_COEF: f64 = 0.2;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_rlhf_ppo_qwen")?;
    let high = rl::ppo_adapt_kl_coef(INITIAL_COEF, 0.5, TARGET_KL);
    let low = rl::ppo_adapt_kl_coef(INITIAL_COEF, 0.01, TARGET_KL);
    let mid = rl::ppo_adapt_kl_coef(INITIAL_COEF, 0.1, TARGET_KL);
    println!(
        "✓ PPO adaptive KL: high→{:.3} (×2), low→{:.3} (×0.5), mid→{:.3} (=)",
        high, low, mid
    );
    assert_eq!(high, INITIAL_COEF * 2.0);
    assert_eq!(low, INITIAL_COEF * 0.5);
    assert_eq!(mid, INITIAL_COEF);
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
        assert_eq!(
            rl::ppo_adapt_kl_coef(INITIAL_COEF, 0.5, TARGET_KL),
            INITIAL_COEF * 2.0
        );
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // KL exactly at target → no change.
        assert_eq!(
            rl::ppo_adapt_kl_coef(INITIAL_COEF, TARGET_KL, TARGET_KL),
            INITIAL_COEF
        );
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rl::ppo_adapt_kl_coef(INITIAL_COEF, 0.5, TARGET_KL);
        let b = rl::ppo_adapt_kl_coef(INITIAL_COEF, 0.5, TARGET_KL);
        assert_eq!(a, b);
    }
}
