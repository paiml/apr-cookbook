//! # Tier 4.8 — Async GRPO (qwen3 family)
//!
//! Falsifier: rollouts on stale policy weighted by importance ratio;
//! gradient bias bounded by |1−r| ≤ ε for all ratios.
//!
//! Run with: cargo run --example t4_async_grpo

use apr_cookbook::finetune::tier4_closeout as t4c;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const RATIOS: [f64; 5] = [1.05, 0.95, 1.0, 1.02, 0.98];
const EPS: f64 = 0.1;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_async_grpo")?;
    let bounded = t4c::async_grpo_bias_bounded(&RATIOS, EPS);
    println!(
        "✓ Async GRPO: ratios {:?} ε={EPS} bounded={}",
        RATIOS, bounded
    );
    assert!(bounded);
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
        assert!(t4c::async_grpo_bias_bounded(&RATIOS, EPS));
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        let unbounded = [2.0, 1.0, 0.5];
        assert!(!t4c::async_grpo_bias_bounded(&unbounded, EPS));
    }

    #[test]
    fn deterministic_across_runs() {
        let a = t4c::async_grpo_bias_bounded(&RATIOS, EPS);
        let b = t4c::async_grpo_bias_bounded(&RATIOS, EPS);
        assert_eq!(a, b);
    }
}
