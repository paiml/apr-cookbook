//! # Tier 4.5 — RLHF/PPO with clipping (llama family)
//!
//! Falsifier: PPO clipping at ε=0.2 keeps the importance ratio in
//! [0.8, 1.2] regardless of the raw policy ratio.
//!
//! Run with: cargo run --example t4_rlhf_ppo_llama

use apr_cookbook::finetune::rl_alignment as rl;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const EPS: f64 = 0.2;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_rlhf_ppo_llama")?;
    for &ratio in &[0.5_f64, 0.7, 1.0, 1.5, 3.0] {
        let clipped = rl::ppo_clipped_ratio(ratio, EPS);
        assert!((1.0 - EPS..=1.0 + EPS).contains(&clipped));
    }
    println!(
        "✓ PPO clip ε={EPS} keeps ratio ∈ [{}, {}]",
        1.0 - EPS,
        1.0 + EPS
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
        for &r in &[0.5_f64, 1.0, 2.0] {
            let c = rl::ppo_clipped_ratio(r, EPS);
            assert!((1.0 - EPS..=1.0 + EPS).contains(&c));
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // ε=0 means clipping to {1.0} only.
        assert_eq!(rl::ppo_clipped_ratio(2.0, 0.0), 1.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rl::ppo_clipped_ratio(1.5, EPS);
        let b = rl::ppo_clipped_ratio(1.5, EPS);
        assert_eq!(a, b);
    }
}
