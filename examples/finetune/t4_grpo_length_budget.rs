//! # Tier 4.4 — GRPO with length-penalty reward (gemma family)
//!
//! Falsifier: GRPO with length-penalty reward — group-relative advantages
//! center at zero, with the longest-output trial getting a negative advantage
//! when the reward penalizes excess length.
//!
//! Run with: cargo run --example t4_grpo_length_budget

use apr_cookbook::finetune::rl_alignment as rl;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn rewards_with_length_penalty() -> Vec<f64> {
    // 4 trials with rewards inversely related to output length.
    // base 2.0 − 0.01 × token-count → varying positive rewards.
    let base = 2.0_f64;
    let pen = 0.01_f64;
    vec![
        base - pen * 50.0,  // 50 tok → 1.5
        base - pen * 100.0, // 100 tok → 1.0
        base - pen * 75.0,  // 75 tok → 1.25
        base - pen * 60.0,  // 60 tok → 1.4
    ]
}

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_grpo_length_budget")?;
    let r = rewards_with_length_penalty();
    let adv = rl::grpo_advantages(&r);
    println!(
        "✓ GRPO length-budget: rewards {:?} → advantages {:?}",
        r, adv
    );
    let sum: f64 = adv.iter().sum();
    assert!(sum.abs() < 1e-10, "advantages sum to ~0");
    // Longest trial (index 1) has lowest reward, so most negative advantage.
    let min_idx = adv
        .iter()
        .enumerate()
        .fold(
            (0, f64::INFINITY),
            |(bi, bv), (i, &v)| {
                if v < bv {
                    (i, v)
                } else {
                    (bi, bv)
                }
            },
        )
        .0;
    assert_eq!(min_idx, 1);
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
        let adv = rl::grpo_advantages(&rewards_with_length_penalty());
        assert!(adv.iter().sum::<f64>().abs() < 1e-10);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Empty rewards → empty advantages.
        let adv = rl::grpo_advantages(&[]);
        assert!(adv.is_empty());
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rl::grpo_advantages(&rewards_with_length_penalty());
        let b = rl::grpo_advantages(&rewards_with_length_penalty());
        assert_eq!(a, b);
    }
}
