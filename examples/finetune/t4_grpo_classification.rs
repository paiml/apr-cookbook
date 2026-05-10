//! # Tier 4.4 — GRPO with accuracy reward (mistral family)
//!
//! Falsifier: GRPO with accuracy reward on multi-choice — top-1 accuracy
//! increases over training vs SFT baseline (modeled as monotone trajectory).
//!
//! Run with: cargo run --example t4_grpo_classification

use apr_cookbook::finetune::rl_alignment as rl;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const SFT_BASELINE: f64 = 0.55;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_grpo_classification")?;
    let traj = rl::grpo_simulate_trajectory(40, SFT_BASELINE, 0.005);
    let final_ = *traj.last().unwrap();
    println!(
        "✓ GRPO classification: SFT={:.3} → GRPO final={:.3}",
        SFT_BASELINE, final_
    );
    assert!(final_ > SFT_BASELINE);
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
        let traj = rl::grpo_simulate_trajectory(40, SFT_BASELINE, 0.005);
        assert!(*traj.last().unwrap() > SFT_BASELINE);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Zero slope — accuracy doesn't improve.
        let traj = rl::grpo_simulate_trajectory(40, SFT_BASELINE, 0.0);
        assert_eq!(*traj.last().unwrap(), SFT_BASELINE);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rl::grpo_simulate_trajectory(40, SFT_BASELINE, 0.005);
        let b = rl::grpo_simulate_trajectory(40, SFT_BASELINE, 0.005);
        assert_eq!(a, b);
    }
}
