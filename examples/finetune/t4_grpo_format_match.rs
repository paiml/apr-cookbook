//! # Tier 4.4 — GRPO with regex-format reward (llama family)
//!
//! Falsifier: GRPO with regex-format reward — % JSON-conforming outputs
//! ≥ 95% after 30 simulated steps.
//!
//! Run with: cargo run --example t4_grpo_format_match

use apr_cookbook::finetune::rl_alignment as rl;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_grpo_format_match")?;
    let traj = rl::grpo_simulate_trajectory(30, 0.50, 0.02);
    let final_ = *traj.last().unwrap();
    println!(
        "✓ GRPO format: % conforming after 30 steps = {:.1}%",
        final_ * 100.0
    );
    assert!(final_ >= 0.95);
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
        let traj = rl::grpo_simulate_trajectory(30, 0.50, 0.02);
        assert!(*traj.last().unwrap() >= 0.95);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Slow slope — final < 95%.
        let traj = rl::grpo_simulate_trajectory(30, 0.50, 0.005);
        assert!(*traj.last().unwrap() < 0.95);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rl::grpo_simulate_trajectory(30, 0.50, 0.02);
        let b = rl::grpo_simulate_trajectory(30, 0.50, 0.02);
        assert_eq!(a, b);
    }
}
