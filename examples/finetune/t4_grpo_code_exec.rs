//! # Tier 4.4 — GRPO with code-execution reward (phi family)
//!
//! Falsifier: GRPO with code-execution reward raises % runnable code by
//! ≥ 20 percentage points over the simulated trajectory.
//!
//! Run with: cargo run --example t4_grpo_code_exec

use apr_cookbook::finetune::rl_alignment as rl;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_grpo_code_exec")?;
    let traj = rl::grpo_simulate_trajectory(50, 0.40, 0.01);
    let initial = traj[0];
    let final_ = *traj.last().unwrap();
    let lift_pp = (final_ - initial) * 100.0;
    println!(
        "✓ GRPO code-exec: runnable% {:.1}% → {:.1}% (+{:.1}pp)",
        initial * 100.0,
        final_ * 100.0,
        lift_pp
    );
    assert!(lift_pp >= 20.0);
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
        let traj = rl::grpo_simulate_trajectory(50, 0.40, 0.01);
        let lift_pp = (traj.last().unwrap() - traj[0]) * 100.0;
        assert!(lift_pp >= 20.0);
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Tiny slope → lift < 20 pp.
        let traj = rl::grpo_simulate_trajectory(50, 0.40, 0.001);
        let lift_pp = (traj.last().unwrap() - traj[0]) * 100.0;
        assert!(lift_pp < 20.0);
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rl::grpo_simulate_trajectory(50, 0.40, 0.01);
        let b = rl::grpo_simulate_trajectory(50, 0.40, 0.01);
        assert_eq!(a, b);
    }
}
