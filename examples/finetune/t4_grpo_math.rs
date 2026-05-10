//! # Tier 4.4 — GRPO on math (qwen3 family)
//!
//! Falsifier: GRPO on math problems with verifiable rewards — reward
//! trajectory grows monotonically over 50 simulated steps.
//!
//! Run with: cargo run --example t4_grpo_math

use apr_cookbook::finetune::rl_alignment as rl;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const N_STEPS: u32 = 50;
const BASE: f64 = 0.0;
const SLOPE: f64 = 0.02;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_grpo_math")?;
    let traj = rl::grpo_simulate_trajectory(N_STEPS, BASE, SLOPE);
    println!(
        "✓ GRPO math: {} steps, reward {:.3} → {:.3}",
        N_STEPS,
        traj[0],
        traj.last().unwrap()
    );
    for w in traj.windows(2) {
        assert!(w[1] >= w[0]);
    }
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
        let traj = rl::grpo_simulate_trajectory(N_STEPS, BASE, SLOPE);
        for w in traj.windows(2) {
            assert!(w[1] >= w[0]);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Negative slope — trajectory decreases.
        let traj = rl::grpo_simulate_trajectory(N_STEPS, BASE, -0.01);
        for w in traj.windows(2) {
            assert!(w[1] <= w[0]);
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let a = rl::grpo_simulate_trajectory(N_STEPS, BASE, SLOPE);
        let b = rl::grpo_simulate_trajectory(N_STEPS, BASE, SLOPE);
        assert_eq!(a, b);
    }
}
