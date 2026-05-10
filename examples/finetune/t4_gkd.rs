//! # Tier 4.10 — GKD on-policy distillation (gemma family)
//!
//! Falsifier: GKD on-policy distillation — student-teacher KL drops
//! monotonically across simulated training steps.
//!
//! Run with: cargo run --example t4_gkd

use apr_cookbook::finetune::tier4_closeout as t4c;
use apr_cookbook::recipe::RecipeContext;
use apr_cookbook::Result;

const INITIAL_KL: f64 = 0.5;
const DECAY: f64 = 0.05;
const N_STEPS: u32 = 100;

fn main() -> Result<()> {
    let _ctx = RecipeContext::new("t4_gkd")?;
    let traj = t4c::gkd_kl_trajectory(INITIAL_KL, DECAY, N_STEPS);
    println!(
        "✓ GKD: KL {:.4} → {:.4} over {} steps",
        traj[0],
        traj.last().unwrap(),
        N_STEPS
    );
    for w in traj.windows(2) {
        assert!(w[1] <= w[0]);
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
        let traj = t4c::gkd_kl_trajectory(INITIAL_KL, DECAY, N_STEPS);
        for w in traj.windows(2) {
            assert!(w[1] <= w[0]);
        }
    }

    #[test]
    fn falsifier_breaks_on_perturbed_input() {
        // Zero decay → KL constant.
        let traj = t4c::gkd_kl_trajectory(INITIAL_KL, 0.0, N_STEPS);
        for v in &traj {
            assert_eq!(*v, INITIAL_KL);
        }
    }

    #[test]
    fn deterministic_across_runs() {
        let a = t4c::gkd_kl_trajectory(INITIAL_KL, DECAY, N_STEPS);
        let b = t4c::gkd_kl_trajectory(INITIAL_KL, DECAY, N_STEPS);
        assert_eq!(a, b);
    }
}
